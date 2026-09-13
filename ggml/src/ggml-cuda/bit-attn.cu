#include "bit-attn.cuh"

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

bool ggml_cuda_bit_attention_supported(const ggml_tensor * op) {
    GGML_UNUSED(op);
    return false;
}
void ggml_cuda_op_bit_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("BitAttention CUDA kernels require NVIDIA; use CPU fallback");
}
void ggml_cuda_op_bit_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("BitAttention CUDA kernels require NVIDIA; use CPU fallback");
}
void ggml_cuda_op_bit_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED_VARS(ctx, dst);
    GGML_ABORT("BitAttention CUDA kernels require NVIDIA; use CPU fallback");
}

#else

// Layout is passed by value: views/non-contiguous Q, K, V and masks retain their
// byte strides. Packed words use I32 storage but unsigned bitwise arithmetic.
struct bit_view {
    const char * data;
    int64_t ne[4];
    size_t nb[4];
    int type;
};

static bit_view bit_make_view(const ggml_tensor * t) {
    bit_view a = {};
    if (t) {
        a.data = static_cast<const char *>(t->data);
        for (int i = 0; i < 4; ++i) {
            a.ne[i] = t->ne[i];
            a.nb[i] = t->nb[i];
        }
        a.type = t->type;
    }
    return a;
}

static __device__ __forceinline__ const char * bit_ptr(
        const bit_view & a, int64_t d, int64_t n, int64_t h, int64_t b) {
    return a.data + d*a.nb[0] + n*a.nb[1] + h*a.nb[2] + b*a.nb[3];
}

static __device__ __forceinline__ float bit_load(
        const bit_view & a, int64_t d, int64_t n, int64_t h, int64_t b) {
    const char * p = bit_ptr(a, d, n, h, b);
    if (a.type == GGML_TYPE_F16) {
        return __half2float(*reinterpret_cast<const half *>(p));
    }
    if (a.type == GGML_TYPE_BF16) {
        return __uint_as_float(uint32_t(*reinterpret_cast<const uint16_t *>(p)) << 16);
    }
    return *reinterpret_cast<const float *>(p);
}

static __device__ __forceinline__ float bit_dot(
        const bit_view & q, const bit_view & k, int d, int64_t iq, int64_t ik, int64_t h, int64_t b) {
    const int64_t kh = h/(q.ne[2]/k.ne[2]);
    const int64_t kb = b/(q.ne[3]/k.ne[3]);
    const uint32_t tail = (d & 31) ? (uint32_t(1) << (d & 31)) - 1 : 0xffffffffu;
    int distance = 0;
    for (int64_t w = 0; w < q.ne[0]; ++w) {
        const uint32_t qw = *reinterpret_cast<const uint32_t *>(bit_ptr(q, w, iq, h, b));
        const uint32_t kw = *reinterpret_cast<const uint32_t *>(bit_ptr(k, w, ik, kh, kb));
        uint32_t diff = qw ^ kw;
        if (w + 1 == q.ne[0]) {
            diff &= tail;
        }
        distance += __popc(diff);
    }
    return float(d) - 2.0f*float(distance);
}

static __global__ void bit_pack_kernel(bit_view x, uint32_t * out, int64_t words, int64_t total) {
    // One complete warp per word; even padding lanes participate in the ballot.
    const int lane = threadIdx.x & 31;
    const int64_t word = (int64_t(blockIdx.x)*blockDim.x + threadIdx.x)/32;
    if (word >= total) {
        return; // uniform for all 32 lanes
    }
    const int64_t row = word/words;
    const int64_t d = (word % words)*32 + lane;
    const int64_t n = row % x.ne[1];
    const int64_t h = (row/x.ne[1]) % x.ne[2];
    const int64_t b = row/(x.ne[1]*x.ne[2]);
    const bool sign = d < x.ne[0] && bit_load(x, d, n, h, b) >= 0.0f;
    const uint32_t packed = __ballot_sync(0xffffffffu, sign);
    if (lane == 0) {
        out[word] = packed;
    }
}

static __global__ void bit_mul_mat_kernel(bit_view k, bit_view q, float * out, int d, int64_t total) {
    for (int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x; i < total;
            i += int64_t(gridDim.x)*blockDim.x) {
        const int64_t ik = i % k.ne[1];
        const int64_t row = i/k.ne[1];
        const int64_t iq = row % q.ne[1];
        const int64_t h = (row/q.ne[1]) % q.ne[2];
        const int64_t b = row/(q.ne[1]*q.ne[2]);
        out[i] = bit_dot(q, k, d, iq, ik, h, b);
    }
}

static __device__ __forceinline__ float bit_warp_max(float x) {
    for (int delta = 16; delta; delta >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, delta));
    }
    return x;
}
static __device__ __forceinline__ float bit_warp_sum(float x) {
    for (int delta = 16; delta; delta >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, delta);
    }
    return x;
}

// Reference fused CUDA kernel: a warp owns one query and streams 32-key tiles.
// Scores/probabilities stay in registers. V and softmax/accumulation remain F32;
// no Tq*Tk global tensor is created. This is not a Tensor-Core-tuned speed claim.
template <int DV_PAD>
static __global__ void bit_attn_kernel(
        bit_view q, bit_view k, bit_view v, bit_view mask, bit_view sinks, float * out,
        int d, float scale, float bias, float cap, int64_t rows) {
    const int lane = threadIdx.x & 31;
    const int64_t row = (int64_t(blockIdx.x)*blockDim.x + threadIdx.x)/32;
    if (row >= rows) {
        return; // whole warp exits; all live lanes use the same full-warp mask
    }
    const int64_t iq = row % q.ne[1];
    const int64_t h = (row/q.ne[1]) % q.ne[2];
    const int64_t b = row/(q.ne[1]*q.ne[2]);
    const int64_t vh = h/(q.ne[2]/v.ne[2]);
    const int64_t vb = b/(q.ne[3]/v.ne[3]);
    int64_t h_pow2 = 1;
    while (h_pow2 <= q.ne[2]/2) {
        h_pow2 *= 2;
    }
    const float m0 = powf(2.0f, -bias/float(h_pow2));
    const float m1 = powf(2.0f, -0.5f*bias/float(h_pow2));
    const float slope = bias == 0.0f ? 1.0f : h < h_pow2
        ? powf(m0, float(h + 1)) : powf(m1, float(2*(h - h_pow2) + 1));
    float m = sinks.data ? bit_load(sinks, h, 0, 0, 0) : -CUDART_INF_F;
    float sum = m == -CUDART_INF_F ? 0.0f : 1.0f;
    float acc[DV_PAD/32] = {};
    for (int64_t start = 0; start < k.ne[1]; start += 32) {
        const int64_t ik = start + lane;
        float score = -CUDART_INF_F;
        if (ik < k.ne[1]) {
            const float mv = mask.data ? bit_load(mask, ik, iq, h % mask.ne[2], b % mask.ne[3]) : 0.0f;
            if (mv != -CUDART_INF_F) {
                score = scale*bit_dot(q, k, d, iq, ik, h, b);
                if (cap > 0.0f) {
                    score = cap*tanhf(score/cap);
                }
                score += slope*mv;
            }
        }
        const float m_new = fmaxf(m, bit_warp_max(score));
        const float alpha = m == -CUDART_INF_F ? 0.0f : expf(m - m_new);
        const float p = score == -CUDART_INF_F ? 0.0f : expf(score - m_new);
        sum = alpha*sum + bit_warp_sum(p);
#pragma unroll
        for (int j = 0; j < DV_PAD/32; ++j) {
            acc[j] *= alpha;
        }
        for (int t = 0; t < 32 && start + t < k.ne[1]; ++t) {
            const float pt = __shfl_sync(0xffffffffu, p, t);
            if (pt != 0.0f) {
#pragma unroll
                for (int j = 0; j < DV_PAD/32; ++j) {
                    const int dv = lane + 32*j;
                    if (dv < v.ne[0]) {
                        acc[j] += pt*bit_load(v, dv, start + t, vh, vb);
                    }
                }
            }
        }
        m = m_new;
    }
    const float inv_sum = sum > 0.0f ? 1.0f/sum : 0.0f;
#pragma unroll
    for (int j = 0; j < DV_PAD/32; ++j) {
        const int dv = lane + 32*j;
        if (dv < v.ne[0]) {
            out[dv + v.ne[0]*(h + q.ne[2]*(iq + q.ne[1]*b))] = acc[j]*inv_sum;
        }
    }
}

static bool bit_float_type(ggml_type t) {
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16;
}

bool ggml_cuda_bit_attention_supported(const ggml_tensor * op) {
    if (!ggml_is_contiguous(op)) {
        return false;
    }
    switch (op->op) {
        case GGML_OP_BIT_PACK:
            return bit_float_type(op->src[0]->type) && op->type == GGML_TYPE_I32;
        case GGML_OP_BIT_MUL_MAT:
            return op->src[0]->type == GGML_TYPE_I32 && op->src[1]->type == GGML_TYPE_I32;
        case GGML_OP_BIT_ATTN_EXT:
            return op->src[0]->type == GGML_TYPE_I32 && op->src[1]->type == GGML_TYPE_I32 &&
                bit_float_type(op->src[2]->type) && op->src[2]->ne[0] <= 1024 &&
                (!op->src[3] || bit_float_type(op->src[3]->type));
        default:
            return false;
    }
}

void ggml_cuda_op_bit_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t total = ggml_nelements(dst);
    const int64_t blocks = (total + 7)/8;
    GGML_ASSERT(blocks <= INT32_MAX);
    bit_pack_kernel<<<static_cast<unsigned>(blocks), 256, 0, ctx.stream()>>>(
        bit_make_view(dst->src[0]), static_cast<uint32_t *>(dst->data), dst->ne[0], total);
    CUDA_CHECK(cudaGetLastError());
}

void ggml_cuda_op_bit_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t total = ggml_nelements(dst);
    const int blocks = int(std::min<int64_t>((total + 255)/256, 65535));
    bit_mul_mat_kernel<<<blocks, 256, 0, ctx.stream()>>>(
        bit_make_view(dst->src[0]), bit_make_view(dst->src[1]), static_cast<float *>(dst->data),
        ggml_get_op_params_i32(dst, 0), total);
    CUDA_CHECK(cudaGetLastError());
}

template <int D>
static void bit_attn_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int64_t rows = ggml_nrows(dst->src[0]);
    const int64_t blocks = (rows + 3)/4;
    GGML_ASSERT(blocks <= INT32_MAX);
    bit_attn_kernel<D><<<static_cast<unsigned>(blocks), 128, 0, ctx.stream()>>>(
        bit_make_view(dst->src[0]), bit_make_view(dst->src[1]), bit_make_view(dst->src[2]),
        bit_make_view(dst->src[3]), bit_make_view(dst->src[4]), static_cast<float *>(dst->data),
        ggml_get_op_params_i32(dst, 3), ggml_get_op_params_f32(dst, 0),
        ggml_get_op_params_f32(dst, 1), ggml_get_op_params_f32(dst, 2), rows);
}

void ggml_cuda_op_bit_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_bit_attention_supported(dst));
    const int64_t d = dst->src[2]->ne[0];
    if      (d <=   32) { bit_attn_launch<  32>(ctx, dst); }
    else if (d <=   64) { bit_attn_launch<  64>(ctx, dst); }
    else if (d <=  128) { bit_attn_launch< 128>(ctx, dst); }
    else if (d <=  256) { bit_attn_launch< 256>(ctx, dst); }
    else if (d <=  512) { bit_attn_launch< 512>(ctx, dst); }
    else               { bit_attn_launch<1024>(ctx, dst); }
    CUDA_CHECK(cudaGetLastError());
}
#endif
