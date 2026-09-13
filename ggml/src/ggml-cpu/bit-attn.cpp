#include "ops.h"
#include "ggml-cpu.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

// CPU reference backend. All indexing is stride-aware, including dimension 0.
// Each worker owns whole output rows; neither a score matrix nor shared scratch
// is needed by BIT_ATTN_EXT. BIT_MUL_MAT is the explicit unfused alternative.
static const char * bit_ptr(const ggml_tensor * t, int64_t d, int64_t n, int64_t h, int64_t b) {
    return static_cast<const char *>(t->data) + d*t->nb[0] + n*t->nb[1] + h*t->nb[2] + b*t->nb[3];
}

static float bit_float(const ggml_tensor * t, int64_t d, int64_t n, int64_t h, int64_t b) {
    const char * p = bit_ptr(t, d, n, h, b);
    switch (t->type) {
        case GGML_TYPE_F32:  { float       x; memcpy(&x, p, sizeof(x)); return x; }
        case GGML_TYPE_F16:  { ggml_fp16_t x; memcpy(&x, p, sizeof(x)); return ggml_fp16_to_fp32(x); }
        case GGML_TYPE_BF16: { ggml_bf16_t x; memcpy(&x, p, sizeof(x)); return ggml_bf16_to_fp32(x); }
        default: GGML_ABORT("BitAttention: unsupported floating-point type");
    }
}

static uint32_t bit_word(const ggml_tensor * t, int64_t w, int64_t n, int64_t h, int64_t b) {
    uint32_t x;
    memcpy(&x, bit_ptr(t, w, n, h, b), sizeof(x));
    return x;
}

static int bit_popcount(uint32_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_popcount(x);
#else
    // Portable fallback, with no CPU instruction-set assumption (including ARM).
    x -= (x >> 1) & 0x55555555u;
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0f0f0f0fu;
    x += x >> 8;
    x += x >> 16;
    return int(x & 0x3fu);
#endif
}

static float bit_dot(const ggml_tensor * q, const ggml_tensor * k, int32_t d,
                     int64_t iq, int64_t ik, int64_t h, int64_t b) {
    const int64_t kh = h/(q->ne[2]/k->ne[2]);
    const int64_t kb = b/(q->ne[3]/k->ne[3]);
    const uint32_t tail = (d & 31) ? (uint32_t(1) << (d & 31)) - 1 : UINT32_MAX;
    int64_t distance = 0;
    for (int64_t w = 0; w < q->ne[0]; ++w) {
        uint32_t x = bit_word(q, w, iq, h, b) ^ bit_word(k, w, ik, kh, kb);
        if (w + 1 == q->ne[0]) {
            x &= tail; // Ignore padding even for externally supplied packed words.
        }
        distance += bit_popcount(x);
    }
    return float(int64_t(d) - 2*distance);
}

void ggml_compute_forward_bit_pack(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const int64_t rows = ggml_nrows(x);
    auto * out = static_cast<uint32_t *>(dst->data);
    for (int64_t row = params->ith; row < rows; row += params->nth) {
        const int64_t n = row % x->ne[1];
        const int64_t h = (row/x->ne[1]) % x->ne[2];
        const int64_t b = row/(x->ne[1]*x->ne[2]);
        for (int64_t w = 0; w < dst->ne[0]; ++w) {
            uint32_t word = 0;
            for (int j = 0; j < 32 && 32*w + j < x->ne[0]; ++j) {
                if (bit_float(x, 32*w + j, n, h, b) >= 0.0f) {
                    word |= uint32_t(1) << j;
                }
            }
            out[row*dst->ne[0] + w] = word;
        }
    }
}

void ggml_compute_forward_bit_mul_mat(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * k = dst->src[0];
    const ggml_tensor * q = dst->src[1];
    const int32_t d = ggml_get_op_params_i32(dst, 0);
    const int64_t rows = ggml_nrows(q);
    auto * out = static_cast<float *>(dst->data);
    for (int64_t row = params->ith; row < rows; row += params->nth) {
        const int64_t iq = row % q->ne[1];
        const int64_t h  = (row/q->ne[1]) % q->ne[2];
        const int64_t b  = row/(q->ne[1]*q->ne[2]);
        for (int64_t ik = 0; ik < k->ne[1]; ++ik) {
            out[row*k->ne[1] + ik] = bit_dot(q, k, d, iq, ik, h, b);
        }
    }
}

void ggml_compute_forward_bit_attn_ext(const ggml_compute_params * params, ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const float scale = ggml_get_op_params_f32(dst, 0);
    const float bias  = ggml_get_op_params_f32(dst, 1);
    const float cap   = ggml_get_op_params_f32(dst, 2);
    const int32_t d   = ggml_get_op_params_i32(dst, 3);
    const int64_t rows = ggml_nrows(q);
    int64_t h_pow2 = 1;
    while (h_pow2 <= q->ne[2]/2) {
        h_pow2 *= 2;
    }
    const float m0 = std::pow(2.0f, -bias/float(h_pow2));
    const float m1 = std::pow(2.0f, -0.5f*bias/float(h_pow2));
    for (int64_t row = params->ith; row < rows; row += params->nth) {
        const int64_t iq = row % q->ne[1];
        const int64_t h  = (row/q->ne[1]) % q->ne[2];
        const int64_t b  = row/(q->ne[1]*q->ne[2]);
        const int64_t vh = h/(q->ne[2]/v->ne[2]);
        const int64_t vb = b/(q->ne[3]/v->ne[3]);
        const float slope = bias == 0.0f ? 1.0f : h < h_pow2
            ? std::pow(m0, float(h + 1)) : std::pow(m1, float(2*(h - h_pow2) + 1));
        auto * acc = reinterpret_cast<float *>(static_cast<char *>(dst->data)
                   + h*dst->nb[1] + iq*dst->nb[2] + b*dst->nb[3]);
        std::fill(acc, acc + v->ne[0], 0.0f);
        float m = sinks ? bit_float(sinks, h, 0, 0, 0) : -INFINITY;
        float sum = m == -INFINITY ? 0.0f : 1.0f;
        for (int64_t ik = 0; ik < k->ne[1]; ++ik) {
            const float mv = mask ? bit_float(mask, ik, iq, h % mask->ne[2], b % mask->ne[3]) : 0.0f;
            if (mv == -INFINITY) {
                continue;
            }
            float score = scale*bit_dot(q, k, d, iq, ik, h, b);
            if (cap > 0.0f) {
                score = cap*std::tanh(score/cap);
            }
            score += slope*mv;
            const float m_new = std::max(m, score);
            const float alpha = m == -INFINITY ? 0.0f : std::exp(m - m_new);
            const float p = std::exp(score - m_new);
            for (int64_t dv = 0; dv < v->ne[0]; ++dv) {
                acc[dv] = alpha*acc[dv] + p*bit_float(v, dv, ik, vh, vb);
            }
            sum = alpha*sum + p;
            m = m_new;
        }
        const float inv_sum = sum > 0.0f ? 1.0f/sum : 0.0f;
        for (int64_t dv = 0; dv < v->ne[0]; ++dv) {
            acc[dv] *= inv_sum;
        }
    }
}
