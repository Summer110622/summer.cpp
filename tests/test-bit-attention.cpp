#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const std::string & message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

struct input {
    ggml_tensor * base;
    ggml_tensor * t;
    std::vector<uint8_t> data;

    input(ggml_context * ctx, ggml_type type, int d, int n, int h, int b, int layout = 0) {
        if (layout == 1) {
            base = ggml_new_tensor_4d(ctx, type, d, h, n, b);
            t = ggml_permute(ctx, base, 0, 2, 1, 3);
        } else if (layout == 2) {
            base = ggml_new_tensor_4d(ctx, type, n, d, h, b);
            t = ggml_permute(ctx, base, 1, 0, 2, 3);
        } else {
            base = t = ggml_new_tensor_4d(ctx, type, d, n, h, b);
        }
        data.resize(ggml_nbytes(base));
    }
    size_t offset(int64_t d, int64_t n, int64_t h, int64_t b) const {
        return d*t->nb[0] + n*t->nb[1] + h*t->nb[2] + b*t->nb[3];
    }
    void set(int64_t d, int64_t n, int64_t h, int64_t b, float value) {
        uint8_t * p = data.data() + offset(d, n, h, b);
        if (t->type == GGML_TYPE_F32) {
            memcpy(p, &value, sizeof(value));
        } else if (t->type == GGML_TYPE_F16) {
            const auto x = ggml_fp32_to_fp16(value); memcpy(p, &x, sizeof(x));
        } else {
            const auto x = ggml_fp32_to_bf16(value); memcpy(p, &x, sizeof(x));
        }
    }
    float get(int64_t d, int64_t n, int64_t h, int64_t b) const {
        const uint8_t * p = data.data() + offset(d, n, h, b);
        if (t->type == GGML_TYPE_F32) {
            float x; memcpy(&x, p, sizeof(x)); return x;
        }
        if (t->type == GGML_TYPE_F16) {
            ggml_fp16_t x; memcpy(&x, p, sizeof(x)); return ggml_fp16_to_fp32(x);
        }
        ggml_bf16_t x; memcpy(&x, p, sizeof(x)); return ggml_bf16_to_fp32(x);
    }
    void randomize(std::mt19937 & rng) {
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
        for (int64_t b = 0; b < t->ne[3]; ++b)
        for (int64_t h = 0; h < t->ne[2]; ++h)
        for (int64_t n = 0; n < t->ne[1]; ++n)
        for (int64_t d = 0; d < t->ne[0]; ++d) {
            set(d, n, h, b, dist(rng));
        }
        // Zero convention, the sign bit and subnormal underflow are intentional.
        set(0, 0, 0, 0, -0.0f);
        if (t->ne[0] > 1) { set(1, 0, 0, 0, -1.0e-20f); }
        if (t->ne[0] > 31) { set(31, 0, 0, 0, 1.0f); }
    }
    void upload() const { ggml_backend_tensor_set(base, data.data(), 0, data.size()); }
    std::vector<uint32_t> packed() const {
        const int64_t words = (t->ne[0] + 31)/32;
        std::vector<uint32_t> out(size_t(words*ggml_nrows(t)), 0);
        for (int64_t b = 0; b < t->ne[3]; ++b)
        for (int64_t h = 0; h < t->ne[2]; ++h)
        for (int64_t n = 0; n < t->ne[1]; ++n)
        for (int64_t d = 0; d < t->ne[0]; ++d) {
            if (get(d, n, h, b) >= 0.0f) {
                out[size_t(d/32 + words*(n + t->ne[1]*(h + t->ne[2]*b)))] |= uint32_t(1) << (d & 31);
            }
        }
        return out;
    }
};

static std::vector<float> read_f32(ggml_tensor * t) {
    std::vector<float> a(size_t(ggml_nelements(t)));
    ggml_backend_tensor_get(t, a.data(), 0, ggml_nbytes(t));
    return a;
}

static void run_case(ggml_backend_t backend, int d, ggml_type type, int id) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({2*1024*1024, nullptr, true}), ggml_free);
    require(bool(ctx), "ggml_init failed");
    const int nq = id % 3 == 0 ? 1 : 5;
    const int nk = 37;
    const int hq = 6, hk = 2, hv = 3, bq = 4;
    const int dv = id % 4 == 0 ? 65 : 17;
    const int kind = id % 5;
    const bool use_sink = id % 3 == 0;
    const float scale = 1.0f/std::sqrt(float(d));
    const float bias = kind == 4 ? 8.0f : 0.0f;
    const float cap = id % 4 == 1 ? 1.75f : 0.0f;
    input q(ctx.get(), type, d, nq, hq, bq, id % 3);
    input k(ctx.get(), type, d, nk, hk, 2, (id + 1) % 3);
    input v(ctx.get(), type, dv, nk, hv, 1, (id + 2) % 3);
    input mask(ctx.get(), id % 2 ? GGML_TYPE_F16 : GGML_TYPE_F32, nk + 3, nq + 2, 2, 2, id % 3);
    input sinks(ctx.get(), GGML_TYPE_F32, hq, 1, 1, 1);
    std::mt19937 rng(110622 + id);
    q.randomize(rng); k.randomize(rng); v.randomize(rng);
    for (int b = 0; b < 2; ++b)
    for (int h = 0; h < 2; ++h)
    for (int iq = 0; iq < nq + 2; ++iq)
    for (int ik = 0; ik < nk + 3; ++ik) {
        bool blocked = ik > (nk - nq + iq); // absolute decode offset, NOT upper-left causal
        if (kind == 2 && iq == 0) { blocked = true; }
        if (kind == 3) { blocked = ik < 34; } // first 32-key tile is completely masked
        const float finite_bias = kind == 4 ? -0.02f*float(ik + 2*h + b) : 0.0f;
        mask.set(ik, iq, h, b, blocked ? -INFINITY : finite_bias);
    }
    for (int h = 0; h < hq; ++h) { sinks.set(h, 0, 0, 0, h == 0 ? -INFINITY : 0.25f*h); }
    auto * qb = ggml_bit_pack(ctx.get(), q.t);
    auto * kb = ggml_bit_pack(ctx.get(), k.t);
    auto * dot = ggml_bit_mul_mat(ctx.get(), kb, qb, d);
    auto * out = ggml_bit_attn_ext(ctx.get(), qb, kb, v.t, kind ? mask.t : nullptr,
                                  use_sink ? sinks.t : nullptr, d, scale, bias, cap);
    // Also test externally packed tensors with deliberately different garbage in padding.
    auto * qext = ggml_dup_tensor(ctx.get(), qb);
    auto * kext = ggml_dup_tensor(ctx.get(), kb);
    auto * out_ext = ggml_bit_attn_ext(ctx.get(), qext, kext, v.t, kind ? mask.t : nullptr,
                                      use_sink ? sinks.t : nullptr, d, scale, bias, cap);
    auto * dot_ext = ggml_bit_mul_mat(ctx.get(), kext, qext, d);
    auto * gf = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(gf, dot);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, out_ext);
    ggml_build_forward_expand(gf, dot_ext);
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        auto * node = ggml_graph_node(gf, i);
        require(ggml_backend_supports_op(backend, node), std::string("unsupported op: ") + ggml_op_name(node->op));
    }
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    require(bool(buffer), "backend allocation failed");
    ggml_backend_buffer_clear(buffer.get(), 0xa5);
    q.upload(); k.upload(); v.upload(); mask.upload(); sinks.upload();
    auto qwords = q.packed(), kwords = k.packed();
    auto dirty_q = qwords;
    if (d & 31) {
        const uint32_t garbage = ~((uint32_t(1) << (d & 31)) - 1);
        const size_t words = size_t((d + 31)/32);
        for (size_t i = words - 1; i < dirty_q.size(); i += words) { dirty_q[i] |= garbage; }
    }
    ggml_backend_tensor_set(qext, dirty_q.data(), 0, ggml_nbytes(qext));
    ggml_backend_tensor_set(kext, kwords.data(), 0, ggml_nbytes(kext));
    require(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS, "graph compute failed");
    std::vector<uint32_t> actual_q(qwords.size()), actual_k(kwords.size());
    ggml_backend_tensor_get(qb, actual_q.data(), 0, ggml_nbytes(qb));
    ggml_backend_tensor_get(kb, actual_k.data(), 0, ggml_nbytes(kb));
    require(actual_q == qwords && actual_k == kwords, "packing/strides/zero/padding mismatch");
    const auto dots = read_f32(dot), dots_ext = read_f32(dot_ext);
    const auto result = read_f32(out), result_ext = read_f32(out_ext);
    double max_error = 0;
    for (int b = 0; b < bq; ++b)
    for (int h = 0; h < hq; ++h)
    for (int iq = 0; iq < nq; ++iq) {
        std::vector<double> scores(nk, -INFINITY);
        const double sink = use_sink ? sinks.get(h, 0, 0, 0) : -INFINITY;
        double maximum = sink;
        // Same ALiBi head convention as ggml FlashAttention, including non-powers of two.
        const float m0 = std::pow(2.0f, -bias/4.0f), m1 = std::pow(2.0f, -bias/8.0f);
        const float slope = bias == 0 ? 1.0f : h < 4 ? std::pow(m0, float(h+1)) : std::pow(m1, float(2*(h-4)+1));
        for (int ik = 0; ik < nk; ++ik) {
            int reference_dot = 0;
            for (int j = 0; j < d; ++j) {
                const bool sq = q.get(j, iq, h, b) >= 0;
                const bool sk = k.get(j, ik, h/(hq/hk), b/(bq/2)) >= 0;
                reference_dot += sq == sk ? 1 : -1;
            }
            const size_t index = size_t(ik + nk*(iq + nq*(h + hq*b)));
            require(dots[index] == reference_dot && dots_ext[index] == reference_dot, "binary dot/GQA/padding mismatch");
            const float mv = kind ? mask.get(ik, iq, h%2, b%2) : 0.0f;
            if (mv == -INFINITY) { continue; }
            double s = double(scale)*reference_dot;
            if (cap > 0) { s = double(cap)*std::tanh(s/double(cap)); }
            scores[ik] = s + double(slope)*mv;
            maximum = std::max(maximum, scores[ik]);
        }
        double denominator = 0;
        if (sink != -INFINITY) { denominator += std::exp(sink - maximum); }
        for (double s : scores) { if (s != -INFINITY) { denominator += std::exp(s - maximum); } }
        for (int j = 0; j < dv; ++j) {
            double expected = 0;
            for (int ik = 0; ik < nk; ++ik) {
                if (scores[ik] != -INFINITY) {
                    expected += std::exp(scores[ik] - maximum)*v.get(j, ik, h/(hq/hv), 0);
                }
            }
            if (denominator > 0) { expected /= denominator; }
            const size_t index = size_t(j + dv*(h + hq*(iq + nq*b)));
            require(std::isfinite(result[index]) && std::isfinite(result_ext[index]), "non-finite output (all-masked row?)");
            const double error = std::max(std::abs(result[index] - expected), std::abs(result_ext[index] - expected));
            max_error = std::max(max_error, error);
            require(error < 5e-5, "attention error exceeded 5e-5: " + std::to_string(error));
        }
    }
    std::printf("PASS case=%d D=%d dtype=%s layout=%d mask=%d max_abs_error=%.3g\n",
                id, d, ggml_type_name(type), id%3, kind, max_error);
}

int main(int argc, char ** argv) {
    const char * name = "CPU";
    if (argc == 3 && std::strcmp(argv[1], "--backend") == 0) {
        name = argv[2];
    } else if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--backend CPU|CUDA0]\n", argv[0]); return 1;
    }
    ggml_backend_load_all();
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_init_by_name(name, nullptr), ggml_backend_free);
    if (!backend) {
        std::fprintf(stderr, "backend not available: %s\n", name); return 77;
    }
    if (ggml_backend_is_cpu(backend.get())) { ggml_backend_cpu_set_n_threads(backend.get(), 3); }
    try {
        int id = 0;
        for (int d : {1, 7, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 257}) {
            for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
                run_case(backend.get(), d, type, id++);
            }
        }
        std::printf("All %d BitAttention configurations passed on %s.\n", id, name);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
    return 0;
}
