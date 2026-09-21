#include "llama-graph-opt.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static int cases = 0;
static double max_error = 0.0;

static void check(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct arena {
    ggml_context * ctx;
    ggml_cgraph * graph;
    explicit arena(bool no_alloc = false) {
        ctx = ggml_init({16 * 1024 * 1024, nullptr, no_alloc});
        check(ctx != nullptr, "ggml_init failed");
        graph = ggml_new_graph_custom(ctx, 4096, false);
    }
    ~arena() { ggml_free(ctx); }
    arena(const arena &) = delete;
    arena & operator=(const arena &) = delete;
    void run(int threads) {
        check(ggml_graph_compute_with_ctx(ctx, graph, threads) == GGML_STATUS_SUCCESS, "CPU graph failed");
    }
};

static void fill(ggml_tensor * tensor, float phase = 0.0f) {
    check(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor), "invalid test input");
    float * data = static_cast<float *>(tensor->data);
    for (int64_t i = 0; i < ggml_nelements(tensor); ++i) {
        data[i] = 0.4f * std::sin(0.73f * float(i) + phase);
    }
}

static void close(ggml_tensor * reference, ggml_tensor * actual, const char * name) {
    check(ggml_are_same_shape(reference, actual), "output shape mismatch");
    check(ggml_is_contiguous(reference) && ggml_is_contiguous(actual), "output must be contiguous");
    const auto * a = static_cast<const float *>(reference->data);
    const auto * b = static_cast<const float *>(actual->data);
    for (int64_t i = 0; i < ggml_nelements(reference); ++i) {
        const double error = std::fabs(double(a[i]) - double(b[i]));
        max_error = std::max(max_error, error);
        if (!std::isfinite(a[i]) || !std::isfinite(b[i]) || error > 2e-5 + 2e-5 * std::fabs(a[i])) {
            throw std::runtime_error(std::string(name) + " mismatch at " + std::to_string(i) +
                ": " + std::to_string(a[i]) + " versus " + std::to_string(b[i]));
        }
    }
    ++cases;
}

static void test_options() {
    check(llama_graph_parse_chunk(nullptr) == 0, "missing option");
    check(llama_graph_parse_chunk("") == 0, "empty option");
    check(llama_graph_parse_chunk("0") == 0, "disabled option");
    check(llama_graph_parse_chunk("064") == 64, "decimal option");
    check(llama_graph_parse_chunk("2147483647") == INT32_MAX, "maximum option");
    for (const char * value : {"-1", "+1", " 1", "1 ", "1x", "2147483648", "99999999999999999999"}) {
        check(llama_graph_parse_chunk(value) == -1, "invalid option accepted");
    }
    check(llama_moe_topk_softmax_safe(16384, 1), "clamp boundary must be eligible");
    check(!llama_moe_topk_softmax_safe(16385, 1), "clamp semantics must not change");
    check(!llama_moe_topk_softmax_safe(4, 5), "invalid top-k accepted");
    check(!llama_moe_topk_softmax_safe(0, 0), "empty top-k accepted");
    check(llama_moe_topk_softmax_safe(INT64_MAX, INT64_MAX), "overflow in eligibility check");
    check(llama_graph_query_chunk(513, 1) == 33, "query graph growth must be bounded");
    check(llama_graph_query_chunk(11, 0) == 11, "disabled query splitting");
    check(llama_graph_attn_set_safe(1024, 128), "ordinary output rejected");
    check(!llama_graph_attn_set_safe(1 << 20, 256), "unsafe SET offset accepted");
    check(!llama_graph_attn_set_safe(INT64_MAX, INT64_MAX), "overflow in SET check");
    ++cases;
}

static void test_router(int64_t experts, int64_t used, int64_t tokens, int pattern, int threads) {
    arena a;
    auto * logits = ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, experts, tokens);
    fill(logits, 1.3f);
    auto * values = static_cast<float *>(logits->data);
    for (int64_t i = 0; i < experts * tokens; ++i) {
        if (pattern == 1) { values[i] = 10000.0f + 0.125f * float(i % experts); }
        if (pattern == 2) { values[i] = -10000.0f + 0.125f * float(i % experts); }
        if (pattern == 3) { values[i] = 0.0f; } // exact ties
    }
    auto * probs = ggml_soft_max(a.ctx, logits);
    auto * ids = ggml_argsort_top_k(a.ctx, probs, int(used));
    auto * weights = ggml_get_rows(a.ctx, ggml_reshape_3d(a.ctx, probs, 1, experts, tokens), ids);
    weights = ggml_reshape_2d(a.ctx, weights, used, tokens);
    auto * sum = ggml_clamp(a.ctx, ggml_sum_rows(a.ctx, weights), 6.103515625e-5f, INFINITY);
    weights = ggml_reshape_3d(a.ctx, ggml_div(a.ctx, weights, sum), 1, used, tokens);
    const auto fast = llama_build_topk_softmax(a.ctx, logits, used);
    ggml_build_forward_expand(a.graph, weights);
    ggml_build_forward_expand(a.graph, fast.weights);
    a.run(threads);
    close(weights, fast.weights, "top-k softmax");
    // Top-K views can have a row stride larger than K; do not compare raw contiguous bytes.
    for (int64_t t = 0; t < tokens; ++t) {
        check(std::memcmp(static_cast<const char *>(ids->data) + t * ids->nb[1],
                          static_cast<const char *>(fast.ids->data) + t * fast.ids->nb[1],
                          size_t(used) * sizeof(int32_t)) == 0, "router IDs differ");
    }
}

static void test_experts(int activation, bool merged, bool weight_before, int64_t chunk,
                         int64_t tokens, int64_t active, int threads) {
    arena a;
    constexpr int64_t d = 12, f = 16, e = 8, selected = 5;
    auto * input = ggml_new_tensor_3d(a.ctx, GGML_TYPE_F32, d, 1, tokens);
    auto * up_w = ggml_new_tensor_3d(a.ctx, GGML_TYPE_F32, d, f, e);
    auto * gate_w = ggml_new_tensor_3d(a.ctx, GGML_TYPE_F32, d, f, e);
    auto * merged_w = ggml_new_tensor_3d(a.ctx, GGML_TYPE_F32, d, 2 * f, e);
    auto * down_w = ggml_new_tensor_3d(a.ctx, GGML_TYPE_F32, f, d, e);
    auto * up_b = ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, f, e);
    auto * gate_b = ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, f, e);
    auto * down_b = ggml_new_tensor_2d(a.ctx, GGML_TYPE_F32, d, e);
    auto * scale = ggml_new_tensor_3d(a.ctx, GGML_TYPE_F32, 1, e, 1);
    auto * lora_a = ggml_new_tensor_3d(a.ctx, GGML_TYPE_F32, d, 3, e);
    auto * lora_b = ggml_new_tensor_3d(a.ctx, GGML_TYPE_F32, 3, f, e);
    float phase = 0.0f;
    for (auto * t : {input, up_w, gate_w, merged_w, down_w, up_b, gate_b, down_b, scale, lora_a, lora_b}) {
        fill(t, phase += 0.17f);
    }
    // Use padded backing rows like argsort_top_k; compact slicing is part of the regression.
    auto * padded_ids = ggml_new_tensor_2d(a.ctx, GGML_TYPE_I32, e, tokens);
    auto * ids_data = static_cast<int32_t *>(padded_ids->data);
    for (int64_t t = 0; t < tokens; ++t) {
        for (int64_t slot = 0; slot < e; ++slot) {
            ids_data[t * e + slot] = int32_t((slot + 3 * t) % e);
        }
    }
    auto * ids = ggml_view_2d(a.ctx, padded_ids, selected, tokens, padded_ids->nb[1], 0);
    auto * weights = ggml_new_tensor_3d(a.ctx, GGML_TYPE_F32, 1, selected, tokens);
    for (int64_t i = 0; i < selected * tokens; ++i) {
        static_cast<float *>(weights->data)[i] = float(1 + i % selected) / 15.0f;
    }
    auto build = [&](ggml_tensor * route, ggml_tensor * w, int64_t count) {
        auto * x = input;
        if (weight_before) {
            x = ggml_mul(a.ctx, ggml_repeat_4d(a.ctx, x, d, count, tokens, 1), w);
        }
        ggml_tensor * up;
        ggml_tensor * gate;
        if (merged) {
            auto * both = ggml_mul_mat_id(a.ctx, merged_w, x, route);
            gate = ggml_view_3d(a.ctx, both, f, count, tokens, both->nb[1], both->nb[2], 0);
            up = ggml_view_3d(a.ctx, both, f, count, tokens, both->nb[1], both->nb[2], f * both->nb[0]);
        } else {
            up = ggml_mul_mat_id(a.ctx, up_w, x, route);
            gate = ggml_mul_mat_id(a.ctx, gate_w, x, route);
            auto * delta = ggml_mul_mat_id(a.ctx, lora_b, ggml_mul_mat_id(a.ctx, lora_a, x, route), route);
            up = ggml_add(a.ctx, up, ggml_scale(a.ctx, delta, 0.25f));
        }
        up = ggml_add_id(a.ctx, up, up_b, route);
        gate = ggml_add_id(a.ctx, gate, gate_b, route);
        auto * s = ggml_get_rows(a.ctx, ggml_repeat_4d(a.ctx, scale, 1, e, tokens, 1), route);
        up = ggml_mul(a.ctx, up, s);
        switch (activation) {
            case 0: x = ggml_swiglu_split(a.ctx, gate, up); break;
            case 1: x = ggml_geglu_split(a.ctx, gate, up); break;
            case 2: x = ggml_reglu_split(a.ctx, gate, up); break;
            case 3: x = ggml_swiglu_oai(a.ctx, gate, up, 1.702f, 7.0f); break;
            case 4: x = ggml_sqr(a.ctx, ggml_relu(a.ctx, up)); break;
            case 5: {
                auto * act = ggml_mul(a.ctx, ggml_tanh(a.ctx, gate), ggml_sigmoid(a.ctx, gate));
                x = ggml_mul(a.ctx, act, ggml_scale(a.ctx, ggml_tanh(a.ctx, ggml_scale(a.ctx, up, 0.5f)), 2.0f));
            } break;
            case 6: x = ggml_swiglu_clamp(a.ctx, gate, up, 0.2f); break;
            default: throw std::runtime_error("unknown test activation");
        }
        x = ggml_mul_mat_id(a.ctx, down_w, x, route);
        x = ggml_add_id(a.ctx, x, down_b, route);
        x = ggml_mul(a.ctx, x, s);
        return weight_before ? x : ggml_mul(a.ctx, x, w);
    };
    auto * all = build(ids, weights, selected);
    ggml_tensor * reference = nullptr;
    for (int64_t i = 0; i < active; ++i) {
        auto * part = ggml_view_2d(a.ctx, all, d, tokens, all->nb[2], i * all->nb[1]);
        reference = reference ? ggml_add(a.ctx, reference, part) : part;
    }
    reference = ggml_cont(a.ctx, reference);
    ggml_build_forward_expand(a.graph, reference);
    auto * actual = llama_build_expert_chunks(a.ctx, a.graph, ids, weights, active, chunk, build);
    ggml_build_forward_expand(a.graph, actual);
    a.run(threads);
    close(reference, actual, "expert chunks");
}

static ggml_tensor * attention_math(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k,
        ggml_tensor * v, ggml_tensor * mask, ggml_tensor * sinks, float softcap, float alibi) {
    auto * scores = ggml_mul_mat(ctx, k, q);
    ggml_prec_set_acc(scores, GGML_PREC_F32);
    if (softcap > 0) {
        scores = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, scores, 1.0f / softcap)), softcap);
    }
    auto * probs = ggml_soft_max_ext(ctx, scores, mask, 1.0f / std::sqrt(float(q->ne[0])), alibi);
    ggml_soft_max_add_sinks(probs, sinks);
    return ggml_mul_mat(ctx, v, probs);
}

static void test_attention(int64_t kv_heads, int64_t streams, int mask_kind, bool half_mask,
                           bool with_sinks, float softcap, int64_t requested, int threads) {
    arena a;
    constexpr int64_t d = 8, dv = 6, heads = 4, queries = 11, keys = 17, padded_q = 16;
    auto * q_raw = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, d, heads, queries, streams);
    auto * k_raw = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, d, kv_heads, keys, streams);
    auto * v = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, keys, dv, kv_heads, streams);
    fill(q_raw, 0.1f); fill(k_raw, 0.3f); fill(v, 0.5f);
    auto * q = ggml_permute(a.ctx, q_raw, 0, 2, 1, 3);
    auto * k = ggml_permute(a.ctx, k_raw, 0, 2, 1, 3);
    ggml_tensor * mask = nullptr;
    if (mask_kind != 0) {
        mask = ggml_new_tensor_4d(a.ctx, half_mask ? GGML_TYPE_F16 : GGML_TYPE_F32, keys, padded_q, 1, streams);
        for (int64_t stream = 0; stream < streams; ++stream) {
            for (int64_t qi = 0; qi < padded_q; ++qi) {
                const int64_t position = keys - queries + qi;
                for (int64_t ki = 0; ki < keys; ++ki) {
                    const bool masked = ki > position || (mask_kind == 2 && ki < position - 3);
                    const float value = masked ? -INFINITY : float(ki - position - stream);
                    const int64_t index = (stream * padded_q + qi) * keys + ki;
                    if (half_mask) { static_cast<ggml_fp16_t *>(mask->data)[index] = ggml_fp32_to_fp16(value); }
                    else { static_cast<float *>(mask->data)[index] = value; }
                }
            }
        }
    }
    ggml_tensor * sinks = nullptr;
    if (with_sinks) { sinks = ggml_new_tensor_1d(a.ctx, GGML_TYPE_F32, heads); fill(sinks, 1.0f); }
    auto build = [&](ggml_tensor * query, ggml_tensor * query_mask) {
        return attention_math(a.ctx, query, k, v, query_mask, sinks, softcap, mask ? 4.0f : 0.0f);
    };
    auto * reference = ggml_permute(a.ctx, build(q, mask), 0, 2, 1, 3);
    reference = ggml_cont_2d(a.ctx, reference, dv * heads, queries * streams);
    ggml_build_forward_expand(a.graph, reference);
    int chunks = 0;
    auto * actual = llama_build_attn_query_chunks(a.ctx, a.graph, q, mask, requested, build,
        [&](ggml_tensor *) { ++chunks; });
    check(chunks <= LLAMA_ATTN_MAX_QUERY_CHUNKS, "too many query chunks");
    ggml_build_forward_expand(a.graph, actual);
    a.run(threads);
    close(reference, actual, "query chunks");
}

static size_t attention_buffer(int64_t requested) {
    arena a(true);
    constexpr int64_t d = 32, heads = 8, kv_heads = 2, queries = 512, keys = 2048;
    auto * q = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, d, queries, heads, 1);
    auto * k = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, d, keys, kv_heads, 1);
    auto * v = ggml_new_tensor_4d(a.ctx, GGML_TYPE_F32, keys, d, kv_heads, 1);
    for (auto * t : {q, k, v}) { ggml_set_input(t); }
    auto build = [&](ggml_tensor * query, ggml_tensor * mask) {
        return attention_math(a.ctx, query, k, v, mask, nullptr, 0.0f, 0.0f);
    };
    ggml_tensor * out;
    if (requested > 0) {
        out = llama_build_attn_query_chunks(a.ctx, a.graph, q, nullptr, requested, build, [](ggml_tensor *) {});
    } else {
        out = ggml_cont_2d(a.ctx, ggml_permute(a.ctx, build(q, nullptr), 0, 2, 1, 3), d * heads, queries);
    }
    ggml_set_output(out);
    ggml_build_forward_expand(a.graph, out);
    auto allocator = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    check(allocator != nullptr, "allocator creation failed");
    const bool ok = ggml_gallocr_reserve(allocator, a.graph);
    const size_t bytes = ggml_gallocr_get_buffer_size(allocator, 0);
    ggml_gallocr_free(allocator);
    check(ok, "graph reservation failed");
    return bytes;
}


// Exercise the actual scheduler/allocator, including buffer reuse on repeated input batches.
static std::vector<float> scheduled_pipeline(bool optimized, int threads) {
    arena a(true);
    constexpr int64_t d = 32, f = 48, e = 8, used = 3, heads = 4, kv_heads = 2;
    constexpr int64_t queries = 11, streams = 2, tokens = queries * streams, keys = 17;
    std::vector<ggml_tensor *> inputs;
    auto input = [&](std::initializer_list<int64_t> shape) {
        std::vector<int64_t> dims(shape);
        auto * t = ggml_new_tensor(a.ctx, GGML_TYPE_F32, int(dims.size()), dims.data());
        ggml_set_input(t);
        inputs.push_back(t);
        return t;
    };
    auto * x = input({d, tokens});
    auto * router = input({d, e});
    auto * up = input({d, f, e});
    auto * gate = input({d, f, e});
    auto * down = input({f, d, e});
    auto * k_raw = input({d / heads, kv_heads, keys, streams});
    auto * v = input({keys, d / heads, kv_heads, streams});
    auto * mask = input({keys, queries, 1, streams});
    auto * logits = ggml_mul_mat(a.ctx, router, x);
    llama_moe_routing routing;
    if (optimized) {
        routing = llama_build_topk_softmax(a.ctx, logits, used);
    } else {
        auto * p = ggml_soft_max(a.ctx, logits);
        routing.ids = ggml_argsort_top_k(a.ctx, p, used);
        auto * w = ggml_get_rows(a.ctx, ggml_reshape_3d(a.ctx, p, 1, e, tokens), routing.ids);
        w = ggml_reshape_2d(a.ctx, w, used, tokens);
        w = ggml_div(a.ctx, w, ggml_clamp(a.ctx, ggml_sum_rows(a.ctx, w), 6.103515625e-5f, INFINITY));
        routing.weights = ggml_reshape_3d(a.ctx, w, 1, used, tokens);
    }
    auto * x3 = ggml_reshape_3d(a.ctx, x, d, 1, tokens);
    auto build_experts = [&](ggml_tensor * ids, ggml_tensor * weights, int64_t) {
        auto * g = ggml_mul_mat_id(a.ctx, gate, x3, ids);
        auto * u = ggml_mul_mat_id(a.ctx, up, x3, ids);
        auto * act = ggml_swiglu_split(a.ctx, g, u);
        return ggml_mul(a.ctx, ggml_mul_mat_id(a.ctx, down, act, ids), weights);
    };
    ggml_tensor * moe = nullptr;
    if (optimized) {
        moe = llama_build_expert_chunks(a.ctx, a.graph, routing.ids, routing.weights, used, 2, build_experts);
    } else {
        auto * all = build_experts(routing.ids, routing.weights, used);
        for (int64_t i = 0; i < used; ++i) {
            auto * part = ggml_view_2d(a.ctx, all, d, tokens, all->nb[2], i * all->nb[1]);
            moe = moe ? ggml_add(a.ctx, moe, part) : part;
        }
    }
    auto * q = ggml_permute(a.ctx, ggml_reshape_4d(a.ctx, moe, d / heads, heads, queries, streams), 0, 2, 1, 3);
    auto * k = ggml_permute(a.ctx, k_raw, 0, 2, 1, 3);
    auto build_attention = [&](ggml_tensor * query, ggml_tensor * query_mask) {
        return attention_math(a.ctx, query, k, v, query_mask, nullptr, 2.0f, 0.0f);
    };
    ggml_tensor * out;
    if (optimized) {
        out = llama_build_attn_query_chunks(a.ctx, a.graph, q, mask, 3, build_attention, [](ggml_tensor *) {});
    } else {
        out = ggml_cont_2d(a.ctx, ggml_permute(a.ctx, build_attention(q, mask), 0, 2, 1, 3), d, tokens);
    }
    ggml_set_output(out);
    ggml_build_forward_expand(a.graph, out);
    ggml_backend_t backend = ggml_backend_cpu_init();
    check(backend != nullptr, "CPU backend creation failed");
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend_owner(backend, ggml_backend_free);
    ggml_backend_cpu_set_n_threads(backend, threads);
    auto * scheduler = ggml_backend_sched_new(&backend, nullptr, 1, 4096, false, true);
    check(scheduler != nullptr, "scheduler creation failed");
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> owner(scheduler, ggml_backend_sched_free);
    check(ggml_backend_sched_alloc_graph(scheduler, a.graph), "scheduler allocation failed");
    std::vector<float> result;
    for (int pass = 0; pass < 3; ++pass) {
        for (size_t index = 0; index < inputs.size(); ++index) {
            auto * t = inputs[index];
            std::vector<float> data(size_t(ggml_nelements(t)), 0.0f);
            for (size_t i = 0; i < data.size(); ++i) {
                if (t == mask) {
                    const int64_t key = int64_t(i % keys);
                    const int64_t query = int64_t(i / keys) % queries;
                    data[i] = key > keys - queries + query ? -INFINITY : 0.0f;
                } else {
                    data[i] = 0.4f * std::sin(0.31f * float(i) + float(index) + 0.5f * pass);
                }
            }
            ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
        }
        check(ggml_backend_sched_graph_compute(scheduler, a.graph) == GGML_STATUS_SUCCESS, "scheduled compute failed");
        const size_t start = result.size();
        result.resize(start + size_t(ggml_nelements(out)));
        ggml_backend_tensor_get(out, result.data() + start, 0, ggml_nbytes(out));
    }
    return result;
}

static void test_scheduler(int threads) {
    const auto reference = scheduled_pipeline(false, threads);
    const auto actual = scheduled_pipeline(true, threads);
    check(reference.size() == actual.size(), "scheduled output size differs");
    for (size_t i = 0; i < actual.size(); ++i) {
        const double error = std::fabs(double(reference[i]) - double(actual[i]));
        max_error = std::max(max_error, error);
        check(std::isfinite(actual[i]) && error <= 2e-5 + 2e-5 * std::fabs(reference[i]), "scheduled pipeline differs");
    }
    ++cases;
}

int main() {
    try {
        test_options();
        for (int threads : {1, 4}) {
            test_scheduler(threads);
            for (int64_t tokens : {int64_t(1), int64_t(7)}) {
                for (int pattern = 0; pattern < 4; ++pattern) {
                    for (int64_t used : {int64_t(1), int64_t(4), int64_t(16)}) {
                        test_router(16, used, tokens, pattern, threads);
                    }
                }
                test_router(128, 8, tokens, 0, threads);
                for (int activation = 0; activation < 7; ++activation) {
                    for (bool merged : {false, true}) {
                        for (bool weight_before : {false, true}) {
                            test_experts(activation, merged, weight_before, 2, tokens, 5, threads);
                        }
                    }
                }
                for (int64_t chunk : {int64_t(1), int64_t(3), int64_t(8)}) {
                    test_experts(0, false, false, chunk, tokens, 3, threads); // warmup subset
                }
                test_experts(0, false, false, 1, tokens, 1, threads);
            }
            for (int64_t kv_heads : {int64_t(1), int64_t(2), int64_t(4)}) {
                for (int64_t streams : {int64_t(1), int64_t(2)}) {
                    for (int mask = 0; mask < 3; ++mask) {
                        for (bool half : {false, true}) {
                            test_attention(kv_heads, streams, mask, half, false, 0.0f, 3, threads);
                            test_attention(kv_heads, streams, mask, half, true, 2.0f, 4, threads);
                        }
                    }
                }
            }
            for (int64_t chunk : {int64_t(1), int64_t(11), int64_t(64)}) {
                test_attention(2, 2, 1, false, true, 0.0f, chunk, threads);
            }
        }
        const size_t dense = attention_buffer(0);
        const size_t chunked = attention_buffer(64);
        check(chunked < dense, "query chunks did not reduce the synthetic graph buffer");
        ++cases;
        std::printf("PASS: %d numerical/guard/allocation cases; maximum absolute difference %.9g\n", cases, max_error);
        std::printf("Synthetic CPU attention graph buffer: dense=%zu bytes, chunk64=%zu bytes (%.2f%% less)\n",
            dense, chunked, 100.0 * (1.0 - double(chunked) / double(dense)));
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL after %d cases: %s\n", cases, error.what());
        return 1;
    }
    return 0;
}
