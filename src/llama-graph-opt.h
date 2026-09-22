#pragma once

#include "ggml.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Bound graph growth even when a very small query chunk is requested.
static constexpr int64_t LLAMA_ATTN_MAX_QUERY_CHUNKS = 16;

inline int32_t llama_graph_parse_chunk(const char * value) {
    if (value == nullptr || *value == '\0') {
        return 0;
    }
    int32_t result = 0;
    for (const char * p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || result > (INT32_MAX - (*p - '0')) / 10) {
            return -1;
        }
        result = result * 10 + (*p - '0');
    }
    return result;
}

inline bool llama_graph_env_flag(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] == '1' && value[1] == '\0';
}

inline int32_t llama_graph_env_chunk(const char * name) {
    const int32_t value = llama_graph_parse_chunk(std::getenv(name));
    if (value < 0) {
        std::fprintf(stderr, "%s: expected an integer in [0, %d]; disabled\n", name, INT32_MAX);
        return 0;
    }
    return value;
}

struct llama_graph_opt_config {
    bool    moe_topk_softmax;
    int32_t moe_expert_chunk;
    int32_t attn_query_chunk;
};

inline const llama_graph_opt_config & llama_graph_opt_get() {
    // Freeze options before graph reservation: changing them on a reused graph is unsafe.
    static const llama_graph_opt_config config = {
        llama_graph_env_flag("LLAMA_MOE_TOPK_SOFTMAX"),
        std::getenv("LLAMA_MOE_EXPERT_CHUNK") ? llama_graph_env_chunk("LLAMA_MOE_EXPERT_CHUNK") :
            (llama_graph_env_flag("LLAMA_SERIAL_EXPERTS") ? 1 : 0),
        llama_graph_env_chunk("LLAMA_ATTN_QUERY_CHUNK"),
    };
    return config;
}

inline int64_t llama_graph_query_chunk(int64_t n_queries, int64_t requested) {
    if (requested <= 0 || n_queries <= requested) {
        return n_queries;
    }
    return std::max(requested, 1 + (n_queries - 1) / LLAMA_ATTN_MAX_QUERY_CHUNKS);
}

inline bool llama_graph_attn_set_safe(int64_t width, int64_t n_queries) {
    // GGML_OP_SET stores 32-bit strides and requires its byte offset to be < 1 GiB.
    return width > 0 && n_queries > 0 &&
        width <= ((int64_t(1) << 30) - 1) / int64_t(sizeof(float)) / n_queries;
}

inline bool llama_moe_topk_softmax_safe(int64_t n_expert, int64_t n_used) {
    // Top-K probability mass is >= K/E. Preserve the legacy F16-min denominator clamp.
    // Divide instead of multiplying K by 16384, so the check cannot overflow.
    return n_expert > 0 && n_used > 0 && n_used <= n_expert &&
        n_used >= 1 + (n_expert - 1) / 16384;
}

// Layout-only packing for read-only inputs/views. Unlike ggml_cont, this is not
// an isolation copy: preserve the source dependency and let GGML track its lifetime.
// Singleton dimensions may have padded strides and still be contiguous.
inline ggml_tensor * llama_graph_cont_if_needed(ggml_context * ctx, ggml_tensor * tensor) {
    return ggml_is_contiguous(tensor) ? tensor : ggml_cont(ctx, tensor);
}

struct llama_moe_routing {
    ggml_tensor * ids;
    ggml_tensor * weights;
};

inline llama_moe_routing llama_build_topk_softmax(
        ggml_context * ctx, ggml_tensor * logits, int64_t n_used) {
    GGML_ASSERT(llama_moe_topk_softmax_safe(logits->ne[0], n_used));
    const int64_t n_tokens = logits->ne[1];
    ggml_tensor * ids = ggml_argsort_top_k(ctx, logits, n_used);
    ggml_tensor * selected = ggml_get_rows(ctx,
        ggml_reshape_3d(ctx, logits, 1, logits->ne[0], n_tokens), ids);
    ggml_tensor * weights = ggml_soft_max(ctx, ggml_reshape_2d(ctx, selected, n_used, n_tokens));
    return { ids, ggml_reshape_3d(ctx, weights, 1, n_used, n_tokens) };
}

template <typename BuildExperts>
ggml_tensor * llama_build_expert_chunks(
        ggml_context * ctx, ggml_cgraph * graph,
        ggml_tensor * ids, ggml_tensor * weights,
        int64_t n_active, int64_t chunk_size, const BuildExperts & build_experts) {
    GGML_ASSERT(chunk_size > 0 && n_active > 0 && n_active <= ids->ne[0]);
    GGML_ASSERT(weights->ne[0] == 1 && weights->ne[1] >= n_active && weights->ne[2] == ids->ne[1]);
    ggml_tensor * result = nullptr;
    for (int64_t first = 0; first < n_active; first += chunk_size) {
        const int64_t count = std::min(chunk_size, n_active - first);
        // Multi-token slot views can have gaps. Pack only when needed; decode
        // views are already contiguous and must not pay for two copy kernels.
        ggml_tensor * chunk_ids = llama_graph_cont_if_needed(ctx, ggml_view_2d(ctx, ids,
            count, ids->ne[1], ids->nb[1], first * ids->nb[0]));
        ggml_tensor * chunk_weights = llama_graph_cont_if_needed(ctx, ggml_view_3d(ctx, weights,
            1, count, weights->ne[2], weights->nb[1], weights->nb[2], first * weights->nb[1]));
        ggml_tensor * experts = build_experts(chunk_ids, chunk_weights, count);
        GGML_ASSERT(experts->ne[1] == count && experts->ne[2] == ids->ne[1]);
        ggml_build_forward_expand(graph, experts);
        // Keep the same left-to-right reduction order across chunk boundaries.
        for (int64_t slot = 0; slot < count; ++slot) {
            ggml_tensor * expert = ggml_view_2d(ctx, experts,
                experts->ne[0], experts->ne[2], experts->nb[2], slot * experts->nb[1]);
            result = result ? ggml_add(ctx, result, expert) : expert;
            ggml_build_forward_expand(graph, result);
        }
    }
    return n_active == 1 ? ggml_cont(ctx, result) : result;
}

template <typename BuildAttention, typename FinishChunk>
ggml_tensor * llama_build_attn_query_chunks(
        ggml_context * ctx, ggml_cgraph * graph,
        ggml_tensor * q, ggml_tensor * mask, int64_t requested,
        const BuildAttention & build_attention, const FinishChunk & finish_chunk) {
    // Q is [head_dim, queries, heads, streams]; masks retain global key positions.
    const int64_t n_queries = q->ne[1];
    const int64_t chunk_size = llama_graph_query_chunk(n_queries, requested);
    GGML_ASSERT(n_queries > 0 && chunk_size > 0);
    GGML_ASSERT(mask == nullptr || mask->ne[1] >= n_queries);
    ggml_tensor * result = nullptr;
    for (int64_t first = 0; first < n_queries; first += chunk_size) {
        const int64_t count = std::min(chunk_size, n_queries - first);
        ggml_tensor * chunk_q = ggml_view_4d(ctx, q, q->ne[0], count, q->ne[2], q->ne[3],
            q->nb[1], q->nb[2], q->nb[3], first * q->nb[1]);
        ggml_tensor * chunk_mask = nullptr;
        if (mask) {
            // A single head/stream slice is contiguous even with padded queries.
            // Multiple heads/streams retain gaps and still require packing.
            chunk_mask = llama_graph_cont_if_needed(ctx, ggml_view_4d(ctx, mask,
                mask->ne[0], count, mask->ne[2], mask->ne[3],
                mask->nb[1], mask->nb[2], mask->nb[3], first * mask->nb[1]));
        }
        ggml_tensor * chunk = build_attention(chunk_q, chunk_mask);
        GGML_ASSERT(chunk->ne[1] == count && chunk->ne[3] == q->ne[3]);
        // A one-query or one-head permutation is layout-only, with no copy.
        chunk = llama_graph_cont_if_needed(ctx, ggml_permute(ctx, chunk, 0, 2, 1, 3));
        chunk = ggml_reshape_4d(ctx, chunk, chunk->ne[0] * chunk->ne[1], count, 1, chunk->ne[3]);
        if (!result) {
            GGML_ASSERT(llama_graph_attn_set_safe(chunk->ne[0], n_queries));
            result = ggml_fill(ctx, ggml_new_tensor_4d(ctx, chunk->type,
                chunk->ne[0], n_queries, 1, q->ne[3]), 0.0f);
        }
        // A dependency-linked in-place set avoids quadratic concatenation/copying.
        result = ggml_set_inplace(ctx, result, chunk,
            result->nb[1], result->nb[2], result->nb[3], first * result->nb[1]);
        finish_chunk(result);
        ggml_build_forward_expand(graph, result);
    }
    return ggml_reshape_2d(ctx, result, result->ne[0], n_queries * q->ne[3]);
}
