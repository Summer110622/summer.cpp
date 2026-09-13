#include "llama.h"
#include "ggml.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

// Tiny deterministic, randomly initialized Llama fixture. This tests wiring and
// cache correctness, not language-model quality. No downloaded weights required.
static void write_model(const std::string & path) {
    constexpr int e = 64, ff = 128, heads = 4, kv = 2, vocab = 32, layers = 2;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({4*1024*1024, nullptr, false}), ggml_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_empty(), gguf_free);
    check(bool(ctx) && bool(meta), "fixture allocation failed");
    gguf_set_val_str(meta.get(), "general.architecture", "llama");
    gguf_set_val_str(meta.get(), "general.name", "BitAttention integration fixture (random)");
    gguf_set_val_u32(meta.get(), "llama.context_length", 128);
    gguf_set_val_u32(meta.get(), "llama.embedding_length", e);
    gguf_set_val_u32(meta.get(), "llama.block_count", layers);
    gguf_set_val_u32(meta.get(), "llama.feed_forward_length", ff);
    gguf_set_val_u32(meta.get(), "llama.attention.head_count", heads);
    gguf_set_val_u32(meta.get(), "llama.attention.head_count_kv", kv);
    gguf_set_val_f32(meta.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);
    gguf_set_val_u32(meta.get(), "llama.rope.dimension_count", e/heads);
    gguf_set_val_f32(meta.get(), "llama.rope.freq_base", 10000.0f);
    gguf_set_val_str(meta.get(), "tokenizer.ggml.model", "llama");
    std::vector<std::string> tokens = {"<unk>", "<s>", "</s>"};
    for (int i = 3; i < vocab; ++i) { tokens.push_back("token" + std::to_string(i)); }
    std::vector<const char *> token_ptrs;
    for (const auto & token : tokens) { token_ptrs.push_back(token.c_str()); }
    std::vector<float> scores(vocab, 0);
    std::vector<int32_t> types(vocab, 1);
    types[0] = 2; types[1] = types[2] = 3;
    gguf_set_arr_str(meta.get(), "tokenizer.ggml.tokens", token_ptrs.data(), vocab);
    gguf_set_arr_data(meta.get(), "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), vocab);
    gguf_set_arr_data(meta.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), vocab);
    gguf_set_val_u32(meta.get(), "tokenizer.ggml.unknown_token_id", 0);
    gguf_set_val_u32(meta.get(), "tokenizer.ggml.bos_token_id", 1);
    gguf_set_val_u32(meta.get(), "tokenizer.ggml.eos_token_id", 2);
    std::mt19937 rng(110622);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    auto add = [&](const std::string & name, int n0, int n1 = 1) {
        ggml_tensor * t = n1 == 1 ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n0)
                                 : ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n0, n1);
        ggml_set_name(t, name.c_str());
        auto * data = static_cast<float *>(t->data);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) { data[i] = n1 == 1 ? 1.0f : dist(rng); }
        gguf_add_tensor(meta.get(), t);
    };
    add("token_embd.weight", e, vocab);
    add("output_norm.weight", e);
    add("output.weight", e, vocab);
    for (int i = 0; i < layers; ++i) {
        const auto prefix = "blk." + std::to_string(i) + ".";
        add(prefix + "attn_norm.weight", e);
        add(prefix + "attn_q.weight", e, e);
        add(prefix + "attn_k.weight", e, (e/heads)*kv);
        add(prefix + "attn_v.weight", e, (e/heads)*kv);
        add(prefix + "attn_output.weight", e, e);
        add(prefix + "ffn_norm.weight", e);
        add(prefix + "ffn_gate.weight", e, ff);
        add(prefix + "ffn_up.weight", e, ff);
        add(prefix + "ffn_down.weight", ff, e);
    }
    check(gguf_write_to_file(meta.get(), path.c_str(), false), "writing fixture failed");
}

struct observed { int bit = 0; int pack = 0; };
static bool observe(ggml_tensor * t, bool ask, void * user) {
    auto & count = *static_cast<observed *>(user);
    if (ask) {
        if (t->op == GGML_OP_BIT_ATTN_EXT || t->op == GGML_OP_BIT_MUL_MAT) { ++count.bit; }
        if (t->op == GGML_OP_BIT_PACK) { ++count.pack; }
    }
    return false; // no tensor download needed
}

static std::vector<float> decode(llama_model * model, bool bit, bool flash, int seqs, int chunk) {
    auto params = llama_context_default_params();
    params.n_ctx = 128;
    params.n_batch = 32;
    params.n_ubatch = 32;
    params.n_seq_max = seqs;
    params.n_threads = params.n_threads_batch = 2;
    params.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.bit_attn = bit;
    observed counts;
    params.cb_eval = observe;
    params.cb_eval_user_data = &counts;
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, params), llama_free);
    check(bool(ctx), "context creation failed");
    llama_batch batch = llama_batch_init(32, 0, seqs);
    std::vector<float> last(size_t(32*seqs));
    for (int pos = 0; pos < 6; pos += chunk) {
        batch.n_tokens = 0;
        for (int s = 0; s < seqs; ++s)
        for (int p = pos; p < std::min(pos + chunk, 6); ++p) {
            const int i = batch.n_tokens++;
            batch.token[i] = 3 + (p + 3*s)%29;
            batch.pos[i] = p;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = s;
            batch.logits[i] = true;
        }
        const int result = llama_decode(ctx.get(), batch);
        if (result != 0) { llama_batch_free(batch); throw std::runtime_error("llama_decode failed"); }
        const int tokens_per_seq = std::min(chunk, 6 - pos);
        for (int s = 0; s < seqs; ++s) {
            const float * logits = llama_get_logits_ith(ctx.get(), (s + 1)*tokens_per_seq - 1);
            check(logits != nullptr, "missing logits");
            for (int i = 0; i < 32; ++i) {
                check(std::isfinite(logits[i]), "non-finite logits");
                last[size_t(s*32 + i)] = logits[i];
            }
        }
    }
    llama_batch_free(batch);
    check(bit ? counts.bit > 0 && counts.pack > 0 : counts.bit == 0 && counts.pack == 0,
          "per-context opt-in did not select the expected ggml ops");
    return last;
}

int main(int argc, char ** argv) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path()/ ("test-bit-attention-" + std::to_string(stamp) + ".gguf");
    try {
        const std::string file = argc == 3 && std::string(argv[1]) == "--write-fixture" ? argv[2] : path.string();
        write_model(file);
        if (argc == 3 && std::string(argv[1]) == "--write-fixture") { return 0; }
        check(argc == 1, "usage: test-bit-attention-model [--write-fixture FILE]");
        llama_backend_init();
        check(!llama_context_default_params().bit_attn, "BitAttention must be disabled by default");
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(file.c_str(), mp), llama_model_free);
        check(bool(model), "loading fixture failed");
        for (bool flash : {false, true}) {
            for (int seqs : {1, 2}) {
                const auto bit_token = decode(model.get(), true, flash, seqs, 1);
                const auto bit_chunk = decode(model.get(), true, flash, seqs, 3);
                for (size_t i = 0; i < bit_token.size(); ++i) {
                    check(std::abs(bit_token[i] - bit_chunk[i]) < 3e-4f,
                          "prefill/decode logits differ: causal offset or KV cache wiring bug");
                }
                decode(model.get(), false, flash, seqs, 3); // disabled path regression
                std::printf("PASS model flash=%d sequences=%d (token/chunk decode and disabled path)\n", flash, seqs);
            }
        }
        model.reset();
        llama_backend_free();
        std::filesystem::remove(path);
    } catch (const std::exception & e) {
        std::error_code ignored; std::filesystem::remove(path, ignored);
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
    return 0;
}
