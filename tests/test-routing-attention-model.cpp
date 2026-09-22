#include "llama.h"
#include "gguf.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) { throw std::runtime_error(message); }
}

static void initialize_tensor(ggml_tensor * tensor, void *) {
    require(tensor->type == GGML_TYPE_F32, "synthetic weights must be F32");
    const char * name = ggml_get_name(tensor);
    uint32_t hash = 2166136261u;
    for (const char * p = name; *p; ++p) { hash = (hash ^ uint8_t(*p)) * 16777619u; }
    std::vector<float> data(size_t(ggml_nelements(tensor)));
    const bool norm = std::strstr(name, "norm.weight") != nullptr;
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = norm ? 1.0f : 0.04f * std::sin(0.37f * float(i) + float(hash % 997));
    }
    ggml_backend_tensor_set(tensor, data.data(), 0, ggml_nbytes(tensor));
}

struct backend_owner {
    backend_owner() { llama_backend_init(); }
    ~backend_owner() { llama_backend_free(); }
};

struct batch_owner {
    llama_batch batch = llama_batch_init(32, 0, 1);
    ~batch_owner() { llama_batch_free(batch); }
};

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s REFERENCE_FILE write|compare dense|flash\n", argv[0]);
        return 2;
    }
    try {
        const bool write = std::string(argv[2]) == "write";
        const bool flash = std::string(argv[3]) == "flash";
        require(write || std::string(argv[2]) == "compare", "invalid reference mode");
        require(flash || std::string(argv[3]) == "dense", "invalid attention mode");
        backend_owner backend;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> metadata(gguf_init_empty(), gguf_free);
        require(metadata != nullptr, "metadata allocation failed");
        gguf_set_val_str(metadata.get(), "general.architecture", "qwen3moe");
        gguf_set_val_str(metadata.get(), "tokenizer.ggml.model", "none");
        for (const auto & item : std::vector<std::pair<const char *, uint32_t>>{
                {"vocab_size", 32}, {"context_length", 128}, {"embedding_length", 128},
                {"block_count", 2}, {"feed_forward_length", 192}, {"expert_feed_forward_length", 64},
                {"attention.head_count", 4}, {"attention.head_count_kv", 2},
                {"rope.dimension_count", 32}, {"expert_count", 8}, {"expert_used_count", 3}}) {
            gguf_set_val_u32(metadata.get(), (std::string("qwen3moe.") + item.first).c_str(), item.second);
        }
        gguf_set_val_f32(metadata.get(), "qwen3moe.attention.layer_norm_rms_epsilon", 1e-6f);
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_init_from_user(metadata.get(), initialize_tensor, nullptr, mp), llama_model_free);
        require(model != nullptr, "synthetic Qwen3 MoE model initialization failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 128;
        cp.n_batch = 32;
        cp.n_ubatch = 32;
        cp.n_threads = 2;
        cp.n_threads_batch = 2;
        cp.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(ctx != nullptr, "synthetic context initialization failed");
        batch_owner owner;
        auto & batch = owner.batch;
        std::vector<float> logits;
        int position = 0;
        // Two differently sized prefills followed by repeated one-token decode (graph reuse).
        for (int count : {13, 7, 1, 1, 1, 1}) {
            batch.n_tokens = count;
            for (int i = 0; i < count; ++i) {
                batch.token[i] = (position + i) % 32;
                batch.pos[i] = position + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = true;
            }
            require(llama_decode(ctx.get(), batch) == 0, "synthetic model decode failed");
            for (int i = 0; i < count; ++i) {
                const float * out = llama_get_logits_ith(ctx.get(), i);
                require(out != nullptr, "missing model logits");
                logits.insert(logits.end(), out, out + 32);
            }
            position += count;
        }
        for (float value : logits) { require(std::isfinite(value), "non-finite model logits"); }
        double max_error = 0.0;
        if (write) {
            std::ofstream file(argv[1], std::ios::binary | std::ios::trunc);
            file.write(reinterpret_cast<const char *>(logits.data()), std::streamsize(logits.size() * sizeof(float)));
            require(bool(file), "cannot write reference logits");
        } else {
            std::ifstream file(argv[1], std::ios::binary);
            std::vector<float> reference(logits.size());
            file.read(reinterpret_cast<char *>(reference.data()), std::streamsize(reference.size() * sizeof(float)));
            require(bool(file) && file.peek() == std::ifstream::traits_type::eof(), "invalid reference logits");
            for (size_t i = 0; i < logits.size(); ++i) {
                const double error = std::fabs(double(logits[i]) - reference[i]);
                max_error = std::max(max_error, error);
                require(std::isfinite(reference[i]) && error <= 2e-5 + 2e-5 * std::fabs(reference[i]), "model logits differ");
            }
        }
        std::printf("PASS: synthetic Qwen3 MoE (%s), %zu logits, max absolute difference %.9g\n",
                    argv[3], logits.size(), max_error);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    return 0;
}
