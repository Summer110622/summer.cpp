#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

static void set_bit_env(const char * value) {
#ifdef _WIN32
    _putenv_s("LLAMA_ARG_BIT_ATTN", value);
#else
    setenv("LLAMA_ARG_BIT_ATTN", value, 1);
#endif
}

static void check_args(std::vector<std::string> words, bool expected) {
    // Parsing only; no file is opened or model loaded by this test.
    words.insert(words.end(), {"--model", "unused-test-fixture.gguf"});
    common_params params;
    std::vector<char *> args;
    for (auto & word : words) { args.push_back(&word[0]); }
    if (!common_params_parse(int(args.size()), args.data(), params, LLAMA_EXAMPLE_COMMON)) {
        throw std::runtime_error("argument parsing failed");
    }
    if (params.bit_attn != expected || common_context_params_to_llama(params).bit_attn != expected) {
        throw std::runtime_error("BitAttention option was not propagated to llama_context_params");
    }
}

int main() {
    try {
        set_bit_env("0");
        check_args({"test"}, false);
        check_args({"test", "--bit-attn"}, true);
        check_args({"test", "--bit-attn", "--no-bit-attn"}, false);
        set_bit_env("1");
        check_args({"test"}, true);
        check_args({"test", "--no-bit-attn"}, false);
        check_args({"test", "--no-bit-attn", "--bit-attn"}, true);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1;
    }
    std::puts("PASS: default, CLI, environment, negation and context propagation (6 cases)");
    return 0;
}
