# Experimental routing and attention graph optimizations

This follow-up to Serial MoE (PR #2) shares the expert implementation instead of duplicating it, adds configurable expert chunks, and adds opt-in normalized Top-K softmax and non-Flash query chunking. All new optimizations are **disabled by default**. No model weights are unloaded, streamed, or cached by this change.

## Controls

Set environment variables **before starting the process**. They are captured once, before graph reservation, so changing the environment on a running/reused context does not change its graph.

| Variable | Meaning |
| --- | --- |
| `LLAMA_SERIAL_EXPERTS=1` | Backward-compatible spelling of expert chunk size 1. |
| `LLAMA_MOE_EXPERT_CHUNK=N` | Execute up to N selected expert slots per chunk; 0 uses the original parallel graph. An explicit value takes precedence over `LLAMA_SERIAL_EXPERTS`. |
| `LLAMA_MOE_TOPK_SOFTMAX=1` | Use selected-logit softmax only when the equivalence guards below hold. |
| `LLAMA_ATTN_QUERY_CHUNK=N` | Preferred query chunk size for the non-Flash MHA/GQA/MQA path; 0 disables chunking. |

Boolean switches require exactly `1`. Chunk sizes accept decimal integers from 0 through `INT32_MAX`; invalid values warn and disable that setting. An empty chunk setting is treated as 0. Query chunking is capped at **16 chunks per attention call**, so very small requests are raised to `ceil(number_of_queries / 16)`. This is a graph-size bound, not a promise that the requested size is a strict memory limit.

For example, after building the normal CLI:

```sh
LLAMA_MOE_EXPERT_CHUNK=2 LLAMA_MOE_TOPK_SOFTMAX=1 \
LLAMA_ATTN_QUERY_CHUNK=64 \
./build/bin/llama-cli -m model.gguf -ngl 999 -fa off
```

Do not disable Flash Attention solely to enable this experiment: the existing Flash Attention path is unchanged and bypasses query chunking. With `-fa on`, the MoE options still work independently.

## 1. Normalized Top-K routing

Let `z` contain E expert logits and `S` be the selected K largest logits. In exact arithmetic:

```text
p_i = exp(z_i) / sum_j exp(z_j)
w_i = p_i / sum_{j in S} p_j
    = exp(z_i) / sum_{j in S} exp(z_j), i in S
```

Thus, for normalized softmax routing, selection can operate directly on logits and only the selected K logits need softmax. The implementation uses GGML's stable softmax, including its maximum subtraction; it does not explicitly exponentiate unshifted logits.

The fast path requires `norm_w`, softmax gating, no selection bias, no grouped routing, no externally supplied expert IDs, and neither Llama 4 nor GroveMoE routing. Everything else uses the original router. Router projection/bias and the final expert weight scale are preserved.

The original denominator is clamped to `2^-14`. Since selected Top-K probability mass is at least `K/E`, the fast path additionally requires `K/E >= 2^-14`; otherwise it falls back. External IDs are excluded because they need not select the largest probabilities and would invalidate this bound.

This reduces softmax/normalization work from E entries to K entries. **It does not change GGML's argsort implementation or the cost of the router projection.** Floating-point rounding can change near-tied rankings; mathematical equivalence does not imply bitwise identity or identical generated text for every model.

## 2. Shared expert chunks

Both parallel and chunked execution call the same expert builder, preserving gate/up/down projections, separate and merged weights, expert bias/scale, LoRA, activation variants, and Llama 4's input-before-FFN weighting rule. Expert count and selected IDs are not pruned or modified.

Strided ID and weight slices are packed before dispatch; this matters for multiple tokens and backend indexing. The last chunk may be smaller. Outputs are summed left-to-right in the same slot order across chunk boundaries. Graph expansion follows each chunk/reduction so intermediate activation buffers can be reused. Kernel shapes and therefore rounding/performance can still differ.

A chunk of C experts reduces the width of each routed intermediate from K expert slots to `min(C,K)` slots. Smaller chunks trade parallelism and extra graph/kernel work for potentially lower activation memory. They do **not** reduce resident expert weight memory. The existing warmup distinction between selected slots and the layer's aggregated active count is retained.

Graph node reservation is increased conservatively for expert chunks, adapters, and attention chunks. The legacy serial switch now benefits from the same reservation accounting.

## 3. Query-chunked non-Flash attention

Each query block evaluates the original full-key attention expression, including the original scale, causal/sliding-window mask, ALiBi, sinks, and logit soft-capping. Keys and values are not truncated or approximated. Query slicing does not change the softmax reduction axis.

For S streams, H query heads, Nq queries and Nk keys, the score scratch size changes from `O(S H Nq Nk)` to `O(S H C Nk)`, with C the effective chunk size. Full outputs, KV storage, and the original input mask still exist. Mask slices are made contiguous as required by `ggml_soft_max_ext`; V is transposed/packed once, outside the chunk loop.

Output blocks are assembled into their original stream-major positions with dependency-linked in-place SET operations. This avoids repeatedly concatenating/copying the full growing output. GGML SET has 32-bit stride/offset constraints: output shapes reaching 1 GiB per stream fall back to the original attention path. KQ-bias and MLA cases also deliberately retain their original paths. Existing CPU placement is applied to chunk outputs when KQV offload is disabled.

This is not a Flash Attention kernel, sparse attention, KV eviction, or an asymptotic reduction in attention FLOPs. It can be slower because of extra launches, packing and smaller matrix multiplications, especially for short sequences. Single-query decode uses the original path.

## 4. Layout-only packing maintenance

`ggml_cont` creates a copy operation even when its input is already contiguous. The chunk builders now use `llama_graph_cont_if_needed` at four layout-only sites: expert IDs, expert weights, attention mask slices, and the attention output permutation.

Contiguous views keep their original data dependency and are not copied. In particular, single-token expert slices and single-head/single-stream mask slices can be contiguous despite padded strides on singleton dimensions. Non-contiguous multi-token IDs and multi-head/multi-stream mask slices still require packing. A one-query or one-head output permutation also needs no data movement.

This helper is **not an isolation copy** and must not replace `ggml_cont` globally: callers that need independent mutable storage must retain their explicit copy. Expert reduction order, full-key masks, softmax guards, output initialization and dependency-linked SET assembly remain unchanged. No new environment setting is required; this maintenance affects the existing opt-in chunk paths only.

## Validation without external models

```sh
cmake -S . -B build-routing \
  -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DGGML_CCACHE=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_SERVER=OFF \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF
cmake --build build-routing --target llama test-routing-attention test-routing-attention-model -j 2
ctest --test-dir build-routing -R '^test-routing-attention' --output-on-failure --no-tests=error
```

`test-routing-attention` exercises 341 numerical/guard/allocation cases with 1 and 4 CPU threads: extreme logits and ties, K=1/K=E, multiple tokens, strided expert IDs, remainder chunks, warmup subsets, merged/separate projections, biases/scales/LoRA, seven activations, GQA/MQA/MHA, streams, F16/F32 masks, causal/sliding masks, ALiBi, sinks, soft-capping, and real scheduler buffer reuse over changing inputs. Structural checks verify F32/F16/I32 contiguous offset views, required strided packing, and attention copy-node counts. Scheduler coverage includes 1/11 queries and 1/2 streams. Numerical checks use `abs_error <= 2e-5 + 2e-5 * abs(reference)`.

`test-routing-attention-model` builds a deterministic, **untrained** two-layer Qwen3 MoE model in memory (128 embedding dimensions, 8 experts, Top-3, 4 query/2 KV heads). It exercises the production llama model/context/graph code, two prefills and four single-token decodes, comparing 768 logits for every mode. CTest runs separate baseline/serial/chunk/router/attention/combined processes with Flash Attention both disabled and enabled. These tests need neither a model download nor Python dependencies.

In the initial PR validation, all 13 CTests passed on the development CPU Release build. The 334 helper/scheduler cases also passed a Debug AddressSanitizer + UndefinedBehaviorSanitizer build with leak detection. Separately compiling PR #2's original `llama-graph.cpp` and running the synthetic model produced identical baseline logits in all four dense/Flash and parallel/serial comparisons. The helper/scheduler tests' maximum absolute difference was `5.96046448e-8`; the synthetic model comparisons' largest difference was `2.38418579e-7`.

The allocator-only attention experiment uses F32, 512 queries, 2048 keys, 8 query heads, 2 KV heads, head dimension 32, one stream, no mask, and query chunk 64:

| Reserved CPU graph buffer | Bytes |
| --- | ---: |
| Original dense graph | 35,127,296 |
| Query chunk 64 | 6,356,992 |

That is **81.90% less for this synthetic graph buffer**, not total process RAM or real-model GPU VRAM. The test allocates/reserves this graph but does not time it. Results can vary with backend and allocator changes.

GPU kernels, quantized real-model accuracy/perplexity, the user's local Qwen3.6-35B-A3B GGUF, total VRAM, and tokens/second remain unmeasured. These opt-in features must not be presented as verified real-model speedups or memory guarantees.

## Maintenance validation (2026-09-22)

Against PR #3 commit `ef9a5ebcc0db59f09d05aac37aac118895cd5022`, the maintenance passed all 13 CTests in both CPU Release and Debug ASan + UBSan builds with leak detection. Both llama and GGML were instrumented. The helper suite now passes 341 numerical/guard/allocation cases; maximum absolute error remains `5.96046448e-8`.

Separately executing the original and maintained binaries with identical options produced bitwise-identical outputs in 12 synthetic-model comparisons: dense/Flash attention times baseline/serial/chunk/router/attention/combined, each with 768 logits. This is evidence for this untrained CPU fixture, not a guarantee for arbitrary models or backends. A mutation that restores unconditional packing is rejected by the new structural tests.

An allocator-only probe of the old and maintained chunk helpers gave:

| Synthetic graph | CONT nodes before / after | Reserved bytes before / after |
| --- | ---: | ---: |
| MoE, 1 token, Top-3, expert chunk 1 | 6 / 0 | 148,640 / 148,640 |
| MoE, 1 token, Top-3, expert chunk 2 | 4 / 0 | 149,024 / 149,024 |
| MoE, 7 tokens, Top-3, expert chunk 1 | 6 / 6 | 154,528 / 154,528 |
| Masked attention, 1 stream, query chunk 64 | 16 / 8 | 11,010,048 / 10,551,296 |
| Masked attention, 2 streams, query chunk 64 | 16 / 16 | 22,020,096 / 22,020,096 |

The MoE probe uses F32, embedding 32, FFN 48 and 8 experts. The attention probe uses F32 Q/K/V and masks, 512 queries, 2048 keys, 8 query heads, 2 KV heads and head dimension 32. Inputs and output are included in these graph reservations. The single-stream masked attention saving is 458,752 bytes (448 KiB, 4.17%) **relative to the previous chunked implementation**, not relative to dense attention. The no-mask 81.90% result above is the earlier dense-to-chunk comparison and is unchanged. No execution latency, total process RAM or GPU VRAM was measured by this probe.

The permanent read-only CI covers Linux static, Linux dynamic, Windows dynamic and Linux Debug ASan/UBSan. It also watches root CMake, `cmake/**` and public headers, cancels superseded runs for the same PR, and fails if CTest discovers no matching tests. To reproduce the sanitizer configuration, add the following to the CPU CMake command and use a separate build directory:

```sh
-DCMAKE_BUILD_TYPE=Debug \
-DLLAMA_SANITIZE_ADDRESS=ON -DLLAMA_SANITIZE_UNDEFINED=ON \
-DGGML_SANITIZE_ADDRESS=ON -DGGML_SANITIZE_UNDEFINED=ON
```

Run CTest with `ASAN_OPTIONS=detect_leaks=1:halt_on_error=1` and `UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`. GPU and quantized real-model performance/accuracy remain unverified; defaults remain off.
