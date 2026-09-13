# Experimental 1-bit Q/K attention

`--bit-attn` enables a native ggml attention operator in the shared
`llm_graph_context::build_attn_mha` path. It is **disabled by default**.
This changes the model's mathematical function; it is not a numerically
identical implementation of standard attention. Quality and speed must be
measured for each model and workload. No performance improvement is claimed.

## Computation

After the model's normal Q/K transformations, including RoPE where present,
map each component to +1 for `x >= 0` and -1 otherwise. Pack 32 components into
a `uint32_t`; padding bits are zero. Positive and negative zero map to +1,
and negative subnormal values retain their negative sign.

For each query/key pair compute `dot = D - 2 * popcount(q_bits ^ k_bits)`.
Multiply by the model's existing attention scale. Apply the configured logit
softcap, then add the ALiBi-scaled attention mask and optional additional bias.
Online softmax and high-precision V accumulation produce FP32 output. Optional
attention sinks contribute to the softmax denominator only. Fully masked rows
without a finite sink return zero.

Only the temporary packed Q/K workspace is linear in sequence length. The
operator never stores a full QK score/probability matrix in global memory.
**The attention computation is still quadratic in sequence length**, and PV
still uses floating-point arithmetic. Existing externally supplied masks can
also have quadratic storage. This is not an integer-only Transformer.

## Build and use

```sh
cmake -S . -B build-bit -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=OFF # NVIDIA CUDA対応のビルドを構成する。
cmake --build build-bit --target llama-completion test-bit-attention test-bit-attention-model test-llama-archs -j 4 # 推論ツールとカーネル・モデル統合テストをビルドする。
ctest --test-dir build-bit -R '^test-bit-attention' --output-on-failure # 数値参照と小型モデルによる検証を実行する。
./build-bit/bin/llama-completion -m /path/to/model.gguf -p 'Hello' -n 32 --bit-attn --cache-type-k f16 --cache-type-v f16 # 符号化Q/Kを明示的に有効化して生成する。
./build-bit/bin/llama-completion -m /path/to/model.gguf -p 'Hello' -n 32 --no-bit-attn # 従来のAttentionで比較用に生成する。
```

For CPU-only builds omit `-DGGML_CUDA=ON`. The common argument parser also
exposes the option to other inference tools and the server; the equivalent
environment variable is `LLAMA_ARG_BIT_ATTN=1`. The API setting is:

```cpp
auto params = llama_context_default_params(); // 従来と同じ既定設定を取得する。
params.bit_attn = true; // このコンテキストの共通MHA層だけ符号化Attentionへ切り替える。
auto * ctx = llama_init_from_model(model, params); // 対応型を検証して推論コンテキストを構築する。
```

`model` must already be loaded; check the returned context for null and free
it using the normal llama lifecycle. The public context-parameter structure
has an appended field: rebuild consumers/bindings together with the library.
The ggml operator count and RPC patch version are updated; use matching builds
on RPC peers. This is a fork-level API change, not upstream ABI compatibility.

## Backends and integration

The CPU implementation packs each Q/K row once per graph execution, shares the
workspace between CPU workers, and evaluates streaming softmax. The CUDA
implementation packs on-device with warp ballot, evaluates XOR plus `__popc`,
and fuses tiled online softmax with V accumulation on the same stream. Its
initial configuration is a correctness-oriented 128-key tile, not an optimized
replacement for vendor Tensor Core attention kernels.

The public operator accepts F32/F16/BF16 Q/K/V, independent K and V dimensions,
strided views, grouped-query/multi-query heads, grouped batch sharing,
F32/F16/BF16 additive masks, ALiBi, softcaps, additional biases, and F32 sinks.
Output is `[D_v, H_q, N_q, B_q]`, matching the existing fused graph path. Decode
and sequence boundaries use the existing absolute-position masks, not a new
local triangular mask. Causal and sliding-window restrictions remain in those
masks.

NVIDIA CUDA currently requires `D_q <= 4096`, `D_v <= 1024`, and supported CUDA
grid dimensions. Unsupported accelerators (including HIP, MUSA, Metal and
Vulkan) do not claim support for this new operator; the graph scheduler can
place it on the CPU, potentially causing transfers and slowdowns.

The common MHA path is connected, including its existing V layout handling.
Separate linear-attention, SSM and gated-delta operators are **not** replaced.
Not every supported model architecture has been evaluated. Tensor-split mode
and Grok's special logit transforms are explicitly rejected when this option
is enabled. Unsupported cache types fail initialization rather than silently
falling back to standard attention. Quantized model weights are not prohibited;
block-quantized **K/V caches** are not supported by this operator.

## Cache, training, and accuracy limitations

Persistent K and V caches remain in their configured F32/F16/BF16 formats.
The kernel repacks the current K on each invocation into temporary workspace.
It does **not** provide a persistent 1-bit cache or a 16x reduction of resident
KV memory. Packing and workspace allocation also have costs.

Sign quantization removes magnitudes and can significantly alter RoPE-based
similarities. This code does not calibrate scales, retrain weights, implement
an STE backward pass, or establish language-model quality. No backward kernel
is provided. Compare perplexity, retrieval accuracy, generation quality and
latency separately before using this experimental path in production.

## Validation

`test-bit-attention` compares against an independent double-precision dense
reference using explicit +/-1 multiplications, not XOR/popcount. The CPU test
covers 98 cases across one and four workers, dimension tails, F32/F16/BF16,
subnormal/negative-zero signs, GQA/MQA, differing query/key lengths and V
dimensions, noncontiguous views, masks, fully masked rows/tiles, sinks, ALiBi
and softcaps. It automatically runs the corresponding 49 numerical cases on
available CUDA devices; absence of a CUDA device is reported as a skip, not a
GPU pass.

`test-bit-attention-generate` creates a reproducible random miniature Llama
GGUF using `test-llama-archs`; no external model download is needed.
`test-bit-attention-model` verifies actual bit-op dispatch, batched prefill
versus cached incremental decode, both V cache layouts, opt-out dispatch and
rejection of unsupported quantized K caches. These are integration tests, not
a benchmark or a model-quality evaluation.

The CI CUDA job compiles the changed CUDA translation units with CUDA 12.8.1
for SM80, including the main dispatch unit with CUDA graphs enabled. It does
not run kernels or validate a full CUDA backend link on a GPU-less runner.
Run the numerical tests and Compute Sanitizer on actual target hardware before
claiming CUDA runtime correctness or performance.

Every added or changed C/C++/CUDA, Python, YAML and CMake code line has an
explanatory Japanese comment. To audit the patch against its starting revision:

```sh
python3 scripts/check-bit-attention-comments.py 5265486cb491ea5b160a3d20ca1e105ae5422312 # 基準コミット以降に追加・変更されたコード行の説明コメントを検査する。
```
