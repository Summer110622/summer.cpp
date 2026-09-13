# Experimental native BitAttention

BitAttention is an **opt-in, inference-only model transformation**, not an exact
implementation of ordinary softmax attention. It replaces Q/K magnitudes with
signs, keeps V floating point and retains softmax, masks and attention sinks.
There are no claims of pretrained-model quality, CUDA speedup or KV-cache
compression. Evaluate perplexity and task quality before using a checkpoint.

## Enable

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=ON
cmake --build build --parallel --target llama-cli llama-server llama-bench test-bit-attention test-bit-attention-model test-bit-attention-args

build/bin/llama-cli -m model.gguf --bit-attn -p "Hello"
build/bin/llama-server -m model.gguf --bit-attn
```

Omit `-DGGML_CUDA=ON` for CPU. The regular command-line tools also accept
`LLAMA_ARG_BIT_ATTN=1` and `--no-bit-attn`. This setting is per context, not a
process-global backend switch. `--flash-attn` still selects the normal KV layout;
BitAttention uses its own fused operator when enabled, even with `-fa off`.

The C API defaults to disabled:

```c
struct llama_context_params params = llama_context_default_params();
params.bit_attn = true;
struct llama_context * ctx = llama_init_from_model(model, params);
```

Rebuild the library and clients together: `llama_context_params` has an appended
field, and ggml has new operation IDs. This is not an ABI-compatible binary-only
replacement for an older build. No GGUF weight conversion is performed.

## Mathematics and graph integration

For `s(x) = +1` when `x >= 0`, otherwise `-1`, a packed word stores 32 signs in
little-bit order (feature 0 is the least-significant bit). Q and K are packed
**after the model's existing positional/rotary processing**. For KV-cache models,
K is packed from the existing cache representation.

```text
binary_dot(q, k) = D - 2 * popcount((q_bits XOR k_bits) AND valid_bits)
score            = scale * binary_dot
score            = cap * tanh(score / cap)             # only when cap > 0
score            = score + alibi_slope * additive_mask
output           = softmax(score, optional_sink) @ V   # sink has zero value
```

`ggml_bit_pack` emits I32 tensors `[ceil(D/32), N, H, B]`. `ggml_bit_attn_ext`
consumes packed Q/K and returns F32 `[Dv, Hq, Nq, Bq]`, the same output layout as
`ggml_flash_attn_ext`. It supports unequal query/key lengths, grouped Q/K/V heads,
batch broadcasting, arbitrary input strides, F32/F16/BF16 V and masks, ALiBi,
logit softcaps and F32 attention sinks. Quantized K/V are explicitly cast before
the binary operations; packed words themselves are never interpreted as floats.

Masks remain the caller's responsibility. Decode uses the existing absolute
positions, sequence ownership and sliding-window masks; there is no implicit
upper-left causal mask. Mask values must be finite or negative infinity. Inputs
and sink logits must be finite (negative-infinity sinks are allowed). All-masked
rows return zero rather than NaN. Unused bits in the final word are always ignored,
including when externally supplied packed tensors contain nonzero padding.

The common model graph builder covers ordinary MHA/GQA/MQA attention and the
existing MLA output projection. MiniMax-M3's direct MSA attention call and the
WavTokenizer decoder's attention call are wired separately. When a model needs
special KQ bias or Grok-specific score transforms, `ggml_bit_mul_mat` replaces
only QK and leaves the original unfused score/softmax/V graph in place. Those
unfused paths still materialize their score matrix. SSM, Gated DeltaNet, recurrent
state updates, indexer/block selection and non-attention matrix products are not
replaced. This is not a replacement for every operator in a hybrid model or for
separate multimodal projector backends.

## Backends and limits

CPU provides stride-aware reference kernels. NVIDIA CUDA provides native
`__ballot_sync` packing, XOR/`__popc` dot products and a 32-key tiled online-softmax
forward kernel. The fused kernel keeps probabilities/accumulators in registers;
it never allocates a global `Nq * Nk` attention matrix. Its V multiply is scalar
FP32 accumulation, **not a tuned Tensor Core implementation**. It may be slower
than existing FlashAttention; fewer logical dot-product operations are not an
end-to-end speed guarantee. CUDA supports fused value dimensions up to 1024;
other value dimensions and non-NVIDIA accelerator backends use CPU fallback.
RPC does not negotiate per-op backend capabilities, so the new operations run
on the local CPU rather than being sent to an unsupported remote backend.

The persistent KV cache is unchanged. K is repacked on each graph evaluation;
there is no new 1-bit cache type, incremental packed-cache update, serialized
packed cache, or claimed 16x total memory saving. Packed temporary storage adds
`4 * ceil(D/32)` bytes per token/head, in addition to the existing cache. An
incremental cache would need to track slot reuse, sequence copies, shifts,
rollback and RoPE changes before it could be safely introduced.

Tensor-parallel split mode is explicitly rejected. Training/backpropagation,
STE, dropout, Metal/HIP/Vulkan-specific kernels and pretrained-checkpoint
calibration are not implemented. CPU fallback preserves binary semantics rather
than silently reverting to ordinary floating-point QK.

## Validation and benchmarking

```sh
ctest --test-dir build -R '^test-bit-attention' --output-on-failure
build/bin/test-bit-attention --backend CUDA0
```

The kernel test compares against an independent double-precision dense oracle,
not another XOR implementation. Cases cover F32/F16/BF16, 1/31/32/33/.../257 head
dimensions, sign-bit/negative-zero conventions, dirty padding, non-contiguous
views, GQA/MQA-style broadcasting, non-square decode masks, entirely masked rows
and tiles, ALiBi, softcaps and sinks. A missing explicitly requested backend
returns 77; a numerical failure returns 1.

The model test generates a tiny deterministic random Llama GGUF locally, checks
that only enabled contexts execute the new ops, and compares token-wise versus
chunked decode with one and two sequences and both KV layouts. It checks wiring,
not real-model quality. CI CPU execution and CUDA translation-unit compilation
are separate; CUDA compilation is not a GPU runtime test.

Run matched benchmarks; the benchmark tool records `bit_attn` in its result data:

```sh
build/bin/llama-bench -m model.gguf --no-bit-attn -p 512 -n 128 -r 5 -o json
build/bin/llama-bench -m model.gguf --bit-attn    -p 512 -n 128 -r 5 -o json
```

Use the same model, device/offload, cache types, context sizes and warmup policy.
Also compare perplexity/task quality: comparing speed alone compares different
mathematical models. The earlier Python/Triton sketches are not dependencies of
this implementation, and this C++/CUDA code does not make those sketches
production-ready.
