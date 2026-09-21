# Serial MoE experts

`LLAMA_SERIAL_EXPERTS=1` makes the shared MoE graph execute the already-selected Top-K expert slots one at a time and immediately accumulate each weighted result. Router logits, Top-K selection, weight normalization, expert count, shared experts, and the final mathematical sum are unchanged.

```text
router -> top-k ids/weights -> expert[0] -> weighted add -> expert[1] -> weighted add -> ... -> expert[K-1] -> final MoE output
```

Enable it with:

```sh
LLAMA_SERIAL_EXPERTS=1 ./build/bin/llama-cli -m model.gguf -ngl 999
```

The mode is experimental and disabled by default. Its purpose is to reduce simultaneously-live routed-expert intermediate activations; it does not unload expert weights, stream weights from CPU to GPU, change the model's configured Top-K, or guarantee a lower total VRAM footprint on every backend. Floating-point addition order differs from the parallel graph, so bit-exact output is not guaranteed even though the mathematical expression is the same.
