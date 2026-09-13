#pragma once // CUDAカーネル宣言の多重読み込みを防ぐ。
#include "common.cuh" // CUDAバックエンドのコンテキストとテンソル型を取得する。
bool ggml_cuda_bit_attn_ext_supported(const ggml_tensor * dst); // このCUDA実装で処理できる形状と型を判定する。
void ggml_cuda_bit_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst); // CUDAストリーム上でBitAttention forwardを実行する。
