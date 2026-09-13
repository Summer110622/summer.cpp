#pragma once // CUDA公開宣言の重複インクルードを防ぐ。
#include "common.cuh" // CUDAの共通型とバックエンド定義を読み込む。

// Native NVIDIA CUDA implementations. Other accelerator backends fall back to CPU.
bool ggml_cuda_bit_attention_supported(const ggml_tensor * op); // CUDAバックエンドが扱える型と出力形状を判定する。
void ggml_cuda_op_bit_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst); // CUDAストリーム上へ符号パックカーネルを起動する。
void ggml_cuda_op_bit_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst); // CUDAストリーム上へ二値内積カーネルを起動する。
void ggml_cuda_op_bit_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst); // Value次元に合う融合CUDAカーネルを選択する。
