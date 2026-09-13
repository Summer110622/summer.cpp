#pragma once
#include "common.cuh"

// Native NVIDIA CUDA implementations. Other accelerator backends fall back to CPU.
bool ggml_cuda_bit_attention_supported(const ggml_tensor * op);
void ggml_cuda_op_bit_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_bit_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_bit_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
