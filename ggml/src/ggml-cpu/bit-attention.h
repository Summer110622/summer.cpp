#pragma once // CPUカーネル宣言の重複読み込みを防ぐ。
#include "ggml.h" // テンソル型とsize_tを取得する。
#ifdef __cplusplus // CとC++の両方から同じシンボル名で参照する。
extern "C" { // ggml-cpu.cから呼べるCリンケージを指定する。
#endif // C++向けリンケージ開始の条件を閉じる。
struct ggml_compute_params; // CPUの実行パラメータを前方宣言する。
size_t ggml_bit_attn_ext_work_size(const struct ggml_tensor * dst); // Q/Kの一時ビット列に必要なバイト数を取得する。
void ggml_compute_forward_bit_attn_ext(const struct ggml_compute_params * params, struct ggml_tensor * dst); // マルチスレッドCPU forwardを公開する。
#ifdef __cplusplus // C++コンパイル時だけリンケージブロックを閉じる。
} // Cリンケージの宣言範囲を終了する。
#endif // C++専用の終了処理を閉じる。
