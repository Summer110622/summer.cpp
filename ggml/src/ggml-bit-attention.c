#include "ggml.h" // 公開テンソルAPIと検証マクロを使用する。
#include <math.h> // スケール値の有限性を検証する。
#include <string.h> // 演算パラメータを型安全にコピーする。

static bool bit_attn_float_type(enum ggml_type type) { // 符号化と値読み取りに対応する非量子化型を判定する。
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16; // FP32・FP16・BF16だけを受け付ける。
} // 対応型の判定を終了する。

static void bit_attn_check_bias(const struct ggml_tensor * x, const struct ggml_tensor * q, int64_t nk) { // 加算マスクとバイアスのブロードキャスト条件を検証する。
    if (!x) { return; } // 省略された加算テンソルには検証を行わない。
    GGML_ASSERT(bit_attn_float_type(x->type)); // 整数やブロック量子化マスクを誤って読み取らない。
    GGML_ASSERT(x->ne[0] >= nk && x->ne[1] >= q->ne[1]); // KV長とクエリ長をカバーする領域を要求する。
    GGML_ASSERT(x->ne[2] > 0 && q->ne[2] % x->ne[2] == 0); // ヘッド方向の周期的ブロードキャストを検証する。
    GGML_ASSERT(x->ne[3] > 0 && q->ne[3] % x->ne[3] == 0); // バッチ方向の周期的ブロードキャストを検証する。
} // 加算テンソルの検証を終了する。

struct ggml_tensor * ggml_bit_attn_ext( // 符号化Q/Kを使用する独立したforward専用演算を構築する。
        struct ggml_context * ctx, // 出力テンソルを所有するggmlコンテキストを受け取る。
        struct ggml_tensor * q, // Qを[DK,NQ,HQ,BQ]形式で受け取る。
        struct ggml_tensor * k, // Kを[DK,NK,HK,BK]形式で受け取る。
        struct ggml_tensor * v, // Vを[DV,NK,HV,BV]形式で受け取る。
        struct ggml_tensor * mask, // 未来トークンや系列境界を隠す加算マスクを受け取る。
        struct ggml_tensor * sinks, // 各クエリヘッドのsoftmax分母だけに加えるsink logitを受け取る。
        struct ggml_tensor * bias, // スケール適用済みの追加logitバイアスを受け取る。
        float scale, // 符号内積に掛けるスケールを受け取る。
        float max_bias, // ALiBiの最大バイアス係数を受け取る。
        float logit_softcap) { // 正数の場合に適用するtanhソフトキャップを受け取る。
    GGML_ASSERT(ctx && q && k && v); // 必須入力の欠落を早期に検出する。
    GGML_ASSERT(bit_attn_float_type(q->type) && bit_attn_float_type(k->type) && bit_attn_float_type(v->type)); // 読み取り可能な浮動小数点型だけを許可する。
    GGML_ASSERT(q->ne[0] > 0 && q->ne[0] <= INT32_MAX && q->ne[0] == k->ne[0]); // 符号内積の次元と32bit距離累積の範囲を検証する。
    GGML_ASSERT(q->ne[1] > 0 && k->ne[1] > 0 && v->ne[0] > 0 && k->ne[1] == v->ne[1]); // 非空系列とK/Vのトークン対応を要求する。
    GGML_ASSERT(k->ne[2] > 0 && v->ne[2] > 0 && q->ne[2] > 0); // GQAの除数となるヘッド数を検証する。
    GGML_ASSERT(q->ne[2] % k->ne[2] == 0 && q->ne[2] % v->ne[2] == 0); // KとVそれぞれのGQA共有率を検証する。
    GGML_ASSERT(k->ne[3] > 0 && v->ne[3] > 0 && q->ne[3] > 0); // バッチ共有の除数を検証する。
    GGML_ASSERT(q->ne[3] % k->ne[3] == 0 && q->ne[3] % v->ne[3] == 0); // K/Vバッチをクエリバッチへグループ共有できることを確認する。
    GGML_ASSERT(isfinite(scale) && isfinite(max_bias) && max_bias >= 0.0f); // 非有限スケールや負のALiBiパラメータを拒否する。
    GGML_ASSERT(isfinite(logit_softcap) && logit_softcap >= 0.0f); // ゼロによる無効化か正のソフトキャップだけを受け付ける。
    bit_attn_check_bias(mask, q, k->ne[1]); // マスクの型と寸法を検証する。
    bit_attn_check_bias(bias, q, k->ne[1]); // 追加バイアスの型と寸法を検証する。
    if (sinks) { // sinkが指定された場合だけその形状を検証する。
        GGML_ASSERT(sinks->type == GGML_TYPE_F32 && ggml_is_vector(sinks) && sinks->ne[0] == q->ne[2]); // ヘッドごとに1個のFP32 sink logitを要求する。
    } // sinkの検証を終了する。
    const int64_t ne[4] = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] }; // FlashAttentionと同じ[DV,HQ,NQ,BQ]出力順を使用する。
    struct ggml_tensor * result = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne); // softmax加重和をFP32テンソルへ出力する。
    const float params[3] = { scale, max_bias, logit_softcap }; // CPUとCUDAが共有する演算パラメータを並べる。
    memcpy(result->op_params, params, sizeof(params)); // エイリアシング違反を避けてパラメータを格納する。
    result->op = GGML_OP_BIT_ATTN_EXT; // 未対応バックエンドが通常Attentionとして誤実行しない専用opを指定する。
    result->src[0] = q; // クエリへのグラフ依存関係を登録する。
    result->src[1] = k; // キーへのグラフ依存関係を登録する。
    result->src[2] = v; // 値へのグラフ依存関係を登録する。
    result->src[3] = mask; // マスク生成へのグラフ依存関係を登録する。
    result->src[4] = sinks; // sinkへのグラフ依存関係を登録する。
    result->src[5] = bias; // 追加バイアスへのグラフ依存関係を登録する。
    return result; // 構築したforwardノードを呼び出し側へ返す。
} // BitAttentionノードの構築を終了する。
