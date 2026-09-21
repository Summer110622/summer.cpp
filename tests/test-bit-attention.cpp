#include "ggml.h" // GGMLのテンソル型と演算APIを読み込む。
#include "ggml-alloc.h" // バックエンドバッファの割り当てAPIを読み込む。
#include "ggml-backend.h" // 演算グラフの実行とテンソル転送APIを読み込む。
#include "ggml-cpu.h" // CPUバックエンドの公開APIを読み込む。

#include <algorithm> // 最大値・範囲初期化などの標準アルゴリズムを読み込む。
#include <cmath> // 指数関数・tanh・有限値検査を読み込む。
#include <cstdint> // 幅を固定した整数型を読み込む。
#include <cstdio> // テスト結果とエラーの標準出力を読み込む。
#include <cstring> // 型エイリアスに依存しないバイトコピーを読み込む。
#include <memory> // リソースを自動解放するスマートポインタを読み込む。
#include <random> // 再現可能なテストデータの生成を読み込む。
#include <stdexcept> // 失敗理由を伝える例外型を読み込む。
#include <string> // 診断文とファイル名の構築を読み込む。
#include <vector> // 入力と参照計算結果の可変長配列を読み込む。

static void require(bool ok, const std::string & message) { // 数値テストの条件違反を例外として報告する。
    if (!ok) { // 参照結果と一致しない場合に検証失敗を通知する。
        throw std::runtime_error(message); // 失敗理由を保持した例外を送出してテストを中断する。
    } // この処理または定義のブロックを閉じる。
} // この処理または定義のブロックを閉じる。

struct input { // ホスト上の検証データと非連続GGMLビューを一緒に管理する。
    ggml_tensor * base; // 実際のストレージを所有する基底テンソルを保持する。
    ggml_tensor * t; // カーネルへ渡す転置を含む論理ビューを保持する。
    std::vector<uint8_t> data; // バックエンドへ転送する生バイト列を保持する。

    input(ggml_context * ctx, ggml_type type, int d, int n, int h, int b, int layout = 0) { // 指定された型・形状・レイアウトの入力を作る。
        if (layout == 1) { // トークン軸とヘッド軸を転置するレイアウトを選ぶ。
            base = ggml_new_tensor_4d(ctx, type, d, h, n, b); // 物理的にヘッドがトークンより先に並ぶストレージを作る。
            t = ggml_permute(ctx, base, 0, 2, 1, 3); // 論理的には[D,N,H,B]となるようビューを転置する。
        } else if (layout == 2) { // 特徴軸が非連続となるレイアウトを選ぶ。
            base = ggml_new_tensor_4d(ctx, type, n, d, h, b); // 特徴とトークンを逆順に並べた物理ストレージを作る。
            t = ggml_permute(ctx, base, 1, 0, 2, 3); // 特徴軸の非単位ストライドを持つ論理ビューを作る。
        } else { // 直前の条件に該当しない場合の経路へ切り替える。
            base = t = ggml_new_tensor_4d(ctx, type, d, n, h, b); // 通常レイアウトでは物理テンソルをそのまま入力にする。
        } // この処理または定義のブロックを閉じる。
        data.resize(ggml_nbytes(base)); // 基底テンソル全体を収容するホスト領域を確保する。
    } // この処理または定義のブロックを閉じる。
    size_t offset(int64_t d, int64_t n, int64_t h, int64_t b) const { // 論理座標からホスト配列上のバイト位置を求める関数を定義する。
        return d*t->nb[0] + n*t->nb[1] + h*t->nb[2] + b*t->nb[3]; // ビューの全4軸のストライドを使って位置を計算する。
    } // この処理または定義のブロックを閉じる。
    void set(int64_t d, int64_t n, int64_t h, int64_t b, float value) { // 指定座標へ入力型に丸めた値を書く補助関数を定義する。
        uint8_t * p = data.data() + offset(d, n, h, b); // 指定座標のホストストレージへ移動する。
        if (t->type == GGML_TYPE_F32) { // F32入力では精度変換が不要な経路を選ぶ。
            memcpy(p, &value, sizeof(value)); // 与えられたF32値をバイト列へ安全にコピーする。
        } else if (t->type == GGML_TYPE_F16) { // FP16へ丸める経路を選ぶ。
            const auto x = ggml_fp32_to_fp16(value); memcpy(p, &x, sizeof(x)); // FP16の表現に丸めた値を格納する。
        } else { // 直前の条件に該当しない場合の経路へ切り替える。
            const auto x = ggml_fp32_to_bf16(value); memcpy(p, &x, sizeof(x)); // BF16の表現に丸めた値を格納する。
        } // この処理または定義のブロックを閉じる。
    } // この処理または定義のブロックを閉じる。
    float get(int64_t d, int64_t n, int64_t h, int64_t b) const { // カーネルと同じストレージ精度の値を参照計算へ返す関数を定義する。
        const uint8_t * p = data.data() + offset(d, n, h, b); // 指定座標の格納済みバイト列へ移動する。
        if (t->type == GGML_TYPE_F32) { // F32ストレージの読み出し経路を選ぶ。
            float x; memcpy(&x, p, sizeof(x)); return x; // バイトコピーによりアラインメントに依存せずF32値を読む。
        } // この処理または定義のブロックを閉じる。
        if (t->type == GGML_TYPE_F16) { // FP16ストレージの復号経路を選ぶ。
            ggml_fp16_t x; memcpy(&x, p, sizeof(x)); return ggml_fp16_to_fp32(x); // FP16を安全に読み出してF32へ復号する。
        } // この処理または定義のブロックを閉じる。
        ggml_bf16_t x; memcpy(&x, p, sizeof(x)); return ggml_bf16_to_fp32(x); // BF16を安全に読み出してF32へ復号する。
    } // この処理または定義のブロックを閉じる。
    void randomize(std::mt19937 & rng) { // ランダム値と符号境界の値を入力に設定する。
        std::uniform_real_distribution<float> dist(-2.0f, 2.0f); // 正負両方を含む一様分布を指定する。
        for (int64_t b = 0; b < t->ne[3]; ++b) // 入力の全バッチを走査する。
        for (int64_t h = 0; h < t->ne[2]; ++h) // 各バッチの全ヘッドを走査する。
        for (int64_t n = 0; n < t->ne[1]; ++n) // 各ヘッドの全トークンを走査する。
        for (int64_t d = 0; d < t->ne[0]; ++d) { // 各トークンの全特徴を走査する。
            set(d, n, h, b, dist(rng)); // 指定型へ丸めながら乱数値を書き込む。
        } // この処理または定義のブロックを閉じる。
        // Zero convention, the sign bit and subnormal underflow are intentional.
        set(0, 0, 0, 0, -0.0f); // 負のゼロも非負としてパックされる規約を検証する。
        if (t->ne[0] > 1) { set(1, 0, 0, 0, -1.0e-20f); } // FP16でのアンダーフローを含む非常に小さい負値を検証する。
        if (t->ne[0] > 31) { set(31, 0, 0, 0, 1.0f); } // 32ビット語の最上位ビットが正しく扱われるか検証する。
    } // この処理または定義のブロックを閉じる。
    void upload() const { ggml_backend_tensor_set(base, data.data(), 0, data.size()); } // 基底テンソルへホストの入力データを転送する。
    std::vector<uint32_t> packed() const { // 各要素を直接検査する独立な符号パック参照値を作る。
        const int64_t words = (t->ne[0] + 31)/32; // 実特徴次元を収容する32ビット語の必要数を求める。
        std::vector<uint32_t> out(size_t(words*ggml_nrows(t)), 0); // パディングがゼロである参照ビット列の領域を用意する。
        for (int64_t b = 0; b < t->ne[3]; ++b) // 参照パック処理の全バッチを走査する。
        for (int64_t h = 0; h < t->ne[2]; ++h) // 参照パック処理の全ヘッドを走査する。
        for (int64_t n = 0; n < t->ne[1]; ++n) // 参照パック処理の全トークンを走査する。
        for (int64_t d = 0; d < t->ne[0]; ++d) { // 有効な実特徴だけを参照ビット列へ格納する。
            if (get(d, n, h, b) >= 0.0f) { // 元の型へ丸めた後の符号を調べる。
                out[size_t(d/32 + words*(n + t->ne[1]*(h + t->ne[2]*b)))] |= uint32_t(1) << (d & 31); // 語番号と語内のビット位置を求めて非負の特徴を1にする。
            } // この処理または定義のブロックを閉じる。
        } // この処理または定義のブロックを閉じる。
        return out; // パックしたビット列を返す。
    } // この処理または定義のブロックを閉じる。
}; // この処理または定義のブロックを閉じる。

static std::vector<float> read_f32(ggml_tensor * t) { // バックエンド上のF32結果をホストへ回収する。
    std::vector<float> a(size_t(ggml_nelements(t))); // 演算結果の全要素を収容するF32配列を用意する。
    ggml_backend_tensor_get(t, a.data(), 0, ggml_nbytes(t)); // デバイスまたはCPUバックエンドから結果を読み出す。
    return a; // 構築したビューまたは読み出した配列を返す。
} // この処理または定義のブロックを閉じる。

static void run_case(ggml_backend_t backend, int d, ggml_type type, int id) { // 一つの型・次元・レイアウトを独立な参照計算と照合する。
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx( // テスト用グラフを自動解放する所有ポインタを用意する。
        ggml_init({2*1024*1024, nullptr, true}), ggml_free); // テンソルの実データはバックエンドへ割り当てるためメタデータだけを確保する。
    require(bool(ctx), "ggml_init failed"); // GGMLコンテキストの確保成功を確認する。
    const int nq = id % 3 == 0 ? 1 : 5; // デコード相当の1Queryと複数Queryの両方を検証する。
    const int nk = 37; // warp幅32を越える非整数タイル長のKey列を使用する。
    const int hq = 6, hk = 2, hv = 3, bq = 4; // Q/K/Vでヘッド数と共有率を変えてGQAとバッチ共有を検証する。
    const int dv = id % 4 == 0 ? 65 : 17; // Value次元にも32の倍数でない形状を選ぶ。
    const int kind = id % 5; // 五種類のマスク条件を順に切り替える。
    const bool use_sink = id % 3 == 0; // sinkあり・なしの両方を検証する。
    const float scale = 1.0f/std::sqrt(float(d)); // 通常のAttentionと同じ逆平方根スケールを使う。
    const float bias = kind == 4 ? 8.0f : 0.0f; // ALiBiを含むケースだけ最大バイアスを設定する。
    const float cap = id % 4 == 1 ? 1.75f : 0.0f; // 一部ケースでtanhソフトキャップを有効にする。
    input q(ctx.get(), type, d, nq, hq, bq, id % 3); // Queryにケースごとの物理レイアウトを設定する。
    input k(ctx.get(), type, d, nk, hk, 2, (id + 1) % 3); // Queryとは異なるレイアウトと共有率でKeyを作る。
    input v(ctx.get(), type, dv, nk, hv, 1, (id + 2) % 3); // さらに異なるValueヘッド数・バッチ数・レイアウトを使う。
    input mask(ctx.get(), id % 2 ? GGML_TYPE_F16 : GGML_TYPE_F32, nk + 3, nq + 2, 2, 2, id % 3); // 余裕を持つ非正方形マスクでF16/F32と軸ストライドを検証する。
    input sinks(ctx.get(), GGML_TYPE_F32, hq, 1, 1, 1); // 各Queryヘッドに一つのF32 sinkロジットを用意する。
    std::mt19937 rng(110622 + id); // ケースごとに固定seedを使い失敗を再現可能にする。
    q.randomize(rng); k.randomize(rng); v.randomize(rng); // Q/K/Vを丸め後の型でランダム初期化する。
    for (int b = 0; b < 2; ++b) // マスクの両バッチを初期化する。
    for (int h = 0; h < 2; ++h) // マスクの両ヘッドを初期化する。
    for (int iq = 0; iq < nq + 2; ++iq) // 余剰Query領域も含めてマスクを初期化する。
    for (int ik = 0; ik < nk + 3; ++ik) { // 余剰Key領域も含めてマスクを初期化する。
        bool blocked = ik > (nk - nq + iq); // absolute decode offset, NOT upper-left causal
        if (kind == 2 && iq == 0) { blocked = true; } // 最初のQueryが全マスクになる境界ケースを作る。
        if (kind == 3) { blocked = ik < 34; } // first 32-key tile is completely masked
        const float finite_bias = kind == 4 ? -0.02f*float(ik + 2*h + b) : 0.0f; // ALiBiケースでは位置・ヘッド・バッチに依存する有限バイアスを作る。
        mask.set(ik, iq, h, b, blocked ? -INFINITY : finite_bias); // 無効な対には負無限大、有効な対には有限バイアスを書き込む。
    } // この処理または定義のブロックを閉じる。
    for (int h = 0; h < hq; ++h) { sinks.set(h, 0, 0, 0, h == 0 ? -INFINITY : 0.25f*h); } // 無効なsinkと有限sinkの両方を一つの入力で検証する。
    auto * qb = ggml_bit_pack(ctx.get(), q.t); // テスト対象のGGML Queryパック演算を作る。
    auto * kb = ggml_bit_pack(ctx.get(), k.t); // テスト対象のGGML Keyパック演算を作る。
    auto * dot = ggml_bit_mul_mat(ctx.get(), kb, qb, d); // 融合Attentionとは別に二値内積だけの結果も検証する。
    auto * out = ggml_bit_attn_ext(ctx.get(), qb, kb, v.t, kind ? mask.t : nullptr, // 通常パック入力でマスクあり・なしの融合Attentionを構築する。
                                  use_sink ? sinks.t : nullptr, d, scale, bias, cap); // sink・実特徴次元・スケール・ALiBi・ソフトキャップを指定する。
    // Also test externally packed tensors with deliberately different garbage in padding.
    auto * qext = ggml_dup_tensor(ctx.get(), qb); // 外部でパック済みのQueryを渡す経路も用意する。
    auto * kext = ggml_dup_tensor(ctx.get(), kb); // 外部でパック済みのKeyを渡す経路も用意する。
    auto * out_ext = ggml_bit_attn_ext(ctx.get(), qext, kext, v.t, kind ? mask.t : nullptr, // 外部パック列でも同じ融合Attentionを構築する。
                                      use_sink ? sinks.t : nullptr, d, scale, bias, cap); // 通常パック経路と同じロジット変換条件を与える。
    auto * dot_ext = ggml_bit_mul_mat(ctx.get(), kext, qext, d); // 外部パック列でも二値内積だけの結果を検証する。
    auto * gf = ggml_new_graph(ctx.get()); // 四つの検証出力をまとめるGGMLグラフを作る。
    ggml_build_forward_expand(gf, dot); // 通常パックから内積までの依存グラフを追加する。
    ggml_build_forward_expand(gf, out); // 通常パックから融合Attentionまでの依存グラフを追加する。
    ggml_build_forward_expand(gf, out_ext); // 外部パックから融合Attentionまでの依存グラフを追加する。
    ggml_build_forward_expand(gf, dot_ext); // 外部パックから内積までの依存グラフを追加する。
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) { // グラフ中のすべての演算が指定バックエンドに対応しているか確認する。
        auto * node = ggml_graph_node(gf, i); // 現在確認するグラフノードを取り出す。
        require(ggml_backend_supports_op(backend, node), std::string("unsupported op: ") + ggml_op_name(node->op)); // 別バックエンドへの暗黙フォールバックで検証を通過させない。
    } // この処理または定義のブロックを閉じる。
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer( // 確保したバックエンドバッファを自動解放する所有ポインタを用意する。
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free); // グラフの実データを対象バックエンドへ割り当てる。
    require(bool(buffer), "backend allocation failed"); // データ領域の確保に失敗していないか確認する。
    ggml_backend_buffer_clear(buffer.get(), 0xa5); // 未初期化領域への依存を見つけやすい非ゼロ値で領域を埋める。
    q.upload(); k.upload(); v.upload(); mask.upload(); sinks.upload(); // すべてのテスト入力を対象バックエンドへ転送する。
    auto qwords = q.packed(), kwords = k.packed(); // ホスト側の独立な要素単位パックでQ/Kの期待値を求める。
    auto dirty_q = qwords; // 未使用ビットを汚すQuery列を参照値から複製する。
    if (d & 31) { // 最後の語にパディングがある場合だけ不正な未使用ビットを作る。
        const uint32_t garbage = ~((uint32_t(1) << (d & 31)) - 1); // 実特徴ではない上位ビットだけを1にするマスクを作る。
        const size_t words = size_t((d + 31)/32); // 一つのQueryベクトルが占める語数を求める。
        for (size_t i = words - 1; i < dirty_q.size(); i += words) { dirty_q[i] |= garbage; } // 各Queryの末尾語にだけゴミビットを立てる。
    } // この処理または定義のブロックを閉じる。
    ggml_backend_tensor_set(qext, dirty_q.data(), 0, ggml_nbytes(qext)); // パディングを汚したQueryを外部入力テンソルへ転送する。
    ggml_backend_tensor_set(kext, kwords.data(), 0, ggml_nbytes(kext)); // Key側のパディングはゼロのままにして不一致を意図的に作る。
    require(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS, "graph compute failed"); // 指定バックエンドで全演算を実行し成功を確認する。
    std::vector<uint32_t> actual_q(qwords.size()), actual_k(kwords.size()); // 実際のQuery/Keyパック結果を回収する領域を用意する。
    ggml_backend_tensor_get(qb, actual_q.data(), 0, ggml_nbytes(qb)); // カーネルが作ったQueryビット列を読み出す。
    ggml_backend_tensor_get(kb, actual_k.data(), 0, ggml_nbytes(kb)); // カーネルが作ったKeyビット列を読み出す。
    require(actual_q == qwords && actual_k == kwords, "packing/strides/zero/padding mismatch"); // 型・ストライド・負のゼロ・末尾ビットのパック結果を厳密比較する。
    const auto dots = read_f32(dot), dots_ext = read_f32(dot_ext); // 通常入力と汚れた外部入力の内積結果を回収する。
    const auto result = read_f32(out), result_ext = read_f32(out_ext); // 両入力経路の融合Attention出力を回収する。
    double max_error = 0; // このケースの最大絶対誤差を初期化する。
    for (int b = 0; b < bq; ++b) // 全Queryバッチについて参照計算を行う。
    for (int h = 0; h < hq; ++h) // 全Queryヘッドについて参照計算を行う。
    for (int iq = 0; iq < nq; ++iq) { // 全Query位置について参照計算を行う。
        std::vector<double> scores(nk, -INFINITY); // 参照実装では検証のため明示的なdoubleスコア列を持つ。
        const double sink = use_sink ? sinks.get(h, 0, 0, 0) : -INFINITY; // sinkを使う場合だけ対応するロジットを参照計算へ含める。
        double maximum = sink; // softmaxの最大値探索をsinkまたは負無限大から始める。
        // Same ALiBi head convention as ggml FlashAttention, including non-powers of two.
        const float m0 = std::pow(2.0f, -bias/4.0f), m1 = std::pow(2.0f, -bias/8.0f); // 6ヘッドに対し4ヘッド群と残りの群のALiBi底を求める。
        const float slope = bias == 0 ? 1.0f : h < 4 ? std::pow(m0, float(h+1)) : std::pow(m1, float(2*(h-4)+1)); // 既存GGMLと同じ非2冪ヘッド数のALiBi規約を適用する。
        for (int ik = 0; ik < nk; ++ik) { // 各Keyのスコアを独立に計算する。
            int reference_dot = 0; // ビット演算を使わない内積の参照値をゼロ初期化する。
            for (int j = 0; j < d; ++j) { // パックを介さず実特徴を一つずつ比較する。
                const bool sq = q.get(j, iq, h, b) >= 0; // 現在のQuery特徴が非負か確認する。
                const bool sk = k.get(j, ik, h/(hq/hk), b/(bq/2)) >= 0; // GQAとバッチ共有先のKey特徴が非負か確認する。
                reference_dot += sq == sk ? 1 : -1; // 符号一致なら+1、不一致なら-1を足して内積を作る。
            } // この処理または定義のブロックを閉じる。
            const size_t index = size_t(ik + nk*(iq + nq*(h + hq*b))); // 明示的内積出力の[Nk,Nq,Hq,Bq]位置を求める。
            require(dots[index] == reference_dot && dots_ext[index] == reference_dot, "binary dot/GQA/padding mismatch"); // 通常入力もパディング不一致入力も同じ厳密な整数内積になるか確認する。
            const float mv = kind ? mask.get(ik, iq, h%2, b%2) : 0.0f; // マスクのヘッドとバッチを剰余で繰り返す規約を適用する。
            if (mv == -INFINITY) { continue; } // マスク済みのKeyは参照softmaxから除外する。
            double s = double(scale)*reference_dot; // 内積に指定スケールをdoubleで掛ける。
            if (cap > 0) { s = double(cap)*std::tanh(s/double(cap)); } // 指定時だけdoubleでtanhソフトキャップを適用する。
            scores[ik] = s + double(slope)*mv; // ALiBiで調整した加算マスクをロジットへ加える。
            maximum = std::max(maximum, scores[ik]); // 有効なKeyとsinkの最大値を探索する。
        } // この処理または定義のブロックを閉じる。
        double denominator = 0; // softmaxの分母をdoubleで集計する。
        if (sink != -INFINITY) { denominator += std::exp(sink - maximum); } // 有効なsinkだけを分母へ加える。
        for (double s : scores) { if (s != -INFINITY) { denominator += std::exp(s - maximum); } } // 有効なKeyの指数重みを最大値基準で分母へ加える。
        for (int j = 0; j < dv; ++j) { // Valueの各特徴について参照出力を作る。
            double expected = 0; // このValue特徴の参照累積値を初期化する。
            for (int ik = 0; ik < nk; ++ik) { // 全Keyの重み付きValueを集計する。
                if (scores[ik] != -INFINITY) { // マスクされていないキーだけValueの加算へ参加させる。
                    expected += std::exp(scores[ik] - maximum)*v.get(j, ik, h/(hq/hv), 0); // Valueヘッドの共有先を選びdoubleで重み付き和を計算する。
                } // この処理または定義のブロックを閉じる。
            } // この処理または定義のブロックを閉じる。
            if (denominator > 0) { expected /= denominator; } // 空の行ではゼロを保ち、有効な分母があれば正規化する。
            const size_t index = size_t(j + dv*(h + hq*(iq + nq*b))); // 融合演算の[Dv,Hq,Nq,Bq]配置に対応する出力位置を求める。
            require(std::isfinite(result[index]) && std::isfinite(result_ext[index]), "non-finite output (all-masked row?)"); // 全マスク行を含めNaNや無限大を一切許容しない。
            const double error = std::max(std::abs(result[index] - expected), std::abs(result_ext[index] - expected)); // 通常入力と外部パック入力のうち大きい方の参照誤差を求める。
            max_error = std::max(max_error, error); // このケースで観測した最大絶対誤差を更新する。
            require(error < 5e-5, "attention error exceeded 5e-5: " + std::to_string(error)); // 両方の経路が許容誤差5e-5以内であることを要求する。
        } // この処理または定義のブロックを閉じる。
    } // この処理または定義のブロックを閉じる。
    std::printf("PASS case=%d D=%d dtype=%s layout=%d mask=%d max_abs_error=%.3g\n", // 成功したケースの条件と最大誤差を表示する。
                id, d, ggml_type_name(type), id%3, kind, max_error); // ケース番号・実次元・型・レイアウト・マスク種別を記録する。
} // この処理または定義のブロックを閉じる。

int main(int argc, char ** argv) { // テスト用の入力と実行条件を設定して検証を開始する。
    const char * name = "CPU"; // 指定がなければCPUバックエンドを検証する。
    if (argc == 3 && std::strcmp(argv[1], "--backend") == 0) { // 明示的なバックエンド選択オプションがあるか確認する。
        name = argv[2]; // CUDA0など指定されたバックエンド名を採用する。
    } else if (argc != 1) { // 定義していない引数形式を拒否する。
        std::fprintf(stderr, "usage: %s [--backend CPU|CUDA0]\n", argv[0]); return 1; // 正しい起動形式を示して非ゼロ終了する。
    } // この処理または定義のブロックを閉じる。
    ggml_backend_load_all(); // ビルド済みのバックエンドを登録する。
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend( // バックエンドを例外時も自動解放する所有ポインタを用意する。
        ggml_backend_init_by_name(name, nullptr), ggml_backend_free); // 名前を指定して対象バックエンドを初期化する。
    if (!backend) { // 指定バックエンドが使用可能か確認する。
        std::fprintf(stderr, "backend not available: %s\n", name); return 77; // 実行不能を成功と区別しCTestのスキップコード77で返す。
    } // この処理または定義のブロックを閉じる。
    if (ggml_backend_is_cpu(backend.get())) { ggml_backend_cpu_set_n_threads(backend.get(), 3); } // CPU実行では3スレッドに制限し行の分担も検証する。
    try { // 失敗を捕捉して診断を返せるよう検証を開始する。
        int id = 0; // 実行ケース番号をゼロから開始する。
        for (int d : {1, 7, 31, 32, 33, 63, 64, 65, 96, 127, 128, 129, 257}) { // 32ビット境界の前後と広い次元を含む13形状を選ぶ。
            for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) { // F32・F16・BF16の三つのストレージ型を検証する。
                run_case(backend.get(), d, type, id++); // 全39設定を実行しケース番号を進める。
            } // この処理または定義のブロックを閉じる。
        } // この処理または定義のブロックを閉じる。
        std::printf("All %d BitAttention configurations passed on %s.\n", id, name); // 全ケースが成功したことと使用バックエンドを報告する。
    } catch (const std::exception & e) { // 検証中の例外を捕捉して失敗を報告する。
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; // 失敗した検査の理由を表示して非ゼロ終了する。
    } // この処理または定義のブロックを閉じる。
    return 0; // 検証または処理の正常終了を返す。
} // この処理または定義のブロックを閉じる。
