#include "ops.h" // CPU演算ディスパッチの関数宣言を読み込む。
#include "ggml-cpu.h" // CPUバックエンドの公開APIを読み込む。
#include "ggml-cpu-impl.h" // CPU計算パラメータと内部定義を読み込む。
#include "ggml-impl.h" // GGML内部ヘルパーと演算パラメータを読み込む。

#include <algorithm> // 最大値・範囲初期化などの標準アルゴリズムを読み込む。
#include <cmath> // 指数関数・tanh・有限値検査を読み込む。
#include <cstdint> // 幅を固定した整数型を読み込む。
#include <cstring> // 型エイリアスに依存しないバイトコピーを読み込む。

// CPU reference backend. All indexing is stride-aware, including dimension 0.
// Each worker owns whole output rows; neither a score matrix nor shared scratch
// is needed by BIT_ATTN_EXT. BIT_MUL_MAT is the explicit unfused alternative.
static const char * bit_ptr(const ggml_tensor * t, int64_t d, int64_t n, int64_t h, int64_t b) { // 全4軸のバイトストライドを使って要素アドレスを求める。
    return static_cast<const char *>(t->data) + d*t->nb[0] + n*t->nb[1] + h*t->nb[2] + b*t->nb[3]; // 全4軸のストライドから目的要素のアドレスを求める。
} // この処理または定義のブロックを閉じる。

static float bit_float(const ggml_tensor * t, int64_t d, int64_t n, int64_t h, int64_t b) { // CPU入力を対応する型からF32へ読み出す。
    const char * p = bit_ptr(t, d, n, h, b); // CPU入力の対象要素のアドレスを取得する。
    switch (t->type) { // テンソル型または演算番号に対応する実装へ分岐する。
        case GGML_TYPE_F32:  { float       x; memcpy(&x, p, sizeof(x)); return x; } // F32のバイト列をアラインメントに依存せず読み出す。
        case GGML_TYPE_F16:  { ggml_fp16_t x; memcpy(&x, p, sizeof(x)); return ggml_fp16_to_fp32(x); } // FP16を安全にコピーしてからF32へ変換する。
        case GGML_TYPE_BF16: { ggml_bf16_t x; memcpy(&x, p, sizeof(x)); return ggml_bf16_to_fp32(x); } // BF16を安全にコピーしてからF32へ変換する。
        default: GGML_ABORT("BitAttention: unsupported floating-point type"); // 対応しない型を誤った浮動小数点として読まないよう停止する。
    } // この処理または定義のブロックを閉じる。
} // この処理または定義のブロックを閉じる。

static uint32_t bit_word(const ggml_tensor * t, int64_t w, int64_t n, int64_t h, int64_t b) { // CPU入力から型エイリアスに依存せず32ビット語を読む。
    uint32_t x; // 読み出す32ビット語の格納先を用意する。
    memcpy(&x, bit_ptr(t, w, n, h, b), sizeof(x)); // アラインメントや型エイリアスへ依存せず32ビット語を読む。
    return x; // 読み出しまたは縮約の結果を返す。
} // この処理または定義のブロックを閉じる。

static int bit_popcount(uint32_t x) { // 32ビット語に立っているビットの数を求める。
#if defined(__GNUC__) || defined(__clang__) // 組み込みpopcountを利用できるコンパイラを選別する。
    return __builtin_popcount(x); // コンパイラ組み込みの32ビットpopcountを利用する。
#else // 代替プラットフォーム用の実装へ切り替える。
    // Portable fallback, with no CPU instruction-set assumption (including ARM).
    x -= (x >> 1) & 0x55555555u; // SWAR法で各2ビット区間の1の個数を数える。
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u); // 各4ビット区間へ個数をまとめる。
    x = (x + (x >> 4)) & 0x0f0f0f0fu; // 各バイトの1の個数を下位4ビットへ集計する。
    x += x >> 8; // 隣接するバイトの個数を加算する。
    x += x >> 16; // 語全体の個数を下位バイトへまとめる。
    return int(x & 0x3fu); // 0から32までのpopcount結果を下位6ビットから返す。
#endif // プラットフォーム別実装の条件分岐を閉じる。
} // この処理または定義のブロックを閉じる。

static float bit_dot(const ggml_tensor * q, const ggml_tensor * k, int32_t d, // XORとpopcountで符号ベクトルの内積を計算する。
                     int64_t iq, int64_t ik, int64_t h, int64_t b) { // Query位置・Key位置・出力ヘッド・出力バッチを受け取る。
    const int64_t kh = h/(q->ne[2]/k->ne[2]); // GQAの共有率に従いQueryヘッドをKeyヘッドへ対応付ける。
    const int64_t kb = b/(q->ne[3]/k->ne[3]); // バッチの共有率に従いKeyのバッチ番号を求める。
    const uint32_t tail = (d & 31) ? (uint32_t(1) << (d & 31)) - 1 : UINT32_MAX; // 末尾語の有効ビットだけを残し32ビットシフトの未定義動作を避ける。
    int64_t distance = 0; // 特徴全体のHamming距離をゼロから数える。
    for (int64_t w = 0; w < q->ne[0]; ++w) { // 特徴を32ビット語ごとに走査する。
        uint32_t x = bit_word(q, w, iq, h, b) ^ bit_word(k, w, ik, kh, kb); // 対応するQueryとKeyの語をXORして符号の不一致を検出する。
        if (w + 1 == q->ne[0]) { // 最後の語だけパディング除去を適用する。
            x &= tail; // Ignore padding even for externally supplied packed words.
        } // この処理または定義のブロックを閉じる。
        distance += bit_popcount(x); // この語の不一致ビット数を距離へ加える。
    } // この処理または定義のブロックを閉じる。
    return float(int64_t(d) - 2*distance); // 一致を+1、不一致を-1として二値内積へ変換する。
} // この処理または定義のブロックを閉じる。

void ggml_compute_forward_bit_pack(const ggml_compute_params * params, ggml_tensor * dst) { // CPUスレッドで入力の符号を32ビット語へパックする。
    const ggml_tensor * x = dst->src[0]; // パック対象または復号対象の入力を取得・計算する。
    const int64_t rows = ggml_nrows(x); // 処理するベクトルの総数を取得・計算する。
    auto * out = static_cast<uint32_t *>(dst->data); // 出力ストレージの先頭を取得・計算する。
    for (int64_t row = params->ith; row < rows; row += params->nth) { // スレッド番号を起点として行を重複なく分担する。
        const int64_t n = row % x->ne[1]; // 現在のトークン番号を取得・計算する。
        const int64_t h = (row/x->ne[1]) % x->ne[2]; // 現在のヘッド番号を取得・計算する。
        const int64_t b = row/(x->ne[1]*x->ne[2]); // 現在のバッチ番号を取得・計算する。
        for (int64_t w = 0; w < dst->ne[0]; ++w) { // 特徴を32ビット語ごとに走査する。
            uint32_t word = 0; // パディングを含む出力語をゼロで初期化する。
            for (int j = 0; j < 32 && 32*w + j < x->ne[0]; ++j) { // 末尾の余剰領域を読まず有効な特徴の符号を調べる。
                if (bit_float(x, 32*w + j, n, h, b) >= 0.0f) { // ゼロを含む非負の特徴を符号ビット1として扱う。
                    word |= uint32_t(1) << j; // 非負の特徴に対応するビットを1にする。
                } // この処理または定義のブロックを閉じる。
            } // この処理または定義のブロックを閉じる。
            out[row*dst->ne[0] + w] = word; // 連続したI32ストレージへ完成した語を書き込む。
        } // この処理または定義のブロックを閉じる。
    } // この処理または定義のブロックを閉じる。
} // この処理または定義のブロックを閉じる。

void ggml_compute_forward_bit_mul_mat(const ggml_compute_params * params, ggml_tensor * dst) { // CPUで二値Query-Keyスコア行列を計算する。
    const ggml_tensor * k = dst->src[0]; // Key入力テンソルを取得・計算する。
    const ggml_tensor * q = dst->src[1]; // Query入力テンソルを取得・計算する。
    const int32_t d = ggml_get_op_params_i32(dst, 0); // 実特徴次元またはこのlaneが読む特徴番号を取得・計算する。
    const int64_t rows = ggml_nrows(q); // 処理するベクトルの総数を取得・計算する。
    auto * out = static_cast<float *>(dst->data); // 出力ストレージの先頭を取得・計算する。
    for (int64_t row = params->ith; row < rows; row += params->nth) { // スレッド番号を起点として行を重複なく分担する。
        const int64_t iq = row % q->ne[1]; // Queryのトークン番号を取得・計算する。
        const int64_t h  = (row/q->ne[1]) % q->ne[2]; // 現在のヘッド番号を取得・計算する。
        const int64_t b  = row/(q->ne[1]*q->ne[2]); // 現在のバッチ番号を取得・計算する。
        for (int64_t ik = 0; ik < k->ne[1]; ++ik) { // 全てのKeyトークンを順に処理する。
            out[row*k->ne[1] + ik] = bit_dot(q, k, d, iq, ik, h, b); // GGMLの[Nk,Nq,Hq,Bq]順で二値内積スコアを保存する。
        } // この処理または定義のブロックを閉じる。
    } // この処理または定義のブロックを閉じる。
} // この処理または定義のブロックを閉じる。

void ggml_compute_forward_bit_attn_ext(const ggml_compute_params * params, ggml_tensor * dst) { // CPUで二値Q/Kとonline softmaxとV集計を融合する。
    const ggml_tensor * q = dst->src[0]; // Query入力テンソルを取得・計算する。
    const ggml_tensor * k = dst->src[1]; // Key入力テンソルを取得・計算する。
    const ggml_tensor * v = dst->src[2]; // 浮動小数点Value入力テンソルを取得・計算する。
    const ggml_tensor * mask  = dst->src[3]; // 省略可能な加算マスクを取得・計算する。
    const ggml_tensor * sinks = dst->src[4]; // softmax分母用の省略可能なsinkを取得・計算する。
    const float scale = ggml_get_op_params_f32(dst, 0); // 二値内積に掛けるロジットのスケールを取得・計算する。
    const float bias  = ggml_get_op_params_f32(dst, 1); // ALiBiヘッド傾斜の制御値を取得・計算する。
    const float cap   = ggml_get_op_params_f32(dst, 2); // 任意のtanhソフトキャップ幅を取得・計算する。
    const int32_t d   = ggml_get_op_params_i32(dst, 3); // 実特徴次元またはこのlaneが読む特徴番号を取得・計算する。
    const int64_t rows = ggml_nrows(q); // 処理するベクトルの総数を取得・計算する。
    int64_t h_pow2 = 1; // ヘッド数以下の最大の2の累乗を探索する初期値を取得・計算する。
    while (h_pow2 <= q->ne[2]/2) { // ヘッド数を超えない範囲で2の累乗を大きくする。
        h_pow2 *= 2; // ALiBiのヘッド群を区切る2の累乗を更新する。
    } // この処理または定義のブロックを閉じる。
    const float m0 = std::pow(2.0f, -bias/float(h_pow2)); // ALiBiの先頭ヘッド群に使う傾斜の底を取得・計算する。
    const float m1 = std::pow(2.0f, -0.5f*bias/float(h_pow2)); // 残りのALiBiヘッド群に使う傾斜の底を取得・計算する。
    for (int64_t row = params->ith; row < rows; row += params->nth) { // スレッド番号を起点として行を重複なく分担する。
        const int64_t iq = row % q->ne[1]; // Queryのトークン番号を取得・計算する。
        const int64_t h  = (row/q->ne[1]) % q->ne[2]; // 現在のヘッド番号を取得・計算する。
        const int64_t b  = row/(q->ne[1]*q->ne[2]); // 現在のバッチ番号を取得・計算する。
        const int64_t vh = h/(q->ne[2]/v->ne[2]); // 共有先のValueヘッド番号を取得・計算する。
        const int64_t vb = b/(q->ne[3]/v->ne[3]); // 共有先のValueバッチ番号を取得・計算する。
        const float slope = bias == 0.0f ? 1.0f : h < h_pow2 // ALiBiが無効なら傾斜を1とし有効ならヘッド群を選ぶ。
            ? std::pow(m0, float(h + 1)) : std::pow(m1, float(2*(h - h_pow2) + 1)); // 2の累乗ヘッド群と残りの群で既存GGMLの傾斜規則を適用する。
        auto * acc = reinterpret_cast<float *>(static_cast<char *>(dst->data) // 出力行をF32の累積ストレージとして直接使用する。
                   + h*dst->nb[1] + iq*dst->nb[2] + b*dst->nb[3]); // [Dv,Hq,Nq,Bq]の出力配置からこのQuery行の先頭を求める。
        std::fill(acc, acc + v->ne[0], 0.0f); // このQuery行の出力累積値をゼロ初期化する。
        float m = sinks ? bit_float(sinks, h, 0, 0, 0) : -INFINITY; // sinkがあれば初期最大値とし、なければ負の無限大にする。
        float sum = m == -INFINITY ? 0.0f : 1.0f; // 有限sinkの寄与を分母へ1として入れる。
        for (int64_t ik = 0; ik < k->ne[1]; ++ik) { // 全てのKeyトークンを順に処理する。
            const float mv = mask ? bit_float(mask, ik, iq, h % mask->ne[2], b % mask->ne[3]) : 0.0f; // 現在のQuery-Key対の加算マスク値を取得・計算する。
            if (mv == -INFINITY) { // マスクされたキーは指数計算より前に除外する。
                continue; // この要素を集計せず次の反復へ進む。
            } // この処理または定義のブロックを閉じる。
            float score = scale*bit_dot(q, k, d, iq, ik, h, b); // 二値内積にAttentionスケールを掛けてロジットを得る。
            if (cap > 0.0f) { // 正のソフトキャップが指定された場合だけtanhを適用する。
                score = cap*std::tanh(score/cap); // 指定時は既存Attentionと同じtanhソフトキャップを適用する。
            } // この処理または定義のブロックを閉じる。
            score += slope*mv; // マスクの有限バイアスへALiBiヘッド傾斜を掛けて加える。
            const float m_new = std::max(m, score); // 新しいキーを含む最大ロジットへ更新する。
            const float alpha = m == -INFINITY ? 0.0f : std::exp(m - m_new); // 過去の累積値を新しい最大値の基準へ換算する。
            const float p = std::exp(score - m_new); // 最大値を引いて現在のキーの安定した指数重みを求める。
            for (int64_t dv = 0; dv < v->ne[0]; ++dv) { // Valueの有効な特徴を順に処理する。
                acc[dv] = alpha*acc[dv] + p*bit_float(v, dv, ik, vh, vb); // 過去の出力を再スケールし新しいVの重み付き値を足す。
            } // この処理または定義のブロックを閉じる。
            sum = alpha*sum + p; // 同じ再スケールをsoftmaxの分母にも適用する。
            m = m_new; // 次のキーまたはタイルへ最大値を引き継ぐ。
        } // この処理または定義のブロックを閉じる。
        const float inv_sum = sum > 0.0f ? 1.0f/sum : 0.0f; // 全マスク行では逆数をゼロにしてゼロ出力を保つ。
        for (int64_t dv = 0; dv < v->ne[0]; ++dv) { // Valueの有効な特徴を順に処理する。
            acc[dv] *= inv_sum; // 累積したVをsoftmaxの分母で正規化する。
        } // この処理または定義のブロックを閉じる。
    } // この処理または定義のブロックを閉じる。
} // この処理または定義のブロックを閉じる。
