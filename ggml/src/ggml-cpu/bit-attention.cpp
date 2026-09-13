#include "bit-attention.h" // CPUカーネルのCリンケージ宣言を取り込む。
#include "ggml-cpu-impl.h" // CPUワーカーパラメータとバリアを使用する。
#include "ggml-impl.h" // ggml内部の実行定義を取り込む。
#include <algorithm> // 最大値と範囲制限を計算する。
#include <cmath> // 安定化softmaxとソフトキャップを計算する。
#include <cstring> // 非連続テンソルから安全にスカラーを読み取る。
#include <limits> // 負の無限大とサイズ上限を取得する。

static float bit_load(const ggml_tensor * t, int64_t d, int64_t n, int64_t h, int64_t b) { // 任意のbyte strideを尊重してスカラーを読み取る。
    const char * p = static_cast<const char *>(t->data) + d*t->nb[0] + n*t->nb[1] + h*t->nb[2] + b*t->nb[3]; // 各軸のbyte offsetを足し合わせる。
    if (t->type == GGML_TYPE_F32) { // FP32入力では変換せず値を取得する。
        float x; // アラインメントに依存しない読み取り先を確保する。
        std::memcpy(&x, p, sizeof(x)); // 未整列アクセスとstrict-aliasing違反を避ける。
        return x; // FP32の値を返す。
    } // FP32の読み取りを終了する。
    if (t->type == GGML_TYPE_F16) { // FP16入力をFP32へ復元する。
        ggml_fp16_t x; // FP16のビット表現を保持する。
        std::memcpy(&x, p, sizeof(x)); // ストライドで指定された半精度値を読み取る。
        return ggml_fp16_to_fp32(x); // ggmlの標準変換を使用する。
    } // FP16の読み取りを終了する。
    ggml_bf16_t x; // コンストラクタで検証済みのBF16値を保持する。
    std::memcpy(&x, p, sizeof(x)); // BF16のビット表現を安全に読み取る。
    return ggml_bf16_to_fp32(x); // BF16値をFP32へ変換する。
} // 型とストライドに対応した読み取りを終了する。

static bool bit_nonnegative(float x) { // 非正規化数のflush設定に依存せずx>=0の符号規則を実装する。
    uint32_t bits; // IEEE754のビット表現を保持する。
    std::memcpy(&bits, &x, sizeof(bits)); // 浮動小数点演算を行わず符号と大きさを取得する。
    const uint32_t magnitude = bits & 0x7fffffffu; // 符号を除いた指数と仮数を取得する。
    return magnitude <= 0x7f800000u && ((bits & 0x80000000u) == 0 || magnitude == 0); // 負の微小値を0、両符号のゼロを1、NaNを0に分類する。
} // subnormalと負のゼロに対応する符号判定を終了する。

static uint32_t bit_popcount(uint32_t x) { // CPU固有命令を必須としない32bit人口カウントを実装する。
    x -= (x >> 1) & 0x55555555u; // 各2bit領域の1の数を求める。
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u); // 各4bit領域へ個数を集約する。
    x = (x + (x >> 4)) & 0x0f0f0f0fu; // 各byteの1の数を下位4bitへ配置する。
    x += x >> 8; // 隣接byteの個数を加算する。
    x += x >> 16; // 上下16bitの個数を加算する。
    return x & 0x3fu; // 0から32の範囲のカウントだけを返す。
} // SWARによる人口カウントを終了する。

size_t ggml_bit_attn_ext_work_size(const ggml_tensor * dst) { // 全ワーカーが共有する一時パック領域のサイズを計算する。
    const ggml_tensor * q = dst->src[0]; // クエリの形状を参照する。
    const ggml_tensor * k = dst->src[1]; // キーの形状を参照する。
    const size_t words = static_cast<size_t>((q->ne[0] + 31)/32); // 末尾の端数を含む32bitワード数を求める。
    const size_t qr = static_cast<size_t>(ggml_nrows(q)); // クエリの全行数を求める。
    const size_t kr = static_cast<size_t>(ggml_nrows(k)); // キーの全行数を求める。
    GGML_ASSERT(qr <= SIZE_MAX - kr && qr + kr <= SIZE_MAX/sizeof(uint32_t)/words); // 加算とサイズ乗算のオーバーフローを検出する。
    return (qr + kr)*words*sizeof(uint32_t); // 系列長に対して線形なスクラッチ容量を返す。
} // CPUスクラッチ容量の計算を終了する。

static void bit_pack_rows(const ggml_tensor * t, uint32_t * packed, int ith, int nth) { // ワーカー間で行を分割して符号ビットを作成する。
    const int64_t words = (t->ne[0] + 31)/32; // 各行に必要なワード数を求める。
    for (int64_t row = ith; row < ggml_nrows(t); row += nth) { // 同じ行を複数ワーカーが書かないよう循環分割する。
        const int64_t n = row % t->ne[1]; // 行番号からトークン番号を復元する。
        const int64_t h = row/t->ne[1] % t->ne[2]; // 行番号からヘッド番号を復元する。
        const int64_t b = row/(t->ne[1]*t->ne[2]); // 行番号からバッチ番号を復元する。
        for (int64_t w = 0; w < words; ++w) { // 各32次元の符号を1ワードに詰める。
            uint32_t value = 0; // パディング部分も含めてゼロで初期化する。
            for (int bit = 0; bit < 32 && w*32 + bit < t->ne[0]; ++bit) { // 有効次元だけを読み取りシフト幅32を避ける。
                value |= static_cast<uint32_t>(bit_nonnegative(bit_load(t, w*32 + bit, n, h, b))) << bit; // 非負を1、負を0として格納する。
            } // ワード内の符号化を終了する。
            packed[row*words + w] = value; // 完成したワードを共有スクラッチへ保存する。
        } // 現在行のパッキングを終了する。
    } // このワーカーが担当する行の処理を終了する。
} // QまたはKの並列パッキングを終了する。

void ggml_compute_forward_bit_attn_ext(const ggml_compute_params * params, ggml_tensor * dst) { // パックとonline softmaxを組み合わせてCPU forwardを実行する。
    const ggml_tensor * q = dst->src[0]; // クエリ入力を参照する。
    const ggml_tensor * k = dst->src[1]; // キー入力を参照する。
    const ggml_tensor * v = dst->src[2]; // 高精度の値入力を参照する。
    const ggml_tensor * mask = dst->src[3]; // 系列境界と因果制約の加算マスクを参照する。
    const ggml_tensor * sinks = dst->src[4]; // オプションのsink logitを参照する。
    const ggml_tensor * bias = dst->src[5]; // オプションの追加バイアスを参照する。
    const int64_t words = (q->ne[0] + 31)/32; // 符号ベクトルのワード数を求める。
    GGML_ASSERT(params->wsize >= ggml_bit_attn_ext_work_size(dst)); // 実行計画が十分なスクラッチを確保したことを確認する。
    auto * qb = static_cast<uint32_t *>(params->wdata); // 共有スクラッチ先頭をQのパック領域とする。
    uint32_t * kb = qb + ggml_nrows(q)*words; // Qの直後にKのパック領域を配置する。
    bit_pack_rows(q, qb, params->ith, params->nth); // 各クエリをグラフ実行ごとに1回だけ符号化する。
    bit_pack_rows(k, kb, params->ith, params->nth); // 全クエリで共有するキーを1回だけ符号化する。
    ggml_barrier(params->threadpool); // すべてのQ/Kワードの書き込み完了を待ち合わせる。
    float config[3]; // スケール・ALiBi・ソフトキャップを保持する。
    std::memcpy(config, dst->op_params, sizeof(config)); // op_paramsから浮動小数点設定を読み取る。
    const float scale = config[0]; // 符号内積用スケールを取得する。
    const float max_bias = config[1]; // ALiBiの係数を取得する。
    const float cap = config[2]; // logitのソフトキャップ値を取得する。
    int64_t hpow = 1; // ヘッド数以下の最大2冪を求める準備をする。
    while (hpow <= q->ne[2]/2) { hpow <<= 1; } // 非2冪のヘッド数にも対応するALiBi基準を求める。
    const float m0 = std::pow(2.0f, -max_bias/hpow); // ALiBi前半ヘッドの指数基数を計算する。
    const float m1 = std::pow(2.0f, -0.5f*max_bias/hpow); // ALiBi追加ヘッドの指数基数を計算する。
    const float neg_inf = -std::numeric_limits<float>::infinity(); // 未選択行のsoftmax最大値を表す。
    for (int64_t row = params->ith; row < ggml_nrows(q); row += params->nth) { // クエリ行をワーカー間で重複なく分割する。
        const int64_t n = row % q->ne[1]; // クエリのトークン番号を求める。
        const int64_t h = row/q->ne[1] % q->ne[2]; // クエリのヘッド番号を求める。
        const int64_t b = row/(q->ne[1]*q->ne[2]); // クエリのバッチ番号を求める。
        const int64_t kh = h/(q->ne[2]/k->ne[2]); // GQAで共有するKヘッドを特定する。
        const int64_t vh = h/(q->ne[2]/v->ne[2]); // GQAで共有するVヘッドを特定する。
        const int64_t kbch = b/(q->ne[3]/k->ne[3]); // 共有Kバッチを特定する。
        const int64_t vbch = b/(q->ne[3]/v->ne[3]); // 共有Vバッチを特定する。
        const float slope = max_bias > 0 ? (h < hpow ? std::pow(m0, h + 1) : std::pow(m1, 2*(h - hpow) + 1)) : 1.0f; // ggmlと同じヘッド別ALiBi傾きを求める。
        float * out = reinterpret_cast<float *>(static_cast<char *>(dst->data) + h*dst->nb[1] + n*dst->nb[2] + b*dst->nb[3]); // [DV,HQ,NQ,BQ]順の出力行を特定する。
        std::fill(out, out + v->ne[0], 0.0f); // weighted sumの累積値をゼロにする。
        float maximum = sinks ? bit_load(sinks, h, 0, 0, 0) : neg_inf; // sinkがある場合はそのlogitを初期最大値に含める。
        float denominator = maximum == neg_inf ? 0.0f : 1.0f; // sinkは値を持たずsoftmax分母だけに寄与させる。
        for (int64_t j = 0; j < k->ne[1]; ++j) { // KV系列を1回走査してAttention行列の保存を避ける。
            const float mv = mask ? bit_load(mask, j, n, h % mask->ne[2], b % mask->ne[3]) : 0.0f; // 対応する加算マスクをストライド付きで読み取る。
            const float bv = bias ? bit_load(bias, j, n, h % bias->ne[2], b % bias->ne[3]) : 0.0f; // 追加バイアスを周期的にブロードキャストする。
            if (mv == neg_inf || bv == neg_inf) { continue; } // 無効キーを指数計算とV読み取りの前に除外する。
            const int64_t krow = j + k->ne[1]*(kh + k->ne[2]*kbch); // 対象キーのパック行を特定する。
            uint32_t distance = 0; // Hamming距離をゼロから累積する。
            for (int64_t w = 0; w < words; ++w) { distance += bit_popcount(qb[row*words + w] ^ kb[krow*words + w]); } // XORとpopcountで32次元ずつ不一致数を求める。
            float score = static_cast<float>(q->ne[0] - 2*static_cast<int64_t>(distance))*scale; // 符号内積D-2Hをスケールする。
            if (cap > 0.0f) { score = cap*std::tanh(score/cap); } // softmaxの前に必要なソフトキャップを適用する。
            score += slope*mv + bv; // ALiBi付きマスクとスケール済み追加バイアスを加える。
            const float next_max = std::max(maximum, score); // これまでの最大logitを更新する。
            const float alpha = maximum == neg_inf ? 0.0f : std::exp(maximum - next_max); // 既存累積値を新しい最大値に合わせて縮小する。
            const float probability = std::exp(score - next_max); // 新しいキーの非正規化確率を求める。
            for (int64_t d = 0; d < v->ne[0]; ++d) { out[d] = alpha*out[d] + probability*bit_load(v, d, j, vh, vbch); } // FP32で確率と高精度Vの積を逐次累積する。
            denominator = alpha*denominator + probability; // 同じ再スケーリングでsoftmax分母を更新する。
            maximum = next_max; // 次のキーで使う基準最大値を保存する。
        } // 全KVトークンのonline更新を終了する。
        const float inv = denominator > 0.0f ? 1.0f/denominator : 0.0f; // 全要素マスク行をゼロ出力にして0除算を防ぐ。
        for (int64_t d = 0; d < v->ne[0]; ++d) { out[d] *= inv; } // weighted sumを最終的なsoftmax分母で正規化する。
    } // このワーカーの全クエリ出力を完了する。
} // CPUのBitAttention forwardを終了する。
