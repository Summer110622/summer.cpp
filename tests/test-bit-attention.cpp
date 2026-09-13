#include "ggml.h" // テンソルとBitAttentionの公開APIを使用する。
#include "ggml-alloc.h" // バックエンド上にテスト用テンソルを確保する。
#include "ggml-backend.h" // CPUと利用可能なCUDAデバイスを同じ方法で検証する。
#include <algorithm> // softmax最大値と最大誤差を計算する。
#include <cmath> // 独立した倍精度softmax参照実装を構築する。
#include <cstdio> // ケース数と誤差およびスキップ理由を表示する。
#include <cstring> // 浮動小数点型のbyte表現を安全にコピーする。
#include <functional> // テスト入力の生成関数を渡す。
#include <limits> // 無効logitに使う負の無限大を取得する。
#include <memory> // コンテキストとバッファの解放を例外時にも保証する。
#include <random> // 再現可能な固定seedのテスト入力を作成する。
#include <stdexcept> // Releaseビルドでも無効化されない検証失敗を通知する。
#include <string> // バックエンド名とエラーメッセージを保持する。
#include <vector> // 独立した参照値と生byte領域を保持する。

struct bit_case { // 数値検証で変化させる形状とオプションを定義する。
    int dk = 33, dv = 17, nq = 3, nk = 9, hq = 4, hk = 2, bq = 2, bk = 1; // 端数次元とGQAおよびバッチ共有を既定ケースとする。
    ggml_type qt = GGML_TYPE_F32, kt = GGML_TYPE_F16, vt = GGML_TYPE_F16; // Q/K/Vの型を独立に指定する。
    bool strided = false; // trueなら特徴軸も非連続なテンソルビューを検証する。
    int mask_mode = 0; // 0はマスクなし、1はcausal、2は全マスク行、3は先頭128キーを無効にする。
    bool sink = false, bias = false; // sinkと追加logitバイアスの有無を指定する。
    float alibi = 0.0f, cap = 0.0f; // ALiBiとlogit softcapの数値を指定する。
}; // 検証ケースの定義を終了する。

static void require(bool condition, const char * message) { // assertとは異なりReleaseビルドでも条件を検証する。
    if (!condition) { throw std::runtime_error(message); } // 条件を満たさない場合はテストを失敗させる。
} // 共通の条件検証を終了する。

static ggml_tensor * tensor4(ggml_context * ctx, ggml_type type, int d, int n, int h, int b, bool strided) { // 連続または転置ビューの入力を生成する。
    if (!strided) { return ggml_new_tensor_4d(ctx, type, d, n, h, b); } // 通常の連続レイアウトを作成する。
    ggml_tensor * backing = ggml_new_tensor_4d(ctx, type, n, d, h, b); // 特徴次元とトークン次元を入れ替えた実体を確保する。
    return ggml_permute(ctx, backing, 1, 0, 2, 3); // 論理形状を戻しnb[0]が型サイズより大きいビューを返す。
} // 入力レイアウトの構築を終了する。

static std::vector<float> fill_tensor(ggml_tensor * t, const std::function<float(int64_t, int64_t, int64_t, int64_t)> & generate) { // 量子化後と同じ値を参照実装にも保存する。
    std::vector<uint8_t> raw(ggml_nbytes(t), 0); // paddingとstrideを含むテンソルのbyte領域を用意する。
    std::vector<float> values(static_cast<size_t>(ggml_nelements(t))); // 論理的な軸順の参照値を確保する。
    for (int64_t b = 0; b < t->ne[3]; ++b) { // バッチ軸を走査する。
        for (int64_t h = 0; h < t->ne[2]; ++h) { // ヘッド軸を走査する。
            for (int64_t n = 0; n < t->ne[1]; ++n) { // トークン軸を走査する。
                for (int64_t d = 0; d < t->ne[0]; ++d) { // 特徴軸を走査する。
                    float x = generate(d, n, h, b); // 指定した規則で入力値を作成する。
                    uint8_t * dst = raw.data() + d*t->nb[0] + n*t->nb[1] + h*t->nb[2] + b*t->nb[3]; // 論理座標を実際のbyte位置へ写像する。
                    if (t->type == GGML_TYPE_F32) { std::memcpy(dst, &x, sizeof(x)); } // FP32入力をそのまま保存する。
                    else if (t->type == GGML_TYPE_F16) { // FP16への丸めを参照値にも反映する。
                        const ggml_fp16_t f = ggml_fp32_to_fp16(x); // テスト入力をFP16へ変換する。
                        std::memcpy(dst, &f, sizeof(f)); // 半精度の実際のbyte列を保存する。
                        x = ggml_fp16_to_fp32(f); // 参照計算にも同じ丸め後の値を使用する。
                    } else { // BF16入力を作成する。
                        const ggml_bf16_t f = ggml_fp32_to_bf16(x); // テスト入力をBF16へ変換する。
                        std::memcpy(dst, &f, sizeof(f)); // BF16のbyte列を保存する。
                        x = ggml_bf16_to_fp32(f); // BF16丸めを参照計算に反映する。
                    } // 入力型ごとの保存を終了する。
                    values[d + t->ne[0]*(n + t->ne[1]*(h + t->ne[2]*b))] = x; // 独立した論理順序で参照値を保持する。
                } // 特徴軸の初期化を終了する。
            } // トークン軸の初期化を終了する。
        } // ヘッド軸の初期化を終了する。
    } // 全バッチの初期化を終了する。
    ggml_backend_tensor_set(t, raw.data(), 0, raw.size()); // CPUまたはGPUバッファへ同じbyte列を書き込む。
    return values; // カーネルの内部パック処理に依存しない参照値を返す。
} // テスト入力の転送と参照値生成を終了する。

static double at(const ggml_tensor * t, const std::vector<float> & data, int64_t d, int64_t n, int64_t h, int64_t b) { // 論理的な座標から独立参照値を取得する。
    return data[d + t->ne[0]*(n + t->ne[1]*(h + t->ne[2]*b))]; // 実際のbyte strideを使わず保存済みの論理順値を読み取る。
} // 独立参照配列の読み取りを終了する。

static float run_case(ggml_backend_t backend, const bit_case & c) { // 1つのバックエンドで1つの形状を検証する。
    const ggml_init_params init = { 2*1024*1024, nullptr, true }; // メタデータだけをggmlコンテキストに確保する。
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(init), ggml_free); // テスト終了時にコンテキストを必ず解放する。
    require(ctx != nullptr, "ggml_init failed"); // メタデータ確保の失敗を報告する。
    ggml_tensor * q = tensor4(ctx.get(), c.qt, c.dk, c.nq, c.hq, c.bq, c.strided); // クエリを指定したレイアウトで構築する。
    ggml_tensor * k = tensor4(ctx.get(), c.kt, c.dk, c.nk, c.hk, c.bk, c.strided); // GQA共有キーを構築する。
    ggml_tensor * v = tensor4(ctx.get(), c.vt, c.dv, c.nk, c.hk, c.bk, c.strided); // 異なる値次元を持つVを構築する。
    ggml_tensor * mask = c.mask_mode ? tensor4(ctx.get(), GGML_TYPE_F16, c.nk, c.nq, 2, 1, c.strided) : nullptr; // ヘッドとバッチへbroadcastするマスクを作成する。
    ggml_tensor * sinks = c.sink ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, c.hq) : nullptr; // ヘッドごとのsinkを作成する。
    ggml_tensor * bias = c.bias ? tensor4(ctx.get(), GGML_TYPE_F32, c.nk, c.nq, 1, c.bq, c.strided) : nullptr; // バッチごとに異なる追加バイアスを作成する。
    const float scale = 1.0f/std::sqrt(static_cast<float>(c.dk)); // 符号内積の標準スケールを設定する。
    ggml_tensor * output = ggml_bit_attn_ext(ctx.get(), q, k, v, mask, sinks, bias, scale, c.alibi, c.cap); // 実際の公開APIから検証対象ノードを構築する。
    require(ggml_backend_supports_op(backend, output), "backend unexpectedly rejects a supported BitAttention case"); // テスト対象バックエンドが正しくopを公開していることを確認する。
    ggml_cgraph * graph = ggml_new_graph(ctx.get()); // 計算グラフを作成する。
    ggml_build_forward_expand(graph, output); // 入力ビューを含む依存関係を展開する。
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free); // バックエンドの実メモリを確保してRAII管理する。
    require(buffer != nullptr, "backend allocation failed"); // バックエンドメモリの確保失敗を検出する。
    std::mt19937 rng(1234 + c.dk + c.dv); // 各ケースの入力を再現可能にする。
    std::uniform_real_distribution<float> distribution(-2.0f, 2.0f); // 符号と値が偏らない乱数分布を使用する。
    const auto random = [&](int64_t d, int64_t n, int64_t h, int64_t b) { // 通常乱数と符号境界の入力を混在させる。
        if (d == 0 && n == 0 && h == 0 && b == 0) { return -0.0f; } // 負のゼロを非負として扱う規則を検証する。
        if (d == 1) { return n % 2 ? std::numeric_limits<float>::min()/2.0f : -std::numeric_limits<float>::min()/2.0f; } // FP32/BF16の正負subnormalとFP16の丸め後ゼロを検証する。
        return distribution(rng); // 残りの要素には再現可能な符号付き乱数を使う。
    }; // 入力値生成関数の定義を終了する。
    const auto qv = fill_tensor(q, random); // Qを入力して丸め後の参照値を取得する。
    const auto kv = fill_tensor(k, random); // Kを入力して丸め後の参照値を取得する。
    const auto vv = fill_tensor(v, random); // Vを入力して丸め後の参照値を取得する。
    const float neg_inf = -std::numeric_limits<float>::infinity(); // 無効位置を表す加算マスク値を用意する。
    std::vector<float> mv, sv, bv; // オプション入力の参照配列を保持する。
    if (mask) { // 必要な場合だけマスクを生成する。
        mv = fill_tensor(mask, [&](int64_t j, int64_t n, int64_t h, int64_t) { // 既存キャッシュを考慮した位置指定でマスクを作る。
            if (c.mask_mode == 2 && n == 0) { return neg_inf; } // 最初のクエリを完全にマスクしてNaNの有無を調べる。
            if (c.mask_mode == 3 && j < 128) { return neg_inf; } // 最初のCUDAタイル全体を無効にしてonline初期化を検証する。
            if (j > c.nk - c.nq + n) { return neg_inf; } // NQとNKが異なるdecodeでも絶対位置に従って未来を隠す。
            return -0.01f*static_cast<float>(j + 2*h); // 有限の加算バイアスとヘッドbroadcastを同時に検証する。
        }); // マスクの生成と転送を終了する。
    } // オプションマスクの処理を終了する。
    if (sinks) { sv = fill_tensor(sinks, [](int64_t h, int64_t, int64_t, int64_t) { return 0.25f*static_cast<float>(h); }); } // sinkによる分母だけの補正を検証する。
    if (bias) { bv = fill_tensor(bias, [](int64_t j, int64_t, int64_t, int64_t b) { return 0.02f*static_cast<float>(j % 7) - 0.1f*static_cast<float>(b); }); } // 追加バイアスのbatch broadcastを検証する。
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "backend graph compute failed"); // 実際のCPUまたはCUDAディスパッチを通してforwardを実行する。
    std::vector<float> actual(static_cast<size_t>(ggml_nelements(output))); // 出力取得用のCPU領域を確保する。
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size()*sizeof(float)); // 非同期GPU処理を含む出力をバックエンドAPI経由で取得する。
    float max_error = 0.0f; // このケースの最大絶対誤差を初期化する。
    const int hpow = 1 << static_cast<int>(std::floor(std::log2(c.hq))); // ALiBiの2冪ヘッド基準を独立に計算する。
    for (int b = 0; b < c.bq; ++b) { // 各バッチの参照結果を計算する。
        for (int h = 0; h < c.hq; ++h) { // 各クエリヘッドの参照結果を計算する。
            const double base = std::pow(2.0, (h < hpow ? -c.alibi : -c.alibi/2.0)/hpow); // 倍精度でALiBi基数を求める。
            const double slope = c.alibi > 0 ? std::pow(base, h < hpow ? h + 1 : 2*(h - hpow) + 1) : 1.0; // 非2冪ヘッドでも正しい傾きを求める。
            for (int n = 0; n < c.nq; ++n) { // 各クエリ行を独立した密な参照計算で検証する。
                std::vector<double> scores(c.nk, -std::numeric_limits<double>::infinity()); // 参照実装だけで1行分のlogitを保持する。
                double maximum = sinks ? sv[h] : -std::numeric_limits<double>::infinity(); // sinkを参照softmaxの最大値に含める。
                for (int j = 0; j < c.nk; ++j) { // 全キーとの密な符号内積を求める。
                    const double m = mask ? at(mask, mv, j, n, h % mask->ne[2], b % mask->ne[3]) : 0.0; // 論理配列からマスクを読み取る。
                    if (m == neg_inf) { continue; } // 無効キーは参照計算でも除外する。
                    double dot = 0; // XORやpopcountを使わない独立した内積を初期化する。
                    for (int d = 0; d < c.dk; ++d) { // 全特徴次元を明示的に走査する。
                        const double qs = at(q, qv, d, n, h, b) >= 0 ? 1.0 : -1.0; // Qの符号を±1へ変換する。
                        const double ks = at(k, kv, d, j, h/(c.hq/c.hk), b/(c.bq/c.bk)) >= 0 ? 1.0 : -1.0; // GQA共有Kの符号を±1へ変換する。
                        dot += qs*ks; // 通常の乗算加算で符号内積を計算する。
                    } // 独立した符号内積計算を終了する。
                    double score = dot*scale; // 公開APIと同じスケールを掛ける。
                    if (c.cap > 0) { score = c.cap*std::tanh(score/c.cap); } // 倍精度でソフトキャップを適用する。
                    score += slope*m + (bias ? at(bias, bv, j, n, 0, b) : 0.0); // マスクと追加バイアスを加算する。
                    scores[j] = score; // 参照softmax用にlogitを保存する。
                    maximum = std::max(maximum, score); // 安定化に必要な最大logitを求める。
                } // 全キーの参照logitを確定する。
                std::vector<double> probabilities(c.nk, 0.0); // 参照確率の格納領域を初期化する。
                double sum = 0.0; // softmax分母を初期化する。
                if (std::isfinite(maximum)) { // 全マスク行で-inf同士を減算しない。
                    sum = sinks ? std::exp(sv[h] - maximum) : 0.0; // sinkを分母だけに含める。
                    for (int j = 0; j < c.nk; ++j) { probabilities[j] = std::exp(scores[j] - maximum); sum += probabilities[j]; } // 安定化した通常softmaxを倍精度で計算する。
                } // 非空softmax行の正規化準備を終了する。
                for (int d = 0; d < c.dv; ++d) { // すべての出力成分を検証する。
                    double reference = 0.0; // 参照V加重和を初期化する。
                    for (int j = 0; j < c.nk; ++j) { reference += probabilities[j]*at(v, vv, d, j, h/(c.hq/c.hk), b/(c.bq/c.bk)); } // 元のV値を密な確率で重み付けする。
                    reference = sum > 0 ? reference/sum : 0.0; // 全マスク行は明示的にゼロへ正規化する。
                    const float value = actual[d + c.dv*(h + c.hq*(n + c.nq*b))]; // 公開APIの[DV,HQ,NQ,BQ]順で結果を取り出す。
                    require(std::isfinite(value), "BitAttention produced NaN or infinity"); // 非有限出力を誤差比較より先に検出する。
                    const double error = std::abs(value - reference); // 倍精度参照からの絶対誤差を求める。
                    if (error > 4e-5 + 4e-5*std::abs(reference)) { // 累積順序と浮動小数点精度の差だけを許容する。
                        std::fprintf(stderr, "mismatch dk=%d dv=%d nq=%d nk=%d h=%d b=%d strided=%d: actual=%g expected=%.12g\n", c.dk, c.dv, c.nq, c.nk, h, b, c.strided, value, reference); // 再現に必要な形状と誤差を報告する。
                        throw std::runtime_error("BitAttention numerical mismatch"); // Releaseビルドでも検証を失敗させる。
                    } // 数値許容誤差の検証を終了する。
                    max_error = std::max(max_error, static_cast<float>(error)); // 最大誤差を更新する。
                } // このクエリ行の全出力を検証する。
            } // 全クエリの参照比較を終了する。
        } // 全ヘッドの参照比較を終了する。
    } // 全バッチの参照比較を終了する。
    return max_error; // このケースで観測した最大絶対誤差を返す。
} // 1ケースの数値検証を終了する。

int main() { // 利用可能なCPU/CUDAバックエンドで回帰テストを実行する。
    try { // 例外を明示的な非ゼロ終了コードへ変換する。
        ggml_backend_load_all(); // 静的または動的に登録されたバックエンドを読み込む。
        std::vector<bit_case> cases; // 境界値と機能の検証ケースを保持する。
        for (int dk : { 1, 31, 32, 33, 64, 65, 128 }) { // 32bitパック境界とその前後を網羅する。
            for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16 }) { // 対応する全浮動小数点型を検証する。
                for (bool strided : { false, true }) { // 連続テンソルと特徴軸も非連続なビューを検証する。
                    bit_case c; // 標準のGQAケースから開始する。
                    c.dk = dk; c.qt = type; c.kt = type; c.vt = type; c.strided = strided; // 符号次元・入力型・配置を変更する。
                    cases.push_back(c); // マスクなしの標準ケースを追加する。
                } // レイアウトの組み合わせを終了する。
            } // 入力型の組み合わせを終了する。
        } // パック境界ケースの生成を終了する。
        for (int mode : { 1, 2, 3 }) { // causal・全マスク行・先頭タイル全マスクを検証する。
            for (bool sink : { false, true }) { // sinkの有無によるonline初期状態を検証する。
                bit_case c; // GQAとbatch broadcastを含む既定ケースを使用する。
                c.dk = 65; c.dv = 129; c.nq = 5; c.nk = 131; c.hq = 6; c.hk = 2; // CUDAタイル境界と非2冪ヘッドを同時に検証する。
                c.mask_mode = mode; c.sink = sink; c.bias = true; c.alibi = 4.0f; c.cap = 2.5f; c.strided = true; // 全オプションと非連続配置を組み合わせる。
                c.qt = GGML_TYPE_F32; c.kt = GGML_TYPE_BF16; c.vt = GGML_TYPE_F16; // Q/K/Vの混合型を検証する。
                cases.push_back(c); // 複合機能の検証ケースを追加する。
            } // sink有無の組み合わせを終了する。
        } // マスク関連の複合ケースを終了する。
        bit_case decode; // 単一クエリで既存キャッシュを参照するケースを作る。
        decode.nq = 1; decode.nk = 129; decode.hk = 1; decode.mask_mode = 1; // NQ!=NKかつMQAのdecodeを検証する。
        cases.push_back(decode); // decode専用の境界ケースを追加する。
        int backends = 0, gpu_backends = 0; // 実際に検証したバックエンド数を記録する。
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) { // 検出された全デバイスを確認する。
            ggml_backend_dev_t dev = ggml_backend_dev_get(i); // 現在のデバイスを取得する。
            const std::string name = ggml_backend_dev_name(dev); // ログとデバイス選択に使用する名前を取得する。
            const bool cpu = ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU; // CPUデバイスを識別する。
            if (!cpu && name.find("CUDA") != 0) { continue; } // 未実装のMetal/HIP等をCUDA検証済みとは扱わない。
            std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(dev, nullptr), ggml_backend_free); // バックエンドを初期化し終了時の解放を保証する。
            require(backend != nullptr, "backend initialization failed"); // 初期化に失敗した利用可能デバイスをスキップで隠さない。
            const auto set_threads = reinterpret_cast<ggml_backend_set_n_threads_t>(ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads")); // 動的ロードにも対応したCPUスレッド設定関数を取得する。
            float error = 0.0f; // デバイス上の全ケースの最大誤差を初期化する。
            int checks = 0; // 実際に実行したケース数を数える。
            for (int threads : { 1, 4 }) { // CPUの単一スレッドと共有スクラッチの並列実行を検証する。
                if (!cpu && threads == 4) { continue; } // CUDAではCPUスレッド数だけを変える重複実行を避ける。
                if (set_threads) { set_threads(backend.get(), threads); } // CPUワーカー数を指定する。
                for (const bit_case & c : cases) { error = std::max(error, run_case(backend.get(), c)); ++checks; } // 全ケースを実バックエンド経由で検証する。
            } // 単一スレッドと並列実行の検証を終了する。
            std::printf("PASS %s: %d cases, maximum absolute error %.9g\n", name.c_str(), checks, error); // 実行数と実測した最大誤差を表示する。
            ++backends; // 検証済みバックエンド数を加算する。
            if (!cpu) { ++gpu_backends; } // GPU実機検証が行われたかを記録する。
        } // 利用可能バックエンドの検証を終了する。
        require(backends > 0, "no testable CPU or CUDA backend found"); // 何も検証せず成功と報告しない。
        if (gpu_backends == 0) { std::puts("SKIP CUDA runtime: no NVIDIA CUDA device detected"); } // GPU未実行を成功と混同しないよう明示する。
        return 0; // 実行した数値検証がすべて通過したことを示す。
    } catch (const std::exception & e) { // 数値不一致や確保失敗を捕捉する。
        std::fprintf(stderr, "FAIL: %s\n", e.what()); // 検証失敗の理由を表示する。
        return 1; // CIとCTestへ失敗を返す。
    } // テスト全体の例外処理を終了する。
} // BitAttention回帰テストを終了する。
