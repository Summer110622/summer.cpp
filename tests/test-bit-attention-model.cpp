#include "llama.h" // モデルと推論コンテキストの公開APIを読み込む。
#include "ggml.h" // GGMLのテンソル型と演算APIを読み込む。
#include "gguf.h" // テスト用GGUFを作成するAPIを読み込む。

#include <chrono> // 一時ファイル名に使う単調時計を読み込む。
#include <cmath> // 指数関数・tanh・有限値検査を読み込む。
#include <cstdio> // テスト結果とエラーの標準出力を読み込む。
#include <filesystem> // 一時ディレクトリとファイルの操作を読み込む。
#include <memory> // リソースを自動解放するスマートポインタを読み込む。
#include <random> // 再現可能なテストデータの生成を読み込む。
#include <stdexcept> // 失敗理由を伝える例外型を読み込む。
#include <string> // 診断文とファイル名の構築を読み込む。
#include <vector> // 入力と参照計算結果の可変長配列を読み込む。

static void check(bool ok, const char * message) { // 統合テストの条件違反を例外として報告する。
    if (!ok) { throw std::runtime_error(message); } // 条件を満たさない場合に診断付き例外を送出する。
} // この処理または定義のブロックを閉じる。

// Tiny deterministic, randomly initialized Llama fixture. This tests wiring and
// cache correctness, not language-model quality. No downloaded weights required.
static void write_model(const std::string & path) { // 外部の重みを取得せず小型のランダムLlamaモデルを作る。
    constexpr int e = 64, ff = 128, heads = 4, kv = 2, vocab = 32, layers = 2; // GQAを含む小さな2層Llamaの形状を固定してテスト負荷を抑える。
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx( // GGMLコンテキストを自動解放する所有ポインタを用意する。
        ggml_init({4*1024*1024, nullptr, false}), ggml_free); // 小型モデルの重みも格納する4MiBのGGML領域を確保する。
    std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_empty(), gguf_free); // GGUFメタデータを作成し例外時も解放されるよう所有する。
    check(bool(ctx) && bool(meta), "fixture allocation failed"); // モデル生成に必要な二つの領域が確保できたか確認する。
    gguf_set_val_str(meta.get(), "general.architecture", "llama"); // モデルローダーがLlamaグラフを選択するよう種類を指定する。
    gguf_set_val_str(meta.get(), "general.name", "BitAttention integration fixture (random)"); // 学習済みモデルではなくランダムな検証用重みだと明記する。
    gguf_set_val_u32(meta.get(), "llama.context_length", 128); // テストモデルの最大コンテキスト長を128に設定する。
    gguf_set_val_u32(meta.get(), "llama.embedding_length", e); // 埋め込みベクトルの次元を記録する。
    gguf_set_val_u32(meta.get(), "llama.block_count", layers); // Attentionを含むTransformer層数を記録する。
    gguf_set_val_u32(meta.get(), "llama.feed_forward_length", ff); // FFN中間層の幅を記録する。
    gguf_set_val_u32(meta.get(), "llama.attention.head_count", heads); // Queryヘッド数を記録する。
    gguf_set_val_u32(meta.get(), "llama.attention.head_count_kv", kv); // Queryより少ないKVヘッド数を記録しGQA経路を使用させる。
    gguf_set_val_f32(meta.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f); // RMSNormの数値安定化定数を指定する。
    gguf_set_val_u32(meta.get(), "llama.rope.dimension_count", e/heads); // ヘッド次元に一致するRoPEの回転次元を指定する。
    gguf_set_val_f32(meta.get(), "llama.rope.freq_base", 10000.0f); // 通常のRoPE経路を通すため回転周波数の底を設定する。
    gguf_set_val_str(meta.get(), "tokenizer.ggml.model", "llama"); // テスト用の簡単なLlama語彙を指定する。
    std::vector<std::string> tokens = {"<unk>", "<s>", "</s>"}; // 未知語・文頭・文末の特殊トークンを先頭に置く。
    for (int i = 3; i < vocab; ++i) { tokens.push_back("token" + std::to_string(i)); } // 32語の小さな語彙を埋める通常トークンを追加する。
    std::vector<const char *> token_ptrs; // GGUFへ文字列配列を渡すためのポインタ列を用意する。
    for (const auto & token : tokens) { token_ptrs.push_back(token.c_str()); } // 寿命が維持されるトークン文字列のアドレスを集める。
    std::vector<float> scores(vocab, 0); // 検証用語彙のスコアをすべてゼロに初期化する。
    std::vector<int32_t> types(vocab, 1); // 各語を既定で通常トークンとして登録する。
    types[0] = 2; types[1] = types[2] = 3; // 先頭3語だけ未知語と制御トークンの種類へ変更する。
    gguf_set_arr_str(meta.get(), "tokenizer.ggml.tokens", token_ptrs.data(), vocab); // トークン文字列の配列をGGUFへ保存する。
    gguf_set_arr_data(meta.get(), "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), vocab); // トークンのスコア配列をF32メタデータとして保存する。
    gguf_set_arr_data(meta.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), vocab); // トークン種別を整数配列として保存する。
    gguf_set_val_u32(meta.get(), "tokenizer.ggml.unknown_token_id", 0); // 未知語トークンの番号を指定する。
    gguf_set_val_u32(meta.get(), "tokenizer.ggml.bos_token_id", 1); // 文頭トークンの番号を指定する。
    gguf_set_val_u32(meta.get(), "tokenizer.ggml.eos_token_id", 2); // 文末トークンの番号を指定する。
    std::mt19937 rng(110622); // 乱数系列を固定して同じテスト重みを再生成可能にする。
    std::normal_distribution<float> dist(0.0f, 0.05f); // 行列重みに使う平均ゼロ・標準偏差0.05の分布を定義する。
    auto add = [&](const std::string & name, int n0, int n1 = 1) { // 名前と形状から重みを生成してGGUFへ追加する補助関数を定義する。
        ggml_tensor * t = n1 == 1 ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n0) // 1次元の正規化重みにはF32ベクトルを確保する。
                                 : ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n0, n1); // 行列重みには指定された2次元のF32テンソルを確保する。
        ggml_set_name(t, name.c_str()); // Llamaローダーが解釈する名前を重みに付ける。
        auto * data = static_cast<float *>(t->data); // 生成した重みを初期化するためF32データへアクセスする。
        for (int64_t i = 0; i < ggml_nelements(t); ++i) { data[i] = n1 == 1 ? 1.0f : dist(rng); } // 正規化の係数は1、行列は小さいランダム値に初期化する。
        gguf_add_tensor(meta.get(), t); // テンソルをGGUFの書き込み対象へ追加する。
    }; // この処理または定義のブロックを閉じる。
    add("token_embd.weight", e, vocab); // トークンIDを埋め込みへ変換する重みを追加する。
    add("output_norm.weight", e); // 最終RMSNormの係数を追加する。
    add("output.weight", e, vocab); // 埋め込みから語彙ロジットへ変換する重みを追加する。
    for (int i = 0; i < layers; ++i) { // すべてのTransformer層の重みを生成する。
        const auto prefix = "blk." + std::to_string(i) + "."; // GGUF規約の層別テンソル名の接頭辞を作る。
        add(prefix + "attn_norm.weight", e); // Attention直前のRMSNorm係数を追加する。
        add(prefix + "attn_q.weight", e, e); // Query射影の行列を追加する。
        add(prefix + "attn_k.weight", e, (e/heads)*kv); // 共有KVヘッド数に合わせたKey射影を追加する。
        add(prefix + "attn_v.weight", e, (e/heads)*kv); // 同じ共有KV幅のValue射影を追加する。
        add(prefix + "attn_output.weight", e, e); // Attention結果をモデル次元へ戻す射影を追加する。
        add(prefix + "ffn_norm.weight", e); // FFN直前のRMSNorm係数を追加する。
        add(prefix + "ffn_gate.weight", e, ff); // FFNゲートの射影行列を追加する。
        add(prefix + "ffn_up.weight", e, ff); // FFNの中間特徴を増やす射影行列を追加する。
        add(prefix + "ffn_down.weight", ff, e); // FFNの中間特徴をモデル次元へ戻す行列を追加する。
    } // この処理または定義のブロックを閉じる。
    check(gguf_write_to_file(meta.get(), path.c_str(), false), "writing fixture failed"); // メタデータだけでなく重みもGGUFへ書き込み成功を確認する。
} // この処理または定義のブロックを閉じる。

struct observed { int bit = 0; int pack = 0; }; // 二値Attention演算と符号パック演算を別々に数える。
static bool observe(ggml_tensor * t, bool ask, void * user) { // 実行グラフが二値演算を選択したか監視する。
    auto & count = *static_cast<observed *>(user); // コールバックへ渡されたカウンタを復元する。
    if (ask) { // 実行前の問い合わせ段階でグラフの演算種類を調べる。
        if (t->op == GGML_OP_BIT_ATTN_EXT || t->op == GGML_OP_BIT_MUL_MAT) { ++count.bit; } // 融合・非融合のどちらでも二値Attentionを検出して数える。
        if (t->op == GGML_OP_BIT_PACK) { ++count.pack; } // 符号パック演算が選ばれた回数を数える。
    } // この処理または定義のブロックを閉じる。
    return false; // no tensor download needed
} // この処理または定義のブロックを閉じる。

static std::vector<float> decode(llama_model * model, bool bit, bool flash, int seqs, int chunk) { // 指定したAttentionモードと系列分割でデコードを実行する。
    auto params = llama_context_default_params(); // 公開APIの既定値からテスト用コンテキスト設定を作る。
    params.n_ctx = 128; // モデル定義に合わせてコンテキスト容量を設定する。
    params.n_batch = 32; // テストで処理する論理バッチの容量を設定する。
    params.n_ubatch = 32; // 物理マイクロバッチがテスト入力を収容できるようにする。
    params.n_seq_max = seqs; // 同時に扱う系列数をテスト条件へ合わせる。
    params.n_threads = params.n_threads_batch = 2; // CPU使用量を抑えつつ複数スレッドで実行する。
    params.flash_attn_type = flash ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED; // 通常FlashAttentionの有効・無効と二値化の組み合わせを検証する。
    params.bit_attn = bit; // 二値Attentionをこのコンテキストだけに設定する。
    observed counts; // グラフ内の二値演算の観測カウンタを初期化する。
    params.cb_eval = observe; // 実行グラフを確認するコールバックを登録する。
    params.cb_eval_user_data = &counts; // コールバックへ観測カウンタの参照を渡す。
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, params), llama_free); // 作成したコンテキストを例外時も自動解放する。
    check(bool(ctx), "context creation failed"); // 指定モードでコンテキストを生成できたことを確認する。
    llama_batch batch = llama_batch_init(32, 0, seqs); // トークン・位置・系列IDを収容するバッチを確保する。
    std::vector<float> last(size_t(32*seqs)); // 各系列の32語分の最終ロジットを保持する領域を用意する。
    for (int pos = 0; pos < 6; pos += chunk) { // 6トークンを指定されたチャンク幅で順にデコードする。
        batch.n_tokens = 0; // 新しいチャンクを詰める前にバッチの使用数をリセットする。
        for (int s = 0; s < seqs; ++s) // 同時に処理する各系列をバッチへ追加する。
        for (int p = pos; p < std::min(pos + chunk, 6); ++p) { // 最後のチャンクが6トークンの範囲を越えないようにする。
            const int i = batch.n_tokens++; // 現在のトークンのバッチ内位置を確保する。
            batch.token[i] = 3 + (p + 3*s)%29; // 系列ごとに異なる有効な通常トークンIDを決定的に生成する。
            batch.pos[i] = p; // キャッシュの因果マスクに必要な絶対位置を渡す。
            batch.n_seq_id[i] = 1; // このトークンを一つの系列だけへ所属させる。
            batch.seq_id[i][0] = s; // KVキャッシュ上で対応する系列IDを指定する。
            batch.logits[i] = true; // 比較用に各トークンのロジットを取得できるようにする。
        } // この処理または定義のブロックを閉じる。
        const int result = llama_decode(ctx.get(), batch); // 実際のモデルグラフを通して現在のチャンクをデコードする。
        if (result != 0) { llama_batch_free(batch); throw std::runtime_error("llama_decode failed"); } // デコード失敗時はバッチを解放してテストを失敗させる。
        const int tokens_per_seq = std::min(chunk, 6 - pos); // 最後のチャンクを含めた実際の系列当たりトークン数を求める。
        for (int s = 0; s < seqs; ++s) { // 各系列の最後の出力を回収する。
            const float * logits = llama_get_logits_ith(ctx.get(), (s + 1)*tokens_per_seq - 1); // この系列で最後に処理したトークンのロジットを取得する。
            check(logits != nullptr, "missing logits"); // 要求したロジットが返されたことを確認する。
            for (int i = 0; i < 32; ++i) { // 小さな語彙の全32要素を確認する。
                check(std::isfinite(logits[i]), "non-finite logits"); // NaNや無限大がデコード結果に混入していないか確認する。
                last[size_t(s*32 + i)] = logits[i]; // 単一トークン実行とチャンク実行の比較用に値を保存する。
            } // この処理または定義のブロックを閉じる。
        } // この処理または定義のブロックを閉じる。
    } // この処理または定義のブロックを閉じる。
    llama_batch_free(batch); // デコード用バッチのストレージを解放する。
    check(bit ? counts.bit > 0 && counts.pack > 0 : counts.bit == 0 && counts.pack == 0, // 有効時だけパックと二値Attentionが実際に選ばれたか検査する。
          "per-context opt-in did not select the expected ggml ops"); // モードの伝播がグラフ構築へ届かなかった場合の診断を指定する。
    return last; // 各系列の最後のロジットを返す。
} // この処理または定義のブロックを閉じる。

int main(int argc, char ** argv) { // テスト用の入力と実行条件を設定して検証を開始する。
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count(); // テスト用一時ファイル名に使う単調時計の値を得る。
    const auto path = std::filesystem::temp_directory_path()/ ("test-bit-attention-" + std::to_string(stamp) + ".gguf"); // 作業用ディレクトリに衝突しにくいGGUFパスを作る。
    try { // 失敗を捕捉して診断を返せるよう検証を開始する。
        const std::string file = argc == 3 && std::string(argv[1]) == "--write-fixture" ? argv[2] : path.string(); // 明示的な保存先があればそれを使い通常テストでは一時ファイルを使う。
        write_model(file); // ランダム小型モデルを生成してローカルファイルへ保存する。
        if (argc == 3 && std::string(argv[1]) == "--write-fixture") { return 0; } // fixture生成専用モードではファイルを残して正常終了する。
        check(argc == 1, "usage: test-bit-attention-model [--write-fixture FILE]"); // 通常検証に不正な引数が混ざっていないか確認する。
        llama_backend_init(); // モデル実行前に利用するバックエンドを初期化する。
        check(!llama_context_default_params().bit_attn, "BitAttention must be disabled by default"); // 公開APIの既定動作が通常Attentionのままであることを検査する。
        auto mp = llama_model_default_params(); // テストモデルを読み込む既定のパラメータを取得する。
        mp.n_gpu_layers = 0; // この統合テストではモデル全層をCPUへ配置する。
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model( // モデルを自動解放する所有ポインタを準備する。
            llama_model_load_from_file(file.c_str(), mp), llama_model_free); // 作成したGGUFを実際のLlamaモデルローダーで読み込む。
        check(bool(model), "loading fixture failed"); // fixtureをモデルとして読み込めたことを確認する。
        for (bool flash : {false, true}) { // 通常FlashAttention設定が有効でも無効でも二値経路を検証する。
            for (int seqs : {1, 2}) { // 1系列と2系列の両方でキャッシュの分離を検証する。
                const auto bit_token = decode(model.get(), true, flash, seqs, 1); // 1トークンずつ二値Attentionでデコードして基準値を得る。
                const auto bit_chunk = decode(model.get(), true, flash, seqs, 3); // 同じ系列を3トークンのチャンクで処理する。
                for (size_t i = 0; i < bit_token.size(); ++i) { // すべての系列・語彙要素で二つの分割方法を比較する。
                    check(std::abs(bit_token[i] - bit_chunk[i]) < 3e-4f, // キャッシュ経路や因果位置が一致しロジット差が許容範囲内か確認する。
                          "prefill/decode logits differ: causal offset or KV cache wiring bug"); // 不一致時にキャッシュ接続または絶対位置の不具合として報告する。
                } // この処理または定義のブロックを閉じる。
                decode(model.get(), false, flash, seqs, 3); // disabled path regression
                std::printf("PASS model flash=%d sequences=%d (token/chunk decode and disabled path)\n", flash, seqs); // 成功したFlashAttention設定と系列数をログへ残す。
            } // この処理または定義のブロックを閉じる。
        } // この処理または定義のブロックを閉じる。
        model.reset(); // バックエンドを終了する前にモデルの所有権を解放する。
        llama_backend_free(); // 初期化したバックエンド資源を解放する。
        std::filesystem::remove(path); // 通常テストの一時GGUFを削除する。
    } catch (const std::exception & e) { // 検証中の例外を捕捉して失敗を報告する。
        std::error_code ignored; std::filesystem::remove(path, ignored); // 例外時にも追加例外を出さず一時GGUFを片付ける。
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; // 失敗理由を標準エラーへ出し非ゼロで終了する。
    } // この処理または定義のブロックを閉じる。
    return 0; // 検証または処理の正常終了を返す。
} // この処理または定義のブロックを閉じる。
