#include "llama.h" // 実際のモデル読み込みと推論コンテキストAPIを検証する。
#include "ggml.h" // 計算グラフ内のBitAttentionノードを識別する。
#include <algorithm> // logits間の最大誤差を求める。
#include <cmath> // logitsの有限性と絶対誤差を検証する。
#include <cstdio> // モデル統合テストの結果を表示する。
#include <memory> // モデルとコンテキストを例外時にも解放する。
#include <stdexcept> // Releaseビルドでも失敗を確実に報告する。
#include <vector> // 入力トークンとlogitsを保持する。

static void require(bool ok, const char * reason) { // 無効化されないテスト条件を定義する。
    if (!ok) { throw std::runtime_error(reason); } // 条件に違反した場合はテストを中止する。
} // テスト条件の検証を終了する。

static bool observe(ggml_tensor * tensor, bool ask, void * data) { // 実際に実行された符号化Attentionを数える。
    if (ask) { return tensor->op == GGML_OP_BIT_ATTN_EXT; } // BitAttentionノードだけ実行後コールバックを要求する。
    if (tensor->op == GGML_OP_BIT_ATTN_EXT) { ++*static_cast<int *>(data); } // 予約グラフではなく実行済みノードだけを記録する。
    return true; // 観測によって推論実行を中断しない。
} // 実行ノードの観測を終了する。

static std::vector<float> infer(llama_model * model, bool bit, bool batched, llama_flash_attn_type fa, int & observed) { // 同じモデルでprefillとdecodeの両経路を検証する。
    llama_context_params params = llama_context_default_params(); // 公開APIの既定設定から開始する。
    require(!params.bit_attn, "BitAttention must be disabled by default"); // 既存モデルの既定動作が変更されていないことを確認する。
    params.n_ctx = 64; params.n_batch = 16; params.n_ubatch = 16; // 小さなコンテキストでキャッシュ付きAttentionを実行する。
    params.n_threads = 2; params.n_threads_batch = 2; // CPU並列実行でもグラフ接続が機能することを調べる。
    params.bit_attn = bit; params.flash_attn_type = fa; params.offload_kqv = false; // 符号化切替とVキャッシュ配置を独立に変化させる。
    params.cb_eval = observe; params.cb_eval_user_data = &observed; // 本当に符号化opへ到達したことをコールバックで検証する。
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model, params), llama_free); // コンテキストの確保と解放を管理する。
    require(ctx != nullptr, "failed to create BitAttention model context"); // 未対応形状や初期化失敗を検出する。
    std::vector<llama_token> tokens = { 1, 2, 3, 4, 5, 6, 7, 8 }; // tokenizerを介さない再現可能なトークン列を用意する。
    if (batched) { // 複数クエリを含むprefill経路を実行する。
        require(llama_decode(ctx.get(), llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()))) == 0, "prefill failed"); // 8トークンをまとめて評価する。
    } else { // 既存KVキャッシュを伸長するdecode経路を実行する。
        for (llama_token & token : tokens) { require(llama_decode(ctx.get(), llama_batch_get_one(&token, 1)) == 0, "cached decode failed"); } // 1トークンずつキャッシュへ追加する。
    } // prefillとdecodeの選択を終了する。
    const int32_t nv = llama_vocab_n_tokens(llama_model_get_vocab(model)); // 出力語彙数をモデルから取得する。
    const float * logits = llama_get_logits_ith(ctx.get(), -1); // 最終トークンのlogitsを取得する。
    require(logits != nullptr, "missing output logits"); // 出力取得失敗を検出する。
    std::vector<float> result(logits, logits + nv); // コンテキスト解放前に結果をコピーする。
    for (float value : result) { require(std::isfinite(value), "non-finite model logits"); } // NaNや無限大がモデル全体へ伝播していないことを検証する。
    require(bit ? observed > 0 : observed == 0, "graph did not honor the BitAttention option"); // 有効時の接続と無効時の非侵入性を検証する。
    return result; // 最終トークンのlogitsを返す。
} // 1つのモデル推論経路の検証を終了する。

int main(int argc, char ** argv) { // 生成された小型Llamaモデルで全体統合を検証する。
    if (argc != 2) { std::fprintf(stderr, "usage: %s model.gguf\n", argv[0]); return 2; } // 必要なモデルファイルを明示する。
    llama_backend_init(); // ggmlバックエンドの初期化を行う。
    try { // モデルと推論のエラーを失敗終了コードへ変換する。
        llama_model_params mp = llama_model_default_params(); // 公開APIの標準読み込み設定を取得する。
        mp.n_gpu_layers = 0; // GPUのないCIでもモデル統合テストを実行できるようCPUへ固定する。
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free); // 同一モデルを複数コンテキストで共有する。
        require(model != nullptr, "failed to load fixture model"); // フィクスチャの読み込み失敗を検出する。
        float maximum = 0.0f; // prefillとcached decodeの最大logit誤差を記録する。
        for (llama_flash_attn_type fa : { LLAMA_FLASH_ATTN_TYPE_ENABLED, LLAMA_FLASH_ATTN_TYPE_DISABLED }) { // 非転置Vキャッシュと転置Vキャッシュの両方を検証する。
            int prefill_nodes = 0, decode_nodes = 0; // 各経路で実行された符号化op数を初期化する。
            const auto prefill = infer(model.get(), true, true, fa, prefill_nodes); // 複数クエリの因果マスクを含む符号化Attentionを実行する。
            const auto decode = infer(model.get(), true, false, fa, decode_nodes); // NQ=1かつNKが増えるキャッシュ付きAttentionを実行する。
            require(prefill.size() == decode.size(), "logit shapes differ"); // 両経路の出力形状が一致することを確認する。
            for (size_t i = 0; i < prefill.size(); ++i) { // 語彙方向の全logitを比較する。
                const float error = std::abs(prefill[i] - decode[i]); // バッチ化の有無による数値誤差を求める。
                require(error <= 2e-5f + 1e-4f*std::abs(prefill[i]), "prefill and cached decode disagree"); // バッチ分割によって因果関係が変わっていないことを検証する。
                maximum = std::max(maximum, error); // 観測した最大誤差を更新する。
            } // 全logitの一致検証を終了する。
        } // 両Vキャッシュ配置の検証を終了する。
        int normal_nodes = 0; // 通常Attentionで符号化opが混入しないことを検証する。
        infer(model.get(), false, true, LLAMA_FLASH_ATTN_TYPE_DISABLED, normal_nodes); // opt-out時の標準経路を実行する。
        llama_context_params bad = llama_context_default_params(); // 不正なキャッシュ設定の検証用パラメータを作る。
        bad.bit_attn = true; bad.type_k = GGML_TYPE_Q8_0; // 未対応の量子化Kキャッシュを明示的に要求する。
        std::unique_ptr<llama_context, decltype(&llama_free)> rejected(llama_init_from_model(model.get(), bad), llama_free); // 未対応入力を安全に拒否することを確認する。
        require(rejected == nullptr, "quantized K cache was silently accepted"); // 非対応型を黙って通常Attentionで実行していないことを確認する。
        std::printf("PASS model integration: bit-op dispatch, prefill/decode, both V layouts, opt-out, invalid cache; max logit error %.9g\n", maximum); // 実際に検証した項目と誤差を表示する。
    } catch (const std::exception & e) { // 全体統合の失敗を捕捉する。
        std::fprintf(stderr, "FAIL: %s\n", e.what()); // エラー原因をCIへ表示する。
        llama_backend_free(); // 失敗時にもバックエンドを終了する。
        return 1; // モデル統合テストの失敗を通知する。
    } // 正常時のモデルとコンテキストのRAII解放を完了する。
    llama_backend_free(); // バックエンド全体を終了する。
    return 0; // 全体統合の成功を通知する。
} // モデル統合テストを終了する。
