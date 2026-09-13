#include "arg.h" // 共通CLI引数パーサーを読み込む。
#include "common.h" // 共通設定とコンテキストへの変換APIを読み込む。
#include "llama.h" // モデルと推論コンテキストの公開APIを読み込む。

#include <cstdio> // テスト結果とエラーの標準出力を読み込む。
#include <cstdlib> // 環境変数の操作を読み込む。
#include <stdexcept> // 失敗理由を伝える例外型を読み込む。
#include <string> // 診断文とファイル名の構築を読み込む。
#include <vector> // 入力と参照計算結果の可変長配列を読み込む。

static void set_bit_env(const char * value) { // BitAttentionの環境変数をテスト用に設定する。
#ifdef _WIN32 // Windows用の環境変数設定APIを選ぶ。
    _putenv_s("LLAMA_ARG_BIT_ATTN", value); // WindowsのAPIでテスト用環境変数を設定する。
#else // 代替プラットフォーム用の実装へ切り替える。
    setenv("LLAMA_ARG_BIT_ATTN", value, 1); // POSIXのAPIで既存の環境変数も上書きして条件を固定する。
#endif // プラットフォーム別実装の条件分岐を閉じる。
} // この処理または定義のブロックを閉じる。

static void check_args(std::vector<std::string> words, bool expected) { // CLIと環境変数からコンテキストまでの設定伝播を確認する。
    // Parsing only; no file is opened or model loaded by this test.
    words.insert(words.end(), {"--model", "unused-test-fixture.gguf"}); // モデル取得を起こさないダミーのモデル引数を与える。
    common_params params; // パーサーから受け取る共通パラメータを用意する。
    std::vector<char *> args; // C形式のargvへ渡す文字列ポインタ列を用意する。
    for (auto & word : words) { args.push_back(&word[0]); } // 元の文字列の寿命を保ったままポインタを配列へ詰める。
    if (!common_params_parse(int(args.size()), args.data(), params, LLAMA_EXAMPLE_COMMON)) { // 実際の共通引数パーサーで入力を解釈する。
        throw std::runtime_error("argument parsing failed"); // 引数が正常に解釈されなかった場合にテストを失敗させる。
    } // この処理または定義のブロックを閉じる。
    if (params.bit_attn != expected || common_context_params_to_llama(params).bit_attn != expected) { // 共通設定と公開コンテキストの両方へ期待値が届いたか確認する。
        throw std::runtime_error("BitAttention option was not propagated to llama_context_params"); // 設定伝播が途中で失われたことを診断する。
    } // この処理または定義のブロックを閉じる。
} // この処理または定義のブロックを閉じる。

int main() { // テスト用の入力と実行条件を設定して検証を開始する。
    try { // 失敗を捕捉して診断を返せるよう検証を開始する。
        set_bit_env("0"); // 環境変数側で二値Attentionを無効にする。
        check_args({"test"}, false); // オプションなしでは通常Attentionになることを検証する。
        check_args({"test", "--bit-attn"}, true); // 有効化フラグが無効の環境変数より優先されることを検証する。
        check_args({"test", "--bit-attn", "--no-bit-attn"}, false); // 後から指定した無効化フラグが優先されることを検証する。
        set_bit_env("1"); // 環境変数側で二値Attentionを有効にする。
        check_args({"test"}, true); // CLI指定がなければ環境変数の有効値が反映されることを検証する。
        check_args({"test", "--no-bit-attn"}, false); // 無効化フラグが有効の環境変数より優先されることを検証する。
        check_args({"test", "--no-bit-attn", "--bit-attn"}, true); // 後から指定した有効化フラグが優先されることを検証する。
    } catch (const std::exception & e) { // 検証中の例外を捕捉して失敗を報告する。
        std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; // 失敗理由を標準エラーに表示して異常終了する。
    } // この処理または定義のブロックを閉じる。
    std::puts("PASS: default, CLI, environment, negation and context propagation (6 cases)"); // 六つの引数・環境変数ケースがすべて成功したことを記録する。
    return 0; // 検証または処理の正常終了を返す。
} // この処理または定義のブロックを閉じる。
