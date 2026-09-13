#!/usr/bin/env python3 # 追加・変更したコード行の説明コメントを機械検査する。
import pathlib # ファイル名と拡張子からコメント構文を判別する。
import re # 日本語の説明を含む行末コメントを検出する。
import subprocess # 指定した基準コミットとの差分を取得する。
import sys # 基準コミットと検査結果を入出力する。
base = sys.argv[1] if len(sys.argv) > 1 else "5265486cb491ea5b160a3d20ca1e105ae5422312" # 既存コードを対象外にする比較基準を指定する。
diff = subprocess.check_output(["git", "diff", "--no-ext-diff", "--no-color", "--unified=0", "--diff-filter=ACM", base, "--"], text=True) # 未変更行を含めず追加された実コードだけを取得する。
path = "" # 現在検査しているファイルの相対パスを保持する。
line_number = 0 # 新しいファイルにおける行番号を追跡する。
checked = 0 # コメントを検査したコード行数を数える。
errors = [] # 説明コメントがない行をまとめて報告する。
for line in diff.splitlines(): # unified diffを行単位で解析する。
    if line.startswith("+++ b/"): # 新しいファイルの見出しを検出する。
        path = line[6:] # リポジトリからの相対パスを取得する。
        continue # 見出し自身をコードとして数えない。
    if line.startswith("@@"): # 次の変更区間の見出しを検出する。
        line_number = int(re.search(r"\+(\d+)", line).group(1)) # 変更後ファイルの開始行を取得する。
        continue # 差分区間の見出しを検査対象から外す。
    if not line.startswith("+"): # 追加行以外を選別する。
        if line.startswith(" "): line_number += 1 # 文脈行がある場合は変更後の行番号を進める。
        continue # 削除行やメタデータを検査しない。
    code = line[1:] # 差分の追加記号を取り除く。
    suffix = pathlib.Path(path).suffix # コードの言語を拡張子で判別する。
    marker = "//" if suffix in {".c", ".cpp", ".h", ".hpp", ".cu", ".cuh"} else "#" if suffix in {".py", ".yml", ".yaml", ".cmake", ".sh"} or pathlib.Path(path).name == "CMakeLists.txt" else None # C系とスクリプト系のコメント記号を区別する。
    if marker and code.strip() and not code.lstrip().startswith(marker): # 空行とコメント専用行以外の実コードを検査する。
        checked += 1 # 新規コード行を数える。
        if not re.search(re.escape(marker) + r"[^\n]*[ぁ-んァ-ン一-龥]", code): # コメント内に具体的な日本語説明があることを確認する。
            errors.append(f"{path}:{line_number}: missing explanatory comment") # 説明がない箇所をファイル名と行番号で保存する。
    line_number += 1 # 追加行の次の行番号へ進む。
if errors: # 不備が1つでもある場合にCIを失敗させる。
    print("\n".join(errors), file=sys.stderr) # 全不備をまとめて表示する。
    sys.exit(1) # 検査失敗を呼び出し元へ通知する。
print(f"PASS explanatory comments: {checked} added/changed code lines") # 検査したコード行数と成功を表示する。
