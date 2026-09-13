import argparse  # 検証対象のビルドディレクトリと実行モードを受け取る。
import json  # ベンチマーク結果とCMakeのコンパイルデータベースを読み込む。
from pathlib import Path  # 作業ディレクトリと出力ファイルを安全に組み立てる。
import shlex  # シェル形式のコンパイルコマンドを引数列へ分解する。
import subprocess  # CMakeが生成したコマンドをシェル展開せず実行する。


def read_json(path: Path):  # UTF-8のJSONファイルを読み込み、解析結果を返す。
    with path.open(encoding="utf-8") as stream:  # ファイル記述子を確実に解放する。
        return json.load(stream)  # 不正なJSONは例外としてCIを失敗させる。


def check_benchmarks(directory: Path) -> None:  # 有効時と既定値の両方のベンチマーク出力を検証する。
    for filename, expected in (("bit-bench.json", True), ("normal-bench.json", False)):  # 二つのモードを同じ基準で確認する。
        rows = read_json(directory / filename)  # 対象モードのJSON出力を読み込む。
        if not rows or not all(row.get("bit_attn") is expected for row in rows):  # 空の結果や設定値の取り違えを検出する。
            raise RuntimeError(f"Invalid bit_attn field in {filename}")  # モードを区別できない出力をテスト失敗とする。
    print("Both benchmark modes passed.", flush=True)  # 両方の検証が成功したことをログに記録する。


def compile_cuda(directory: Path) -> None:  # GPUを実行せず二つのCUDA翻訳単位をコンパイルする。
    commands = read_json(directory / "compile_commands.json")  # 実際のCMake設定からコンパイル条件を取得する。
    names = {"bit-attn.cu", "ggml-cuda.cu"}  # 新カーネルと共通ディスパッチの両方を検証対象にする。
    selected = [item for item in commands if Path(item["file"]).name in names]  # 対象翻訳単位だけを抽出する。
    if len(selected) != 2 or {Path(item["file"]).name for item in selected} != names:  # 重複や検証対象の欠落を拒否する。
        raise RuntimeError("Expected one command for each CUDA translation unit")  # 不完全な検証を成功として扱わない。
    for item in selected:  # カーネルと呼び出し元を順番にコンパイルする。
        command = item.get("arguments") or shlex.split(item["command"])  # 両方のコンパイルデータベース表現を処理する。
        if "-o" in command:  # コンパイラの出力先指定が存在する場合だけ親ディレクトリを作る。
            output = Path(item["directory"]) / command[command.index("-o") + 1]  # CMakeの作業ディレクトリを基準に出力先を解決する。
            output.parent.mkdir(parents=True, exist_ok=True)  # 個別コンパイルでも必要な出力ディレクトリを用意する。
        print("Compiling", item["file"], flush=True)  # 失敗した翻訳単位をログから識別できるようにする。
        subprocess.run(command, cwd=item["directory"], check=True)  # コンパイラの失敗を呼び出し元へ伝える。
    print("CUDA compilation passed; GPU execution and performance are NOT tested.", flush=True)  # 検証範囲をコンパイルに限定して明記する。


def main() -> None:  # CIから呼び出す検証モードを選択する。
    parser = argparse.ArgumentParser(description="BitAttention CI checks")  # 使用方法と不正な引数の診断を用意する。
    parser.add_argument("mode", choices=("benchmarks", "cuda-compile"))  # 実行可能な検証を二種類に制限する。
    parser.add_argument("directory", type=Path)  # JSON出力またはCMakeビルドの場所を指定する。
    args = parser.parse_args()  # コマンドライン引数を解析する。
    if args.mode == "benchmarks":  # ベンチマークのモード識別を検証する経路を選ぶ。
        check_benchmarks(args.directory)  # 二つの実行結果のbit_attn値を照合する。
    else:  # CUDAコンパイル検証を選択した場合に進む。
        compile_cuda(args.directory)  # 対象カーネルとディスパッチをコンパイルする。


if __name__ == "__main__":  # インポート時には検証を勝手に実行しない。
    main()  # スクリプトとして実行された場合だけ引数を処理する。
