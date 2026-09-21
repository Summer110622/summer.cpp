import argparse  # 比較元コミットをコマンドラインから指定できるようにする。
import difflib  # 比較元と現在のソースの追加・置換行を求める。
import io  # Pythonトークナイザーへ文字列入力を渡す。
from pathlib import Path  # リポジトリ内のパスを扱う。
import re  # C/C++の文字列と実際のコメントを区別する。
import subprocess  # Gitから比較元の内容と変更ファイルを取得する。
import tokenize  # Python文字列内のコメント記号を誤認しないようにする。


ROOT = Path(__file__).resolve().parents[1]  # 実行ディレクトリによらずリポジトリルートを確定する。
QUOTED = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\\n])*\''  # エスケープを含む文字列と文字リテラルを認識する。
RAW = r'R"(?P<tag>[^ ()\\\t\r\n]{0,16})\(.*?\)(?P=tag)"'  # C++のraw文字列内にあるコメント記号を無視する。
CPP = re.compile(RAW + '|' + QUOTED + r'|(?P<block>/\*.*?\*/)|(?P<note>//[^\n]*)', re.DOTALL)  # コメントより先に文字列全体を消費する。
HASH = re.compile(QUOTED + r'|(?P<note>\#[^\n]*)', re.DOTALL)  # シェル・YAML・CMakeの引用外のコメントを認識する。


def git(*args: str) -> str:  # コマンド文字列ではなく引数列でGitを呼び出す。
    result = subprocess.run(['git', *args], cwd=ROOT, text=True, capture_output=True, check=True)  # 失敗を隠さず呼び出し元へ伝える。
    return result.stdout  # 標準出力だけを解析対象として返す。


def inspect(text: str, language: str) -> tuple[list[str], set[int]]:  # コード本文と同じ行の説明コメントの位置を返す。
    spans = []  # 実際のコメントの開始・終了位置を集める。
    if language == 'python':  # Pythonでは正規表現でなく言語トークンを使う。
        offsets = [0]  # トークンの行・列を文字列位置へ変換するため先頭位置を準備する。
        for line in text.splitlines(keepends=True):  # 各物理行の長さを集計する。
            offsets.append(offsets[-1] + len(line))  # 次の物理行の開始オフセットを記録する。
        for item in tokenize.generate_tokens(io.StringIO(text).readline):  # 文字列とコメントを構文に沿って識別する。
            if item.type == tokenize.COMMENT:  # 実際のPythonコメントだけを抽出する。
                start = offsets[item.start[0] - 1] + item.start[1]  # コメント先頭の絶対位置を求める。
                end = offsets[item.end[0] - 1] + item.end[1]  # コメント終端の絶対位置を求める。
                spans.append((start, end, bool(item.string[1:].strip())))  # 空でない説明があることも記録する。
    else:  # C/C++または引用符付きハッシュコメントを処理する。
        pattern = CPP if language == 'cpp' else HASH  # 言語に合うコメント構文を選択する。
        for match in pattern.finditer(text):  # 文字列リテラルも消費して誤ったコメント検出を防ぐ。
            if match.lastgroup in ('block', 'note'):  # 文字列ではなく本物のコメントだけを扱う。
                width = 2 if language == 'cpp' else 1  # コメント開始記号の長さを選ぶ。
                explained = match.lastgroup == 'note' and bool(match.group()[width:].strip())  # 同じ行の空でない説明を必須にする。
                spans.append((match.start(), match.end(), explained))  # コードから除く範囲を記録する。
    code = list(text)  # 改行位置を保持したままコメントを空白へ置き換える。
    documented = set()  # 説明が付いた物理行をゼロ起点で記録する。
    for start, end, explained in spans:  # 検出した各コメントを処理する。
        if explained:  # 空コメントや単なるブロックラベルを説明として数えない。
            documented.add(text.count('\n', 0, start))  # 元のソースの物理行番号を記録する。
        code[start:end] = ['\n' if char == '\n' else ' ' for char in text[start:end]]  # 行数を変えずコメント部分だけを除去する。
    return ''.join(code).splitlines(), documented  # コードの有無とコメントの有無を独立に判定可能にする。


def rows(path: str, text: str) -> tuple[set[int], set[int]]:  # ファイル形式に応じて検査するコード行を選ぶ。
    suffix = Path(path).suffix  # ソースの言語を拡張子から選ぶ。
    if suffix == '.md':  # 文書ではコードフェンス内だけを検査する。
        active, start, body = None, 0, []  # 現在のコードフェンスの状態を初期化する。
        code_rows, documented = set(), set()  # 文書全体でのコード行と説明行を集める。
        for index, line in enumerate(text.splitlines()):  # 文書の元の行番号を維持して走査する。
            if line.startswith('```'):  # コードフェンスの開始または終端を検出する。
                if active is not None:  # 終了したフェンスの内容を検査する。
                    code, notes = inspect('\n'.join(body), 'cpp' if active in ('c', 'cpp', 'cuda') else 'hash')  # サンプルの言語を使う。
                    code_rows.update(start + i for i, row in enumerate(code) if row.strip())  # 空行とコメントだけの行はコードに数えない。
                    documented.update(start + i for i in notes)  # フェンス内位置を文書全体の行番号へ変換する。
                    active = None  # フェンスの外へ戻る。
                else:  # 新しいコードフェンスを開始する。
                    active, start, body = line[3:].strip(), index + 1, []  # 言語と開始位置を保存する。
            elif active is not None:  # フェンス内のコードをそのまま保持する。
                body.append(line)  # 解析対象のコードへ現在行を追加する。
        return code_rows, documented  # 文書の説明文そのものは対象外とする。
    language = 'cpp' if suffix in ('.c', '.cpp', '.h', '.cu', '.cuh') else 'python' if suffix == '.py' else 'hash'  # コードのコメント形式を選ぶ。
    code, documented = inspect(text, language)  # 文字列内の記号と実際のコメントを分離する。
    return {i for i, line in enumerate(code) if line.strip()}, documented  # 実際にコードがある物理行だけを返す。


def main() -> None:  # 指定した比較元からの追加・変更を監査する。
    parser = argparse.ArgumentParser(description='Check same-line explanations on feature changes')  # 比較対象の指定方法を公開する。
    parser.add_argument('--base', required=True)  # 比較元を固定し未検査の範囲を暗黙に変えない。
    args = parser.parse_args()  # 指定された比較元を取得する。
    base = git('rev-parse', '--verify', args.base + '^{commit}').strip()  # 実在するコミットだけを比較元にする。
    changed = set(git('diff', '--name-only', '--diff-filter=AM', base, '--').splitlines())  # 新規・変更されたファイルを取得する。
    changed.update(git('ls-files', '--others', '--exclude-standard').splitlines())  # コミット前の新規ファイルも監査する。
    failures, checked = [], 0  # 説明不足と確認済みコード行数を集計する。
    for path in sorted(changed):  # 出力が再現可能になる順序で検査する。
        if Path(path).suffix not in ('.c', '.cpp', '.h', '.cu', '.cuh', '.py', '.yml', '.yaml', '.cmake', '.md') and Path(path).name != 'CMakeLists.txt':  # バイナリやコードでないデータを除外する。
            continue  # 対象外のファイルを飛ばす。
        text = (ROOT / path).read_text(encoding='utf-8')  # 現在のファイルをUTF-8で読む。
        exists = subprocess.run(['git', 'cat-file', '-e', f'{base}:{path}'], cwd=ROOT, capture_output=True).returncode == 0  # 比較元に同じファイルがあるか確認する。
        old = git('show', f'{base}:{path}') if exists else ''  # 新規ファイルは空の比較元として扱う。
        code_rows, documented = rows(path, text)  # ファイル全体を解析し差分外から続くコメントも考慮する。
        diff = difflib.SequenceMatcher(a=old.splitlines(), b=text.splitlines(), autojunk=False)  # 同じ記号が多いコードでも差分検出を省略しない。
        for tag, _, _, begin, end in diff.get_opcodes():  # 差分の現在側に存在する行を調べる。
            if tag in ('insert', 'replace'):  # 削除行と変更のない行へ説明を要求しない。
                for index in range(begin, end):  # 追加・置換範囲の全物理行を検査する。
                    if index in code_rows:  # 空行とコメントだけの行は数えない。
                        checked += 1  # 実際に確認したコード行数を加算する。
                        if index not in documented:  # 同じ行の説明が欠けているか確認する。
                            failures.append(f'{path}:{index + 1}')  # 人が修正できるファイル名と行番号を記録する。
    if failures:  # 一行でも説明が不足していればCIを失敗させる。
        raise SystemExit('Missing same-line explanation:\n' + '\n'.join(failures))  # 不足箇所をすべてまとめて通知する。
    print(f'PASS: {checked} added/changed code lines have same-line explanations.')  # 確認範囲を行数付きで報告する。


if __name__ == '__main__':  # インポートだけでGit監査を始めない。
    main()  # スクリプトとして呼ばれた場合だけ監査を実行する。
