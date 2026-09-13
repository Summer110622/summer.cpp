#include "bit-attention.cuh" // CUDAディスパッチ用の公開宣言を取り込む。
#include <climits> // CUDAグリッド寸法の整数上限を検証する。
#include <cstdint> // パックする32bit符号ワードを定義する。
#include <cstring> // 演算設定をホスト側で安全に読み取る。

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) // warp幅32のNVIDIA専用実装を他バックエンドでコンパイルしない。
namespace { // 実装用のCUDAシンボルを翻訳単位内に限定する。
constexpr int BIT_THREADS = 128; // KVタイルの128キーを128スレッドへ割り当てる。
constexpr int BIT_VALUES = 8; // 各スレッドで最大8要素を保持してDV<=1024を処理する。
struct bit_tensor { // デバイスへ値渡しするテンソル記述子を定義する。
    const char * data; // テンソル先頭のデバイスポインタを保持する。
    int64_t ne[4]; // 各軸の論理的な要素数を保持する。
    size_t nb[4]; // 各軸のbyte strideを保持する。
    int type; // ggmlの浮動小数点型を識別する。
}; // デバイス用テンソル記述子の定義を終了する。

static bit_tensor bit_describe(const ggml_tensor * t) { // ggmlテンソルをカーネル引数へ変換する。
    bit_tensor result = {}; // 省略されたマスクやsinkをnull記述子にする。
    if (t) { // 実在するテンソルだけのメタデータをコピーする。
        result.data = static_cast<const char *>(t->data); // 元のデバイスメモリをそのまま参照する。
        result.type = static_cast<int>(t->type); // デバイスで使う型識別子を設定する。
        for (int i = 0; i < 4; ++i) { result.ne[i] = t->ne[i]; result.nb[i] = t->nb[i]; } // 非連続テンソルの形状とstrideを保持する。
    } // テンソル記述子へのコピーを終了する。
    return result; // 値渡し可能な小さな記述子を返す。
} // ホスト側テンソル記述子の構築を終了する。

__device__ __forceinline__ float bit_read(const bit_tensor & t, int64_t d, int64_t n, int64_t h, int64_t b) { // 任意のstrideでFP32値を読み取る。
    const char * p = t.data + d*t.nb[0] + n*t.nb[1] + h*t.nb[2] + b*t.nb[3]; // 各軸のbyte offsetから読み取り位置を求める。
    if (t.type == GGML_TYPE_F32) { return *reinterpret_cast<const float *>(p); } // FP32はそのまま読み取る。
    if (t.type == GGML_TYPE_F16) { return __half2float(*reinterpret_cast<const half *>(p)); } // FP16をCUDAの標準変換でFP32へ展開する。
    return __uint_as_float(static_cast<uint32_t>(*reinterpret_cast<const uint16_t *>(p)) << 16); // BF16の上位16bitをFP32へ復元する。
} // 浮動小数点型別の読み取りを終了する。

__device__ __forceinline__ bool bit_nonnegative(float x) { // CUDAのfast-mathでも負のsubnormalを負のゼロと混同しない。
    const uint32_t bits = __float_as_uint(x); // 浮動小数点比較を行わずIEEE754の表現を取得する。
    const uint32_t magnitude = bits & 0x7fffffffu; // 符号を除いた指数と仮数を取得する。
    return magnitude <= 0x7f800000u && ((bits & 0x80000000u) == 0 || magnitude == 0); // x>=0と同じNaN・負のゼロ規則で1bitへ分類する。
} // CUDA上の厳密な符号分類を終了する。

__global__ void bit_pack_kernel(bit_tensor t, uint32_t * out, int64_t words, int64_t count) { // 1warpにつき32次元の符号を1ワードへパックする。
    const int64_t thread = static_cast<int64_t>(blockIdx.x)*blockDim.x + threadIdx.x; // 大規模テンソルでも溢れない全体スレッド番号を求める。
    const int64_t word = thread/32; // 32laneで共同生成するワード番号を求める。
    const int lane = threadIdx.x & 31; // warp内のbit位置を求める。
    if (word >= count) { return; } // 全laneが同じ条件で範囲外warpを終了する。
    const int64_t row = word/words; // ワード番号からテンソル行を求める。
    const int64_t d = (word % words)*32 + lane; // このlaneが符号化する特徴次元を求める。
    const int64_t n = row % t.ne[1]; // 行番号からトークン番号を求める。
    const int64_t h = row/t.ne[1] % t.ne[2]; // 行番号からヘッド番号を求める。
    const int64_t b = row/(t.ne[1]*t.ne[2]); // 行番号からバッチ番号を求める。
    const bool positive = d < t.ne[0] && bit_nonnegative(bit_read(t, d, n, h, b)); // 端数パディングを0とし非負値だけを1とする。
    const uint32_t packed = __ballot_sync(0xffffffffu, positive); // warp内の32個の真偽値を1つの32bitワードへ集約する。
    if (lane == 0) { out[word] = packed; } // 各warpの代表laneだけがパック結果を書き込む。
} // GPU上の符号パッキングを終了する。

__device__ __forceinline__ float bit_reduce(float value, float * scratch, bool maximum) { // 1ブロックの最大値または総和を求める。
    const int t = threadIdx.x; // ブロック内の要素位置を取得する。
    scratch[t] = value; // 各スレッドの部分値を共有メモリへ書き込む。
    __syncthreads(); // 全部分値の書き込み完了を揃える。
    for (int width = BIT_THREADS/2; width > 0; width >>= 1) { // 木構造で128要素を1要素へ集約する。
        if (t < width) { scratch[t] = maximum ? fmaxf(scratch[t], scratch[t + width]) : scratch[t] + scratch[t + width]; } // 現在の半区間どうしを合成する。
        __syncthreads(); // 次の段階で未完成の部分値を読まないよう同期する。
    } // ブロック内の縮約を完了する。
    const float result = scratch[0]; // すべてのスレッドが同じ縮約結果を取得する。
    __syncthreads(); // 次の縮約による上書きより前に全スレッドの読み取りを完了させる。
    return result; // ブロック最大値または総和を返す。
} // ブロック縮約を終了する。

__global__ void bit_attention_kernel( // logit・online softmax・V加重和を1カーネルに融合する。
        const uint32_t * qb, const uint32_t * kb, // 32次元ずつ詰めたQ/Kビット列を受け取る。
        bit_tensor q, bit_tensor k, bit_tensor v, // Q/K形状と高精度Vのstrideを受け取る。
        bit_tensor mask, bit_tensor sinks, bit_tensor bias, // 系列制約・sink・追加バイアスを受け取る。
        float * output, float scale, float max_bias, float cap) { // FP32出力とlogit変換設定を受け取る。
    const int t = threadIdx.x; // KVタイル内の担当キーを決める。
    const int64_t n = blockIdx.x; // このブロックが処理するクエリ番号を取得する。
    const int64_t h = blockIdx.y; // このブロックが処理するクエリヘッドを取得する。
    const int64_t b = blockIdx.z; // このブロックが処理するクエリバッチを取得する。
    const int64_t words = (q.ne[0] + 31)/32; // 符号ベクトルのパックワード数を求める。
    const int64_t qrow = n + q.ne[1]*(h + q.ne[2]*b); // Qパック領域の行位置を求める。
    const int64_t kh = h/(q.ne[2]/k.ne[2]); // GQAで共有するKヘッド番号を求める。
    const int64_t vh = h/(q.ne[2]/v.ne[2]); // GQAで共有するVヘッド番号を求める。
    const int64_t kbch = b/(q.ne[3]/k.ne[3]); // グループ共有されたKバッチ番号を求める。
    const int64_t vbch = b/(q.ne[3]/v.ne[3]); // グループ共有されたVバッチ番号を求める。
    int64_t hpow = 1; // ALiBiで使う2冪ヘッド数を初期化する。
    while (hpow <= q.ne[2]/2) { hpow <<= 1; } // 実際のヘッド数以下の最大2冪を求める。
    const float base = exp2f((h < hpow ? -max_bias : -0.5f*max_bias)/hpow); // ヘッド区間に対応するALiBi基数を計算する。
    const float slope = max_bias > 0 ? powf(base, h < hpow ? h + 1 : 2*(h - hpow) + 1) : 1.0f; // ggmlと同じALiBi傾きを求める。
    __shared__ float scratch[BIT_THREADS]; // 最大値と確率総和のブロック縮約に使う領域を確保する。
    __shared__ float probabilities[BIT_THREADS]; // 現在のKVタイルだけの非正規化確率を保持する。
    float accum[BIT_VALUES] = {}; // 各スレッドが担当するV成分をレジスタでFP32累積する。
    float maximum = sinks.data ? bit_read(sinks, h, 0, 0, 0) : -CUDART_INF_F; // sinkを含む初期logit最大値を設定する。
    float denominator = maximum == -CUDART_INF_F ? 0.0f : 1.0f; // sinkの分母への寄与を設定する。
    for (int64_t start = 0; start < k.ne[1]; start += BIT_THREADS) { // KVを128キーずつ走査し二乗サイズの行列保存を避ける。
        const int64_t j = start + t; // このスレッドが計算するキー番号を求める。
        float score = -CUDART_INF_F; // 範囲外キーやマスクされたキーのlogitを無効化する。
        if (j < k.ne[1]) { // 最後の端数タイルで領域外読み取りを防ぐ。
            const float mv = mask.data ? bit_read(mask, j, n, h % mask.ne[2], b % mask.ne[3]) : 0.0f; // ヘッド・バッチ方向にマスクをブロードキャストする。
            const float bv = bias.data ? bit_read(bias, j, n, h % bias.ne[2], b % bias.ne[3]) : 0.0f; // 追加バイアスを同じルールで読み取る。
            if (mv != -CUDART_INF_F && bv != -CUDART_INF_F) { // マスクを通過するキーだけ類似度を計算する。
                const int64_t krow = j + k.ne[1]*(kh + k.ne[2]*kbch); // 対応する共有キーのパック行を求める。
                int distance = 0; // Hamming距離を初期化する。
                for (int64_t w = 0; w < words; ++w) { distance += __popc(qb[qrow*words + w] ^ kb[krow*words + w]); } // XORとNVIDIA popcount命令で符号の不一致を数える。
                score = static_cast<float>(q.ne[0] - 2LL*distance)*scale; // D-2Hで符号内積を復元してスケールする。
                if (cap > 0) { score = cap*tanhf(score/cap); } // 必要な場合だけlogitを滑らかに制限する。
                score += slope*mv + bv; // ALiBi付きマスクと追加バイアスを加算する。
            } // 有効キーのlogit計算を終了する。
        } // 最終タイルの範囲検証を終了する。
        const float tile_max = bit_reduce(score, scratch, true); // 現在タイルの最大logitを求める。
        const float next_max = fmaxf(maximum, tile_max); // 過去タイルと現在タイルの最大値を統合する。
        const float alpha = maximum == -CUDART_INF_F ? 0.0f : expf(maximum - next_max); // 過去の分母と加重和を再スケーリングする係数を求める。
        const float probability = score == -CUDART_INF_F ? 0.0f : expf(score - next_max); // 全マスク行でも-inf同士を減算せず0を返す。
        probabilities[t] = probability; // 現在タイルの確率を全V成分の計算で共有する。
        const float tile_sum = bit_reduce(probability, scratch, false); // 非正規化確率をブロック全体で加算する。
        for (int slot = 0; slot < BIT_VALUES; ++slot) { accum[slot] *= alpha; } // 既存のV加重和を新しい最大logitに合わせて縮小する。
        for (int jlocal = 0; jlocal < BIT_THREADS && start + jlocal < k.ne[1]; ++jlocal) { // 有効なタイル内キーのVを加重する。
            const float weight = probabilities[jlocal]; // 共有メモリからキーの確率を取得する。
            if (weight == 0.0f) { continue; } // マスクされたVを読まず不要な演算を避ける。
            for (int slot = 0; slot < BIT_VALUES; ++slot) { // 各スレッドが受け持つ最大8成分を更新する。
                const int d = t + slot*BIT_THREADS; // 連続lane間でVの連続次元を読み取る。
                if (d < v.ne[0]) { accum[slot] = fmaf(weight, bit_read(v, d, start + jlocal, vh, vbch), accum[slot]); } // 元のV精度を読み取りFP32 FMAで加重和を累積する。
            } // このキーのV成分をすべて累積する。
        } // 現在KVタイルのV集約を終了する。
        denominator = alpha*denominator + tile_sum; // online softmax分母を更新する。
        maximum = next_max; // 次タイルで使用するlogit基準を保存する。
        __syncthreads(); // 全スレッドのV集約完了前に確率領域が上書きされることを防ぐ。
    } // KV全体の融合処理を終了する。
    const float inv = denominator > 0.0f ? 1.0f/denominator : 0.0f; // 全マスク行では出力を0にしてNaNを防ぐ。
    const int64_t outrow = h + q.ne[2]*(n + q.ne[1]*b); // FlashAttentionと同じ出力行順を求める。
    for (int slot = 0; slot < BIT_VALUES; ++slot) { // 担当する全V成分を出力する。
        const int d = t + slot*BIT_THREADS; // 出力成分の次元番号を求める。
        if (d < v.ne[0]) { output[outrow*v.ne[0] + d] = accum[slot]*inv; } // 端数を保護して正規化済みのFP32出力を書き込む。
    } // 担当成分の書き込みを終了する。
} // fused BitAttentionのデバイス実行を終了する。

static void bit_launch_pack(const ggml_tensor * t, uint32_t * out, cudaStream_t stream) { // パックカーネルを同じCUDAストリームへ投入する。
    const int64_t words = (t->ne[0] + 31)/32; // 端数を含むワード数を求める。
    const int64_t count = ggml_nrows(t)*words; // 全行を格納するワード総数を求める。
    const int64_t blocks = (count + 7)/8; // 256スレッドの8warpで1ブロックあたり8ワードを生成する。
    GGML_ASSERT(blocks <= INT_MAX); // CUDAのx方向グリッド上限を超える起動を防ぐ。
    bit_pack_kernel<<<static_cast<unsigned>(blocks), 256, 0, stream>>>(bit_describe(t), out, words, count); // 符号パッキングを非同期に起動する。
    CUDA_CHECK(cudaGetLastError()); // カーネルの起動エラーを即座に検出する。
} // パックカーネルのホスト側起動を終了する。
} // CUDA内部実装の名前空間を終了する。
#endif // NVIDIA専用のデバイスコードを終了する。

bool ggml_cuda_bit_attn_ext_supported(const ggml_tensor * dst) { // スケジューラへ正確な対応条件を返す。
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) // NVIDIA以外のCUDA互換バックエンドを識別する。
    GGML_UNUSED(dst); // 未使用引数の警告を抑える。
    return false; // HIP/MUSAでは黙って通常AttentionにせずCPUへフォールバックさせる。
#else // NVIDIA CUDAでの対応条件を調べる。
    if (dst->op != GGML_OP_BIT_ATTN_EXT || dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst)) { return false; } // 専用opと連続FP32出力を要求する。
    for (int i = 0; i < 6; ++i) { // Q/K/Vとオプション入力の型を確認する。
        const ggml_tensor * t = dst->src[i]; // 現在の入力テンソルを取得する。
        if (!t) { if (i < 3) { return false; } else { continue; } } // 必須入力の欠落だけを拒否する。
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) { return false; } // デバイス読み取り関数にない量子化型を拒否する。
    } // 各入力の型検証を終了する。
    const ggml_tensor * q = dst->src[0]; // グリッド計算に使用するクエリ形状を取得する。
    const ggml_tensor * k = dst->src[1]; // パック起動の大きさを検証するキー形状を取得する。
    const int64_t words = (q->ne[0] + 31)/32; // ワード数を求める。
    if (q->ne[0] > 4096 || dst->src[2]->ne[0] > BIT_THREADS*BIT_VALUES) { return false; } // レジスタ消費とパック処理の対応範囲を制限する。
    if (q->ne[1] > INT_MAX || q->ne[2] > 65535 || q->ne[3] > 65535) { return false; } // CUDAの各グリッド次元上限を守る。
    return ggml_nrows(q) <= (static_cast<int64_t>(INT_MAX)*8)/words && ggml_nrows(k) <= (static_cast<int64_t>(INT_MAX)*8)/words; // パックカーネルのx方向グリッドも検証する。
#endif // NVIDIA向け対応判定を終了する。
} // CUDAサポート判定を終了する。

void ggml_cuda_bit_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) { // 一時パック領域を確保して融合カーネルを起動する。
#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) // 未対応バックエンドの誤ディスパッチを明示的に検出する。
    GGML_UNUSED(ctx); // 未使用コンテキストの警告を抑える。
    GGML_UNUSED(dst); // 未使用テンソルの警告を抑える。
    GGML_ABORT("BitAttention CUDA kernel requires NVIDIA; use the CPU backend"); // 異なる計算を黙って返さず実行を停止する。
#else // NVIDIA上で符号化と融合Attentionを起動する。
    GGML_ASSERT(ggml_cuda_bit_attn_ext_supported(dst)); // 対応判定と実際のディスパッチを一致させる。
    const ggml_tensor * q = dst->src[0]; // クエリテンソルを参照する。
    const ggml_tensor * k = dst->src[1]; // キーテンソルを参照する。
    const size_t words = static_cast<size_t>((q->ne[0] + 31)/32); // 1行あたりのパックワード数を求める。
    ggml_cuda_pool_alloc<uint32_t> qb(ctx.pool(), static_cast<size_t>(ggml_nrows(q))*words); // ストリーム順序で再利用可能なQパック領域を確保する。
    ggml_cuda_pool_alloc<uint32_t> kb(ctx.pool(), static_cast<size_t>(ggml_nrows(k))*words); // すべてのクエリで共有するKパック領域を確保する。
    const cudaStream_t stream = ctx.stream(); // ggmlスケジューラと同じCUDAストリームを使用する。
    bit_launch_pack(q, qb.get(), stream); // RoPE適用済みQの符号をデバイス上でパックする。
    bit_launch_pack(k, kb.get(), stream); // 現在のKVキャッシュからKの符号をデバイス上でパックする。
    float config[3]; // logit設定を保持するホスト変数を確保する。
    std::memcpy(config, dst->op_params, sizeof(config)); // スケール・ALiBi・ソフトキャップを読み取る。
    const dim3 grid(static_cast<unsigned>(q->ne[1]), static_cast<unsigned>(q->ne[2]), static_cast<unsigned>(q->ne[3])); // クエリ・ヘッド・バッチごとに1ブロックを起動する。
    bit_attention_kernel<<<grid, BIT_THREADS, 0, stream>>>( // 二乗サイズのglobal-memoryスコア行列を生成しない融合カーネルを起動する。
        qb.get(), kb.get(), bit_describe(q), bit_describe(k), bit_describe(dst->src[2]), // パック済みQ/Kと高精度Vの形状を渡す。
        bit_describe(dst->src[3]), bit_describe(dst->src[4]), bit_describe(dst->src[5]), // マスク・sink・追加バイアスを渡す。
        static_cast<float *>(dst->data), config[0], config[1], config[2]); // FP32出力とlogit設定を渡す。
    CUDA_CHECK(cudaGetLastError()); // 融合カーネルの起動エラーを検出する。
#endif // NVIDIA上での起動処理を終了する。
} // CUDAのBitAttention forwardを終了する。
