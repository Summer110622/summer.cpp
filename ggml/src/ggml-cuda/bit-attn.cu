#include "bit-attn.cuh" // 二値AttentionのCUDA関数宣言を読み込む。

#if defined(GGML_USE_HIP) || defined(GGML_USE_MUSA) // NVIDIA専用命令をHIPやMUSAでコンパイルしないよう分岐する。

bool ggml_cuda_bit_attention_supported(const ggml_tensor * op) { // CUDAバックエンドが扱える型と出力形状を判定する。
    GGML_UNUSED(op); // 未対応バックエンドの引数を意図的な未使用として扱う。
    return false; // この条件では機能を利用しないことを返す。
} // この処理または定義のブロックを閉じる。
void ggml_cuda_op_bit_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst) { // CUDAストリーム上へ符号パックカーネルを起動する。
    GGML_UNUSED_VARS(ctx, dst); // NVIDIA以外のスタブで未使用引数の警告を抑える。
    GGML_ABORT("BitAttention CUDA kernels require NVIDIA; use CPU fallback"); // 未対応GPUへ誤配送された場合は静かに誤計算せず停止する。
} // この処理または定義のブロックを閉じる。
void ggml_cuda_op_bit_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) { // CUDAストリーム上へ二値内積カーネルを起動する。
    GGML_UNUSED_VARS(ctx, dst); // NVIDIA以外のスタブで未使用引数の警告を抑える。
    GGML_ABORT("BitAttention CUDA kernels require NVIDIA; use CPU fallback"); // 未対応GPUへ誤配送された場合は静かに誤計算せず停止する。
} // この処理または定義のブロックを閉じる。
void ggml_cuda_op_bit_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) { // Value次元に合う融合CUDAカーネルを選択する。
    GGML_UNUSED_VARS(ctx, dst); // NVIDIA以外のスタブで未使用引数の警告を抑える。
    GGML_ABORT("BitAttention CUDA kernels require NVIDIA; use CPU fallback"); // 未対応GPUへ誤配送された場合は静かに誤計算せず停止する。
} // この処理または定義のブロックを閉じる。

#else // 代替プラットフォーム用の実装へ切り替える。

#include <math_constants.h> // CUDART_INF_F (NVIDIA only)

// Layout is passed by value: views/non-contiguous Q, K, V and masks retain their
// byte strides. Packed words use I32 storage but unsigned bitwise arithmetic.
struct bit_view { // テンソルの形状とバイトストライドを値渡しする構造体を定義する。
    const char * data; // テンソルストレージの先頭バイトを保持する。
    int64_t ne[4]; // 特徴・トークン・ヘッド・バッチのサイズを保持する。
    size_t nb[4]; // 各軸の移動に必要なバイト数を保持する。
    int type; // F32、F16、BF16などの要素型識別子を保持する。
}; // この処理または定義のブロックを閉じる。

static bit_view bit_make_view(const ggml_tensor * t) { // テンソルの型・形状・ストライドをデバイスへ渡せる値にする。
    bit_view a = {}; // 欠省テンソルを表現できるようビューをゼロ初期化する。
    if (t) { // 省略されていないテンソルだけビューへ複写する。
        a.data = static_cast<const char *>(t->data); // バイト単位のアドレス計算に使う先頭ポインタを記録する。
        for (int i = 0; i < 4; ++i) { // GGMLの全4軸のサイズとストライドを複写する。
            a.ne[i] = t->ne[i]; // 各軸の要素数をデバイスへ渡すビューに複写する。
            a.nb[i] = t->nb[i]; // 非連続ビューを保つため各軸のバイトストライドを複写する。
        } // この処理または定義のブロックを閉じる。
        a.type = t->type; // デバイス側で値を復号するため要素型を記録する。
    } // この処理または定義のブロックを閉じる。
    return a; // 構築したビューまたは読み出した配列を返す。
} // この処理または定義のブロックを閉じる。

static __device__ __forceinline__ const char * bit_ptr( // 全4軸のバイトストライドを使って要素アドレスを求める。
        const bit_view & a, int64_t d, int64_t n, int64_t h, int64_t b) { // 入力ビューと特徴・トークン・ヘッド・バッチ座標を受け取る。
    return a.data + d*a.nb[0] + n*a.nb[1] + h*a.nb[2] + b*a.nb[3]; // 転置やビューを保持したまま目的要素へアクセスする。
} // この処理または定義のブロックを閉じる。

static __device__ __forceinline__ float bit_load( // CUDA入力のF16・BF16・F32をF32値として読む。
        const bit_view & a, int64_t d, int64_t n, int64_t h, int64_t b) { // 読み出すビューと全4軸の座標を受け取る。
    const char * p = bit_ptr(a, d, n, h, b); // デバイス入力の対象要素のアドレスを取得する。
    if (a.type == GGML_TYPE_F16) { // FP16入力の変換経路を選ぶ。
        return __half2float(*reinterpret_cast<const half *>(p)); // FP16ストレージをFP32値へ変換する。
    } // この処理または定義のブロックを閉じる。
    if (a.type == GGML_TYPE_BF16) { // BF16入力の復元経路を選ぶ。
        return __uint_as_float(uint32_t(*reinterpret_cast<const uint16_t *>(p)) << 16); // BF16をFP32の上位16ビットとして復元する。
    } // この処理または定義のブロックを閉じる。
    return *reinterpret_cast<const float *>(p); // F32入力は変換せず読み出す。
} // この処理または定義のブロックを閉じる。

static __device__ __forceinline__ float bit_dot( // XORとpopcountで符号ベクトルの内積を計算する。
        const bit_view & q, const bit_view & k, int d, int64_t iq, int64_t ik, int64_t h, int64_t b) { // 二値Q/Kと実特徴次元および出力側座標を受け取る。
    const int64_t kh = h/(q.ne[2]/k.ne[2]); // GQAの共有率に従いKeyヘッドを選ぶ。
    const int64_t kb = b/(q.ne[3]/k.ne[3]); // ブロードキャスト先に対応するKeyバッチを選ぶ。
    const uint32_t tail = (d & 31) ? (uint32_t(1) << (d & 31)) - 1 : 0xffffffffu; // 32の倍数でない特徴次元のパディングを無視するマスクを作る。
    int distance = 0; // XORで異なるビットの総数を初期化する。
    for (int64_t w = 0; w < q.ne[0]; ++w) { // 特徴を32ビット語ごとに走査する。
        const uint32_t qw = *reinterpret_cast<const uint32_t *>(bit_ptr(q, w, iq, h, b)); // Queryの対応する32ビット語をストライド付きで読む。
        const uint32_t kw = *reinterpret_cast<const uint32_t *>(bit_ptr(k, w, ik, kh, kb)); // 共有されたKeyの対応語を読む。
        uint32_t diff = qw ^ kw; // 32特徴の符号の不一致を一度のXORで求める。
        if (w + 1 == q.ne[0]) { // 最後の語の未使用ビットを除去する。
            diff &= tail; // 外部から渡された語の未使用ビットを距離に数えない。
        } // この処理または定義のブロックを閉じる。
        distance += __popc(diff); // CUDAの32ビットpopcountで不一致数を加算する。
    } // この処理または定義のブロックを閉じる。
    return float(d) - 2.0f*float(distance); // Hamming距離から符号ベクトルの内積を求める。
} // この処理または定義のブロックを閉じる。

static __global__ void bit_pack_kernel(bit_view x, uint32_t * out, int64_t words, int64_t total) { // 各warpで32個の符号を一つの語に集約する。
    // One complete warp per word; even padding lanes participate in the ballot.
    const int lane = threadIdx.x & 31; // warp内のスレッド番号を下位5ビットから求める。
    const int64_t word = (int64_t(blockIdx.x)*blockDim.x + threadIdx.x)/32; // このwarpが出力する32ビット語番号を取得・計算する。
    if (word >= total) { // 無効な語を担当するwarp全体を終了させる。
        return; // uniform for all 32 lanes
    } // この処理または定義のブロックを閉じる。
    const int64_t row = word/words; // Queryまたは入力ベクトルの線形行番号を取得・計算する。
    const int64_t d = (word % words)*32 + lane; // 実特徴次元またはこのlaneが読む特徴番号を取得・計算する。
    const int64_t n = row % x.ne[1]; // 現在のトークン番号を取得・計算する。
    const int64_t h = (row/x.ne[1]) % x.ne[2]; // 現在のヘッド番号を取得・計算する。
    const int64_t b = row/(x.ne[1]*x.ne[2]); // 現在のバッチ番号を取得・計算する。
    const bool sign = d < x.ne[0] && bit_load(x, d, n, h, b) >= 0.0f; // 有効次元だけ符号を判定しパディングのビットはゼロにする。
    const uint32_t packed = __ballot_sync(0xffffffffu, sign); // 全32laneの符号判定を1個の32ビット語へ集約する。
    if (lane == 0) { // 一つの語を複数laneが重複書き込みしないようにする。
        out[word] = packed; // warpの代表laneだけが完成した語を保存する。
    } // この処理または定義のブロックを閉じる。
} // この処理または定義のブロックを閉じる。

static __global__ void bit_mul_mat_kernel(bit_view k, bit_view q, float * out, int d, int64_t total) { // 出力要素ごとに二値内積を求めるCUDAカーネルを定義する。
    for (int64_t i = int64_t(blockIdx.x)*blockDim.x + threadIdx.x; i < total; // CUDAの各スレッドから異なるスコア要素の処理を開始する。
            i += int64_t(gridDim.x)*blockDim.x) { // グリッド全体の幅で進めて巨大なスコア行列も分担する。
        const int64_t ik = i % k.ne[1]; // Keyのトークン番号を取得・計算する。
        const int64_t row = i/k.ne[1]; // Queryまたは入力ベクトルの線形行番号を取得・計算する。
        const int64_t iq = row % q.ne[1]; // Queryのトークン番号を取得・計算する。
        const int64_t h = (row/q.ne[1]) % q.ne[2]; // 現在のヘッド番号を取得・計算する。
        const int64_t b = row/(q.ne[1]*q.ne[2]); // 現在のバッチ番号を取得・計算する。
        out[i] = bit_dot(q, k, d, iq, ik, h, b); // 各スレッドが担当するスコア要素へ二値内積を保存する。
    } // この処理または定義のブロックを閉じる。
} // この処理または定義のブロックを閉じる。

static __device__ __forceinline__ float bit_warp_max(float x) { // warp内の最大値を全laneへ共有する。
    for (int delta = 16; delta; delta >>= 1) { // 交換距離を半減させながら32laneを縮約する。
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, delta)); // butterfly交換でwarp全体の最大値を求める。
    } // この処理または定義のブロックを閉じる。
    return x; // 読み出しまたは縮約の結果を返す。
} // この処理または定義のブロックを閉じる。
static __device__ __forceinline__ float bit_warp_sum(float x) { // warp内の総和を全laneへ共有する。
    for (int delta = 16; delta; delta >>= 1) { // 交換距離を半減させながら32laneを縮約する。
        x += __shfl_xor_sync(0xffffffffu, x, delta); // butterfly交換でwarp全体の和を求める。
    } // この処理または定義のブロックを閉じる。
    return x; // 読み出しまたは縮約の結果を返す。
} // この処理または定義のブロックを閉じる。

// Reference fused CUDA kernel: a warp owns one query and streams 32-key tiles.
// Scores/probabilities stay in registers. V and softmax/accumulation remain F32;
// no Tq*Tk global tensor is created. This is not a Tensor-Core-tuned speed claim.
template <int DV_PAD> // Value次元の切り上げサイズごとにカーネルを特殊化する。
static __global__ void bit_attn_kernel( // 32キーずつ走査するonline softmaxのCUDA本体を定義する。
        bit_view q, bit_view k, bit_view v, bit_view mask, bit_view sinks, float * out, // Q/K/V・マスク・sinkのビューと出力アドレスを受け取る。
        int d, float scale, float bias, float cap, int64_t rows) { // 実特徴次元・スケール・ALiBi・ソフトキャップ・Query行数を受け取る。
    const int lane = threadIdx.x & 31; // warp内のスレッド番号を下位5ビットから求める。
    const int64_t row = (int64_t(blockIdx.x)*blockDim.x + threadIdx.x)/32; // Queryまたは入力ベクトルの線形行番号を取得・計算する。
    if (row >= rows) { // 範囲外Queryのwarp全体を終了させる。
        return; // whole warp exits; all live lanes use the same full-warp mask
    } // この処理または定義のブロックを閉じる。
    const int64_t iq = row % q.ne[1]; // Queryのトークン番号を取得・計算する。
    const int64_t h = (row/q.ne[1]) % q.ne[2]; // 現在のヘッド番号を取得・計算する。
    const int64_t b = row/(q.ne[1]*q.ne[2]); // 現在のバッチ番号を取得・計算する。
    const int64_t vh = h/(q.ne[2]/v.ne[2]); // 共有先のValueヘッド番号を取得・計算する。
    const int64_t vb = b/(q.ne[3]/v.ne[3]); // 共有先のValueバッチ番号を取得・計算する。
    int64_t h_pow2 = 1; // ヘッド数以下の最大の2の累乗を探索する初期値を取得・計算する。
    while (h_pow2 <= q.ne[2]/2) { // ヘッド数を超えない範囲で2の累乗を大きくする。
        h_pow2 *= 2; // ALiBiのヘッド群を区切る2の累乗を更新する。
    } // この処理または定義のブロックを閉じる。
    const float m0 = powf(2.0f, -bias/float(h_pow2)); // ALiBiの先頭ヘッド群に使う傾斜の底を取得・計算する。
    const float m1 = powf(2.0f, -0.5f*bias/float(h_pow2)); // 残りのALiBiヘッド群に使う傾斜の底を取得・計算する。
    const float slope = bias == 0.0f ? 1.0f : h < h_pow2 // ALiBiを使わないときは1、使うときはヘッド群を選ぶ。
        ? powf(m0, float(h + 1)) : powf(m1, float(2*(h - h_pow2) + 1)); // 2の累乗でないヘッド数でも既存GGMLと同じ傾斜を計算する。
    float m = sinks.data ? bit_load(sinks, h, 0, 0, 0) : -CUDART_INF_F; // sinkのロジットをonline softmaxの初期最大値へ入れる。
    float sum = m == -CUDART_INF_F ? 0.0f : 1.0f; // 有限sinkだけを初期分母へ含める。
    float acc[DV_PAD/32] = {}; // 各laneが担当するV特徴の累積値をゼロにする。
    for (int64_t start = 0; start < k.ne[1]; start += 32) { // Keyをwarp幅の32トークン単位で走査する。
        const int64_t ik = start + lane; // Keyのトークン番号を取得・計算する。
        float score = -CUDART_INF_F; // 無効キーやマスクされたキーを初期状態で除外する。
        if (ik < k.ne[1]) { // 末尾タイルの範囲内キーだけスコアを計算する。
            const float mv = mask.data ? bit_load(mask, ik, iq, h % mask.ne[2], b % mask.ne[3]) : 0.0f; // 現在のQuery-Key対の加算マスク値を取得・計算する。
            if (mv != -CUDART_INF_F) { // マスクされていないキーだけスコアへ参加させる。
                score = scale*bit_dot(q, k, d, iq, ik, h, b); // XORとpopcountからこのQuery-Key対のロジットを計算する。
                if (cap > 0.0f) { // 正のソフトキャップが指定された場合だけtanhを適用する。
                    score = cap*tanhf(score/cap); // ロジットの絶対値をtanhで滑らかに抑える。
                } // この処理または定義のブロックを閉じる。
                score += slope*mv; // マスクの有限バイアスへALiBiヘッド傾斜を掛けて加える。
            } // この処理または定義のブロックを閉じる。
        } // この処理または定義のブロックを閉じる。
        const float m_new = fmaxf(m, bit_warp_max(score)); // この32キーの最大値と過去の最大値を合わせる。
        const float alpha = m == -CUDART_INF_F ? 0.0f : expf(m - m_new); // 初期の負無限大を特別扱いして再スケール時のNaNを避ける。
        const float p = score == -CUDART_INF_F ? 0.0f : expf(score - m_new); // マスク済みキーは厳密にゼロ重みとする。
        sum = alpha*sum + bit_warp_sum(p); // ブロック内32キーの指数重みを分母へ加える。
#pragma unroll // 固定回数の特徴ループをコンパイル時に展開する。
        for (int j = 0; j < DV_PAD/32; ++j) { // このlaneへ割り当てられたValue特徴を走査する。
            acc[j] *= alpha; // 保存済みの出力を新しい最大値の基準へ換算する。
        } // この処理または定義のブロックを閉じる。
        for (int t = 0; t < 32 && start + t < k.ne[1]; ++t) { // 末尾を越えない範囲でタイル内のキーを集計する。
            const float pt = __shfl_sync(0xffffffffu, p, t); // 対象キーの重みをwarpの全laneへ配る。
            if (pt != 0.0f) { // 重みがゼロのキーについて不要なVの読み出しを省く。
#pragma unroll // 固定回数の特徴ループをコンパイル時に展開する。
                for (int j = 0; j < DV_PAD/32; ++j) { // このlaneへ割り当てられたValue特徴を走査する。
                    const int dv = lane + 32*j; // このlaneが担当するVの特徴番号を求める。
                    if (dv < v.ne[0]) { // 特殊化で追加された余剰特徴へのアクセスを防ぐ。
                        acc[j] += pt*bit_load(v, dv, start + t, vh, vb); // Vを二値化せずFP32で重み付き累積する。
                    } // この処理または定義のブロックを閉じる。
                } // この処理または定義のブロックを閉じる。
            } // この処理または定義のブロックを閉じる。
        } // この処理または定義のブロックを閉じる。
        m = m_new; // 次のキーまたはタイルへ最大値を引き継ぐ。
    } // この処理または定義のブロックを閉じる。
    const float inv_sum = sum > 0.0f ? 1.0f/sum : 0.0f; // 全マスク行では逆数をゼロにしてゼロ出力を保つ。
#pragma unroll // 固定回数の特徴ループをコンパイル時に展開する。
    for (int j = 0; j < DV_PAD/32; ++j) { // このlaneへ割り当てられたValue特徴を走査する。
        const int dv = lane + 32*j; // このlaneが担当するVの特徴番号を求める。
        if (dv < v.ne[0]) { // 特殊化で追加された余剰特徴へのアクセスを防ぐ。
            out[dv + v.ne[0]*(h + q.ne[2]*(iq + q.ne[1]*b))] = acc[j]*inv_sum; // GGMLの[Dv,Hq,Nq,Bq]配置へ正規化結果を書く。
        } // この処理または定義のブロックを閉じる。
    } // この処理または定義のブロックを閉じる。
} // この処理または定義のブロックを閉じる。

static bool bit_float_type(ggml_type t) { // カーネルが直接読み出せる浮動小数点型を判定する。
    return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16; // 対応する浮動小数点のストレージ型か判定する。
} // この処理または定義のブロックを閉じる。

bool ggml_cuda_bit_attention_supported(const ggml_tensor * op) { // CUDAバックエンドが扱える型と出力形状を判定する。
    if (!ggml_is_contiguous(op)) { // このカーネルが要求する連続出力配置か検査する。
        return false; // この条件では機能を利用しないことを返す。
    } // この処理または定義のブロックを閉じる。
    switch (op->op) { // テンソル型または演算番号に対応する実装へ分岐する。
        case GGML_OP_BIT_PACK: // 符号パック演算の対応条件を調べる。
            return bit_float_type(op->src[0]->type) && op->type == GGML_TYPE_I32; // パック入力が対応浮動小数点型で出力がI32か確認する。
        case GGML_OP_BIT_MUL_MAT: // 明示的な二値内積行列の対応条件を調べる。
            return op->src[0]->type == GGML_TYPE_I32 && op->src[1]->type == GGML_TYPE_I32; // 両入力が32ビット語のI32ストレージか確認する。
        case GGML_OP_BIT_ATTN_EXT: // 融合二値Attentionの対応条件を調べる。
            return op->src[0]->type == GGML_TYPE_I32 && op->src[1]->type == GGML_TYPE_I32 && // 融合経路のQとKがパック済みI32か確認する。
                bit_float_type(op->src[2]->type) && op->src[2]->ne[0] <= 1024 && // Vの型を検査し今回特殊化した1024特徴以内へ制限する。
                (!op->src[3] || bit_float_type(op->src[3]->type)); // マスクがある場合は対応する浮動小数点型に限る。
        default: // 対応表にない型または演算を扱う経路を選ぶ。
            return false; // この条件では機能を利用しないことを返す。
    } // この処理または定義のブロックを閉じる。
} // この処理または定義のブロックを閉じる。

void ggml_cuda_op_bit_pack(ggml_backend_cuda_context & ctx, ggml_tensor * dst) { // CUDAストリーム上へ符号パックカーネルを起動する。
    const int64_t total = ggml_nelements(dst); // 処理する出力要素の総数を取得・計算する。
    const int64_t blocks = (total + 7)/8; // 必要なCUDAブロック数を取得・計算する。
    GGML_ASSERT(blocks <= INT32_MAX); // CUDAグリッドの表現範囲を超える起動を拒否する。
    bit_pack_kernel<<<static_cast<unsigned>(blocks), 256, 0, ctx.stream()>>>( // 8warpを持つブロックで各warpに一つの語をパックさせる。
        bit_make_view(dst->src[0]), static_cast<uint32_t *>(dst->data), dst->ne[0], total); // 入力ビュー・出力・1ベクトルの語数・総語数を渡す。
    CUDA_CHECK(cudaGetLastError()); // 非同期カーネル起動時のCUDAエラーを検出する。
} // この処理または定義のブロックを閉じる。

void ggml_cuda_op_bit_mul_mat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) { // CUDAストリーム上へ二値内積カーネルを起動する。
    const int64_t total = ggml_nelements(dst); // 処理する出力要素の総数を取得・計算する。
    const int blocks = int(std::min<int64_t>((total + 255)/256, 65535)); // 必要なCUDAブロック数を取得・計算する。
    bit_mul_mat_kernel<<<blocks, 256, 0, ctx.stream()>>>( // 256スレッドのブロックで二値スコア行列を計算する。
        bit_make_view(dst->src[0]), bit_make_view(dst->src[1]), static_cast<float *>(dst->data), // パック済みK/QのビューとF32出力を渡す。
        ggml_get_op_params_i32(dst, 0), total); // パディングを除いた実特徴次元と出力要素数を渡す。
    CUDA_CHECK(cudaGetLastError()); // 非同期カーネル起動時のCUDAエラーを検出する。
} // この処理または定義のブロックを閉じる。

template <int D> // 起動するValue次元に対応したカーネルを選べるようにする。
static void bit_attn_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) { // Value次元の特殊化を選んだCUDAカーネルを起動する。
    const int64_t rows = ggml_nrows(dst->src[0]); // 処理するベクトルの総数を取得・計算する。
    const int64_t blocks = (rows + 3)/4; // 必要なCUDAブロック数を取得・計算する。
    GGML_ASSERT(blocks <= INT32_MAX); // CUDAグリッドの表現範囲を超える起動を拒否する。
    bit_attn_kernel<D><<<static_cast<unsigned>(blocks), 128, 0, ctx.stream()>>>( // 4warpのブロックで4本のQueryを分担する融合カーネルを起動する。
        bit_make_view(dst->src[0]), bit_make_view(dst->src[1]), bit_make_view(dst->src[2]), // 非連続レイアウトを保ったQ/K/Vビューを渡す。
        bit_make_view(dst->src[3]), bit_make_view(dst->src[4]), static_cast<float *>(dst->data), // 省略可能なマスク・sinkとF32出力ポインタを渡す。
        ggml_get_op_params_i32(dst, 3), ggml_get_op_params_f32(dst, 0), // 実特徴次元とロジットスケールを演算パラメータから復元する。
        ggml_get_op_params_f32(dst, 1), ggml_get_op_params_f32(dst, 2), rows); // ALiBi・ソフトキャップ・Query総行数を渡す。
} // この処理または定義のブロックを閉じる。

void ggml_cuda_op_bit_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) { // Value次元に合う融合CUDAカーネルを選択する。
    GGML_ASSERT(ggml_cuda_bit_attention_supported(dst)); // 型と形状が対応範囲であることを起動前に検査する。
    const int64_t d = dst->src[2]->ne[0]; // 実特徴次元またはこのlaneが読む特徴番号を取得・計算する。
    if      (d <=   32) { bit_attn_launch<  32>(ctx, dst); } // Valueが32特徴以内ならlaneごとに一つの累積値を使う。
    else if (d <=   64) { bit_attn_launch<  64>(ctx, dst); } // Valueが64特徴以内なら64次元用の特殊化を使う。
    else if (d <=  128) { bit_attn_launch< 128>(ctx, dst); } // Valueが128特徴以内なら128次元用の特殊化を使う。
    else if (d <=  256) { bit_attn_launch< 256>(ctx, dst); } // Valueが256特徴以内なら256次元用の特殊化を使う。
    else if (d <=  512) { bit_attn_launch< 512>(ctx, dst); } // Valueが512特徴以内なら512次元用の特殊化を使う。
    else               { bit_attn_launch<1024>(ctx, dst); } // 対応上限の1024特徴用の特殊化を使う。
    CUDA_CHECK(cudaGetLastError()); // 非同期カーネル起動時のCUDAエラーを検出する。
} // この処理または定義のブロックを閉じる。
#endif // プラットフォーム別実装の条件分岐を閉じる。
