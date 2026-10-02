// Implements: include/ninfer/ops/qsa.h
//
// Phase-2 QSA route. Pooled block keys are recomputed per call from the paged raw index keys
// (query-independent, so once per sequence per call), scored for every column on CUDA cores,
// reduced to the top block budget by an MSB-first radix select, and emitted in ascending token
// order. Sparse attention then streams only the selected K/V rows.
#include "ops/launcher/qsa.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/warp.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kPage       = kPagedKVPageSize;
constexpr int kMaxIndexD  = 128;
constexpr int kScoreBlocks  = 64;  // pooled blocks per score CTA
constexpr int kScoreColumns = 8;   // columns per score CTA
constexpr int kSelectThreads = 512;
constexpr int kAttnTile      = 32;
constexpr int kAttnHeadDim   = 256;
constexpr int kMaxGroup      = 16;
constexpr int kPositionSlots = 4; // three RoPE axes, padded

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// HF Qwen4Exp: inv_freq in FP32, cos/sin rounded to the activation dtype, BF16 elementwise RoPE.
__device__ __forceinline__ void rope_coefficients(int pair, int rotary_dim, float theta,
                                                  std::int64_t position, float& c, float& s) {
    const float inv_freq = 1.0f / powf(theta, static_cast<float>(2 * pair) / rotary_dim);
    const float angle    = static_cast<float>(position) * inv_freq;
    c                    = bf(cosf(angle));
    s                    = bf(sinf(angle));
}

// Text MRoPE positions of one token: pair i rotates with axis i%3 (Qwen3.5/Qwen4Exp interleaved
// sections [11,11,10] over 32 pairs). Text tokens carry three equal axes, which is 1-D RoPE.
struct AxisPositions {
    std::int64_t axis[3];
};

__device__ __forceinline__ AxisPositions uniform_axes(std::int64_t position) {
    return {{position, position, position}};
}

// Rope positions are I32 [T] (one axis) or [T,3] axis-major (element (t,a) at a*T+t).
__device__ __forceinline__ AxisPositions token_axes(const std::int32_t* rope, int axes,
                                                    int tokens, int t) {
    if (axes == 1) return uniform_axes(rope[t]);
    return {{rope[t], rope[tokens + t], rope[2 * tokens + t]}};
}

// Normalizes and rotates one Di-vector held in shared memory `v` (FP32, already BF16-valued).
// Called by one warp; writes BF16 results to `out`.
__device__ void norm_rope_vector(float* v, int di, const __nv_bfloat16* weight, float eps,
                                 AxisPositions position, int rotary_dim, float theta,
                                 __nv_bfloat16* out) {
    const int lane = threadIdx.x & 31;
    float sum      = 0.0f;
    for (int d = lane; d < di; d += 32) sum += v[d] * v[d];
    sum = warp_reduce_sum(sum);
    sum = __shfl_sync(0xffffffffu, sum, 0);
    const float inv = rsqrtf(sum / di + eps);
    __syncwarp();
    for (int d = lane; d < di; d += 32) v[d] = bf(v[d] * inv * (1.0f + __bfloat162float(weight[d])));
    __syncwarp();
    const int half = rotary_dim / 2;
    for (int d = lane; d < di; d += 32) {
        float result = v[d];
        if (d < rotary_dim) {
            const int pair = d < half ? d : d - half;
            float c, s;
            rope_coefficients(pair, rotary_dim, theta, position.axis[pair % 3], c, s);
            const float rotated = d < half ? -v[d + half] : v[d - half];
            result              = bf(bf(v[d] * c) + bf(rotated * s));
        }
        out[d] = __float2bfloat16_rn(result);
    }
}

__global__ void prepare_query_kernel(const __nv_bfloat16* __restrict__ q,
                                     const __nv_bfloat16* __restrict__ weight,
                                     const std::int32_t* __restrict__ positions, int axes,
                                     int di, int heads, int vectors, int rotary_dim, float theta,
                                     float eps, __nv_bfloat16* __restrict__ out) {
    __shared__ float stage[8][kMaxIndexD];
    const int warp   = threadIdx.x >> 5;
    const int vector = blockIdx.x * 8 + warp;
    if (vector >= vectors) return;
    const int column = vector / heads;
    float* v         = stage[warp];
    for (int d = threadIdx.x & 31; d < di; d += 32) v[d] = __bfloat162float(q[vector * di + d]);
    __syncwarp();
    norm_rope_vector(v, di, weight, eps, token_axes(positions, axes, vectors / heads, column),
                     rotary_dim, theta, out + vector * di);
}

__device__ __forceinline__ std::int64_t index_slot(const std::int32_t* block_tables,
                                                   int logical_pages, int row, std::int64_t pos) {
    const std::int32_t page = block_tables[static_cast<std::int64_t>(row) * logical_pages + pos / kPage];
    return static_cast<std::int64_t>(page) * kPage + pos % kPage;
}

__global__ void index_append_kernel(const __nv_bfloat16* __restrict__ keys,
                                    const std::int32_t* __restrict__ positions,
                                    const std::int32_t* __restrict__ rows,
                                    const std::int32_t* __restrict__ block_tables,
                                    int logical_pages, int di, int width,
                                    __nv_bfloat16* __restrict__ pages,
                                    const std::int32_t* __restrict__ rope, int axes, int tokens,
                                    std::int32_t* __restrict__ position_pages) {
    const int column = blockIdx.x;
    const int row    = rows[column / width];
    const std::int64_t slot =
        index_slot(block_tables, logical_pages, row, positions[column]);
    for (int d = threadIdx.x; d < di; d += blockDim.x) {
        pages[slot * di + d] = keys[static_cast<std::int64_t>(column) * di + d];
    }
    if (position_pages != nullptr && threadIdx.x < kPositionSlots) {
        const AxisPositions p = token_axes(rope, axes, tokens, column);
        position_pages[slot * kPositionSlots + threadIdx.x] =
            threadIdx.x < 3 ? static_cast<std::int32_t>(p.axis[threadIdx.x]) : 0;
    }
}

// One warp per (pooled block, sequence). pooled: BF16 [di, nb_cap, B].
__global__ void pool_kernel(const __nv_bfloat16* __restrict__ pages,
                            const std::int32_t* __restrict__ positions,
                            const std::int32_t* __restrict__ rows,
                            const std::int32_t* __restrict__ block_tables, int logical_pages,
                            const __nv_bfloat16* __restrict__ key_norm, int di, int ratio,
                            int width, int nb_cap, int rotary_dim, float theta, float eps,
                            const std::int32_t* __restrict__ position_pages,
                            __nv_bfloat16* __restrict__ pooled) {
    __shared__ float stage[8][kMaxIndexD];
    const int warp  = threadIdx.x >> 5;
    const int block = blockIdx.x * 8 + warp;
    const int seq   = blockIdx.y;
    if (block >= nb_cap) return;
    const std::int64_t last = positions[seq * width + width - 1]; // columns ascend per sequence
    if (block >= (last + 1) / ratio) return;
    const int row = rows[seq];
    float* v      = stage[warp];
    for (int d = threadIdx.x & 31; d < di; d += 32) {
        float sum = 0.0f;
        for (int j = 0; j < ratio; ++j) {
            const std::int64_t slot =
                index_slot(block_tables, logical_pages, row, static_cast<std::int64_t>(block) * ratio + j);
            sum += __bfloat162float(pages[slot * di + d]);
        }
        v[d] = bf(sum / ratio);
    }
    __syncwarp();
    // A pooled block rotates with the RoPE position of its first token.
    AxisPositions start = uniform_axes(static_cast<std::int64_t>(block) * ratio);
    if (position_pages != nullptr) {
        const std::int64_t slot = index_slot(block_tables, logical_pages, row,
                                             static_cast<std::int64_t>(block) * ratio);
        start = {{position_pages[slot * kPositionSlots], position_pages[slot * kPositionSlots + 1],
                  position_pages[slot * kPositionSlots + 2]}};
    }
    norm_rope_vector(v, di, key_norm, eps, start, rotary_dim, theta,
                     pooled + (static_cast<std::int64_t>(seq) * nb_cap + block) * di);
}

// scores: FP32 [nb_cap, W, B]; -1 marks blocks a column cannot see.
__global__ void score_kernel(const __nv_bfloat16* __restrict__ query,
                             const __nv_bfloat16* __restrict__ pooled,
                             const std::int32_t* __restrict__ positions, int di, int heads,
                             int ratio, int width, int nb_cap, float inv_sqrt_di,
                             float* __restrict__ scores) {
    extern __shared__ float shared[];
    float* keys    = shared;                                 // [kScoreBlocks][di]
    float* queries = shared + kScoreBlocks * di;             // [kScoreColumns][heads][di]
    const int seq        = blockIdx.z;
    const int block0     = blockIdx.x * kScoreBlocks;
    const int column0    = blockIdx.y * kScoreColumns;
    const int columns    = min(kScoreColumns, width - column0);
    for (int i = threadIdx.x; i < kScoreBlocks * di; i += blockDim.x) {
        const int b = block0 + i / di;
        keys[i]     = b < nb_cap ? __bfloat162float(pooled[(static_cast<std::int64_t>(seq) * nb_cap + b) * di + i % di]) : 0.0f;
    }
    for (int i = threadIdx.x; i < columns * heads * di; i += blockDim.x) {
        const int c = column0 + i / (heads * di);
        queries[i]  = __bfloat162float(query[(static_cast<std::int64_t>(seq) * width + c) * heads * di + i % (heads * di)]);
    }
    __syncthreads();
    for (int pair = threadIdx.x; pair < kScoreBlocks * columns; pair += blockDim.x) {
        const int b      = pair % kScoreBlocks;
        const int c      = pair / kScoreBlocks;
        const int block  = block0 + b;
        const int column = seq * width + column0 + c;
        if (block >= nb_cap) continue;
        float score = -1.0f;
        if (block < (positions[column] + 1) / ratio) {
            score = 0.0f;
            for (int h = 0; h < heads; ++h) {
                const float* qv = queries + (c * heads + h) * di;
                const float* kv = keys + b * di;
                float dot       = 0.0f;
                for (int d = 0; d < di; ++d) dot += qv[d] * kv[d];
                score += fmaxf(dot, 0.0f);
            }
            score *= inv_sqrt_di;
        }
        scores[static_cast<std::int64_t>(column) * nb_cap + block] = score;
    }
}

__device__ int block_exclusive_scan(int value, int* scratch) {
    // kSelectThreads-wide inclusive scan via warp scans, returned exclusive.
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int inclusive = value;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const int n = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += n;
    }
    __syncthreads();
    if (lane == 31) scratch[warp] = inclusive;
    __syncthreads();
    if (warp == 0) {
        int w = lane < kSelectThreads / 32 ? scratch[lane] : 0;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const int n = __shfl_up_sync(0xffffffffu, w, offset);
            if (lane >= offset) w += n;
        }
        if (lane < kSelectThreads / 32) scratch[lane] = w;
    }
    __syncthreads();
    const int base = warp == 0 ? 0 : scratch[warp - 1];
    const int result = base + inclusive - value;
    __syncthreads();
    return result;
}

// One CTA per column.
__global__ void __launch_bounds__(kSelectThreads)
    select_kernel(const float* __restrict__ scores, const std::int32_t* __restrict__ positions,
                  int ratio, int budget, int nb_cap, int max_selected,
                  std::int32_t* __restrict__ selected, std::int32_t* __restrict__ counts) {
    __shared__ unsigned histogram[256];
    __shared__ int scratch[32];
    __shared__ unsigned prefix_shared, need_shared;
    const int column        = blockIdx.x;
    const std::int64_t pos  = positions[column];
    const int visible       = static_cast<int>(pos + 1);
    const int nb            = visible / ratio;
    std::int32_t* out       = selected + static_cast<std::int64_t>(column) * max_selected;
    if (nb <= budget) {
        for (int t = threadIdx.x; t < visible; t += blockDim.x) out[t] = t;
        if (threadIdx.x == 0) counts[column] = visible;
        return;
    }
    const float* row = scores + static_cast<std::int64_t>(column) * nb_cap;
    // Radix select of the budget-th largest score (non-negative FP32 ordered as uint32).
    unsigned prefix = 0, mask = 0, need = budget;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) histogram[i] = 0;
        __syncthreads();
        for (int b = threadIdx.x; b < nb; b += blockDim.x) {
            const unsigned bits = __float_as_uint(row[b]);
            if ((bits & mask) == prefix) atomicAdd(&histogram[(bits >> shift) & 255], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned remaining = need;
            int digit          = 255;
            for (; digit > 0; --digit) {
                if (histogram[digit] >= remaining) break;
                remaining -= histogram[digit];
            }
            prefix_shared = prefix | (static_cast<unsigned>(digit) << shift);
            need_shared   = remaining;
        }
        __syncthreads();
        prefix = prefix_shared;
        need   = need_shared;
        mask |= 255u << shift;
        __syncthreads();
    }
    const unsigned threshold = prefix; // exact bits of the budget-th largest score
    // `need` ties at the threshold are taken in ascending block order.
    const int per = (nb + blockDim.x - 1) / blockDim.x;
    const int begin = min(nb, static_cast<int>(threadIdx.x) * per), end = min(nb, begin + per);
    int greater = 0, equal = 0;
    for (int b = begin; b < end; ++b) {
        const unsigned bits = __float_as_uint(row[b]);
        greater += bits > threshold;
        equal += bits == threshold;
    }
    const int greater_before = block_exclusive_scan(greater, scratch);
    const int equal_before   = block_exclusive_scan(equal, scratch);
    int ties_left            = max(0, static_cast<int>(need) - equal_before);
    int ties_taken_before    = min(static_cast<int>(need), equal_before);
    int slot                 = (greater_before + ties_taken_before) * ratio;
    for (int b = begin; b < end; ++b) {
        const unsigned bits = __float_as_uint(row[b]);
        bool take           = bits > threshold;
        if (!take && bits == threshold && ties_left > 0) {
            take = true;
            --ties_left;
        }
        if (take) {
            for (int j = 0; j < ratio; ++j) out[slot + j] = b * ratio + j;
            slot += ratio;
        }
    }
    const int tail_begin = nb * ratio;
    for (int t = tail_begin + threadIdx.x; t < visible; t += blockDim.x) {
        out[budget * ratio + (t - tail_begin)] = t;
    }
    if (threadIdx.x == 0) counts[column] = budget * ratio + (visible - tail_begin);
}

// Tensor-core sparse attention: one CTA per (column, KV head). The group's query heads form
// a 16-row operand (zero-padded); each 32-token tile is gathered into shared memory (FP8 decoded
// on load), scored with BF16 WMMA, softmaxed online in FP32 and accumulated with FP16 WMMA.
constexpr int kRowsPad = 16;
constexpr int kTileLd  = kAttnHeadDim + 8;

struct SparseShared {
    __nv_bfloat16 q[kRowsPad][kTileLd];
    __nv_bfloat16 k[kAttnTile][kTileLd];
    __half v[kAttnTile][kTileLd];
    float scores[kRowsPad][kAttnTile];
    __half p[kRowsPad][kAttnTile + 8];
    float pv[kRowsPad][kAttnHeadDim + 4];
    float o[kRowsPad][kAttnHeadDim];
    float alpha[kRowsPad], running_max[kRowsPad], running_sum[kRowsPad];
    std::int64_t slots[kAttnTile];
};

template <bool Fp8>
__global__ void __launch_bounds__(256)
    sparse_attention_kernel(const __nv_bfloat16* __restrict__ q,
                            const std::int32_t* __restrict__ selected,
                            const std::int32_t* __restrict__ counts,
                            const std::int32_t* __restrict__ rows,
                            const std::int32_t* __restrict__ block_tables, int logical_pages,
                            const void* __restrict__ k_pages, const void* __restrict__ v_pages,
                            const __half* __restrict__ k_scales,
                            const __half* __restrict__ v_scales, int kv_heads, int q_heads,
                            int width, int max_selected, float scale,
                            __nv_bfloat16* __restrict__ out) {
    using namespace nvcuda;
    constexpr int D = kAttnHeadDim;
    extern __shared__ __align__(32) unsigned char raw_shared[];
    auto& sh         = *reinterpret_cast<SparseShared*>(raw_shared);
    const int column = blockIdx.x;
    const int kvh    = blockIdx.y;
    const int group  = q_heads / kv_heads;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row    = rows[column / width];
    const int count  = counts[column];
    const std::int32_t* sel = selected + static_cast<std::int64_t>(column) * max_selected;

    // Query rows (FP8 K is Hadamard-prepared, so the same transform applies to q).
    for (int g = warp; g < kRowsPad; g += 8) {
        float values[8];
        for (int r = 0; r < 8; ++r) {
            values[r] = g < group ? __bfloat162float(q[(static_cast<std::int64_t>(column) * q_heads + kvh * group + g) * D + lane + 32 * r]) : 0.0f;
        }
        if constexpr (Fp8) {
            if (g < group) normalized_hadamard_d256_inplace(values, lane);
        }
        for (int r = 0; r < 8; ++r) sh.q[g][lane + 32 * r] = __float2bfloat16_rn(values[r]);
    }
    for (int i = threadIdx.x; i < kRowsPad * D; i += blockDim.x) sh.o[i / D][i % D] = 0.0f;
    if (threadIdx.x < kRowsPad) {
        sh.running_max[threadIdx.x] = -INFINITY;
        sh.running_sum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    for (int tile = 0; tile < count; tile += kAttnTile) {
        const int tokens = min(kAttnTile, count - tile);
        if (threadIdx.x < kAttnTile) {
            std::int64_t slot = -1;
            if (static_cast<int>(threadIdx.x) < tokens) {
                const std::int64_t pos = sel[tile + threadIdx.x];
                const std::int32_t page =
                    block_tables[static_cast<std::int64_t>(row) * logical_pages + pos / kPage];
                slot = (static_cast<std::int64_t>(page) * kv_heads + kvh) * kPage + pos % kPage;
            }
            sh.slots[threadIdx.x] = slot;
        }
        __syncthreads();
        // Gather K (BF16) and V (FP16) rows: 8 values per task.
        for (int task = threadIdx.x; task < kAttnTile * (D / 8); task += blockDim.x) {
            const int t = task / (D / 8), c = (task % (D / 8)) * 8;
            const std::int64_t slot = sh.slots[t];
            if (slot < 0) {
                *reinterpret_cast<uint4*>(&sh.k[t][c]) = make_uint4(0, 0, 0, 0);
                *reinterpret_cast<uint4*>(&sh.v[t][c]) = make_uint4(0, 0, 0, 0);
                continue;
            }
            if constexpr (Fp8) {
                const auto* kc   = static_cast<const __nv_fp8_e4m3*>(k_pages) + slot * D + c;
                const auto* vc   = static_cast<const __nv_fp8_e4m3*>(v_pages) + slot * D + c;
                const float ks   = __half2float(k_scales[slot]);
                const float vs   = __half2float(v_scales[slot]);
                for (int j = 0; j < 8; ++j) {
                    sh.k[t][c + j] = __float2bfloat16_rn(static_cast<float>(kc[j]) * ks);
                    sh.v[t][c + j] = __float2half_rn(static_cast<float>(vc[j]) * vs);
                }
            } else {
                *reinterpret_cast<uint4*>(&sh.k[t][c]) = *reinterpret_cast<const uint4*>(
                    static_cast<const __nv_bfloat16*>(k_pages) + slot * D + c);
                *reinterpret_cast<uint4*>(&sh.v[t][c]) = *reinterpret_cast<const uint4*>(
                    static_cast<const __half*>(v_pages) + slot * D + c);
            }
        }
        __syncthreads();
        // Scores [16 x 32]: warps 0 and 1 each own one 16-token half.
        if (warp < 2) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < D; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
                wmma::load_matrix_sync(a, &sh.q[0][kk], kTileLd);
                wmma::load_matrix_sync(b, &sh.k[warp * 16][kk], kTileLd);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(&sh.scores[0][warp * 16], acc, kAttnTile, wmma::mem_row_major);
        }
        __syncthreads();
        if (threadIdx.x < kRowsPad) {
            const int g = threadIdx.x;
            float m     = sh.running_max[g];
            for (int t = 0; t < tokens; ++t) m = fmaxf(m, sh.scores[g][t] * scale);
            const float a = g < group ? __expf(sh.running_max[g] - m) : 0.0f;
            float sum     = sh.running_sum[g] * a;
            for (int t = 0; t < kAttnTile; ++t) {
                const float pv = (t < tokens && g < group) ? __expf(sh.scores[g][t] * scale - m) : 0.0f;
                sh.p[g][t]     = __float2half_rn(pv);
                sum += pv;
            }
            sh.alpha[g]       = a;
            sh.running_max[g] = m;
            sh.running_sum[g] = sum;
        }
        __syncthreads();
        // P [16 x 32] x V [32 x 256]: each warp owns two 16-column output tiles.
        for (int ct = warp * 2; ct < warp * 2 + 2; ++ct) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            for (int kk = 0; kk < kAttnTile; kk += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                wmma::load_matrix_sync(a, &sh.p[0][kk], kAttnTile + 8);
                wmma::load_matrix_sync(b, &sh.v[kk][ct * 16], kTileLd);
                wmma::mma_sync(acc, a, b, acc);
            }
            wmma::store_matrix_sync(&sh.pv[0][ct * 16], acc, D + 4, wmma::mem_row_major);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < group * D; i += blockDim.x) {
            const int g = i / D, d = i % D;
            sh.o[g][d] = sh.o[g][d] * sh.alpha[g] + sh.pv[g][d];
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < group * D; i += blockDim.x) {
        const int g = i / D, d = i % D;
        out[(static_cast<std::int64_t>(column) * q_heads + kvh * group + g) * D + d] =
            __float2bfloat16_rn(sh.o[g][d] / sh.running_sum[g]);
    }
}

struct SelectLayout {
    std::size_t pooled, scores, total;
};

SelectLayout select_layout(std::int32_t di, std::int32_t ratio, std::int32_t sequences,
                           std::int32_t columns, std::int32_t max_visible) {
    const std::int64_t nb_cap = std::max<std::int32_t>(1, max_visible / ratio);
    SelectLayout l{};
    const auto align = [](std::size_t n) { return (n + 255) / 256 * 256; };
    l.pooled = 0;
    l.scores = align(static_cast<std::size_t>(nb_cap) * di * sequences * 2);
    l.total  = l.scores + align(static_cast<std::size_t>(nb_cap) * columns * 4);
    return l;
}

} // namespace

void qsa_prepare_query_launch(const Tensor& q, const Tensor& weight, const Tensor& positions,
                              std::int32_t rotary_dim, float theta, float eps, Tensor& out,
                              cudaStream_t stream) {
    const int di = q.ne[0], heads = q.ne[1];
    const int vectors = static_cast<int>(q.numel() / di);
    const int axes    = positions.numel() == vectors / heads ? 1 : 3;
    prepare_query_kernel<<<(vectors + 7) / 8, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(weight.data),
        static_cast<const std::int32_t*>(positions.data), axes, di, heads, vectors, rotary_dim,
        theta, eps, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void qsa_index_append_launch(const Tensor& keys, const Tensor& positions, const Tensor& rows,
                             const Tensor& index_pages, const Tensor& block_tables,
                             const Tensor& rope_positions, const Tensor& position_pages,
                             cudaStream_t stream) {
    const int di = keys.ne[0], width = keys.ne[1], sequences = keys.ne[2];
    const int tokens = width * sequences;
    const int axes   = rope_positions.data == nullptr || rope_positions.numel() == tokens ? 1 : 3;
    index_append_kernel<<<tokens, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(keys.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(rows.data),
        static_cast<const std::int32_t*>(block_tables.data), block_tables.ne[0], di, width,
        static_cast<__nv_bfloat16*>(index_pages.data),
        static_cast<const std::int32_t*>(rope_positions.data), axes, tokens,
        static_cast<std::int32_t*>(position_pages.data));
    CUDA_CHECK(cudaGetLastError());
}

std::size_t qsa_select_workspace_launch_bytes(std::int32_t ratio, std::int32_t index_dim,
                                              std::int32_t sequences, std::int32_t columns,
                                              std::int32_t max_visible) {
    return select_layout(index_dim, ratio, sequences, columns, max_visible).total;
}

void qsa_select_launch(const Tensor& query, const Tensor& positions, const Tensor& rows,
                       const Tensor& index_pages, const Tensor& position_pages,
                       const Tensor& block_tables,
                       const Tensor& key_norm, std::int32_t ratio, std::int32_t budget,
                       std::int32_t rotary_dim, float theta, float eps, std::int32_t max_visible,
                       void* workspace, Tensor& selected, Tensor& counts, cudaStream_t stream) {
    const int di = query.ne[0], heads = query.ne[1], width = query.ne[2], sequences = query.ne[3];
    const int nb_cap        = std::max(1, max_visible / ratio);
    const SelectLayout l    = select_layout(di, ratio, sequences, width * sequences, max_visible);
    auto* base              = static_cast<std::byte*>(workspace);
    auto* pooled            = reinterpret_cast<__nv_bfloat16*>(base + l.pooled);
    auto* scores            = reinterpret_cast<float*>(base + l.scores);
    const auto* pos         = static_cast<const std::int32_t*>(positions.data);
    const auto* bt          = static_cast<const std::int32_t*>(block_tables.data);
    pool_kernel<<<dim3((nb_cap + 7) / 8, sequences), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(index_pages.data), pos,
        static_cast<const std::int32_t*>(rows.data), bt, block_tables.ne[0],
        static_cast<const __nv_bfloat16*>(key_norm.data), di, ratio, width, nb_cap, rotary_dim,
        theta, eps, static_cast<const std::int32_t*>(position_pages.data), pooled);
    CUDA_CHECK(cudaGetLastError());
    const std::size_t shared =
        static_cast<std::size_t>(kScoreBlocks + kScoreColumns * heads) * di * sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(score_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    static_cast<int>(shared)));
    score_kernel<<<dim3((nb_cap + kScoreBlocks - 1) / kScoreBlocks,
                        (width + kScoreColumns - 1) / kScoreColumns, sequences),
                   256, shared, stream>>>(static_cast<const __nv_bfloat16*>(query.data), pooled,
                                          pos, di, heads, ratio, width, nb_cap,
                                          1.0f / std::sqrt(static_cast<float>(di)), scores);
    CUDA_CHECK(cudaGetLastError());
    select_kernel<<<width * sequences, kSelectThreads, 0, stream>>>(
        scores, pos, ratio, budget, nb_cap, selected.ne[0],
        static_cast<std::int32_t*>(selected.data), static_cast<std::int32_t*>(counts.data));
    CUDA_CHECK(cudaGetLastError());
}

void qsa_sparse_attention_launch(const Tensor& q, const Tensor& selected, const Tensor& counts,
                                 const Tensor& rows, const PagedKVBatchLayerView& cache,
                                 float scale, Tensor& out, cudaStream_t stream) {
    const int q_heads = q.ne[1], width = q.ne[2], sequences = q.ne[3];
    const dim3 grid(width * sequences, cache.num_kv_heads);
    const auto* bt = static_cast<const std::int32_t*>(cache.block_tables.data);
    const int logical_pages = cache.block_tables.ne[0];
    const std::size_t shared = sizeof(SparseShared);
    static const bool configured = [shared] {
        CUDA_CHECK(cudaFuncSetAttribute(sparse_attention_kernel<true>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
        CUDA_CHECK(cudaFuncSetAttribute(sparse_attention_kernel<false>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
        return true;
    }();
    (void)configured;
    if (cache.storage == KvCacheStorage::Fp8E4M3Row256) {
        sparse_attention_kernel<true><<<grid, 256, shared, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int32_t*>(selected.data),
            static_cast<const std::int32_t*>(counts.data),
            static_cast<const std::int32_t*>(rows.data), bt, logical_pages, cache.k_pages.data,
            cache.v_pages.data, static_cast<const __half*>(cache.k_scale_pages.data),
            static_cast<const __half*>(cache.v_scale_pages.data), cache.num_kv_heads, q_heads,
            width, selected.ne[0], scale, static_cast<__nv_bfloat16*>(out.data));
    } else {
        sparse_attention_kernel<false><<<grid, 256, shared, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int32_t*>(selected.data),
            static_cast<const std::int32_t*>(counts.data),
            static_cast<const std::int32_t*>(rows.data), bt, logical_pages, cache.k_pages.data,
            cache.v_pages.data, nullptr, nullptr, cache.num_kv_heads, q_heads, width,
            selected.ne[0], scale, static_cast<__nv_bfloat16*>(out.data));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
