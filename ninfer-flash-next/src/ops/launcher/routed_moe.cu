// Implements: include/ninfer/ops/routed_moe.h
//
// Phase-1 route for Qwen4Exp's 512-expert NVFP4 MoE. Pairs (token, choice) are grouped by expert
// with a counting sort; each CTA of the projection kernels owns a row tile of one active expert
// and walks that expert's tokens, so an expert's weights are fetched once from DRAM per call and
// re-read from L2 for its other tokens. Per-pair outputs are combined in a fixed order, so results
// do not depend on scheduling.
#include "ops/launcher/routed_moe.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRouteWarps      = 4;    // tokens per routing CTA
constexpr int kMaxExperts      = 1024; // scan CTA width
constexpr int kProjWarps       = 8;
constexpr int kGateRowsPerWarp = 4;    // gate rows; each also computes the matching up row
constexpr int kGateRowsPerCta  = kProjWarps * kGateRowsPerWarp; // 32
constexpr int kDownRowsPerWarp = 8;
constexpr int kDownRowsPerCta  = kProjWarps * kDownRowsPerWarp; // 64
// Below this many tokens per call the memory-bound CUDA-core route is faster.
constexpr int kMmaMinTokens    = 32;

// E2M1 values. Lanes index it with divergent codes, so each CTA copies it to shared memory
// (constant memory would serialize divergent reads).
__device__ __constant__ float kE2m1[16] = {0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                                           -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ void load_e2m1(float* table) {
    if (threadIdx.x < 16) { table[threadIdx.x] = kE2m1[threadIdx.x]; }
}

std::size_t align_up(std::size_t n, std::size_t a) { return (n + a - 1) / a * a; }

struct Layout {
    std::size_t ids, weights, shared_gate, counts, offsets, cursors, active, active_count, tiles,
        tile_count, pairs,
        activation, partial, total;
};

Layout layout(std::int32_t experts, std::int32_t top_k, std::int32_t hidden,
              std::int32_t intermediate, std::int32_t tokens) {
    const std::size_t pairs = static_cast<std::size_t>(top_k) * tokens;
    Layout l{};
    std::size_t at = 0;
    auto take      = [&](std::size_t bytes) {
        const std::size_t begin = at;
        at                      = align_up(at + bytes, 256);
        return begin;
    };
    l.ids          = take(pairs * 4);
    l.weights      = take(pairs * 4);
    l.shared_gate  = take(static_cast<std::size_t>(tokens) * 4);
    l.counts       = take(static_cast<std::size_t>(experts) * 4);
    l.offsets      = take((static_cast<std::size_t>(experts) + 1) * 4);
    l.cursors      = take(static_cast<std::size_t>(experts) * 4);
    l.active       = take(static_cast<std::size_t>(experts) * 4);
    l.active_count = take(4);
    l.tiles        = take((pairs / 64 + static_cast<std::size_t>(experts) + 1) * 8);
    l.tile_count   = take(4);
    l.pairs        = take(pairs * 4);
    l.activation   = take(pairs * intermediate * 2);
    l.partial      = take(pairs * hidden * 4);
    l.total        = at;
    return l;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

__device__ __forceinline__ float fp8_scale(std::uint8_t word) {
    __nv_fp8_e4m3 value;
    value.__x = word;
    return static_cast<float>(value);
}

// Byte offset of the E4M3 scale for (row, group) in a BlockScaleK16M128x4 slice.
__device__ __forceinline__ std::uint64_t scale_index(std::int32_t row, std::int32_t group,
                                                     std::int32_t k) {
    const std::int32_t inner = row & 127;
    return (static_cast<std::uint64_t>(row >> 7) * (k >> 6) + (group >> 2)) * 512 +
           (inner & 31) * 16 + (inner >> 5) * 4 + (group & 3);
}

// Dot product of one NVFP4 row with a BF16 vector held in shared memory; the result excludes
// the expert's weight divisor. Lanes stride over 16-element scale groups.
__device__ __forceinline__ float nvfp4_row_dot(const std::uint8_t* codes, const std::uint8_t* scales,
                                               std::int32_t row, std::int32_t k,
                                               const __nv_bfloat16* x, const float* e2m1) {
    const std::int32_t groups = k >> 4;
    const std::uint8_t* row_codes = codes + static_cast<std::uint64_t>(row) * (k >> 1);
    float acc                 = 0.0f;
    for (std::int32_t g = threadIdx.x & 31; g < groups; g += 32) {
        const uint2 packed  = *reinterpret_cast<const uint2*>(row_codes + g * 8);
        const float scale   = fp8_scale(scales[scale_index(row, g, k)]);
        const __nv_bfloat162* xv = reinterpret_cast<const __nv_bfloat162*>(x + g * 16);
        float part          = 0.0f;
#pragma unroll
        for (int word = 0; word < 2; ++word) {
            const std::uint32_t bits = word == 0 ? packed.x : packed.y;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const std::uint32_t byte = (bits >> (8 * b)) & 0xFF;
                const float2 xf          = __bfloat1622float2(xv[word * 4 + b]);
                part += e2m1[byte & 15] * xf.x + e2m1[byte >> 4] * xf.y;
            }
        }
        acc += part * scale;
    }
    return warp_reduce_sum(acc);
}

__device__ __forceinline__ float bank_divisor(const std::byte* expert, std::uint64_t offset) {
    return *reinterpret_cast<const float*>(expert + offset);
}

// One warp per token: softmax over E logits, top-K (ties to the lower index), renormalize.
__global__ void route_kernel(const __nv_bfloat16* __restrict__ logits, std::int32_t experts,
                             std::int32_t top_k, std::int32_t tokens, std::int32_t* __restrict__ ids,
                             float* __restrict__ weights, float* __restrict__ shared_gate,
                             std::int32_t* __restrict__ counts) {
    const std::int32_t lane  = threadIdx.x & 31;
    const std::int32_t token = blockIdx.x * kRouteWarps + (threadIdx.x >> 5);
    if (token >= tokens) { return; }
    const __nv_bfloat16* row = logits + static_cast<std::int64_t>(token) * (experts + 1);
    float maximum            = -INFINITY;
    for (std::int32_t e = lane; e < experts; e += 32) {
        maximum = fmaxf(maximum, __bfloat162float(row[e]));
    }
    maximum = warp_max(maximum);
    // Selection on logits equals selection on probabilities; weights only need the selected set.
    std::uint32_t taken[kMaxExperts / 32 / 32 + 1] = {}; // lane-local bitmap over e = lane + 32*j
    float selected_sum                               = 0.0f;
    float my_weight                                  = 0.0f;
    std::int32_t my_id                               = 0;
    for (std::int32_t choice = 0; choice < top_k; ++choice) {
        float best          = -INFINITY;
        std::int32_t best_e = experts;
        for (std::int32_t j = 0, e = lane; e < experts; ++j, e += 32) {
            if ((taken[j >> 5] >> (j & 31)) & 1U) { continue; }
            const float v = __bfloat162float(row[e]);
            if (v > best || (v == best && e < best_e)) {
                best   = v;
                best_e = e;
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_v      = __shfl_xor_sync(0xFFFFFFFFu, best, offset);
            const std::int32_t other = __shfl_xor_sync(0xFFFFFFFFu, best_e, offset);
            if (other_v > best || (other_v == best && other < best_e)) {
                best   = other_v;
                best_e = other;
            }
        }
        if ((best_e & 31) == lane) {
            const std::int32_t j = best_e >> 5;
            taken[j >> 5] |= 1U << (j & 31);
        }
        const float p = __expf(best - maximum);
        selected_sum += p;
        if (lane == choice) {
            my_weight = p;
            my_id     = best_e;
        }
    }
    if (lane < top_k) {
        const std::int64_t at = static_cast<std::int64_t>(token) * top_k + lane;
        ids[at]               = my_id;
        weights[at]           = my_weight / selected_sum;
        atomicAdd(counts + my_id, 1);
    }
    if (lane == 0) { shared_gate[token] = sigmoidf_(__bfloat162float(row[experts])); }
}

// Single CTA: exclusive scan of counts, cursor initialization and ordered active-expert list.
__global__ void scan_kernel(const std::int32_t* __restrict__ counts, std::int32_t experts,
                            std::int32_t* __restrict__ offsets, std::int32_t* __restrict__ cursors,
                            std::int32_t* __restrict__ active,
                            std::int32_t* __restrict__ active_count) {
    __shared__ std::int32_t sums[kMaxExperts];
    __shared__ std::int32_t flags[kMaxExperts];
    const std::int32_t e = threadIdx.x;
    const std::int32_t c = e < experts ? counts[e] : 0;
    sums[e]              = c;
    flags[e]             = c > 0 ? 1 : 0;
    __syncthreads();
    for (std::int32_t stride = 1; stride < kMaxExperts; stride <<= 1) {
        const std::int32_t add_sum  = e >= stride ? sums[e - stride] : 0;
        const std::int32_t add_flag = e >= stride ? flags[e - stride] : 0;
        __syncthreads();
        sums[e] += add_sum;
        flags[e] += add_flag;
        __syncthreads();
    }
    if (e < experts) {
        const std::int32_t begin = sums[e] - c;
        offsets[e]               = begin;
        cursors[e]               = begin;
        if (c > 0) { active[flags[e] - 1] = e; }
    }
    if (e == experts - 1) {
        offsets[experts] = sums[e];
        *active_count    = flags[e];
    }
}

__global__ void scatter_kernel(const std::int32_t* __restrict__ ids, std::int32_t pairs,
                               std::int32_t* __restrict__ cursors,
                               std::int32_t* __restrict__ sorted_pairs) {
    const std::int32_t p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < pairs) { sorted_pairs[atomicAdd(cursors + ids[p], 1)] = p; }
}

struct BankArgs {
    const std::byte* data;
    std::uint64_t stride;
    std::uint64_t scale_offset;
    std::uint64_t divisor_offset;
};

// grid.x: 32-row tiles of I; grid.y: active-expert slot. Writes SiLU(gate)*up per pair.
__global__ void __launch_bounds__(kProjWarps * 32)
    gate_up_kernel(const __nv_bfloat16* __restrict__ x, BankArgs gate, BankArgs up,
                   const std::int32_t* __restrict__ active,
                   const std::int32_t* __restrict__ active_count,
                   const std::int32_t* __restrict__ offsets,
                   const std::int32_t* __restrict__ sorted_pairs, std::int32_t top_k,
                   std::int32_t hidden, std::int32_t intermediate,
                   __nv_bfloat16* __restrict__ activation) {
    extern __shared__ __align__(16) unsigned char smem[];
    auto* xs = reinterpret_cast<__nv_bfloat16*>(smem);
    __shared__ float e2m1[16];
    if (static_cast<std::int32_t>(blockIdx.y) >= *active_count) { return; }
    load_e2m1(e2m1);
    const std::int32_t e         = active[blockIdx.y];
    const std::byte* gate_expert = gate.data + static_cast<std::uint64_t>(e) * gate.stride;
    const std::byte* up_expert   = up.data + static_cast<std::uint64_t>(e) * up.stride;
    const auto* gate_codes       = reinterpret_cast<const std::uint8_t*>(gate_expert);
    const auto* up_codes         = reinterpret_cast<const std::uint8_t*>(up_expert);
    const auto* gate_scales = reinterpret_cast<const std::uint8_t*>(gate_expert + gate.scale_offset);
    const auto* up_scales   = reinterpret_cast<const std::uint8_t*>(up_expert + up.scale_offset);
    const float gate_div    = bank_divisor(gate_expert, gate.divisor_offset);
    const float up_div      = bank_divisor(up_expert, up.divisor_offset);
    const std::int32_t warp = threadIdx.x >> 5;
    const std::int32_t row0 = blockIdx.x * kGateRowsPerCta + warp * kGateRowsPerWarp;

    for (std::int32_t i = offsets[e]; i < offsets[e + 1]; ++i) {
        const std::int32_t pair  = sorted_pairs[i];
        const std::int32_t token = pair / top_k;
        __syncthreads();
        const auto* src = reinterpret_cast<const uint4*>(x + static_cast<std::int64_t>(token) * hidden);
        for (std::int32_t v = threadIdx.x; v < hidden / 8; v += blockDim.x) {
            reinterpret_cast<uint4*>(xs)[v] = src[v];
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kGateRowsPerWarp; ++r) {
            const std::int32_t row = row0 + r;
            if (row >= intermediate) { break; }
            const float g =
                nvfp4_row_dot(gate_codes, gate_scales, row, hidden, xs, e2m1) / gate_div;
            const float u = nvfp4_row_dot(up_codes, up_scales, row, hidden, xs, e2m1) / up_div;
            if ((threadIdx.x & 31) == 0) {
                activation[static_cast<std::int64_t>(pair) * intermediate + row] =
                    __float2bfloat16_rn(g * sigmoidf_(g) * u);
            }
        }
    }
}

// grid.x: 64-row tiles of H; grid.y: active-expert slot. Writes FP32 per-pair down outputs.
__global__ void __launch_bounds__(kProjWarps * 32)
    down_kernel(BankArgs down, const std::int32_t* __restrict__ active,
                const std::int32_t* __restrict__ active_count,
                const std::int32_t* __restrict__ offsets,
                const std::int32_t* __restrict__ sorted_pairs, std::int32_t hidden,
                std::int32_t intermediate, const __nv_bfloat16* __restrict__ activation,
                float* __restrict__ partial) {
    extern __shared__ __align__(16) unsigned char smem[];
    auto* as = reinterpret_cast<__nv_bfloat16*>(smem);
    __shared__ float e2m1[16];
    if (static_cast<std::int32_t>(blockIdx.y) >= *active_count) { return; }
    load_e2m1(e2m1);
    const std::int32_t e    = active[blockIdx.y];
    const std::byte* expert = down.data + static_cast<std::uint64_t>(e) * down.stride;
    const auto* codes       = reinterpret_cast<const std::uint8_t*>(expert);
    const auto* scales      = reinterpret_cast<const std::uint8_t*>(expert + down.scale_offset);
    const float divisor     = bank_divisor(expert, down.divisor_offset);
    const std::int32_t warp = threadIdx.x >> 5;
    const std::int32_t row0 = blockIdx.x * kDownRowsPerCta + warp * kDownRowsPerWarp;

    for (std::int32_t i = offsets[e]; i < offsets[e + 1]; ++i) {
        const std::int32_t pair = sorted_pairs[i];
        __syncthreads();
        const auto* src =
            reinterpret_cast<const uint4*>(activation + static_cast<std::int64_t>(pair) * intermediate);
        for (std::int32_t v = threadIdx.x; v < intermediate / 8; v += blockDim.x) {
            reinterpret_cast<uint4*>(as)[v] = src[v];
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < kDownRowsPerWarp; ++r) {
            const std::int32_t row = row0 + r;
            if (row >= hidden) { break; }
            const float y = nvfp4_row_dot(codes, scales, row, intermediate, as, e2m1) / divisor;
            if ((threadIdx.x & 31) == 0) {
                partial[static_cast<std::int64_t>(pair) * hidden + row] = y;
            }
        }
    }
}

// ---- Tensor-core route for larger T -------------------------------------------------------
// A work item is (expert, first routed pair of a 64-pair tile). Weights decode to BF16 as
// e2m1 * E4M3 block scale, which is exact; the expert divisor is applied to FP32 accumulators.
constexpr int kTileRows   = 64;
constexpr int kTilePairs  = 64;
constexpr int kTileK      = 64;
constexpr int kTilePad    = 8;
constexpr int kMmaThreads = 128;

union MmaShared {
    struct {
        __nv_bfloat16 w[kTileRows][kTileK + kTilePad];
        __nv_bfloat16 x[kTilePairs][kTileK + kTilePad];
    } operands;
    float c[kTileRows][kTilePairs + 4];
};

__global__ void tile_list_kernel(const std::int32_t* __restrict__ counts,
                                 const std::int32_t* __restrict__ offsets,
                                 const std::int32_t* __restrict__ active,
                                 const std::int32_t* __restrict__ active_count,
                                 int2* __restrict__ tiles, std::int32_t* __restrict__ tile_count) {
    // One thread per active slot; serial prefix over slots by thread 0 keeps it deterministic.
    if (threadIdx.x != 0) return;
    int n = 0;
    for (int a = 0; a < *active_count; ++a) {
        const int e = active[a];
        for (int t = 0; t < counts[e]; t += kTilePairs) tiles[n++] = make_int2(e, offsets[e] + t);
    }
    *tile_count = n;
}

// pair_table[b] = (e2m1(b & 15), e2m1(b >> 4)) as BF16, filled once per CTA.
__device__ __forceinline__ void load_pair_table(__nv_bfloat162* pair_table) {
    for (int b = threadIdx.x; b < 256; b += blockDim.x) {
        pair_table[b] = __floats2bfloat162_rn(kE2m1[b & 15], kE2m1[b >> 4]);
    }
}

// Decodes tile_rows rows x kTileK columns into shared rows starting at tile_row0. One task is one
// 16-value scale group (8 code bytes); e2m1 * E4M3 scale is exact in BF16.
__device__ __forceinline__ void load_weight_tile(const std::uint8_t* codes, const std::uint8_t* scales,
                                                 int row_base, int rows_valid, int k0, int k,
                                                 const __nv_bfloat162* pair_table,
                                                 __nv_bfloat16 (*tile)[kTileK + kTilePad],
                                                 int tile_row0, int tile_rows) {
    constexpr int kGroups = kTileK / 16;
    for (int task = threadIdx.x; task < tile_rows * kGroups; task += blockDim.x) {
        const int r = task / kGroups, g = task % kGroups;
        uint4 out0 = make_uint4(0, 0, 0, 0), out1 = make_uint4(0, 0, 0, 0);
        if (r < rows_valid) {
            const int row     = row_base + r;
            const uint2 bytes = *reinterpret_cast<const uint2*>(
                codes + static_cast<std::uint64_t>(row) * (k >> 1) + (k0 >> 1) + g * 8);
            const float scale  = fp8_scale(scales[scale_index(row, (k0 >> 4) + g, k)]);
            const __nv_bfloat162 s2 = __floats2bfloat162_rn(scale, scale);
            __nv_bfloat162 v[8];
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                v[b]     = __hmul2(pair_table[(bytes.x >> (8 * b)) & 255], s2);
                v[b + 4] = __hmul2(pair_table[(bytes.y >> (8 * b)) & 255], s2);
            }
            out0 = *reinterpret_cast<const uint4*>(&v[0]);
            out1 = *reinterpret_cast<const uint4*>(&v[4]);
        }
        auto* dst = reinterpret_cast<uint4*>(&tile[tile_row0 + r][g * 16]);
        dst[0]    = out0;
        dst[1]    = out1;
    }
}

// grid.x: row tiles of 32 intermediate rows (gate rows + matching up rows); grid.y: work items.
__global__ void __launch_bounds__(kMmaThreads, 6)
    gate_up_mma_kernel(const __nv_bfloat16* __restrict__ x, BankArgs gate, BankArgs up,
                       const std::int32_t* __restrict__ counts, const std::int32_t* __restrict__ offsets,
                       const int2* __restrict__ tiles, const std::int32_t* __restrict__ tile_count,
                       const std::int32_t* __restrict__ sorted_pairs, std::int32_t top_k,
                       std::int32_t hidden, std::int32_t intermediate,
                       __nv_bfloat16* __restrict__ activation) {
    using namespace nvcuda;
    // The FP32 epilogue tile reuses the operand tiles' storage after the K loop.
    __shared__ __align__(32) MmaShared shared;
    auto& wtile = shared.operands.w;
    auto& xtile = shared.operands.x;
    auto& ctile = shared.c;
    __shared__ __nv_bfloat162 pair_table[256];
    __shared__ int pair_ids[kTilePairs];
    if (static_cast<int>(blockIdx.y) >= *tile_count) return;
    load_pair_table(pair_table);
    const int2 item  = tiles[blockIdx.y];
    const int e      = item.x;
    const int first  = item.y;
    const int pairs  = min(kTilePairs, offsets[e] + counts[e] - first);
    const int row0   = blockIdx.x * (kTileRows / 2);
    if (threadIdx.x < kTilePairs) pair_ids[threadIdx.x] = threadIdx.x < pairs ? sorted_pairs[first + threadIdx.x] : -1;
    const std::byte* gate_expert = gate.data + static_cast<std::uint64_t>(e) * gate.stride;
    const std::byte* up_expert   = up.data + static_cast<std::uint64_t>(e) * up.stride;
    const int warp = threadIdx.x >> 5, wy = warp >> 1, wx = warp & 1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (auto& a : acc) for (auto& f : a) wmma::fill_fragment(f, 0.0f);
    __syncthreads();
    for (int k0 = 0; k0 < hidden; k0 += kTileK) {
        load_weight_tile(reinterpret_cast<const std::uint8_t*>(gate_expert),
                         reinterpret_cast<const std::uint8_t*>(gate_expert + gate.scale_offset), row0,
                         intermediate - row0, k0, hidden, pair_table, wtile, 0, kTileRows / 2);
        load_weight_tile(reinterpret_cast<const std::uint8_t*>(up_expert),
                         reinterpret_cast<const std::uint8_t*>(up_expert + up.scale_offset), row0,
                         intermediate - row0, k0, hidden, pair_table, wtile, kTileRows / 2, kTileRows / 2);
        for (int i = threadIdx.x; i < kTilePairs * (kTileK / 8); i += blockDim.x) {
            const int p = i / (kTileK / 8), c = i % (kTileK / 8);
            uint4 v     = make_uint4(0, 0, 0, 0);
            if (pair_ids[p] >= 0) {
                const std::int64_t token = pair_ids[p] / top_k;
                v = *reinterpret_cast<const uint4*>(x + token * hidden + k0 + c * 8);
            }
            *reinterpret_cast<uint4*>(&xtile[p][c * 8]) = v;
        }
        __syncthreads();
        for (int kk = 0; kk < kTileK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &wtile[wy * 32 + i * 16][kk], kTileK + kTilePad);
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &xtile[wx * 32 + j * 16][kk], kTileK + kTilePad);
            for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
        wmma::store_matrix_sync(&ctile[wy * 32 + i * 16][wx * 32 + j * 16], acc[i][j], kTilePairs + 4, wmma::mem_row_major);
    __syncthreads();
    const float gate_div = bank_divisor(gate_expert, gate.divisor_offset);
    const float up_div   = bank_divisor(up_expert, up.divisor_offset);
    for (int i = threadIdx.x; i < (kTileRows / 2) * kTilePairs; i += blockDim.x) {
        const int r = i % (kTileRows / 2), p = i / (kTileRows / 2);
        if (pair_ids[p] < 0 || row0 + r >= intermediate) continue;
        const float g = ctile[r][p] / gate_div;
        const float u = ctile[kTileRows / 2 + r][p] / up_div;
        activation[static_cast<std::int64_t>(pair_ids[p]) * intermediate + row0 + r] =
            __float2bfloat16_rn(g * sigmoidf_(g) * u);
    }
}

// grid.x: 64-row tiles of H; grid.y: work items.
__global__ void __launch_bounds__(kMmaThreads, 6)
    down_mma_kernel(BankArgs down, const std::int32_t* __restrict__ counts,
                    const std::int32_t* __restrict__ offsets, const int2* __restrict__ tiles,
                    const std::int32_t* __restrict__ tile_count,
                    const std::int32_t* __restrict__ sorted_pairs, std::int32_t hidden,
                    std::int32_t intermediate, const __nv_bfloat16* __restrict__ activation,
                    float* __restrict__ partial) {
    using namespace nvcuda;
    // The FP32 epilogue tile reuses the operand tiles' storage after the K loop.
    __shared__ __align__(32) MmaShared shared;
    auto& wtile = shared.operands.w;
    auto& xtile = shared.operands.x;
    auto& ctile = shared.c;
    __shared__ __nv_bfloat162 pair_table[256];
    __shared__ int pair_ids[kTilePairs];
    if (static_cast<int>(blockIdx.y) >= *tile_count) return;
    load_pair_table(pair_table);
    const int2 item = tiles[blockIdx.y];
    const int e     = item.x;
    const int first = item.y;
    const int pairs = min(kTilePairs, offsets[e] + counts[e] - first);
    const int row0  = blockIdx.x * kTileRows;
    if (threadIdx.x < kTilePairs) pair_ids[threadIdx.x] = threadIdx.x < pairs ? sorted_pairs[first + threadIdx.x] : -1;
    const std::byte* expert = down.data + static_cast<std::uint64_t>(e) * down.stride;
    const int warp = threadIdx.x >> 5, wy = warp >> 1, wx = warp & 1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    for (auto& a : acc) for (auto& f : a) wmma::fill_fragment(f, 0.0f);
    __syncthreads();
    for (int k0 = 0; k0 < intermediate; k0 += kTileK) {
        load_weight_tile(reinterpret_cast<const std::uint8_t*>(expert),
                         reinterpret_cast<const std::uint8_t*>(expert + down.scale_offset), row0,
                         hidden - row0, k0, intermediate, pair_table, wtile, 0, kTileRows);
        for (int i = threadIdx.x; i < kTilePairs * (kTileK / 8); i += blockDim.x) {
            const int p = i / (kTileK / 8), c = i % (kTileK / 8);
            uint4 v     = make_uint4(0, 0, 0, 0);
            if (pair_ids[p] >= 0) {
                v = *reinterpret_cast<const uint4*>(activation + static_cast<std::int64_t>(pair_ids[p]) * intermediate + k0 + c * 8);
            }
            *reinterpret_cast<uint4*>(&xtile[p][c * 8]) = v;
        }
        __syncthreads();
        for (int kk = 0; kk < kTileK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b[2];
            for (int i = 0; i < 2; ++i) wmma::load_matrix_sync(a[i], &wtile[wy * 32 + i * 16][kk], kTileK + kTilePad);
            for (int j = 0; j < 2; ++j) wmma::load_matrix_sync(b[j], &xtile[wx * 32 + j * 16][kk], kTileK + kTilePad);
            for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
        wmma::store_matrix_sync(&ctile[wy * 32 + i * 16][wx * 32 + j * 16], acc[i][j], kTilePairs + 4, wmma::mem_row_major);
    __syncthreads();
    const float divisor = bank_divisor(expert, down.divisor_offset);
    for (int i = threadIdx.x; i < kTileRows * kTilePairs; i += blockDim.x) {
        const int r = i % kTileRows, p = i / kTileRows;
        if (pair_ids[p] < 0 || row0 + r >= hidden) continue;
        partial[static_cast<std::int64_t>(pair_ids[p]) * hidden + row0 + r] = ctile[r][p] / divisor;
    }
}

__global__ void combine_kernel(const float* __restrict__ partial, const float* __restrict__ weights,
                               const float* __restrict__ shared_gate,
                               const __nv_bfloat16* __restrict__ shared, std::int32_t top_k,
                               std::int32_t hidden, std::int64_t n, __nv_bfloat16* __restrict__ out) {
    for (std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t token = i / hidden;
        const std::int64_t h     = i - token * hidden;
        float acc                = shared_gate[token] * __bfloat162float(shared[i]);
        for (std::int32_t k = 0; k < top_k; ++k) {
            const std::int64_t pair = token * top_k + k;
            acc += weights[pair] * partial[pair * hidden + h];
        }
        out[i] = __float2bfloat16_rn(acc);
    }
}

BankArgs bank_args(const Nvfp4Bank& bank) {
    return {bank.data, bank.expert_stride, bank.scale_offset, bank.divisor_offset};
}

} // namespace

std::size_t routed_moe_workspace_launch_bytes(std::int32_t experts, std::int32_t top_k,
                                              std::int32_t hidden, std::int32_t intermediate,
                                              std::int32_t max_tokens) {
    return layout(experts, top_k, hidden, intermediate, max_tokens).total;
}

void routed_moe_launch(const Tensor& x, const Tensor& logits, std::int32_t top_k,
                       const Nvfp4Bank& gate, const Nvfp4Bank& up, const Nvfp4Bank& down,
                       const Tensor& shared, void* workspace, Tensor& out, cudaStream_t stream) {
    const std::int32_t experts      = gate.experts;
    const std::int32_t hidden       = x.ne[0];
    const std::int32_t intermediate = gate.n;
    const std::int32_t tokens       = x.ne[1];
    const std::int32_t pairs        = top_k * tokens;
    if (experts > kMaxExperts) { throw std::invalid_argument("routed_moe: too many experts"); }
    const Layout l = layout(experts, top_k, hidden, intermediate, tokens);
    auto* base     = static_cast<std::byte*>(workspace);
    auto* ids      = reinterpret_cast<std::int32_t*>(base + l.ids);
    auto* weights  = reinterpret_cast<float*>(base + l.weights);
    auto* gates    = reinterpret_cast<float*>(base + l.shared_gate);
    auto* counts   = reinterpret_cast<std::int32_t*>(base + l.counts);
    auto* offsets  = reinterpret_cast<std::int32_t*>(base + l.offsets);
    auto* cursors  = reinterpret_cast<std::int32_t*>(base + l.cursors);
    auto* active   = reinterpret_cast<std::int32_t*>(base + l.active);
    auto* count    = reinterpret_cast<std::int32_t*>(base + l.active_count);
    auto* sorted   = reinterpret_cast<std::int32_t*>(base + l.pairs);
    auto* act      = reinterpret_cast<__nv_bfloat16*>(base + l.activation);
    auto* partial  = reinterpret_cast<float*>(base + l.partial);

    CUDA_CHECK(cudaMemsetAsync(counts, 0, static_cast<std::size_t>(experts) * 4, stream));
    route_kernel<<<(tokens + kRouteWarps - 1) / kRouteWarps, kRouteWarps * 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), experts, top_k, tokens, ids, weights,
        gates, counts);
    CUDA_CHECK(cudaGetLastError());
    scan_kernel<<<1, kMaxExperts, 0, stream>>>(counts, experts, offsets, cursors, active, count);
    CUDA_CHECK(cudaGetLastError());
    scatter_kernel<<<(pairs + 255) / 256, 256, 0, stream>>>(ids, pairs, cursors, sorted);
    CUDA_CHECK(cudaGetLastError());

    if (tokens >= kMmaMinTokens) {
        static const bool carveout = [] {
            CUDA_CHECK(cudaFuncSetAttribute(gate_up_mma_kernel,
                                            cudaFuncAttributePreferredSharedMemoryCarveout, 100));
            CUDA_CHECK(cudaFuncSetAttribute(down_mma_kernel,
                                            cudaFuncAttributePreferredSharedMemoryCarveout, 100));
            return true;
        }();
        (void)carveout;
        auto* tiles      = reinterpret_cast<int2*>(base + l.tiles);
        auto* tile_count = reinterpret_cast<std::int32_t*>(base + l.tile_count);
        tile_list_kernel<<<1, 32, 0, stream>>>(counts, offsets, active, count, tiles, tile_count);
        CUDA_CHECK(cudaGetLastError());
        const unsigned max_tiles = static_cast<unsigned>(pairs / kTilePairs + experts + 1);
        gate_up_mma_kernel<<<dim3(intermediate / (kTileRows / 2), max_tiles), kMmaThreads, 0,
                             stream>>>(static_cast<const __nv_bfloat16*>(x.data), bank_args(gate),
                                       bank_args(up), counts, offsets, tiles, tile_count, sorted,
                                       top_k, hidden, intermediate, act);
        CUDA_CHECK(cudaGetLastError());
        down_mma_kernel<<<dim3(hidden / kTileRows, max_tiles), kMmaThreads, 0, stream>>>(
            bank_args(down), counts, offsets, tiles, tile_count, sorted, hidden, intermediate, act,
            partial);
        CUDA_CHECK(cudaGetLastError());
    } else {
    const unsigned slots = static_cast<unsigned>(std::min(pairs, experts));
    const dim3 gate_grid((intermediate + kGateRowsPerCta - 1) / kGateRowsPerCta, slots);
    gate_up_kernel<<<gate_grid, kProjWarps * 32, static_cast<std::size_t>(hidden) * 2, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), bank_args(gate), bank_args(up), active, count,
        offsets, sorted, top_k, hidden, intermediate, act);
    CUDA_CHECK(cudaGetLastError());
    const dim3 down_grid((hidden + kDownRowsPerCta - 1) / kDownRowsPerCta, slots);
    down_kernel<<<down_grid, kProjWarps * 32, static_cast<std::size_t>(intermediate) * 2,
                  stream>>>(bank_args(down), active, count, offsets, sorted, hidden, intermediate,
                            act, partial);
    CUDA_CHECK(cudaGetLastError());
    }
    const std::int64_t n = static_cast<std::int64_t>(hidden) * tokens;
    const int grid = static_cast<int>(std::clamp<std::int64_t>((n + 255) / 256, 1, 8192));
    combine_kernel<<<grid, 256, 0, stream>>>(partial, weights, gates,
                                             static_cast<const __nv_bfloat16*>(shared.data), top_k,
                                             hidden, n, static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
