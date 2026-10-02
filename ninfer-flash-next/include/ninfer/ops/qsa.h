#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Qwen4Exp query-sparse attention (QSA); see docs/maintainer/qwen4_exp-model.md.
 *
 * Execution columns are grouped per sequence: a [.., W, B] operand holds W consecutive columns of
 * each of B sequences, `positions` is contiguous I32 [W,B] of absolute cache positions, and `rows`
 * is contiguous I32 [B] selecting each sequence's block-table row. A column at position p sees
 * the p+1 tokens [0,p] of its sequence. Paged index keys live in a BF16 plane
 * [Di,64,1,pages] sharing the Main K/V block tables (I32 [logical_pages, rows]).
 *
 * RoPE here is Text 1-D split-half NeoX over the first `rotary_dim` dimensions with angle
 * position * theta^(-2i/rotary_dim); the remaining dimensions are unchanged. RMSNorm is
 * zero-centred: x / sqrt(mean(x^2) + eps) * (1 + w). Oracles evaluate the stated formulas in FP64
 * from the represented inputs, rounding to BF16 where noted.
 */

/// Index queries: out[:,h,c] = BF16(RoPE_{pos(c)}(BF16(RMSNorm(q[:,h,c]; weight)))).
/// `q`/`out` contiguous BF16 [Di,Hi,W,B]; weight BF16 [Di].
void qsa_prepare_query(const Tensor& q, const Tensor& weight, const Tensor& positions,
                       std::int32_t rotary_dim, float theta, float eps, Tensor& out,
                       cudaStream_t stream);

/// Writes raw index keys `keys` (contiguous BF16 [Di,W,B]) to the slots of their positions.
void qsa_index_append(const Tensor& keys, const Tensor& positions, const Tensor& rows,
                      const Tensor& index_pages, const Tensor& block_tables, cudaStream_t stream);

struct QsaSelectGeometry {
    std::int32_t compress_ratio = 4;    // tokens per pooled block
    std::int32_t block_budget   = 512;  // selected blocks (token budget / ratio)
    std::int32_t rotary_dim     = 64;
    float theta                 = 1.0e7F;
    float eps                   = 1.0e-6F;

    [[nodiscard]] constexpr std::int32_t max_selected() const noexcept {
        return block_budget * compress_ratio + compress_ratio - 1;
    }
};

/// Workspace for qsa_select over at most `columns` columns of `sequences` sequences whose
/// visible extents do not exceed `max_visible` (the same bound passed to qsa_select).
[[nodiscard]] std::size_t qsa_select_workspace_bytes(const QsaSelectGeometry& geometry,
                                                     std::int32_t index_dim, std::int32_t sequences,
                                                     std::int32_t columns,
                                                     std::int32_t max_visible);

/**
 * Token selection. For column c at position p with V=p+1 visible tokens and nb=floor(V/R)
 * complete blocks b covering tokens [bR, bR+R):
 *
 *   kbar_b = BF16(RoPE_{bR}(BF16(RMSNorm(BF16(mean_{j<R} key[bR+j]); key_norm))))
 *   score  = sum_h relu(<query[:,h,c], kbar_b>) / sqrt(Di)          (FP32)
 *   chosen = the min(nb, block_budget) blocks of largest score; ties prefer lower b
 *   tokens = every token of each chosen block, then the V - nb*R tail tokens
 *
 * `selected` (I32 [max_selected, W, B]) receives the chosen positions in ascending order and
 * `counts` (I32 [W,B]) their number; slots past a count are unspecified. When nb <= block_budget
 * every visible token is selected. `query` is the prepared BF16 [Di,Hi,W,B] index query.
 */
/// `max_visible` bounds every column's p+1 and fixes the launch shape (CUDA Graph stable).
void qsa_select(const Tensor& query, const Tensor& positions, const Tensor& rows,
                const Tensor& index_pages, const Tensor& block_tables, const Tensor& key_norm,
                const QsaSelectGeometry& geometry, std::int32_t max_visible,
                WorkspaceArena& workspace, Tensor& selected, Tensor& counts, cudaStream_t stream);

/**
 * Attention over selected tokens for grouped heads (Hq query heads per Hkv KV heads, D=256):
 *
 *   s_j        = scale * <q[:,h,c], K[sel_j, h/(Hq/Hkv)]>        j < counts[c]
 *   out[:,h,c] = sum_j softmax_j(s) * V[sel_j, h/(Hq/Hkv)]
 *
 * K/V are read from the paged Main cache `cache` (BFloat16 or Fp8E4M3Row256 storage) through
 * the block-table row of each column's sequence; FP8 K is stored Hadamard-transformed and the
 * transform is applied to q. `q`/`out` contiguous BF16 [D,Hq,W,B].
 */
void qsa_sparse_attention(const Tensor& q, const Tensor& selected, const Tensor& counts,
                          const Tensor& rows, const PagedKVBatchLayerView& cache, float scale,
                          Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
