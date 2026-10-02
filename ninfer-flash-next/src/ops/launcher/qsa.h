#pragma once

// ninfer::ops::detail - private launch prototypes for the QSA Ops.

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

void qsa_prepare_query_launch(const Tensor& q, const Tensor& weight, const Tensor& positions,
                              std::int32_t rotary_dim, float theta, float eps, Tensor& out,
                              cudaStream_t stream);
void qsa_index_append_launch(const Tensor& keys, const Tensor& positions, const Tensor& rows,
                             const Tensor& index_pages, const Tensor& block_tables,
                             const Tensor& rope_positions, const Tensor& position_pages,
                             cudaStream_t stream);
[[nodiscard]] std::size_t qsa_select_workspace_launch_bytes(std::int32_t ratio,
                                                            std::int32_t index_dim,
                                                            std::int32_t sequences,
                                                            std::int32_t columns,
                                                            std::int32_t max_visible);
void qsa_select_launch(const Tensor& query, const Tensor& positions, const Tensor& rows,
                       const Tensor& index_pages, const Tensor& position_pages,
                       const Tensor& block_tables,
                       const Tensor& key_norm, std::int32_t ratio, std::int32_t budget,
                       std::int32_t rotary_dim, float theta, float eps, std::int32_t max_visible,
                       void* workspace, Tensor& selected, Tensor& counts, cudaStream_t stream);
void qsa_sparse_attention_launch(const Tensor& q, const Tensor& selected, const Tensor& counts,
                                 const Tensor& rows, const PagedKVBatchLayerView& cache,
                                 float scale, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
