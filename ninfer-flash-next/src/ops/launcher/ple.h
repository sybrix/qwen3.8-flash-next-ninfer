#pragma once

// ninfer::ops::detail - private launch prototypes for the PLE Ops.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <span>

#include "ninfer/ops/gdn_replay.h"

namespace ninfer::ops::detail {

void ple_dequantize_launch(const Tensor& codes, const Tensor& scale, Tensor& out,
                           cudaStream_t stream);
void ple_gate_launch(const Tensor& key, const Tensor& query, const Tensor& value,
                     std::int32_t streams, Tensor& out, cudaStream_t stream);
// Sequence form is the snapshot form with B=1, W=T, one initial slot and every column valid; the
// launcher receives raw state pointers so both forms share kernels.
void ple_conv_residual_launch(const Tensor& gated, const Tensor& normalized, const Tensor& weight,
                              const void* state_in, void* state_out, Tensor& residual,
                              cudaStream_t stream);
void ple_conv_residual_snapshot_launch(const Tensor& gated, const Tensor& normalized,
                                       const Tensor& weight, Tensor& states,
                                       const Tensor& valid_columns,
                                       const Tensor& initial_state_slots,
                                       const Tensor& snapshot_base_slots, Tensor& residual,
                                       cudaStream_t stream);

void ple_conv_residual_record_launch(const Tensor& gated, const Tensor& normalized,
                                     const Tensor& weight, const Tensor& states,
                                     const Tensor& valid_columns,
                                     const Tensor& initial_state_slots, Tensor& residual,
                                     cudaStream_t stream);
void ple_conv_replay_fold_launch(Tensor& states, const Tensor& record,
                                 std::span<const GdnReplayFoldRow> rows, cudaStream_t stream);
void causal_conv1d_silu_record_launch(const Tensor& x, const Tensor& weight,
                                      const Tensor& conv_states, const Tensor& valid_columns,
                                      const Tensor& initial_state_slots, Tensor& out,
                                      cudaStream_t stream);

} // namespace ninfer::ops::detail
