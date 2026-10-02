#pragma once

#include "core/tensor.h"
#include "ninfer/ops/gdn_replay.h" // GdnReplayFoldRow

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>
#include <span>

namespace ninfer::ops {

/**
 * Qwen4Exp per-layer n-gram embedding (PLE). Hashing and table-row gathering happen on the host;
 * these Ops consume the gathered FP8 rows. See docs/maintainer/qwen4_exp-model.md. None uses
 * workspace. Oracles evaluate each formula naively in FP64 from the represented inputs.
 */

/**
 * Scales gathered FP8 E4M3FN table rows:  ideal[e,t] = codes[e,t] * scale[0].
 * `codes` is contiguous FP8_E4M3FN [E,T], `scale` contiguous BF16 [1], `out` contiguous BF16 [E,T].
 */
void ple_dequantize(const Tensor& codes, const Tensor& scale, Tensor& out, cudaStream_t stream);

/**
 * Stream gate. With S streams of width H:
 *
 *   gamma[s,t] = sum_h key[s*H+h,t] * query[s*H+h,t] / sqrt(H)
 *   gamma'     = sign(gamma) * sqrt(max(|gamma|, 1e-6))
 *   ideal[s*H+h,t] = sigmoid(gamma'[s,t]) * value[h,t].
 *
 * `key` and `query` are contiguous BF16 [S*H,T]; `value` contiguous BF16 [H,T]; `out` contiguous
 * BF16 [S*H,T].
 */
void ple_gate(const Tensor& key, const Tensor& query, const Tensor& value, std::int32_t streams,
              Tensor& out, cudaStream_t stream);

/// Taps and dilation of the PLE depthwise causal convolution; the state holds
/// (taps-1)*dilation = 9 previous columns.
inline constexpr std::int32_t kPleConvTaps     = 4;
inline constexpr std::int32_t kPleConvDilation = 3;
inline constexpr std::int32_t kPleConvState    = (kPleConvTaps - 1) * kPleConvDilation;

/**
 * Dilated depthwise causal convolution with SiLU, added to a residual together with the gated
 * value. Let u[c,-9..-1] be the state (oldest first) and u[c,t] = normalized[c,t] for t >= 0:
 *
 *   conv[c,t]  = sum_{k=0..3} weight[k,c] * u[c, t - (3-k)*3]
 *   ideal[c,t] = residual[c,t] + gated[c,t] + SiLU(conv[c,t]).
 *
 * `gated`, `normalized` and `residual` are contiguous BF16 [C,T]; `residual` is updated in place.
 * `weight` is contiguous BF16 [C,4] (tap-major: ne[0]=C). States are contiguous BF16 [C,9].
 * `state_out` receives the trailing 9 columns of concat(state_in, normalized); it may be disjoint
 * from or exactly equal to `state_in`. Any positive T.
 */
void ple_conv_residual(const Tensor& gated, const Tensor& normalized, const Tensor& weight,
                       const Tensor& state_in, Tensor& state_out, Tensor& residual,
                       cudaStream_t stream);

/**
 * Snapshot form for B independent sequences of W columns. `gated`, `normalized` and `residual` are
 * contiguous BF16 [C,W,B]; `states` is contiguous BF16 [C,9,Slots]; `initial_state_slots` and
 * `snapshot_base_slots` are contiguous I32 [B]; `valid_columns` is contiguous I32 [B] with values
 * in [1,W], or an empty Tensor meaning all W are valid.
 *
 * Row b starts from the window in slot initial_state_slots[b]. After valid column j its new window
 * is written to slot snapshot_base_slots[b]+j. Invalid-tail columns leave `residual` unchanged
 * and write no state. Reservations [base,base+W) of different rows are disjoint and never contain
 * another row's initial slot; a row's own initial slot may lie in its reservation.
 */
void ple_conv_residual_snapshot(const Tensor& gated, const Tensor& normalized,
                                const Tensor& weight, Tensor& states, const Tensor& valid_columns,
                                const Tensor& initial_state_slots,
                                const Tensor& snapshot_base_slots, Tensor& residual,
                                cudaStream_t stream);

/**
 * Speculative verify for Qwen4Exp: ReplaySSM-style records instead of state snapshots.
 *
 * Record form of ple_conv_residual_snapshot. Shapes and the residual update are identical, but the
 * state pool is read only: row b reads the window in initial_state_slots[b], and `record`
 * (contiguous BF16 [C,W,B]) receives a bit copy of `normalized` (the invalid tail included; it is
 * never read back). The window after any committed prefix is rebuilt by ple_conv_replay_fold.
 */
void ple_conv_residual_record(const Tensor& gated, const Tensor& normalized, const Tensor& weight,
                              const Tensor& states, const Tensor& valid_columns,
                              const Tensor& initial_state_slots, Tensor& record, Tensor& residual,
                              cudaStream_t stream);

/**
 * Commits recorded PLE columns. For each row r (at most 8): the window in
 * source_state_slot followed by record[:, 0:commit_columns, r] is reduced to its trailing 9
 * columns and written to destination_state_slot (which may equal the source). commit_columns 0
 * copies the source window. `states` is BF16 [C,9,Slots]; `record` is BF16 [C,W,R] with R >= rows.
 */
void ple_conv_replay_fold(Tensor& states, const Tensor& record,
                          std::span<const GdnReplayFoldRow> rows, cudaStream_t stream);

/**
 * Record form of causal_conv1d_silu_snapshot for the Qwen4Exp GDN convolution: `x`, `out` and
 * `record` are contiguous BF16 [C,W,B]; `conv_states` is BF16 [C,3,Slots] and is read only (row b
 * from initial_state_slots[b]). `out` follows the causal_conv1d_silu formula (invalid-tail
 * columns are exact BF16 zero) and `record` receives a bit copy of `x`, the layout
 * GdnReplayFoldPlan consumes as its conv record.
 */
void causal_conv1d_silu_record(const Tensor& x, const Tensor& weight, const Tensor& conv_states,
                               const Tensor& valid_columns, const Tensor& initial_state_slots,
                               Tensor& record, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
