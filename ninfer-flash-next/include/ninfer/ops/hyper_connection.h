#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Qwen4Exp hyper-connection residual streams. A residual state holds S streams of width H per
 * token as contiguous BF16 [S*H,T], stream-major within each column. The gated-residual math is
 * composed from these Ops and ordinary BF16 projections:
 *
 *   N = grouped_rmsnorm(X)                    zero-centred weight over S groups of H
 *   m = sigmoid(U * hyper_connection_gates(D * N).hidden)
 *   u = hyper_connection_mix(N, U-projection logits)     the block input
 *   c = hyper_connection_gates(B * N).inject             injection weights
 *   X <- hyper_connection_inject(X, block output, c)
 *
 * See docs/maintainer/qwen4_exp-model.md. Every Op accepts any positive T, uses no workspace and
 * has no persistent state. Inputs and outputs must not overlap unless an in-place form says so.
 * Oracles evaluate the stated formula naively in FP64 from the represented inputs; BF16 outputs
 * are compared after promotion under each Op's named criterion.
 */

/**
 * Grouped zero-centred RMS normalization:
 *
 *   inv[g,t]   = 1 / sqrt((1/D) * sum_d x[g*D+d,t]^2 + eps)
 *   ideal[i,t] = x[i,t] * inv[i/D,t] * (1 + weight[i]).
 *
 * `x` and `out` are contiguous BF16 [G*D,T]; `weight` is contiguous BF16 [G*D]. eps is positive and
 * finite.
 */
void grouped_rmsnorm(const Tensor& x, const Tensor& weight, std::int32_t groups, float eps,
                     Tensor& out, cudaStream_t stream);

/**
 * Sigmoid-weighted stream mean:
 *
 *   ideal[h,t] = (1/S) * sum_s sigmoid(logits[s*H+h,t]) * normalized[s*H+h,t].
 *
 * `normalized` and `logits` are contiguous BF16 [S*H,T]; `out` is contiguous BF16 [H,T].
 */
void hyper_connection_mix(const Tensor& normalized, const Tensor& logits, std::int32_t streams,
                          Tensor& out, cudaStream_t stream);

/**
 * Elementwise gate preparation with stream count S:
 *
 *   hidden[r,t] = SiLU(down[r,t] / S)            BF16 [R,T]
 *   inject[s,t] = 2 * sigmoid(inject_logits[s,t] / S)   FP32 [S,T]
 *
 * `down` is contiguous BF16 [R,T] and `inject_logits` contiguous BF16 [S,T]. Either pair may be
 * empty Tensors (no data) to skip it; at least one pair is present.
 */
void hyper_connection_gates(const Tensor& down, const Tensor& inject_logits, std::int32_t streams,
                            Tensor& hidden, Tensor& inject, cudaStream_t stream);

/**
 * In-place stream injection:
 *
 *   ideal[s*H+h,t] = state[s*H+h,t] + inject[s,t] * y[h,t].
 *
 * `state` is contiguous BF16 [S*H,T], updated in place; `y` is contiguous BF16 [H,T]; `inject`
 * is contiguous FP32 [S,T].
 */
void hyper_connection_inject(const Tensor& y, const Tensor& inject, Tensor& state,
                             cudaStream_t stream);

} // namespace ninfer::ops
