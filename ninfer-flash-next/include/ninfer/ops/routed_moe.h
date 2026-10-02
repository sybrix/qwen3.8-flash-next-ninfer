#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/// Workspace for routed_moe over at most `max_tokens` columns. Positive arguments only.
[[nodiscard]] std::size_t routed_moe_workspace_bytes(std::int32_t experts, std::int32_t top_k,
                                                     std::int32_t hidden,
                                                     std::int32_t intermediate,
                                                     std::int32_t max_tokens);

/**
 * Routed SwiGLU experts with NVFP4 banks and a sigmoid-gated shared-expert output. With E
 * experts, routing logits l[0..E,t] (row E is the shared-expert score) and K = top_k:
 *
 *   p[e,t]     = softmax_e(l[0..E-1,t])
 *   S_t        = the K experts of largest p (ties to the lower index)
 *   w[e,t]     = p[e,t] / sum_{e' in S_t} p[e',t]                        e in S_t
 *   a_e(x)[i]  = SiLU(gate_e[i,:] . x) * (up_e[i,:] . x)                 i < I
 *   ideal[h,t] = sum_{e in S_t} w[e,t] * (down_e[h,:] . a_e(x[:,t]))
 *                + sigmoid(l[E,t]) * shared[h,t].
 *
 * Expert matrices decode as `e2m1 * block_scale / weight_divisor` per expert (Nvfp4Bank);
 * activations stay BF16 (A16). `x`, `shared` and `out` are contiguous BF16 [H,T]; `logits` is
 * contiguous BF16 [E+1,T]. Banks: gate and up are [E,I,H], down is [E,H,I], all with the same E.
 * `out` must not overlap any input or the workspace. The oracle evaluates the formula in FP64
 * from the represented inputs and decoded weights; routing uses the same represented logits, so
 * near-tied selections are excluded from qualification inputs. Any positive T up to the
 * workspace's max_tokens. Internal accumulation and intermediate rounding are implementation
 * choices; the BF16 output is compared after promotion.
 */
void routed_moe(const Tensor& x, const Tensor& logits, std::int32_t top_k, const Nvfp4Bank& gate,
                const Nvfp4Bank& up, const Nvfp4Bank& down, const Tensor& shared,
                WorkspaceArena& workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
