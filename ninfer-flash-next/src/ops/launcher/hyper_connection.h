#pragma once

// ninfer::ops::detail - private launch prototypes for the hyper-connection Ops.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void grouped_rmsnorm_launch(const Tensor& x, const Tensor& weight, std::int32_t groups, float eps,
                            Tensor& out, cudaStream_t stream);
void hyper_connection_mix_launch(const Tensor& normalized, const Tensor& logits,
                                 std::int32_t streams, Tensor& out, cudaStream_t stream);
void hyper_connection_gates_launch(const Tensor& down, const Tensor& inject_logits,
                                   std::int32_t streams, Tensor& hidden, Tensor& inject,
                                   cudaStream_t stream);
void hyper_connection_inject_launch(const Tensor& y, const Tensor& inject, Tensor& state,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
