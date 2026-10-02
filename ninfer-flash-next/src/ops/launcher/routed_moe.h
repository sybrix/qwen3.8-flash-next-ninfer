#pragma once

// ninfer::ops::detail - private launch prototypes for routed_moe.

#include "core/tensor.h"
#include "core/weight.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t routed_moe_workspace_launch_bytes(std::int32_t experts,
                                                            std::int32_t top_k,
                                                            std::int32_t hidden,
                                                            std::int32_t intermediate,
                                                            std::int32_t max_tokens);

void routed_moe_launch(const Tensor& x, const Tensor& logits, std::int32_t top_k,
                       const Nvfp4Bank& gate, const Nvfp4Bank& up, const Nvfp4Bank& down,
                       const Tensor& shared, void* workspace, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
