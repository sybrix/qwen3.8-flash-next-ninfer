#pragma once

#include "ops/linear/bf16/bf16_launch.h"

namespace ninfer::ops::detail {

[[nodiscard]] Bf16Launch select_bf16_n14336_k5120(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n5120_k6144(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n256_k5120(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n10240_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n6144_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n96_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n48_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n512_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n128_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n640_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n513_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n1280_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n2560_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n2560_k6144(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n2560_k640(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n248320_k2560(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n320_k10240(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n10240_k320(std::int32_t tokens);
[[nodiscard]] Bf16Launch select_bf16_n4_k10240(std::int32_t tokens);

} // namespace ninfer::ops::detail
