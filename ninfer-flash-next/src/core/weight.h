#pragma once

#include "core/dtype.h"

#include <cstddef>
#include <cstdint>

namespace ninfer {

enum class QType : std::uint16_t {
    Q4_G64_FP16         = 0,
    Q5_G64_FP16         = 1,
    Q6_G64_FP16         = 2,
    Q8_G32_FP16         = 3,
    BF16                = 4,
    FP32                = 5,
    INT32               = 6,
    NVFP4               = 7,
    FP8_E4M3FN_ROW_BF16 = 8,
    FP8_E4M3FN          = 9, // raw E4M3FN words; any multiplier is bound separately
};

enum class QuantLayout : std::uint16_t {
    RowSplit            = 0,
    Contiguous          = 1,
    BlockScaleK16M128x4 = 2,
    RowScale            = 3,
    // Rank-3 [E,N,K]: E complete BlockScaleK16M128x4 encodings at a 256-byte-aligned stride.
    BlockScaleK16M128x4Bank = 4,
};

struct Weight {
    const void* payload            = nullptr;
    std::uint64_t payload_bytes    = 0;
    std::uint64_t high_plane_bytes = 0;
    QType qtype                    = QType::Q4_G64_FP16;
    std::uint32_t group_size       = 0;
    std::int32_t shape[4]          = {1, 1, 1, 1};
    std::int32_t padded_shape[4]   = {1, 1, 1, 1};
    std::uint32_t ndim             = 0;

    const void* qdata          = nullptr;
    const void* qhigh          = nullptr;
    const void* scales         = nullptr;
    std::int32_t n             = 0;
    std::int32_t k             = 0;
    std::int32_t group         = 0;
    QuantLayout layout         = QuantLayout::RowSplit;
    DType scale_dtype          = DType::FP32;
    std::int32_t scale_ne[4]   = {1, 1, 1, 1};
    std::int64_t scale_nb[4]   = {0, 0, 0, 0};
    float weight_scale_divisor = 0.0F;
    float input_scale_divisor  = 0.0F;
};

/// One resident rank-3 NVFP4 bank. Expert e occupies [data + e*expert_stride, +expert_bytes) and
/// is a complete [n,k] BlockScaleK16M128x4 encoding: codes at 0, swizzled E4M3 scales at
/// scale_offset and its own FP32 weight divisor at divisor_offset.
struct Nvfp4Bank {
    const std::byte* data       = nullptr;
    std::uint64_t expert_stride = 0;
    std::uint64_t scale_offset  = 0;
    std::uint64_t divisor_offset = 0;
    std::int32_t experts        = 0;
    std::int32_t n              = 0;
    std::int32_t k              = 0;
};

} // namespace ninfer
