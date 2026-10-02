#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
// K=320 admits no SIMT phase (a multiple of 128) or sliced-K group (a multiple of 128); only
// 64-wide MMA K tiles divide it.
Bf16Launch select_bf16_n10240_k320(std::int32_t tokens) {
    if (tokens <= 16) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T16K64S3, 320>>;
    if (tokens <= 64) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR64T32K64S3, 320>>;
    if (tokens <= 192) return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K64S3, 320>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T128K64S2, 320>>;
}
} // namespace ninfer::ops::detail
