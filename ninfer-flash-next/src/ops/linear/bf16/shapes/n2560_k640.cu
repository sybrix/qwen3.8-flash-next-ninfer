#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// K=640 admits only 128-wide SIMT phases (4 values per lane, one warp per row).
using Gemv = Bf16A16GemvSchedule<4, 1, 4, 4, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
using C2   = Bf16A16SimtSchedule<4, 1, 4, 4, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using C4   = Bf16A16SimtSchedule<4, 1, 2, 4, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
} // namespace

Bf16Launch select_bf16_n2560_k640(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Bf16ScheduleInstance<Gemv, 640>>;
    if (tokens <= 2) return launch_bf16_simt<Bf16ScheduleInstance<C2, 640, 2>>;
    if (tokens <= 4) return launch_bf16_simt<Bf16ScheduleInstance<C4, 640, 4>>;
    if (tokens <= 16)
        return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR16T16W2, 640>>;
    if (tokens <= 64) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 640>>;
    if (tokens <= 192)
        return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K128S2, 640>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T128K64S2, 640>>;
}
} // namespace ninfer::ops::detail
