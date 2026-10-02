#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv = Bf16A16GemvSchedule<8, 2, 2, 8, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 2, 1, 1>;
using C2   = Bf16A16SimtSchedule<4, 1, 4, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
using C4   = Bf16A16SimtSchedule<4, 1, 2, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 2, 1, 2>;
} // namespace

Bf16Launch select_bf16_n2560_k2560(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Bf16ScheduleInstance<Gemv, 2560>>;
    if (tokens <= 2) return launch_bf16_simt<Bf16ScheduleInstance<C2, 2560, 2>>;
    if (tokens <= 4) return launch_bf16_simt<Bf16ScheduleInstance<C4, 2560, 4>>;
    if (tokens <= 16)
        return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR16T16W8, 2560>>;
    if (tokens <= 128) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 2560>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T128K64S2, 2560>>;
}
} // namespace ninfer::ops::detail
