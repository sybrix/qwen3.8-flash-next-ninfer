#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// One CTA per row keeps decode parallel; five warps split K=2560 into two 1280-wide phases.
using Gemv = Bf16A16GemvSchedule<5, 5, 1, 8, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
using Simt = Bf16A16SimtSchedule<5, 5, 1, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2, 8>;
} // namespace

Bf16Launch select_bf16_n96_k2560(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Bf16ScheduleInstance<Gemv, 2560>>;
    if (tokens <= 2) return launch_bf16_simt<Bf16ScheduleInstance<Simt, 2560, 2>>;
    if (tokens <= 4) return launch_bf16_simt<Bf16ScheduleInstance<Simt, 2560, 4>>;
    if (tokens <= 8) return launch_bf16_simt<Bf16ScheduleInstance<Simt, 2560, 8>>;
    if (tokens <= 64)
        return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR16T8W8, 2560>>;
    if (tokens <= 256)
        return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR16T16W8, 2560>>;
    return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K128S3, 2560>>;
}
} // namespace ninfer::ops::detail
