#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Gemv = Bf16A16GemvSchedule<4, 2, 1, 8, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
} // namespace

Bf16Launch select_bf16_n640_k2560(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Bf16ScheduleInstance<Gemv, 2560>>;
    if (tokens <= 64)
        return launch_bf16_sliced_k_mma<Bf16ScheduleInstance<Bf16A16SlicedR16T8W8, 2560>>;
    if (tokens <= 512) return launch_bf16_mma<Bf16ScheduleInstance<Bf16A16MmaR32T32K256S3, 2560>>;
    return launch_bf16_tma_mma<Bf16ScheduleInstance<Bf16A16TmaR64T64K128S2, 2560>>;
}
} // namespace ninfer::ops::detail
