#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// Four rows admit no 16-row MMA tile. Decode gives each row one CTA; larger T keeps all four rows
// in every warp so each activation pack is loaded once per token block.
using Gemv = Bf16A16GemvSchedule<8, 8, 1, 8, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
using Simt = Bf16A16SimtSchedule<8, 8, 4, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2, 8>;
} // namespace

Bf16Launch select_bf16_n4_k10240(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Bf16ScheduleInstance<Gemv, 10240>>;
    if (tokens <= 2) return launch_bf16_simt<Bf16ScheduleInstance<Simt, 10240, 2>>;
    if (tokens <= 4) return launch_bf16_simt<Bf16ScheduleInstance<Simt, 10240, 4>>;
    return launch_bf16_simt<Bf16ScheduleInstance<Simt, 10240>>;
}
} // namespace ninfer::ops::detail
