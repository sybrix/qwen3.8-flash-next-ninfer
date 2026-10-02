#include "ops/linear/bf16/bf16_instances.cuh"
#include "ops/linear/bf16/bf16_shapes.h"
#include "ops/linear/bf16/bf16_launch.cuh"

namespace ninfer::ops::detail {
namespace {
// 513 = 27 * 19 admits SIMT row blocks of three but no 16-row MMA tile. Small T runs all rows
// through SIMT; larger T contracts rows [0,512) with an MMA schedule and the final row with SIMT.
using Gemv = Bf16A16GemvSchedule<6, 2, 1, 8, 4, Bf16ActivationAccess::Direct,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;
using Simt = Bf16A16SimtSchedule<6, 2, 1, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2>;
using Tail = Bf16A16SimtSchedule<5, 5, 1, 8, 1, 4, Bf16SimtActivationAccess::WarpPacked,
                                 Bf16WeightCache::Default, Bf16PhaseOrder::Sequential, 1, 1, 1, 2, 8>;

constexpr int kHeadRows = 512;

template <auto Head>
void launch_head_tail(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const Bf16A16Operands all = bf16_a16_operands(x, w);
    auto* data                = static_cast<__nv_bfloat16*>(out.data);
    Bf16A16Operands head      = all;
    head.rows                 = kHeadRows;
    Bf16A16Operands tail      = all;
    tail.rows                 = all.rows - kHeadRows;
    tail.weight += static_cast<std::int64_t>(kHeadRows) * all.k;
    Head(head, LinearBf16Output{data, all.rows}, LinearIdentityEpilogue{}, stream);
    launch_bf16_a16_simt<Bf16ScheduleInstance<Tail, 2560>>(
        tail, LinearBf16Output{data + kHeadRows, all.rows}, LinearIdentityEpilogue{}, stream);
}

template <class Schedule>
inline constexpr auto kSlicedHead =
    launch_bf16_a16_sliced_k_mma<Bf16ScheduleInstance<Schedule, 2560>, LinearBf16Output,
                                 LinearIdentityEpilogue>;
template <class Schedule>
inline constexpr auto kMmaHead =
    launch_bf16_a16_mma<Bf16ScheduleInstance<Schedule, 2560>, LinearBf16Output,
                        LinearIdentityEpilogue>;
template <class Schedule>
inline constexpr auto kTmaHead =
    launch_bf16_a16_tma_mma<Bf16ScheduleInstance<Schedule, 2560>, LinearBf16Output,
                            LinearIdentityEpilogue>;
} // namespace

Bf16Launch select_bf16_n513_k2560(std::int32_t tokens) {
    if (tokens == 1) return launch_bf16_gemv<Bf16ScheduleInstance<Gemv, 2560>>;
    if (tokens <= 2) return launch_bf16_simt<Bf16ScheduleInstance<Simt, 2560, 2>>;
    if (tokens <= 4) return launch_bf16_simt<Bf16ScheduleInstance<Simt, 2560, 4>>;
    if (tokens <= 64) return launch_head_tail<kSlicedHead<Bf16A16SlicedR16T8W8>>;
    if (tokens <= 512) return launch_head_tail<kMmaHead<Bf16A16MmaR32T32K256S3>>;
    return launch_head_tail<kTmaHead<Bf16A16TmaR64T64K128S2>>;
}
} // namespace ninfer::ops::detail
