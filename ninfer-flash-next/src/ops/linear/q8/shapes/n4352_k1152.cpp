#include "ops/linear/q8/q8_shapes.h"

#include <stdexcept>

namespace ninfer::ops::detail {

Q8Launch select_q8_n4352_k1152(std::int32_t tokens) {
    // Vision calls carry up to ~65K patches; the launchers slice any column extent.
    if (tokens > 131072) throw std::invalid_argument("q8 linear: column extent exceeds 131072");
    if (tokens <= 8 || (tokens <= 12 && tokens % 4 == 0)) return launch_q8_a16_simt_r8_t4;
    if (tokens <= 256) return launch_q8_a16_mma_r32_t128;
    return launch_q8_a16_mma_r64_t128;
}

} // namespace ninfer::ops::detail
