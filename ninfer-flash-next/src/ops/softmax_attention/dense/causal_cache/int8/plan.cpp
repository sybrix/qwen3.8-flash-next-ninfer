#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/operands.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 256;
} // namespace

Int8KvCausalPlan make_int8_kv_causal_plan(int heads, int kv_heads, int width, int batch,
                                          CausalAttentionExecutionEnvelope envelope) {
    if (!causal_geometry_registered(heads, kv_heads) || width < 1 || batch < 1 || batch > 8 ||
        (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("INT8 attention: invalid plan inputs");
    const int group             = heads / kv_heads;
    const int grouped_limit     = causal_grouped_token_tile(group);
    const auto family           = width <= grouped_limit             ? Int8KvFamily::Grouped
                                  : width <= kGroupedPrefillMaxWidth ? Int8KvFamily::ParallelGrouped
                                                                     : Int8KvFamily::Tiled;
    const int tiles =
        family == Int8KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int independent_tiles = batch * kv_heads * tiles;
    constexpr int sms           = kCausalAttentionSmCount;
    const int wave_ctas         = (sms / independent_tiles) * independent_tiles;
    const int budget = kv_heads == 4 || width <= 4 || wave_ctas < sms * 9 / 10 ? 2 * sms : sms;
    CausalKvPartition partition{
        1, std::clamp(budget / independent_tiles, 1, CausalKvPartition::kMaxSplits)};
    // Bound partial traffic by keeping enough KV work in each split.
    partition.key_shift = (width == 1 ? 7 : 8) - (kv_heads == 2 ? 1 : 0);
    partition.capacity  = partition.active(envelope.max_visible_keys);
    return {family, heads, kv_heads, width, batch, envelope, partition};
}

std::size_t int8_kv_workspace_bytes(int heads, int kv_heads, int batch, int min_width,
                                    int max_width, CausalAttentionExecutionEnvelope envelope) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        const auto plan = make_int8_kv_causal_plan(heads, kv_heads, width, batch, envelope);
        if (plan.family == Int8KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

} // namespace ninfer::ops::detail
