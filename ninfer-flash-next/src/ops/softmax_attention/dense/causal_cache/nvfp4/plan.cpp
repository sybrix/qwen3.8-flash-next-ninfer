#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/operands.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
constexpr int kGroupedPrefillMaxWidth = 192;
} // namespace

Nvfp4KvCausalPlan make_nvfp4_kv_causal_plan(int heads, int kv_heads, int width, int batch,
                                            CausalAttentionExecutionEnvelope envelope) {
    if (!causal_geometry_registered(heads, kv_heads) || width < 1 || batch < 1 || batch > 8 ||
        (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("NVFP4 attention: invalid plan inputs");
    const int group             = heads / kv_heads;
    const int grouped_limit     = causal_grouped_token_tile(group);
    const auto family           = width <= grouped_limit ? Nvfp4KvFamily::Grouped
                                  : width <= kGroupedPrefillMaxWidth ? Nvfp4KvFamily::ParallelGrouped
                                                                     : Nvfp4KvFamily::Tiled;
    const int tiles =
        family == Nvfp4KvFamily::ParallelGrouped ? (width + grouped_limit - 1) / grouped_limit : 1;
    const int independent_tiles = batch * kv_heads * tiles;
    const int query_tile        = family == Nvfp4KvFamily::ParallelGrouped &&
                                          width <= 2 * grouped_limit
                                      ? (width + 1) / 2
                                      : std::min(width, grouped_limit);
    const int row_tiles         = (query_tile * group + 15) / 16;
    constexpr int sms           = kCausalAttentionSmCount;
    const int wave_ctas         = (sms / independent_tiles) * independent_tiles;
    const int budget            = row_tiles <= 2 || wave_ctas < sms * 9 / 10 ? 2 * sms : sms;
    CausalKvPartition partition{
        1, std::clamp(budget / independent_tiles, 1, CausalKvPartition::kMaxSplits)};
    // Bound partial traffic by keeping enough KV work in each split.
    partition.key_shift = (row_tiles <= 2 ? 7 : 8) - (kv_heads == 2 ? 1 : 0);
    partition.capacity  = partition.active(envelope.max_visible_keys);
    return {family, heads, kv_heads, width, batch, query_tile, envelope, partition};
}

std::size_t nvfp4_kv_workspace_bytes(int heads, int kv_heads, int batch, int min_width,
                                     int max_width, CausalAttentionExecutionEnvelope envelope) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= std::min(max_width, kGroupedPrefillMaxWidth); ++width) {
        const auto plan = make_nvfp4_kv_causal_plan(heads, kv_heads, width, batch, envelope);
        if (plan.family == Nvfp4KvFamily::Tiled) continue;
        const int splits = plan.partition.capacity;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

} // namespace ninfer::ops::detail
