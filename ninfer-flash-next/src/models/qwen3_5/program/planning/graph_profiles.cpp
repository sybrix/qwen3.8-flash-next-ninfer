#include "models/qwen3_5/program/planning/graph_profiles.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace ninfer::models::qwen3_5::detail {
namespace {
// Resource tiers bound inactive attention work. They are not kernel/topology boundaries.
constexpr std::array<std::uint32_t, 7> kCausalVisibleTiers{128,  512,   2048, 4096,
                                                           8192, 16384, 32768};

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

std::vector<GraphExecutionProfile> causal_resource_profiles(std::uint32_t capacity,
                                                            std::uint32_t visible_offset) {
    std::vector<std::uint32_t> ends;
    for (const auto visible : kCausalVisibleTiers)
        if (visible >= visible_offset) ends.push_back(visible - visible_offset);
    return graph_profiles_through(capacity - 1, ends);
}

} // namespace

std::vector<GraphExecutionProfile> ordinary_graph_profiles(std::uint32_t capacity,
                                                           std::uint32_t sparse_visible_threshold) {
    // E+1 is the one-token visible window; all tiers share one topology per exact B, except that
    // tiers reaching past a sparse-attention threshold form a second topology.
    auto profiles = causal_resource_profiles(capacity, 1);
    for (auto& profile : profiles) {
        profile.topology_class =
            sparse_visible_threshold != 0 && profile.max + 1 > sparse_visible_threshold ? 1U : 0U;
    }
    return profiles;
}

std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                      std::uint32_t draft_window,
                                                      std::uint32_t sparse_visible_threshold) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // The final AR call can see E+2K. Target verify and MTP attention are update-compatible
    // across all resource tiers, independently of the selected KV representation. With a sparse
    // threshold, every call's dense/sparse choice follows its envelope, whose maximum visible
    // extent lies in [E+1, E+2K+1]: tiers entirely below, entirely above and straddling the
    // threshold are distinct topologies.
    auto profiles = causal_resource_profiles(capacity, 2 * draft_window);
    if (sparse_visible_threshold != 0) {
        for (auto& profile : profiles) {
            const std::uint64_t lowest  = std::uint64_t(profile.max) + 1U;
            const std::uint64_t highest = std::uint64_t(profile.max) + 2U * draft_window + 1U;
            profile.topology_class = highest <= sparse_visible_threshold ? 0U
                                     : lowest > sparse_visible_threshold ? 1U
                                                                         : 2U;
        }
    }
    return profiles;
}

std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                         std::uint32_t capacity,
                                                         std::uint32_t draft_window) {
    if (capacity == 0 || draft_window == 0 || draft_window > 15) {
        throw std::invalid_argument("invalid masked draft graph dimensions");
    }
    if (backend == SpeculativeBackend::DFlash2) {
        auto profiles = graph_profiles_through(capacity - 1, {96, 511, 2047, 8191, 32767});
        for (std::size_t i = 0; i < profiles.size(); ++i) {
            profiles[i].topology_class = static_cast<std::uint32_t>(i);
        }
        return profiles;
    }
    // Retain the draft's resource and topology profiles. Target causal attention contributes
    // no context-dependent topology class.
    auto profiles = graph_profiles_through(
        capacity - 1, {96, 127, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65536, 131072, 196608});
    for (GraphExecutionProfile& profile : profiles) {
        profile.topology_class = profile.max > 96U ? 1U : 0U;
    }
    return profiles;
}

} // namespace ninfer::models::qwen3_5::detail
