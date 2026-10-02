#pragma once
#include "models/qwen3_5/program/program.h"

namespace ninfer::models::qwen3_5::detail {

// `sparse_visible_threshold` (0 = none): profiles whose largest visible extent exceeds it run a
// different attention topology (Qwen4Exp query-sparse attention) and get their own class.
[[nodiscard]] std::vector<GraphExecutionProfile>
ordinary_graph_profiles(std::uint32_t capacity, std::uint32_t sparse_visible_threshold = 0);
[[nodiscard]] std::vector<GraphExecutionProfile> mtp_graph_profiles(std::uint32_t capacity,
                                                                    std::uint32_t draft_window,
                                                                    std::uint32_t sparse_visible_threshold = 0);
[[nodiscard]] std::vector<GraphExecutionProfile> dflash_graph_profiles(SpeculativeBackend backend,
                                                                       std::uint32_t capacity,
                                                                       std::uint32_t draft_window);

} // namespace ninfer::models::qwen3_5::detail
