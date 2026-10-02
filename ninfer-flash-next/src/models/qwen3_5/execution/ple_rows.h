#pragma once

#include "models/qwen3_5/execution/parameters.h"

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::models::qwen3_5::execution {

/**
 * Host half of Qwen4Exp PLE: hashes the n-grams ending at `count` consecutive positions
 * [begin, begin+count) of one sequence and copies their FP8 table rows. `tokens` holds the
 * sequence from position 0 through at least begin+count-1. Positions before the start of the
 * token's EOS-delimited segment, or before 0, read as EOS (docs/maintainer/qwen4_exp-model.md).
 * `out` receives `count` columns of heads*row_bytes bytes, head-major within each column.
 */
void gather_ple_rows(const PleParameters& ple, const PleConfig& config,
                     std::span<const int> tokens, std::size_t begin, std::size_t count,
                     std::uint8_t* out);

} // namespace ninfer::models::qwen3_5::execution
