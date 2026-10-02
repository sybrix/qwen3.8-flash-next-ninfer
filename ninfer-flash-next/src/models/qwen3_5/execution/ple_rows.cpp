#include "models/qwen3_5/execution/ple_rows.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

void gather_range(const PleParameters& ple, const PleConfig& config, std::span<const int> tokens,
                  std::size_t begin, std::size_t count, std::uint8_t* out) {
    const auto& layer       = *ple.config;
    const std::uint32_t n   = config.ngram_size;
    const std::uint32_t per = config.heads_per_ngram;
    const std::int64_t eos  = config.eos_token_id;
    if (begin + count > tokens.size() || n > 8) {
        throw std::invalid_argument("PLE row gather exceeds its token history");
    }
    std::int64_t shifted[8];
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t p = begin + i;
        // shifted[j] is the token at p-j when p >= j and no EOS lies in tokens[p-j+1 .. p-1]
        // (the segment containing p starts after the last EOS before p); otherwise EOS.
        shifted[0] = tokens[p];
        for (std::uint32_t j = 1; j < n; ++j) {
            bool valid = p >= j;
            for (std::uint32_t q = 1; valid && q < j; ++q) { valid = tokens[p - q] != eos; }
            shifted[j] = valid ? tokens[p - j] : eos;
        }
        std::uint8_t* column = out + i * static_cast<std::size_t>(config.heads()) * ple.row_bytes;
        for (std::uint32_t order = 2; order <= n; ++order) {
            std::int64_t mix = shifted[0] * layer.multipliers[0];
            for (std::uint32_t j = 1; j < order; ++j) { mix ^= shifted[j] * layer.multipliers[j]; }
            for (std::uint32_t h = 0; h < per; ++h) {
                const std::uint32_t head = (order - 2) * per + h;
                const std::int64_t size  = layer.head_vocab_sizes[head];
                std::int64_t id          = mix % size;
                if (id < 0) { id += size; }
                const auto row   = static_cast<std::uint64_t>(id + layer.head_offsets[head]);
                const auto shard = row / ple.shard_rows;
                if (shard >= ple.table.size()) {
                    throw std::out_of_range("PLE row is outside the mapped table");
                }
                std::memcpy(column + static_cast<std::size_t>(head) * ple.row_bytes,
                            ple.table[shard] + (row - shard * ple.shard_rows) * ple.row_bytes,
                            ple.row_bytes);
            }
        }
    }
}

} // namespace

void gather_ple_rows(const PleParameters& ple, const PleConfig& config,
                     std::span<const int> tokens, std::size_t begin, std::size_t count,
                     std::uint8_t* out) {
    // Every row is a random read from the memory-mapped table; with a cold page cache each is a
    // synchronous NVMe page fault (~60 us). Prefill-sized gathers therefore fan out across
    // threads so the faults overlap (measured ~8x on the 53.7 GB table); each thread writes its
    // own columns, so the result is identical to the serial gather.
    constexpr std::size_t kParallelMinimum = 128;
    constexpr std::size_t kMaximumThreads  = 16;
    if (count < kParallelMinimum) {
        gather_range(ple, config, tokens, begin, count, out);
        return;
    }
    const std::size_t column = static_cast<std::size_t>(config.heads()) * ple.row_bytes;
    const std::size_t threads =
        std::min<std::size_t>(kMaximumThreads, (count + kParallelMinimum / 2 - 1) /
                                                   (kParallelMinimum / 2));
    const std::size_t per = (count + threads - 1) / threads;
    std::vector<std::exception_ptr> errors(threads);
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (std::size_t t = 0; t < threads; ++t) {
        const std::size_t first = t * per;
        if (first >= count) { break; }
        const std::size_t n = std::min(per, count - first);
        workers.emplace_back([&, t, first, n] {
            try {
                gather_range(ple, config, tokens, begin + first, n, out + first * column);
            } catch (...) {
                errors[t] = std::current_exception();
            }
        });
    }
    for (auto& worker : workers) { worker.join(); }
    for (const auto& error : errors) {
        if (error) { std::rethrow_exception(error); }
    }
}

} // namespace ninfer::models::qwen3_5::execution
