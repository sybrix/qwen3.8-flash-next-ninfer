#include "ninfer/ops/routed_moe.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// Activations are rounded to BF16 between the projections and once more at the output.
constexpr ReductionCriterion routed_moe_criterion() {
    return {/*relative_l2*/ 3.0e-3, /*gross_absolute*/ 1.0e-4,
            /*gross_relative_to_max_reference*/ 1.5e-2};
}

constexpr double kE2m1[16] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                              -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

double decode_e4m3(std::uint8_t word) {
    const int exponent = (word >> 3) & 0xF;
    const int fraction = word & 7;
    if (exponent == 0) return fraction * std::ldexp(1.0, -9);
    return (1.0 + fraction / 8.0) * std::ldexp(1.0, exponent - 7);
}

std::uint64_t align256(std::uint64_t n) { return (n + 255) / 256 * 256; }

struct HostBank {
    std::int32_t experts = 0, n = 0, k = 0;
    std::uint64_t scale_offset = 0, divisor_offset = 0, stride = 0;
    std::vector<std::uint8_t> bytes;

    // Decodes one stored element exactly: e2m1 * E4M3 block scale / FP32 expert divisor.
    double at(std::int32_t e, std::int32_t row, std::int32_t col) const;
};

std::uint64_t scale_index(std::int32_t row, std::int32_t group, std::int32_t k) {
    const std::int32_t inner = row % 128;
    return (static_cast<std::uint64_t>(row / 128) * (k / 64) + group / 4) * 512 +
           (inner % 32) * 16 + (inner / 32) * 4 + group % 4;
}

double HostBank::at(std::int32_t e, std::int32_t row, std::int32_t col) const {
    const std::uint8_t* base = bytes.data() + stride * e;
    float divisor;
    std::memcpy(&divisor, base + divisor_offset, 4);
    const std::uint8_t byte = base[static_cast<std::size_t>(row) * (k / 2) + col / 2];
    const int code          = col % 2 ? byte >> 4 : byte & 15;
    return kE2m1[code] * decode_e4m3(base[scale_offset + scale_index(row, col / 16, k)]) / divisor;
}

HostBank make_bank(std::int32_t experts, std::int32_t n, std::int32_t k, std::mt19937& rng) {
    HostBank bank;
    bank.experts        = experts;
    bank.n              = n;
    bank.k              = k;
    bank.scale_offset   = align256(static_cast<std::uint64_t>(n) * k / 2);
    bank.divisor_offset = bank.scale_offset + static_cast<std::uint64_t>(n) * k / 16;
    bank.stride         = align256(bank.divisor_offset + 4);
    bank.bytes.assign(bank.stride * experts, 0);
    std::uniform_int_distribution<int> code(0, 15);
    std::uniform_int_distribution<int> scale_word(0x20, 0x3F); // E4M3 in [2^-3, ~0.94]
    std::uniform_real_distribution<float> divisor(64.0F, 512.0F);
    for (std::int32_t e = 0; e < experts; ++e) {
        std::uint8_t* base = bank.bytes.data() + bank.stride * e;
        const float d      = divisor(rng);
        std::memcpy(base + bank.divisor_offset, &d, 4);
        for (std::int32_t row = 0; row < n; ++row) {
            for (std::int32_t g = 0; g < k / 16; ++g) {
                base[bank.scale_offset + scale_index(row, g, k)] =
                    static_cast<std::uint8_t>(scale_word(rng));
                for (std::int32_t j = 0; j < 16; ++j) {
                    const int c            = code(rng);
                    const std::int32_t col = g * 16 + j;
                    std::uint8_t& byte = base[static_cast<std::size_t>(row) * (k / 2) + col / 2];
                    byte |= static_cast<std::uint8_t>(col % 2 ? c << 4 : c);
                }
            }
        }
    }
    return bank;
}

Nvfp4Bank device_bank(const HostBank& host, const DeviceBuffer& storage) {
    Nvfp4Bank bank;
    bank.data           = static_cast<const std::byte*>(storage.p);
    bank.expert_stride  = host.stride;
    bank.scale_offset   = host.scale_offset;
    bank.divisor_offset = host.divisor_offset;
    bank.experts        = host.experts;
    bank.n              = host.n;
    bank.k              = host.k;
    return bank;
}

std::vector<std::uint16_t> bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) out[i] = f32_to_bf16(values[i]);
    return out;
}

struct Geometry {
    std::int32_t experts, hidden, intermediate, top_k;
};

struct Banks {
    HostBank gate, up, down;
    DeviceBuffer gate_device, up_device, down_device;
};

Banks make_banks(const Geometry& g, std::uint32_t seed) {
    std::mt19937 rng(seed);
    Banks banks{make_bank(g.experts, g.intermediate, g.hidden, rng),
                make_bank(g.experts, g.intermediate, g.hidden, rng),
                make_bank(g.experts, g.hidden, g.intermediate, rng), {}, {}, {}};
    banks.gate_device = to_device(banks.gate.bytes);
    banks.up_device   = to_device(banks.up.bytes);
    banks.down_device = to_device(banks.down.bytes);
    return banks;
}

int run_case(const Geometry& g, Banks& banks, std::int32_t t, std::uint32_t seed) {
    const std::string label = "routed_moe E" + std::to_string(g.experts) + " K" +
                              std::to_string(g.top_k) + " [" + std::to_string(g.hidden) + "x" +
                              std::to_string(g.intermediate) + "] T=" + std::to_string(t);
    std::mt19937 rng(seed);
    std::vector<float> x(static_cast<std::size_t>(g.hidden) * t);
    std::vector<float> shared(x.size());
    fill_uniform(x, seed + 1, -1.0F, 1.0F);
    fill_uniform(shared, seed + 2, -0.5F, 0.5F);
    round_to_bf16(x);
    round_to_bf16(shared);

    // Distinct, well-separated winners so routing is unambiguous; the rest sit below them.
    std::vector<float> logits(static_cast<std::size_t>(g.experts + 1) * t);
    std::uniform_real_distribution<float> low(-3.0F, 1.0F);
    std::vector<std::int32_t> order(g.experts);
    for (std::int32_t c = 0; c < t; ++c) {
        float* row = logits.data() + static_cast<std::size_t>(c) * (g.experts + 1);
        for (std::int32_t e = 0; e < g.experts; ++e) row[e] = low(rng);
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        for (std::int32_t j = 0; j < g.top_k; ++j) row[order[j]] = 1.5F + 0.125F * j;
        row[g.experts] = low(rng);
    }
    round_to_bf16(logits);

    std::vector<double> expected(x.size());
    std::vector<double> act(g.intermediate);
    for (std::int32_t c = 0; c < t; ++c) {
        const float* row = logits.data() + static_cast<std::size_t>(c) * (g.experts + 1);
        std::vector<std::int32_t> ids(g.experts);
        std::iota(ids.begin(), ids.end(), 0);
        std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) { return row[a] > row[b]; });
        double denominator = 0.0;
        for (std::int32_t j = 0; j < g.top_k; ++j) denominator += std::exp(double(row[ids[j]]));
        const double shared_gate = 1.0 / (1.0 + std::exp(-double(row[g.experts])));
        for (std::int32_t h = 0; h < g.hidden; ++h) {
            expected[static_cast<std::size_t>(c) * g.hidden + h] =
                shared_gate * shared[static_cast<std::size_t>(c) * g.hidden + h];
        }
        const float* xc = x.data() + static_cast<std::size_t>(c) * g.hidden;
        for (std::int32_t j = 0; j < g.top_k; ++j) {
            const std::int32_t e = ids[j];
            const double weight  = std::exp(double(row[e])) / denominator;
            for (std::int32_t i = 0; i < g.intermediate; ++i) {
                double gv = 0.0, uv = 0.0;
                for (std::int32_t h = 0; h < g.hidden; ++h) {
                    gv += banks.gate.at(e, i, h) * xc[h];
                    uv += banks.up.at(e, i, h) * xc[h];
                }
                act[i] = gv / (1.0 + std::exp(-gv)) * uv;
            }
            for (std::int32_t h = 0; h < g.hidden; ++h) {
                double y = 0.0;
                for (std::int32_t i = 0; i < g.intermediate; ++i) {
                    y += banks.down.at(e, h, i) * act[i];
                }
                expected[static_cast<std::size_t>(c) * g.hidden + h] += weight * y;
            }
        }
    }

    auto dx      = to_device(bits(x));
    auto dshared = to_device(bits(shared));
    auto dlogits = to_device(bits(logits));
    GuardedDeviceBuffer dout(x.size() * sizeof(std::uint16_t));
    const std::size_t ws_bytes =
        ops::routed_moe_workspace_bytes(g.experts, g.top_k, g.hidden, g.intermediate, t);
    WorkspaceArena workspace(ws_bytes + 4096);
    Tensor tx(dx.p, DType::BF16, {g.hidden, t});
    Tensor tshared(dshared.p, DType::BF16, {g.hidden, t});
    Tensor tlogits(dlogits.p, DType::BF16, {g.experts + 1, t});
    Tensor tout(dout.data(), DType::BF16, {g.hidden, t});
    ops::routed_moe(tx, tlogits, g.top_k, device_bank(banks.gate, banks.gate_device),
                    device_bank(banks.up, banks.up_device),
                    device_bank(banks.down, banks.down_device), tshared, workspace, tout, nullptr);
    cuda_synchronize();
    int failures = verify_reduction(label, from_device_bf16(dout.data(), expected.size()), expected,
                                    routed_moe_criterion());
    failures += dout.verify_guards(label);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    {
        // Small geometry with many tokens per expert exercises the grouped walk.
        const Geometry small{64, 256, 128, 4};
        Banks banks = make_banks(small, 4401U);
        for (std::int32_t t : {1, 5, 300}) failures += run_case(small, banks, t, 4500U + t);
    }
    {
        // Qwen3.8-Flash-Next production geometry.
        const Geometry flash{512, 2560, 640, 10};
        Banks banks = make_banks(flash, 4402U);
        for (std::int32_t t : {1, 2, 9, 64}) failures += run_case(flash, banks, t, 4600U + t);
    }
    std::cout << (failures ? "FAIL" : "OK") << " routed_moe\n";
    return failures ? 1 : 0;
}
