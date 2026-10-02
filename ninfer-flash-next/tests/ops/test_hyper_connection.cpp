#include "ninfer/ops/hyper_connection.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr float kEps = 1.0e-6F;

// Normalized rows and stream means round once to BF16 after an FP32 reduction.
constexpr ReductionCriterion reduction_bf16_criterion() {
    return {/*relative_l2*/ 2.5e-3, /*gross_absolute*/ 1.0e-4,
            /*gross_relative_to_max_reference*/ 4.0e-3};
}

// Single transcendental per element, one BF16 or FP32 output rounding.
constexpr PointwiseCriterion gate_bf16_criterion() { return {/*absolute*/ 1.0e-6, 4.0e-3}; }

constexpr PointwiseCriterion gate_fp32_criterion() { return {/*absolute*/ 1.0e-6, 2.0e-6}; }

std::vector<std::uint16_t> bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) out[i] = f32_to_bf16(values[i]);
    return out;
}

GuardedDeviceBuffer upload_bf16(const std::vector<float>& values) {
    const auto encoded = bits(values);
    GuardedDeviceBuffer buffer(encoded.size() * sizeof(std::uint16_t));
    buffer.copy_from_host(encoded.data(), buffer.bytes());
    return buffer;
}

double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

std::vector<float> random_bf16(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    fill_uniform(v, seed, lo, hi);
    round_to_bf16(v);
    return v;
}

int grouped_rmsnorm_case(std::int32_t groups, std::int32_t d, std::int32_t t, std::uint32_t seed) {
    const std::string label = "grouped_rmsnorm [" + std::to_string(groups) + "x" +
                              std::to_string(d) + "," + std::to_string(t) + "]";
    const std::int32_t width = groups * d;
    const std::size_t count  = static_cast<std::size_t>(width) * t;
    auto x                   = random_bf16(count, seed, -4.0F, 4.0F);
    auto weight              = random_bf16(width, seed + 1, -0.75F, 0.75F);
    std::vector<double> expected(count);
    for (std::int32_t c = 0; c < t; ++c) {
        for (std::int32_t g = 0; g < groups; ++g) {
            const std::size_t base = static_cast<std::size_t>(c) * width + g * d;
            double sum             = 0.0;
            for (std::int32_t i = 0; i < d; ++i) sum += double(x[base + i]) * x[base + i];
            const double inv = 1.0 / std::sqrt(sum / d + kEps);
            for (std::int32_t i = 0; i < d; ++i) {
                expected[base + i] = x[base + i] * inv * (1.0 + double(weight[g * d + i]));
            }
        }
    }
    auto dx = upload_bf16(x);
    auto dw = upload_bf16(weight);
    GuardedDeviceBuffer dout(count * sizeof(std::uint16_t));
    Tensor tx(dx.data(), DType::BF16, {width, t});
    Tensor tw(dw.data(), DType::BF16, {width});
    Tensor to(dout.data(), DType::BF16, {width, t});
    ops::grouped_rmsnorm(tx, tw, groups, kEps, to, nullptr);
    cuda_synchronize();
    int failures = verify_reduction(label, from_device_bf16(dout.data(), count), expected,
                                    reduction_bf16_criterion());
    failures += dout.verify_guards(label);
    return failures;
}

int mix_case(std::int32_t streams, std::int32_t h, std::int32_t t, std::uint32_t seed) {
    const std::string label = "hyper_connection_mix [" + std::to_string(streams) + "x" +
                              std::to_string(h) + "," + std::to_string(t) + "]";
    const std::int32_t width = streams * h;
    const std::size_t count  = static_cast<std::size_t>(width) * t;
    auto normalized          = random_bf16(count, seed, -3.0F, 3.0F);
    auto logits              = random_bf16(count, seed + 1, -6.0F, 6.0F);
    std::vector<double> expected(static_cast<std::size_t>(h) * t);
    for (std::int32_t c = 0; c < t; ++c) {
        for (std::int32_t i = 0; i < h; ++i) {
            double acc = 0.0;
            for (std::int32_t s = 0; s < streams; ++s) {
                const std::size_t at = static_cast<std::size_t>(c) * width + s * h + i;
                acc += sigmoid(logits[at]) * normalized[at];
            }
            expected[static_cast<std::size_t>(c) * h + i] = acc / streams;
        }
    }
    auto dn = upload_bf16(normalized);
    auto dl = upload_bf16(logits);
    GuardedDeviceBuffer dout(expected.size() * sizeof(std::uint16_t));
    Tensor tn(dn.data(), DType::BF16, {width, t});
    Tensor tl(dl.data(), DType::BF16, {width, t});
    Tensor to(dout.data(), DType::BF16, {h, t});
    ops::hyper_connection_mix(tn, tl, streams, to, nullptr);
    cuda_synchronize();
    int failures = verify_reduction(label, from_device_bf16(dout.data(), expected.size()),
                                    expected, reduction_bf16_criterion());
    failures += dout.verify_guards(label);
    return failures;
}

int gates_case(std::int32_t streams, std::int32_t r, std::int32_t t, std::uint32_t seed) {
    const std::string label = "hyper_connection_gates [" + std::to_string(r) + "/" +
                              std::to_string(streams) + "," + std::to_string(t) + "]";
    auto down   = random_bf16(static_cast<std::size_t>(r) * t, seed, -20.0F, 20.0F);
    auto logits = random_bf16(static_cast<std::size_t>(streams) * t, seed + 1, -20.0F, 20.0F);
    std::vector<double> hidden(down.size()), inject(logits.size());
    for (std::size_t i = 0; i < down.size(); ++i) {
        const double v = double(down[i]) / streams;
        hidden[i]      = v * sigmoid(v);
    }
    for (std::size_t i = 0; i < logits.size(); ++i) {
        inject[i] = 2.0 * sigmoid(double(logits[i]) / streams);
    }
    auto dd = upload_bf16(down);
    auto dl = upload_bf16(logits);
    GuardedDeviceBuffer dh(down.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer di(logits.size() * sizeof(float));
    Tensor td(dd.data(), DType::BF16, {r, t});
    Tensor tl(dl.data(), DType::BF16, {streams, t});
    Tensor th(dh.data(), DType::BF16, {r, t});
    Tensor ti(di.data(), DType::FP32, {streams, t});
    ops::hyper_connection_gates(td, tl, streams, th, ti, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(label + " hidden", from_device_bf16(dh.data(), hidden.size()),
                                    hidden, gate_bf16_criterion());
    failures += verify_pointwise(label + " inject", [&] { const auto raw = from_device<float>(di.data(), inject.size()); return std::vector<double>(raw.begin(), raw.end()); }(),
                                 inject, gate_fp32_criterion());
    failures += dh.verify_guards(label) + di.verify_guards(label);
    return failures;
}

int inject_case(std::int32_t streams, std::int32_t h, std::int32_t t, std::uint32_t seed) {
    const std::string label = "hyper_connection_inject [" + std::to_string(streams) + "x" +
                              std::to_string(h) + "," + std::to_string(t) + "]";
    const std::int32_t width = streams * h;
    auto state               = random_bf16(static_cast<std::size_t>(width) * t, seed, -8.0F, 8.0F);
    auto y                   = random_bf16(static_cast<std::size_t>(h) * t, seed + 1, -8.0F, 8.0F);
    std::vector<float> inject(static_cast<std::size_t>(streams) * t);
    fill_uniform(inject, seed + 2, 0.0F, 2.0F);
    std::vector<double> expected(state.size());
    for (std::int32_t c = 0; c < t; ++c) {
        for (std::int32_t s = 0; s < streams; ++s) {
            for (std::int32_t i = 0; i < h; ++i) {
                const std::size_t at = static_cast<std::size_t>(c) * width + s * h + i;
                expected[at]         = double(state[at]) +
                               double(inject[c * streams + s]) * y[static_cast<std::size_t>(c) * h + i];
            }
        }
    }
    auto dstate = upload_bf16(state);
    auto dy     = upload_bf16(y);
    GuardedDeviceBuffer di(inject.size() * sizeof(float));
    di.copy_from_host(inject.data(), di.bytes());
    Tensor tstate(dstate.data(), DType::BF16, {width, t});
    Tensor ty(dy.data(), DType::BF16, {h, t});
    Tensor ti(di.data(), DType::FP32, {streams, t});
    ops::hyper_connection_inject(ty, ti, tstate, nullptr);
    cuda_synchronize();
    int failures = verify_reduction(label, from_device_bf16(dstate.data(), expected.size()),
                                    expected, reduction_bf16_criterion());
    failures += dstate.verify_guards(label);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures = 0;
    std::uint32_t seed = 9100;
    for (std::int32_t t : {1, 3, 8, 64, 257}) {
        failures += grouped_rmsnorm_case(4, 2560, t, seed += 7);
        failures += mix_case(4, 2560, t, seed += 7);
        failures += gates_case(4, 320, t, seed += 7);
        failures += inject_case(4, 2560, t, seed += 7);
    }
    failures += grouped_rmsnorm_case(1, 128, 33, seed += 7);
    failures += grouped_rmsnorm_case(4, 2560, 2048, seed += 7);
    std::cout << (failures ? "FAIL" : "OK") << " hyper_connection\n";
    return failures ? 1 : 0;
}
