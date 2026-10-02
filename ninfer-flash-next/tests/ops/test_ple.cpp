#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/ple.h"
#include "ops/op_tester.h"

#include <array>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr PointwiseCriterion dequantize_criterion() { return {/*absolute*/ 0.0, 4.0e-3}; }

// One FP32 dot of H products feeds a damped sigmoid, then one BF16 rounding.
constexpr PointwiseCriterion gate_criterion() { return {/*absolute*/ 1.0e-5, 5.0e-3}; }

// residual + gated + SiLU(4-tap conv): sums of BF16 values rounded once.
constexpr ReductionCriterion conv_criterion() {
    return {/*relative_l2*/ 2.5e-3, /*gross_absolute*/ 1.0e-4,
            /*gross_relative_to_max_reference*/ 4.0e-3};
}

std::vector<float> random_bf16(std::size_t n, std::uint32_t seed, float lo, float hi) {
    std::vector<float> v(n);
    fill_uniform(v, seed, lo, hi);
    round_to_bf16(v);
    return v;
}

std::vector<std::uint16_t> bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> out(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) out[i] = f32_to_bf16(values[i]);
    return out;
}

GuardedDeviceBuffer upload(const std::vector<float>& values) {
    const auto encoded = bits(values);
    GuardedDeviceBuffer buffer(encoded.size() * sizeof(std::uint16_t));
    buffer.copy_from_host(encoded.data(), buffer.bytes());
    return buffer;
}

double silu(double x) { return x / (1.0 + std::exp(-x)); }

int dequantize_case(std::int32_t e, std::int32_t t, std::uint32_t seed) {
    const std::string label = "ple_dequantize [" + std::to_string(e) + "," + std::to_string(t) + "]";
    const std::size_t count = static_cast<std::size_t>(e) * t;
    std::vector<std::uint8_t> codes(count);
    std::vector<double> expected(count);
    const float scale = 0.0068359375F; // exactly representable in BF16
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < count; ++i) {
        state         = state * 1664525U + 1013904223U;
        std::uint8_t c = static_cast<std::uint8_t>(state >> 24);
        if ((c & 0x7F) == 0x7F) c ^= 0x01; // avoid NaN encodings
        codes[i] = c;
        __nv_fp8_e4m3 fp8;
        fp8.__x     = c;
        expected[i] = static_cast<double>(static_cast<float>(fp8)) * scale;
    }
    GuardedDeviceBuffer dcodes(count);
    dcodes.copy_from_host(codes.data(), count);
    auto dscale = upload({scale});
    GuardedDeviceBuffer dout(count * sizeof(std::uint16_t));
    Tensor tc(dcodes.data(), DType::FP8_E4M3FN, {e, t});
    Tensor ts(dscale.data(), DType::BF16, {1});
    Tensor to(dout.data(), DType::BF16, {e, t});
    ops::ple_dequantize(tc, ts, to, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(label, from_device_bf16(dout.data(), count), expected,
                                    dequantize_criterion());
    return failures + dout.verify_guards(label);
}

int gate_case(std::int32_t streams, std::int32_t h, std::int32_t t, std::uint32_t seed) {
    const std::string label = "ple_gate [" + std::to_string(streams) + "x" + std::to_string(h) +
                              "," + std::to_string(t) + "]";
    const std::int32_t width = streams * h;
    auto key                 = random_bf16(static_cast<std::size_t>(width) * t, seed, -2.0F, 2.0F);
    auto query = random_bf16(static_cast<std::size_t>(width) * t, seed + 1, -2.0F, 2.0F);
    auto value = random_bf16(static_cast<std::size_t>(h) * t, seed + 2, -3.0F, 3.0F);
    std::vector<double> expected(key.size());
    for (std::int32_t c = 0; c < t; ++c) {
        for (std::int32_t s = 0; s < streams; ++s) {
            const std::size_t base = static_cast<std::size_t>(c) * width + s * h;
            double dot             = 0.0;
            for (std::int32_t i = 0; i < h; ++i) dot += double(key[base + i]) * query[base + i];
            double gamma = dot / std::sqrt(double(h));
            gamma        = std::copysign(std::sqrt(std::max(std::abs(gamma), 1.0e-6)), gamma);
            const double gate = 1.0 / (1.0 + std::exp(-gamma));
            for (std::int32_t i = 0; i < h; ++i) {
                expected[base + i] = gate * value[static_cast<std::size_t>(c) * h + i];
            }
        }
    }
    auto dk = upload(key);
    auto dq = upload(query);
    auto dv = upload(value);
    GuardedDeviceBuffer dout(expected.size() * sizeof(std::uint16_t));
    Tensor tk(dk.data(), DType::BF16, {width, t});
    Tensor tq(dq.data(), DType::BF16, {width, t});
    Tensor tv(dv.data(), DType::BF16, {h, t});
    Tensor to(dout.data(), DType::BF16, {width, t});
    ops::ple_gate(tk, tq, tv, streams, to, nullptr);
    cuda_synchronize();
    int failures = verify_pointwise(label, from_device_bf16(dout.data(), expected.size()),
                                    expected, gate_criterion());
    return failures + dout.verify_guards(label);
}

struct ConvReference {
    std::vector<double> residual;
    std::vector<std::vector<float>> windows; // window after each column, [9*C]
};

// One sequence: initial window [9*C] (k-major), columns [T*C].
ConvReference conv_reference(const std::vector<float>& residual, const std::vector<float>& gated,
                             const std::vector<float>& normalized,
                             const std::vector<float>& weight, const std::vector<float>& window,
                             std::int32_t c_count, std::int32_t t) {
    ConvReference out{std::vector<double>(residual.size()), {}};
    auto u = [&](std::int32_t c, std::int32_t at) -> double {
        return at >= 0 ? normalized[static_cast<std::size_t>(at) * c_count + c]
                       : window[static_cast<std::size_t>(ops::kPleConvState + at) * c_count + c];
    };
    for (std::int32_t j = 0; j < t; ++j) {
        for (std::int32_t c = 0; c < c_count; ++c) {
            double conv = 0.0;
            for (std::int32_t k = 0; k < ops::kPleConvTaps; ++k) {
                conv += double(weight[static_cast<std::size_t>(k) * c_count + c]) *
                        u(c, j - (ops::kPleConvTaps - 1 - k) * ops::kPleConvDilation);
            }
            const std::size_t at = static_cast<std::size_t>(j) * c_count + c;
            out.residual[at]     = double(residual[at]) + gated[at] + silu(conv);
        }
        std::vector<float> next(window.size());
        for (std::int32_t k = 0; k < ops::kPleConvState; ++k) {
            for (std::int32_t c = 0; c < c_count; ++c) {
                next[static_cast<std::size_t>(k) * c_count + c] =
                    static_cast<float>(u(c, j + 1 - ops::kPleConvState + k));
            }
        }
        out.windows.push_back(std::move(next));
    }
    return out;
}

int conv_sequence_case(std::int32_t c, std::int32_t t, bool in_place, std::uint32_t seed) {
    const std::string label = std::string("ple_conv_residual ") + (in_place ? "in-place " : "") +
                              "[" + std::to_string(c) + "," + std::to_string(t) + "]";
    const std::size_t count = static_cast<std::size_t>(c) * t;
    auto residual           = random_bf16(count, seed, -4.0F, 4.0F);
    auto gated              = random_bf16(count, seed + 1, -2.0F, 2.0F);
    auto normalized         = random_bf16(count, seed + 2, -3.0F, 3.0F);
    auto weight = random_bf16(static_cast<std::size_t>(c) * ops::kPleConvTaps, seed + 3, -1.0F, 1.0F);
    auto window = random_bf16(static_cast<std::size_t>(c) * ops::kPleConvState, seed + 4, -3.0F, 3.0F);
    const auto reference = conv_reference(residual, gated, normalized, weight, window, c, t);

    auto dres  = upload(residual);
    auto dg    = upload(gated);
    auto dn    = upload(normalized);
    auto dw    = upload(weight);
    auto din   = upload(window);
    GuardedDeviceBuffer dout_state(window.size() * sizeof(std::uint16_t));
    Tensor tres(dres.data(), DType::BF16, {c, t});
    Tensor tg(dg.data(), DType::BF16, {c, t});
    Tensor tn(dn.data(), DType::BF16, {c, t});
    Tensor tw(dw.data(), DType::BF16, {c, ops::kPleConvTaps});
    Tensor tin(din.data(), DType::BF16, {c, ops::kPleConvState});
    Tensor tout(in_place ? din.data() : dout_state.data(), DType::BF16, {c, ops::kPleConvState});
    ops::ple_conv_residual(tg, tn, tw, tin, tout, tres, nullptr);
    cuda_synchronize();
    int failures = verify_reduction(label, from_device_bf16(dres.data(), count), reference.residual,
                                    conv_criterion());
    failures += verify_exact((label + " state").c_str(),
                             from_device<std::uint16_t>(tout.data, window.size()),
                             bits(reference.windows.back()));
    failures += dres.verify_guards(label) + din.verify_guards(label) +
                dout_state.verify_guards(label);
    return failures;
}

int conv_snapshot_case(std::int32_t c, std::int32_t w, std::uint32_t seed) {
    const std::string label = "ple_conv_residual_snapshot [" + std::to_string(c) + "," +
                              std::to_string(w) + ",3]";
    constexpr std::int32_t b = 3;
    // Row 0 starts in slot 0 inside its own reservation [0,w); rows 1 and 2 start just before
    // their reservations [w+1,2w+1) and [2w+2,3w+2).
    const std::vector<std::int32_t> initial{0, w, 2 * w + 1};
    const std::vector<std::int32_t> base{0, w + 1, 2 * w + 2};
    const std::vector<std::int32_t> valid{w, std::max(1, w - 1), 1};
    const std::int32_t slots = 3 * w + 2;
    const std::size_t column = static_cast<std::size_t>(c) * w;
    const std::size_t count  = column * b;
    const std::size_t slot   = static_cast<std::size_t>(c) * ops::kPleConvState;
    auto residual            = random_bf16(count, seed, -4.0F, 4.0F);
    auto gated               = random_bf16(count, seed + 1, -2.0F, 2.0F);
    auto normalized          = random_bf16(count, seed + 2, -3.0F, 3.0F);
    auto weight = random_bf16(static_cast<std::size_t>(c) * ops::kPleConvTaps, seed + 3, -1.0F, 1.0F);
    auto states = random_bf16(slot * slots, seed + 4, -3.0F, 3.0F);

    std::vector<double> expected_residual(residual.begin(), residual.end());
    auto expected_states = bits(states);
    for (std::int32_t row = 0; row < b; ++row) {
        const auto slice = [&](const std::vector<float>& v) {
            return std::vector<float>(v.begin() + row * column, v.begin() + (row + 1) * column);
        };
        const std::vector<float> window(states.begin() + initial[row] * slot,
                                        states.begin() + (initial[row] + 1) * slot);
        const auto reference = conv_reference(slice(residual), slice(gated), slice(normalized),
                                              weight, window, c, w);
        for (std::int32_t j = 0; j < valid[row]; ++j) {
            for (std::int32_t ch = 0; ch < c; ++ch) {
                const std::size_t local = static_cast<std::size_t>(j) * c + ch;
                expected_residual[row * column + local] = reference.residual[local];
            }
            const auto encoded = bits(reference.windows[j]);
            std::copy(encoded.begin(), encoded.end(),
                      expected_states.begin() + (base[row] + j) * slot);
        }
    }

    auto dres    = upload(residual);
    auto dg      = upload(gated);
    auto dn      = upload(normalized);
    auto dw      = upload(weight);
    auto dstates = upload(states);
    auto di      = to_device_i32(initial);
    auto dbase   = to_device_i32(base);
    auto dvalid  = to_device_i32(valid);
    Tensor tres(dres.data(), DType::BF16, {c, w, b});
    Tensor tg(dg.data(), DType::BF16, {c, w, b});
    Tensor tn(dn.data(), DType::BF16, {c, w, b});
    Tensor tw(dw.data(), DType::BF16, {c, ops::kPleConvTaps});
    Tensor tstates(dstates.data(), DType::BF16, {c, ops::kPleConvState, slots});
    Tensor ti(di.p, DType::I32, {b});
    Tensor tb(dbase.p, DType::I32, {b});
    Tensor tv(dvalid.p, DType::I32, {b});
    ops::ple_conv_residual_snapshot(tg, tn, tw, tstates, tv, ti, tb, tres, nullptr);
    cuda_synchronize();
    int failures = verify_reduction(label, from_device_bf16(dres.data(), count), expected_residual,
                                    conv_criterion());
    failures += verify_exact((label + " states").c_str(),
                             from_device<std::uint16_t>(dstates.data(), states.size()),
                             expected_states);
    failures += dres.verify_guards(label) + dstates.verify_guards(label);
    return failures;
}

// The record form must give the snapshot residual bit for bit, and folding any committed prefix
// must reproduce the snapshot window after that column. A second GDN-conv check compares
// causal_conv1d_silu_record against causal_conv1d_silu_snapshot.
int record_fold_case(std::int32_t c, std::int32_t w, std::uint32_t seed) {
    const std::string label = "ple_conv_residual_record [" + std::to_string(c) + "," +
                              std::to_string(w) + ",2]";
    constexpr std::int32_t b = 2;
    const std::vector<std::int32_t> initial{0, 1};
    const std::vector<std::int32_t> valid{w, std::max(1, w - 1)};
    // Snapshot reservations live after the two initial slots; fold destinations after those.
    const std::vector<std::int32_t> base{2, 2 + w};
    const std::int32_t fold_base = 2 + 2 * w;
    const std::int32_t slots     = fold_base + 2;
    const std::size_t count      = static_cast<std::size_t>(c) * w * b;
    const std::size_t slot       = static_cast<std::size_t>(c) * ops::kPleConvState;
    auto residual   = random_bf16(count, seed, -4.0F, 4.0F);
    auto gated      = random_bf16(count, seed + 1, -2.0F, 2.0F);
    auto normalized = random_bf16(count, seed + 2, -3.0F, 3.0F);
    auto weight = random_bf16(static_cast<std::size_t>(c) * ops::kPleConvTaps, seed + 3, -1.0F, 1.0F);
    auto states = random_bf16(slot * slots, seed + 4, -3.0F, 3.0F);

    auto dres_snap = upload(residual);
    auto dres_rec  = upload(residual);
    auto dg        = upload(gated);
    auto dn        = upload(normalized);
    auto dw        = upload(weight);
    auto dstates   = upload(states);
    auto drecord   = upload(std::vector<float>(count, 0.0F));
    auto di        = to_device_i32(initial);
    auto dbase     = to_device_i32(base);
    auto dvalid    = to_device_i32(valid);
    Tensor tsnap(dres_snap.data(), DType::BF16, {c, w, b});
    Tensor trec(dres_rec.data(), DType::BF16, {c, w, b});
    Tensor tg(dg.data(), DType::BF16, {c, w, b});
    Tensor tn(dn.data(), DType::BF16, {c, w, b});
    Tensor tw(dw.data(), DType::BF16, {c, ops::kPleConvTaps});
    Tensor tstates(dstates.data(), DType::BF16, {c, ops::kPleConvState, slots});
    Tensor trecord(drecord.data(), DType::BF16, {c, w, b});
    Tensor ti(di.p, DType::I32, {b});
    Tensor tb(dbase.p, DType::I32, {b});
    Tensor tv(dvalid.p, DType::I32, {b});
    ops::ple_conv_residual_record(tg, tn, tw, tstates, tv, ti, trecord, trec, nullptr);
    ops::ple_conv_residual_snapshot(tg, tn, tw, tstates, tv, ti, tb, tsnap, nullptr);
    cuda_synchronize();
    int failures = verify_exact((label + " residual").c_str(),
                                from_device<std::uint16_t>(dres_rec.data(), count),
                                from_device<std::uint16_t>(dres_snap.data(), count));
    failures += verify_exact((label + " record").c_str(),
                             from_device<std::uint16_t>(drecord.data(), count),
                             from_device<std::uint16_t>(dn.data(), count));
    for (std::int32_t commit = 0; commit <= valid[0]; ++commit) {
        const std::array<ops::GdnReplayFoldRow, 2> rows{
            ops::GdnReplayFoldRow{initial[0], fold_base, commit},
            ops::GdnReplayFoldRow{initial[1], fold_base + 1, std::min(commit, valid[1])}};
        ops::ple_conv_replay_fold(tstates, trecord, rows, nullptr);
        cuda_synchronize();
        const auto all = from_device<std::uint16_t>(dstates.data(), states.size());
        for (std::int32_t r = 0; r < b; ++r) {
            const std::int32_t used = rows[r].commit_columns;
            const std::int32_t from = used == 0 ? initial[r] : base[r] + used - 1;
            const std::vector<std::uint16_t> want(all.begin() + from * slot, all.begin() + (from + 1) * slot);
            const std::vector<std::uint16_t> got(all.begin() + (fold_base + r) * slot,
                                                 all.begin() + (fold_base + r + 1) * slot);
            failures += verify_exact((label + " fold row " + std::to_string(r) + " commit " +
                                      std::to_string(used)).c_str(), got, want);
        }
    }

    // GDN conv: record vs snapshot over the same pool shape (window 3).
    const std::size_t cslot = static_cast<std::size_t>(c) * 3;
    auto cstates  = random_bf16(cslot * slots, seed + 5, -3.0F, 3.0F);
    auto x        = random_bf16(count, seed + 6, -3.0F, 3.0F);
    auto cweight  = random_bf16(static_cast<std::size_t>(c) * 4, seed + 7, -1.0F, 1.0F);
    auto dcs      = upload(cstates);
    auto dx       = upload(x);
    auto dcw      = upload(cweight);
    auto dout_rec = upload(std::vector<float>(count, 1.0F));
    auto dout_snp = upload(std::vector<float>(count, 1.0F));
    auto dcrec    = upload(std::vector<float>(count, 0.0F));
    Tensor tcs(dcs.data(), DType::BF16, {c, 3, slots});
    Tensor tx(dx.data(), DType::BF16, {c, w, b});
    Tensor tcw(dcw.data(), DType::BF16, {c, 4});
    Tensor torec(dout_rec.data(), DType::BF16, {c, w, b});
    Tensor tosnp(dout_snp.data(), DType::BF16, {c, w, b});
    Tensor tcrec(dcrec.data(), DType::BF16, {c, w, b});
    ops::causal_conv1d_silu_record(tx, tcw, tcs, tv, ti, tcrec, torec, nullptr);
    ops::causal_conv1d_silu_snapshot(tx, tcw, tcs, tv, ti, tb, tosnp, nullptr);
    cuda_synchronize();
    const auto conv_rec  = from_device_bf16(dout_rec.data(), count);
    const auto conv_snap = from_device_bf16(dout_snp.data(), count);
    std::vector<double> want(conv_snap.begin(), conv_snap.end());
    failures += verify_reduction(label + " gdn conv", conv_rec, want, conv_criterion());
    failures += verify_exact((label + " gdn record").c_str(),
                             from_device<std::uint16_t>(dcrec.data(), count),
                             from_device<std::uint16_t>(dx.data(), count));
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    int failures       = 0;
    std::uint32_t seed = 9300;
    for (std::int32_t t : {1, 7, 64}) failures += dequantize_case(2560, t, seed += 11);
    for (std::int32_t t : {1, 3, 64}) failures += gate_case(4, 2560, t, seed += 11);
    for (std::int32_t t : {1, 2, 5, 9, 10, 64, 300}) {
        failures += conv_sequence_case(10240, t, false, seed += 11);
        failures += conv_sequence_case(10240, t, true, seed += 11);
    }
    for (std::int32_t w : {1, 2, 4}) failures += conv_snapshot_case(10240, w, seed += 11);
    for (std::int32_t w : {1, 3, 6}) failures += record_fold_case(10240, w, seed += 11);
    std::cout << (failures ? "FAIL" : "OK") << " ple\n";
    return failures ? 1 : 0;
}
