#include "ninfer/ops/qsa.h"
#include "ops/op_tester.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kDi = 128, kHi = 4, kRatio = 4, kBudget = 512, kRotary = 64;
constexpr float kTheta = 1.0e7F, kEps = 1.0e-6F;
constexpr int kPage = kPagedKVPageSize;

float bfr(double x) { return bf16_to_f32(f32_to_bf16(static_cast<float>(x))); }

std::vector<std::uint16_t> bits(const std::vector<float>& v) {
    std::vector<std::uint16_t> out(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) out[i] = f32_to_bf16(v[i]);
    return out;
}

// Mirror of the contract's rounding points: BF16 norm output, BF16 cos/sin and BF16 products.
std::vector<float> norm_rope(const float* x, const std::vector<float>& w, std::int64_t pos) {
    double sum = 0;
    for (int d = 0; d < kDi; ++d) sum += double(x[d]) * x[d];
    const double inv = 1.0 / std::sqrt(sum / kDi + kEps);
    std::vector<float> n(kDi), out(kDi);
    for (int d = 0; d < kDi; ++d) n[d] = bfr(x[d] * inv * (1.0 + w[d]));
    for (int d = 0; d < kDi; ++d) {
        if (d >= kRotary) {
            out[d] = n[d];
            continue;
        }
        const int half = kRotary / 2, pair = d < half ? d : d - half;
        const float inv_freq = 1.0f / std::pow(kTheta, float(2 * pair) / kRotary);
        const float angle    = float(pos) * inv_freq;
        const float c = bfr(std::cos(angle)), s = bfr(std::sin(angle));
        const float rotated = d < half ? -n[d + half] : n[d - half];
        out[d]              = bfr(bfr(double(n[d]) * c) + bfr(double(rotated) * s));
    }
    return out;
}

struct Cache {
    int pages = 0, logical = 0, rows = 0;
    std::vector<int> table; // [logical, rows], row-major by row
    int slot(int row, std::int64_t pos) const {
        return table[row * logical + int(pos / kPage)] * kPage + int(pos % kPage);
    }
};

Cache make_cache(int rows, int logical, std::mt19937& rng) {
    Cache c;
    c.rows = rows, c.logical = logical, c.pages = rows * logical;
    std::vector<int> ids(c.pages);
    std::iota(ids.begin(), ids.end(), 0);
    std::shuffle(ids.begin(), ids.end(), rng);
    c.table = ids;
    return c;
}

int prepare_query_case(std::mt19937& rng) {
    const int width = 37, seqs = 2;
    std::vector<float> q(std::size_t(kDi) * kHi * width * seqs), w(kDi);
    std::vector<int> pos(width * seqs);
    std::uniform_real_distribution<float> u(-3, 3);
    for (auto& v : q) v = bfr(u(rng));
    for (auto& v : w) v = bfr(u(rng) * 0.1f);
    for (int i = 0; i < width * seqs; ++i) pos[i] = 1000 + 97 * i;
    std::vector<double> expected(q.size());
    for (std::size_t v = 0; v < q.size() / kDi; ++v) {
        const auto r = norm_rope(&q[v * kDi], w, pos[v / kHi]);
        for (int d = 0; d < kDi; ++d) expected[v * kDi + d] = r[d];
    }
    auto dq = to_device(bits(q)), dw = to_device(bits(w)), dp = to_device_i32(pos);
    GuardedDeviceBuffer dout(q.size() * 2);
    Tensor tq(dq.p, DType::BF16, {kDi, kHi, width, seqs}), tw(dw.p, DType::BF16, {kDi});
    Tensor tp(dp.p, DType::I32, {width, seqs}), to(dout.data(), DType::BF16, {kDi, kHi, width, seqs});
    ops::qsa_prepare_query(tq, tw, tp, kRotary, kTheta, kEps, to, nullptr);
    cuda_synchronize();
    return verify_pointwise("qsa_prepare_query", from_device_bf16(dout.data(), q.size()), expected,
                            PointwiseCriterion{2.0e-3, 1.6e-2}) +
           dout.verify_guards("qsa_prepare_query");
}

// Reference selection for one column; returns ascending token list and the scores used.
std::vector<int> select_reference(const std::vector<float>& query_col, // [Di,Hi]
                                  const std::vector<std::vector<float>>& pooled, std::int64_t pos,
                                  std::vector<double>* scores_out, double* threshold) {
    const int visible = int(pos + 1), nb = visible / kRatio;
    std::vector<int> tokens;
    if (nb <= kBudget) {
        for (int t = 0; t < visible; ++t) tokens.push_back(t);
        return tokens;
    }
    std::vector<double> score(nb);
    for (int b = 0; b < nb; ++b) {
        double s = 0;
        for (int h = 0; h < kHi; ++h) {
            double dot = 0;
            for (int d = 0; d < kDi; ++d) dot += double(query_col[h * kDi + d]) * pooled[b][d];
            s += std::max(dot, 0.0);
        }
        score[b] = s / std::sqrt(double(kDi));
    }
    std::vector<int> order(nb);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return score[a] > score[b]; });
    std::vector<int> chosen(order.begin(), order.begin() + kBudget);
    std::sort(chosen.begin(), chosen.end());
    for (int b : chosen)
        for (int j = 0; j < kRatio; ++j) tokens.push_back(b * kRatio + j);
    for (int t = nb * kRatio; t < visible; ++t) tokens.push_back(t);
    *scores_out = score;
    *threshold  = score[order[kBudget - 1]];
    return tokens;
}

int select_case(const std::string& label, int width, const std::vector<int>& last_positions,
                std::mt19937& rng) {
    const int seqs        = int(last_positions.size());
    const int max_visible = *std::max_element(last_positions.begin(), last_positions.end()) + 1;
    const int logical     = (max_visible + kPage - 1) / kPage + 1;
    Cache cache           = make_cache(seqs, logical, rng);
    std::uniform_real_distribution<float> u(-2, 2);
    std::vector<float> norm(kDi);
    for (auto& v : norm) v = bfr(u(rng) * 0.1f);
    std::vector<std::vector<float>> raw(seqs);
    for (int s = 0; s < seqs; ++s) {
        raw[s].resize(std::size_t(last_positions[s] + 1) * kDi);
        for (auto& v : raw[s]) v = bfr(u(rng));
    }
    GuardedDeviceBuffer dpages(std::size_t(cache.pages) * kPage * kDi * 2);
    dpages.fill(0);
    auto dtable = to_device_i32(cache.table);
    Tensor tpages(dpages.data(), DType::BF16, {kDi, kPage, 1, cache.pages});
    Tensor ttable(dtable.p, DType::I32, {logical, seqs});
    for (int s = 0; s < seqs; ++s) {
        for (int start = 0; start <= last_positions[s]; start += 512) {
            const int n = std::min(512, last_positions[s] + 1 - start);
            std::vector<float> k(raw[s].begin() + std::size_t(start) * kDi,
                                 raw[s].begin() + std::size_t(start + n) * kDi);
            std::vector<int> p(n);
            std::iota(p.begin(), p.end(), start);
            std::vector<int> r{s};
            auto dk = to_device(bits(k));
            auto dp = to_device_i32(p);
            auto dr = to_device_i32(r);
            Tensor tk(dk.p, DType::BF16, {kDi, n, 1}), tp(dp.p, DType::I32, {n, 1});
            Tensor tr(dr.p, DType::I32, {1});
            ops::qsa_index_append(tk, tp, tr, tpages, ttable, nullptr);
        }
    }
    std::vector<float> query(std::size_t(kDi) * kHi * width * seqs);
    for (auto& v : query) v = bfr(u(rng));
    std::vector<int> positions(width * seqs), rows(seqs);
    std::iota(rows.begin(), rows.end(), 0);
    for (int s = 0; s < seqs; ++s)
        for (int w = 0; w < width; ++w) positions[s * width + w] = last_positions[s] - (width - 1 - w);
    auto dq = to_device(bits(query)), dpos = to_device_i32(positions), drows = to_device_i32(rows);
    auto dnorm = to_device(bits(norm));
    const ops::QsaSelectGeometry geometry{};
    const int max_selected = geometry.max_selected();
    GuardedDeviceBuffer dsel(std::size_t(max_selected) * width * seqs * 4);
    GuardedDeviceBuffer dcount(std::size_t(width) * seqs * 4);
    WorkspaceArena ws(ops::qsa_select_workspace_bytes(geometry, kDi, seqs, width * seqs, max_visible) + 4096);
    Tensor tq(dq.p, DType::BF16, {kDi, kHi, width, seqs}), tpos(dpos.p, DType::I32, {width, seqs});
    Tensor trows(drows.p, DType::I32, {seqs}), tnorm(dnorm.p, DType::BF16, {kDi});
    Tensor tsel(dsel.data(), DType::I32, {max_selected, width, seqs});
    Tensor tcount(dcount.data(), DType::I32, {width, seqs});
    ops::qsa_select(tq, tpos, trows, tpages, ttable, tnorm, geometry, max_visible, ws, tsel, tcount,
                    nullptr);
    cuda_synchronize();
    const auto got    = from_device<std::int32_t>(dsel.data(), std::size_t(max_selected) * width * seqs);
    const auto counts = from_device<std::int32_t>(dcount.data(), std::size_t(width) * seqs);

    int failures = 0, borderline = 0;
    for (int s = 0; s < seqs; ++s) {
        // Pooled keys for this sequence (query independent).
        const int nb_max = (last_positions[s] + 1) / kRatio;
        std::vector<std::vector<float>> pooled(nb_max);
        for (int b = 0; b < nb_max; ++b) {
            std::vector<float> mean(kDi);
            for (int d = 0; d < kDi; ++d) {
                double sum = 0;
                for (int j = 0; j < kRatio; ++j) sum += raw[s][std::size_t(b * kRatio + j) * kDi + d];
                mean[d] = bfr(sum / kRatio);
            }
            pooled[b] = norm_rope(mean.data(), norm, std::int64_t(b) * kRatio);
        }
        for (int w = 0; w < width; ++w) {
            const int column = s * width + w;
            std::vector<float> qcol(query.begin() + std::size_t(column) * kDi * kHi,
                                    query.begin() + std::size_t(column + 1) * kDi * kHi);
            std::vector<double> scores;
            double threshold = 0;
            const auto expected = select_reference(qcol, pooled, positions[column], &scores, &threshold);
            std::vector<int> actual(got.begin() + std::size_t(column) * max_selected,
                                    got.begin() + std::size_t(column) * max_selected + counts[column]);
            if (actual == expected) continue;
            // Accept only swaps of blocks whose scores sit at the threshold within FP32 rounding.
            std::set<int> a(actual.begin(), actual.end()), e(expected.begin(), expected.end());
            bool explained = actual.size() == expected.size() && std::is_sorted(actual.begin(), actual.end());
            for (int t : a) if (!e.count(t)) explained = explained && std::abs(scores[t / kRatio] - threshold) <= 1e-5 * threshold;
            for (int t : e) if (!a.count(t)) explained = explained && std::abs(scores[t / kRatio] - threshold) <= 1e-5 * threshold;
            if (explained) { ++borderline; continue; }
            ++failures;
            if (failures <= 3) {
                std::cerr << label << ": column " << column << " pos " << positions[column]
                          << " count " << counts[column] << " expected " << expected.size() << '\n';
            }
        }
    }
    std::cout << label << ": " << (failures ? "FAIL" : "ok") << " (" << borderline
              << " threshold-tie columns)\n";
    return failures + dsel.verify_guards(label) + dcount.verify_guards(label);
}

// Writes K/V for `count` tokens of row 0 into BF16 or FP8 page planes and checks attention over a
// random ascending subset per column.
int sparse_attention_case(KvCacheStorage storage, std::mt19937& rng) {
    const bool fp8  = storage == KvCacheStorage::Fp8E4M3Row256;
    const int D = 256, Hkv = 2, Hq = 24, group = Hq / Hkv, tokens = 3000, width = 3;
    const int logical = (tokens + kPage - 1) / kPage;
    Cache cache = make_cache(1, logical, rng);
    std::uniform_real_distribution<float> u(-1, 1);
    // Decoded K/V values in logical [token][head][d] order.
    std::vector<double> K(std::size_t(tokens) * Hkv * D), V(K.size());
    const std::size_t plane = std::size_t(cache.pages) * kPage * Hkv;
    std::vector<std::uint8_t> kbytes(plane * D * (fp8 ? 1 : 2)), vbytes(kbytes.size());
    std::vector<std::uint16_t> kscale(plane), vscale(plane);
    // Normalized Sylvester H256 (FP8 K is stored Hadamard-transformed).
    auto hadamard = [](int i, int j) { return (__builtin_popcount(i & j) & 1 ? -1.0 : 1.0) / 16.0; };
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < Hkv; ++h) {
            const std::size_t row = std::size_t(cache.slot(0, t) / kPage) * Hkv * kPage + h * kPage + t % kPage;
            std::vector<float> k(D), v(D);
            for (auto& x : k) x = bfr(u(rng));
            for (auto& x : v) x = bfr(u(rng));
            if (!fp8) {
                for (int d = 0; d < D; ++d) {
                    const std::uint16_t kb = f32_to_bf16(k[d]);
                    const __half vh       = __float2half_rn(v[d]);
                    std::memcpy(&kbytes[(row * D + d) * 2], &kb, 2);
                    std::memcpy(&vbytes[(row * D + d) * 2], &vh, 2);
                    K[(std::size_t(t) * Hkv + h) * D + d] = k[d];
                    V[(std::size_t(t) * Hkv + h) * D + d] = __half2float(vh);
                }
                continue;
            }
            std::vector<double> hk(D, 0.0);
            for (int i = 0; i < D; ++i)
                for (int j = 0; j < D; ++j) hk[i] += hadamard(i, j) * k[j];
            for (int pass = 0; pass < 2; ++pass) {
                const std::vector<double> x = pass == 0 ? hk : std::vector<double>(v.begin(), v.end());
                double a = 0;
                for (double y : x) a = std::max(a, std::abs(y));
                const __half sh = __float2half_rn(float(std::clamp(a / 448.0, 0x1p-24, 65504.0)));
                const float sc  = __half2float(sh);
                for (int d = 0; d < D; ++d) {
                    const __nv_fp8_e4m3 code(float(x[d]) / sc);
                    (pass == 0 ? kbytes : vbytes)[row * D + d] = code.__x;
                    // Oracle uses the decoded stored row; Q is transformed by the same H.
                    (pass == 0 ? K : V)[(std::size_t(t) * Hkv + h) * D + d] = double(float(code)) * sc;
                }
                std::memcpy(&(pass == 0 ? kscale : vscale)[row], &sh, 2);
            }
        }
    }
    std::vector<float> q(std::size_t(D) * Hq * width);
    for (auto& x : q) x = bfr(u(rng));
    const int max_selected = 2051;
    std::vector<int> selected(std::size_t(max_selected) * width, 0), counts(width);
    std::vector<int> all(tokens);
    std::iota(all.begin(), all.end(), 0);
    for (int c = 0; c < width; ++c) {
        std::shuffle(all.begin(), all.end(), rng);
        counts[c] = c == 0 ? 1 : (c == 1 ? 77 : max_selected);
        std::vector<int> pick(all.begin(), all.begin() + counts[c]);
        std::sort(pick.begin(), pick.end());
        std::copy(pick.begin(), pick.end(), selected.begin() + std::size_t(c) * max_selected);
    }
    std::vector<double> expected(q.size());
    const double scale = 1.0 / 16.0;
    for (int c = 0; c < width; ++c) {
        for (int head = 0; head < Hq; ++head) {
            const int h = head / group;
            std::vector<double> qv(D);
            for (int d = 0; d < D; ++d) qv[d] = q[(std::size_t(c) * Hq + head) * D + d];
            if (fp8) {
                std::vector<double> hq(D, 0.0);
                for (int i = 0; i < D; ++i)
                    for (int j = 0; j < D; ++j) hq[i] += hadamard(i, j) * qv[j];
                qv = hq;
            }
            std::vector<double> s(counts[c]);
            double m = -1e300;
            for (int j = 0; j < counts[c]; ++j) {
                const int t = selected[std::size_t(c) * max_selected + j];
                double dot  = 0;
                for (int d = 0; d < D; ++d) dot += qv[d] * K[(std::size_t(t) * Hkv + h) * D + d];
                s[j] = dot * scale;
                m    = std::max(m, s[j]);
            }
            double sum = 0;
            for (auto& x : s) sum += (x = std::exp(x - m));
            for (int d = 0; d < D; ++d) {
                double acc = 0;
                for (int j = 0; j < counts[c]; ++j)
                    acc += s[j] * V[(std::size_t(selected[std::size_t(c) * max_selected + j]) * Hkv + h) * D + d];
                expected[(std::size_t(c) * Hq + head) * D + d] = acc / sum;
            }
        }
    }
    auto dk = to_device(kbytes), dv = to_device(vbytes), dks = to_device(kscale), dvs = to_device(vscale);
    auto dtable = to_device_i32(cache.table), dsel = to_device_i32(selected), dcnt = to_device_i32(counts);
    std::vector<int> rows{0};
    auto drows = to_device_i32(rows);
    auto dq    = to_device(bits(q));
    GuardedDeviceBuffer dout(q.size() * 2);
    PagedKVBatchLayerView view;
    view.k_pages       = Tensor(dk.p, fp8 ? DType::FP8_E4M3FN : DType::BF16, {D, kPage, Hkv, cache.pages});
    view.v_pages       = Tensor(dv.p, fp8 ? DType::FP8_E4M3FN : DType::FP16, {D, kPage, Hkv, cache.pages});
    if (fp8) {
        view.k_scale_pages = Tensor(dks.p, DType::FP16, {1, kPage, Hkv, cache.pages});
        view.v_scale_pages = Tensor(dvs.p, DType::FP16, {1, kPage, Hkv, cache.pages});
    }
    view.block_tables  = Tensor(dtable.p, DType::I32, {logical, 1});
    view.head_dim      = D;
    view.num_kv_heads  = Hkv;
    view.storage       = storage;
    Tensor tq(dq.p, DType::BF16, {D, Hq, width, 1}), tout(dout.data(), DType::BF16, {D, Hq, width, 1});
    Tensor tsel(dsel.p, DType::I32, {max_selected, width, 1}), tcnt(dcnt.p, DType::I32, {width, 1});
    Tensor trows(drows.p, DType::I32, {1});
    ops::qsa_sparse_attention(tq, tsel, tcnt, trows, view, float(scale), tout, nullptr);
    cuda_synchronize();
    const std::string label = std::string("qsa_sparse_attention ") + (fp8 ? "fp8" : "bf16");
    return verify_reduction(label, from_device_bf16(dout.data(), q.size()), expected,
                            ReductionCriterion{4.0e-3, 1.0e-3, 1.0e-2}) +
           dout.verify_guards(label);
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    std::mt19937 rng(20261001);
    int failures = prepare_query_case(rng);
    // Prefill chunk crossing the dense-equivalent boundary (positions 1950..2149).
    failures += select_case("qsa_select prefill", 200, {2149}, rng);
    // Decode rows: dense, exactly at the boundary (2051 visible), first sparse (2052), long.
    failures += select_case("qsa_select decode", 1, {99, 2050, 2051, 9000}, rng);
    failures += sparse_attention_case(KvCacheStorage::BFloat16, rng);
    failures += sparse_attention_case(KvCacheStorage::Fp8E4M3Row256, rng);
    std::cout << (failures ? "FAIL" : "OK") << " qsa\n";
    return failures ? 1 : 0;
}
