// Performance bench for routed_moe at the Qwen3.8-Flash-Next geometry (512 NVFP4 experts,
// top-10, H=2560, I=640). The bytes printed are the selected experts' weight traffic, a decode
// lower bound; prefill reads each active expert once per row tile.
//   ./ninfer_routed_moe_bench [--decode] [--prefill]   (default: both)
#include "ninfer/ops/routed_moe.h"
#include "ninfer_bench_common.h"

#include <cstdint>
#include <cstring>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr int kExperts = 512, kTopK = 10, kHidden = 2560, kInter = 640;

std::uint64_t align256(std::uint64_t n) { return (n + 255) / 256 * 256; }

struct Bank {
    DeviceBuffer storage;
    Nvfp4Bank view;
};

Bank make_bank(int n, int k, std::uint32_t seed) {
    const std::uint64_t codes   = std::uint64_t(n) * k / 2;
    const std::uint64_t scale   = align256(codes);
    const std::uint64_t divisor = scale + std::uint64_t(n) * k / 16;
    const std::uint64_t stride  = align256(divisor + 4);
    std::vector<std::uint8_t> host(stride * kExperts);
    std::uint32_t state = seed;
    for (auto& b : host) {
        state = state * 1664525u + 1013904223u;
        b     = static_cast<std::uint8_t>(state >> 24);
    }
    for (int e = 0; e < kExperts; ++e) {
        std::uint8_t* base = host.data() + stride * e;
        for (std::uint64_t i = 0; i < std::uint64_t(n) * k / 16; ++i) base[scale + i] = 0x38;
        const float d = 64.0f;
        std::memcpy(base + divisor, &d, 4);
    }
    Bank out{DeviceBuffer(host.size()), {}};
    cudaMemcpy(out.storage.p, host.data(), host.size(), cudaMemcpyHostToDevice);
    out.view = {static_cast<const std::byte*>(out.storage.p), stride, scale, divisor, kExperts, n, k};
    return out;
}

void run(int t, const char* tag, const Bank& gate, const Bank& up, const Bank& down) {
    DeviceBuffer x      = make_bf16(std::size_t(kHidden) * t, 11U);
    DeviceBuffer shared = make_bf16(std::size_t(kHidden) * t, 13U);
    DeviceBuffer logits = make_bf16(std::size_t(kExperts + 1) * t, 17U, -4.0F, 4.0F);
    DeviceBuffer out    = make_zeros(std::size_t(kHidden) * t * 2);
    WorkspaceArena ws(ops::routed_moe_workspace_bytes(kExperts, kTopK, kHidden, kInter, t) + 4096);
    Tensor tx(x.p, DType::BF16, {kHidden, t});
    Tensor ts(shared.p, DType::BF16, {kHidden, t});
    Tensor tl(logits.p, DType::BF16, {kExperts + 1, t});
    Tensor to(out.p, DType::BF16, {kHidden, t});
    const double expert_bytes = 3.0 * kHidden * kInter * 0.5625;
    const double active       = std::min<double>(kExperts, double(kTopK) * t);
    const Result r            = bench_loop(
        [&](cudaStream_t s) {
            ops::routed_moe(tx, tl, kTopK, gate.view, up.view, down.view, ts, ws, to, s);
        },
        active * expert_bytes);
    print_result(tag, r);
}

} // namespace

int main(int argc, char** argv) {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    bool prefill = false, decode = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--prefill")) prefill = true;
        else if (!std::strcmp(argv[i], "--decode")) decode = true;
    }
    if (!prefill && !decode) { prefill = decode = true; }
    const Bank gate = make_bank(kInter, kHidden, 1), up = make_bank(kInter, kHidden, 2),
               down = make_bank(kHidden, kInter, 3);
    if (decode) {
        run(1, "routed_moe decode  T=1", gate, up, down);
        run(4, "routed_moe decode  T=4", gate, up, down);
    }
    if (prefill) {
        run(512, "routed_moe prefill T=512", gate, up, down);
        run(2048, "routed_moe prefill T=2048", gate, up, down);
    }
    return 0;
}
