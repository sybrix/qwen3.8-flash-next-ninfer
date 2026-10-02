// ninfer::ops - routed_moe wrapper: validates the public contract and dispatches to the launcher.
#include "ninfer/ops/routed_moe.h"

#include "ops/launcher/routed_moe.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(const Tensor& t, std::int32_t rows, std::int32_t columns, const char* name) {
    if (t.dtype != DType::BF16 || t.ne[0] != rows || t.ne[1] != columns || t.ne[2] != 1 ||
        t.ne[3] != 1 || !t.is_contiguous() || t.data == nullptr ||
        reinterpret_cast<std::uintptr_t>(t.data) % 16) {
        throw std::invalid_argument(std::string("routed_moe: invalid ") + name);
    }
}

void require_bank(const Nvfp4Bank& bank, std::int32_t experts, std::int32_t n, std::int32_t k,
                  const char* name) {
    if (bank.data == nullptr || bank.experts != experts || bank.n != n || bank.k != k ||
        n % 128 || k % 64 || bank.expert_stride % 256 ||
        reinterpret_cast<std::uintptr_t>(bank.data) % 256 ||
        bank.scale_offset < static_cast<std::uint64_t>(n) * (k / 2) ||
        bank.divisor_offset + 4 > bank.expert_stride || bank.divisor_offset % 4) {
        throw std::invalid_argument(std::string("routed_moe: invalid ") + name + " bank");
    }
}

bool overlaps(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const auto* a0 = static_cast<const std::byte*>(a);
    const auto* b0 = static_cast<const std::byte*>(b);
    return a0 < b0 + b_bytes && b0 < a0 + a_bytes;
}

} // namespace

std::size_t routed_moe_workspace_bytes(std::int32_t experts, std::int32_t top_k,
                                       std::int32_t hidden, std::int32_t intermediate,
                                       std::int32_t max_tokens) {
    if (experts <= 0 || top_k <= 0 || top_k > 32 || top_k > experts || hidden <= 0 ||
        intermediate <= 0 || max_tokens <= 0) {
        throw std::invalid_argument("routed_moe workspace: invalid geometry");
    }
    return detail::routed_moe_workspace_launch_bytes(experts, top_k, hidden, intermediate,
                                                     max_tokens);
}

void routed_moe(const Tensor& x, const Tensor& logits, std::int32_t top_k, const Nvfp4Bank& gate,
                const Nvfp4Bank& up, const Nvfp4Bank& down, const Tensor& shared,
                WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    const std::int32_t hidden = x.ne[0];
    const std::int32_t tokens = x.ne[1];
    const std::int32_t experts = gate.experts;
    if (hidden <= 0 || tokens <= 0 || hidden % 16) {
        throw std::invalid_argument("routed_moe: x must be a positive [H,T] with H % 16 == 0");
    }
    require(x, hidden, tokens, "x");
    require(shared, hidden, tokens, "shared");
    require(out, hidden, tokens, "out");
    require(logits, experts + 1, tokens, "logits");
    require_bank(gate, experts, gate.n, hidden, "gate");
    require_bank(up, experts, gate.n, hidden, "up");
    require_bank(down, experts, hidden, gate.n, "down");
    if (gate.n % 16 || top_k <= 0 || top_k > 32 || top_k > experts) {
        throw std::invalid_argument("routed_moe: invalid intermediate width or top_k");
    }
    const std::size_t required =
        routed_moe_workspace_bytes(experts, top_k, hidden, gate.n, tokens);
    if (workspace.base() == nullptr || workspace.used() > workspace.capacity() ||
        workspace.capacity() - workspace.used() < required) {
        throw std::invalid_argument("routed_moe: insufficient workspace capacity");
    }
    for (const Tensor* input : {&x, &logits, &shared}) {
        if (overlaps(input->data, input->bytes(), out.data, out.bytes())) {
            throw std::invalid_argument("routed_moe: out overlaps an input");
        }
    }
    auto scope       = workspace.scope();
    void* scratch    = workspace.alloc_bytes(required).data;
    if (overlaps(scratch, required, out.data, out.bytes())) {
        throw std::invalid_argument("routed_moe: out overlaps the workspace");
    }
    detail::routed_moe_launch(x, logits, top_k, gate, up, down, shared, scratch, out, stream);
}

} // namespace ninfer::ops
