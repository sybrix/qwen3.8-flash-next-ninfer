// ninfer::ops - hyper-connection wrappers: validate the public contracts and dispatch to the
// launcher. Host-compiled; never includes kernel headers.
#include "ninfer/ops/hyper_connection.h"

#include "ops/launcher/hyper_connection.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_matrix(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t columns,
                    const char* op, const char* name) {
    if (t.dtype != dtype || t.ne[0] != rows || t.ne[1] != columns || t.ne[2] != 1 ||
        t.ne[3] != 1 || !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous [" +
                                    std::to_string(rows) + "," + std::to_string(columns) + "]");
    }
}

std::int32_t columns_of(const Tensor& t, const char* op, const char* name) {
    if (t.ne[1] <= 0) {
        throw std::invalid_argument(std::string(op) + ": " + name + " needs positive T");
    }
    return t.ne[1];
}

void require_disjoint(const Tensor& a, const Tensor& b, const char* op) {
    const auto* a0 = static_cast<const std::byte*>(a.data);
    const auto* b0 = static_cast<const std::byte*>(b.data);
    if (a0 < b0 + b.bytes() && b0 < a0 + a.bytes()) {
        throw std::invalid_argument(std::string(op) + ": operands must not overlap");
    }
}

} // namespace

void grouped_rmsnorm(const Tensor& x, const Tensor& weight, std::int32_t groups, float eps,
                     Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "grouped_rmsnorm";
    if (groups <= 0 || x.ne[0] <= 0 || x.ne[0] % groups) {
        throw std::invalid_argument("grouped_rmsnorm: width must split into positive groups");
    }
    if (!(eps > 0.0f) || !std::isfinite(eps)) {
        throw std::invalid_argument("grouped_rmsnorm: eps must be positive and finite");
    }
    const std::int32_t t = columns_of(x, op, "x");
    require_matrix(x, DType::BF16, x.ne[0], t, op, "x");
    require_matrix(weight, DType::BF16, x.ne[0], 1, op, "weight");
    require_matrix(out, DType::BF16, x.ne[0], t, op, "out");
    require_disjoint(x, out, op);
    require_disjoint(weight, out, op);
    detail::grouped_rmsnorm_launch(x, weight, groups, eps, out, stream);
}

void hyper_connection_mix(const Tensor& normalized, const Tensor& logits, std::int32_t streams,
                          Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "hyper_connection_mix";
    const std::int32_t t     = columns_of(out, op, "out");
    if (streams <= 0 || out.ne[0] <= 0) {
        throw std::invalid_argument("hyper_connection_mix: streams and width must be positive");
    }
    const std::int32_t width = streams * out.ne[0];
    require_matrix(normalized, DType::BF16, width, t, op, "normalized");
    require_matrix(logits, DType::BF16, width, t, op, "logits");
    require_matrix(out, DType::BF16, out.ne[0], t, op, "out");
    require_disjoint(normalized, out, op);
    require_disjoint(logits, out, op);
    detail::hyper_connection_mix_launch(normalized, logits, streams, out, stream);
}

void hyper_connection_gates(const Tensor& down, const Tensor& inject_logits, std::int32_t streams,
                            Tensor& hidden, Tensor& inject, cudaStream_t stream) {
    constexpr const char* op = "hyper_connection_gates";
    if (streams <= 0) { throw std::invalid_argument("hyper_connection_gates: streams <= 0"); }
    if (down.data == nullptr && inject_logits.data == nullptr) {
        throw std::invalid_argument("hyper_connection_gates: no gate pair was supplied");
    }
    if (down.data != nullptr) {
        const std::int32_t t = columns_of(down, op, "down");
        require_matrix(down, DType::BF16, down.ne[0], t, op, "down");
        require_matrix(hidden, DType::BF16, down.ne[0], t, op, "hidden");
        require_disjoint(down, hidden, op);
    }
    if (inject_logits.data != nullptr) {
        const std::int32_t t = columns_of(inject_logits, op, "inject_logits");
        require_matrix(inject_logits, DType::BF16, streams, t, op, "inject_logits");
        require_matrix(inject, DType::FP32, streams, t, op, "inject");
        require_disjoint(inject_logits, inject, op);
    }
    detail::hyper_connection_gates_launch(down, inject_logits, streams, hidden, inject, stream);
}

void hyper_connection_inject(const Tensor& y, const Tensor& inject, Tensor& state,
                             cudaStream_t stream) {
    constexpr const char* op = "hyper_connection_inject";
    const std::int32_t t     = columns_of(y, op, "y");
    const std::int32_t s     = inject.ne[0];
    if (s <= 0 || y.ne[0] <= 0) {
        throw std::invalid_argument("hyper_connection_inject: streams and width must be positive");
    }
    require_matrix(y, DType::BF16, y.ne[0], t, op, "y");
    require_matrix(inject, DType::FP32, s, t, op, "inject");
    require_matrix(state, DType::BF16, s * y.ne[0], t, op, "state");
    require_disjoint(y, state, op);
    require_disjoint(inject, state, op);
    detail::hyper_connection_inject_launch(y, inject, state, stream);
}

} // namespace ninfer::ops
