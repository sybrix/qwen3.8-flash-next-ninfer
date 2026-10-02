// ninfer::ops - PLE wrappers: validate the public contracts and dispatch to the launcher.
#include "ninfer/ops/ple.h"

#include "core/device.h" // CUDA_CHECK
#include "ops/launcher/ple.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require(const Tensor& t, DType dtype, std::int32_t n0, std::int32_t n1, std::int32_t n2,
             const char* op, const char* name) {
    if (t.dtype != dtype || t.ne[0] != n0 || t.ne[1] != n1 || t.ne[2] != n2 || t.ne[3] != 1 ||
        !t.is_contiguous() || t.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": invalid " + name + " (expected [" +
                                    std::to_string(n0) + "," + std::to_string(n1) + "," +
                                    std::to_string(n2) + "])");
    }
}

bool overlaps(const Tensor& a, const Tensor& b) {
    const auto* a0 = static_cast<const std::byte*>(a.data);
    const auto* b0 = static_cast<const std::byte*>(b.data);
    return a0 < b0 + b.bytes() && b0 < a0 + a.bytes();
}

void require_disjoint(std::initializer_list<const Tensor*> inputs, const Tensor& out,
                      const char* op) {
    for (const auto* input : inputs) {
        if (input->data != nullptr && overlaps(*input, out)) {
            throw std::invalid_argument(std::string(op) + ": operands must not overlap");
        }
    }
}

} // namespace

void ple_dequantize(const Tensor& codes, const Tensor& scale, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "ple_dequantize";
    if (codes.ne[0] <= 0 || codes.ne[1] <= 0) {
        throw std::invalid_argument("ple_dequantize: codes must be a positive [E,T] matrix");
    }
    require(codes, DType::FP8_E4M3FN, codes.ne[0], codes.ne[1], 1, op, "codes");
    require(scale, DType::BF16, 1, 1, 1, op, "scale");
    require(out, DType::BF16, codes.ne[0], codes.ne[1], 1, op, "out");
    require_disjoint({&codes, &scale}, out, op);
    detail::ple_dequantize_launch(codes, scale, out, stream);
}

void ple_gate(const Tensor& key, const Tensor& query, const Tensor& value, std::int32_t streams,
              Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "ple_gate";
    if (streams <= 0 || value.ne[0] <= 0 || value.ne[1] <= 0) {
        throw std::invalid_argument("ple_gate: streams, H and T must be positive");
    }
    const std::int32_t h = value.ne[0];
    const std::int32_t t = value.ne[1];
    require(value, DType::BF16, h, t, 1, op, "value");
    require(key, DType::BF16, streams * h, t, 1, op, "key");
    require(query, DType::BF16, streams * h, t, 1, op, "query");
    require(out, DType::BF16, streams * h, t, 1, op, "out");
    require_disjoint({&key, &query, &value}, out, op);
    detail::ple_gate_launch(key, query, value, streams, out, stream);
}

void ple_conv_residual(const Tensor& gated, const Tensor& normalized, const Tensor& weight,
                       const Tensor& state_in, Tensor& state_out, Tensor& residual,
                       cudaStream_t stream) {
    constexpr const char* op = "ple_conv_residual";
    const std::int32_t c     = residual.ne[0];
    const std::int32_t t     = residual.ne[1];
    if (c <= 0 || t <= 0) { throw std::invalid_argument("ple_conv_residual: empty residual"); }
    require(residual, DType::BF16, c, t, 1, op, "residual");
    require(gated, DType::BF16, c, t, 1, op, "gated");
    require(normalized, DType::BF16, c, t, 1, op, "normalized");
    require(weight, DType::BF16, c, kPleConvTaps, 1, op, "weight");
    require(state_in, DType::BF16, c, kPleConvState, 1, op, "state_in");
    require(state_out, DType::BF16, c, kPleConvState, 1, op, "state_out");
    require_disjoint({&gated, &normalized, &weight, &state_in, &state_out}, residual, op);
    if (state_in.data != state_out.data && overlaps(state_in, state_out)) {
        throw std::invalid_argument("ple_conv_residual: states must be disjoint or identical");
    }
    require_disjoint({&gated, &normalized, &weight}, state_out, op);
    detail::ple_conv_residual_launch(gated, normalized, weight, state_in.data, state_out.data,
                                     residual, stream);
}

void ple_conv_residual_snapshot(const Tensor& gated, const Tensor& normalized,
                                const Tensor& weight, Tensor& states, const Tensor& valid_columns,
                                const Tensor& initial_state_slots,
                                const Tensor& snapshot_base_slots, Tensor& residual,
                                cudaStream_t stream) {
    constexpr const char* op = "ple_conv_residual_snapshot";
    const std::int32_t c     = residual.ne[0];
    const std::int32_t w     = residual.ne[1];
    const std::int32_t b     = residual.ne[2];
    if (c <= 0 || w <= 0 || b <= 0) {
        throw std::invalid_argument("ple_conv_residual_snapshot: empty residual");
    }
    require(residual, DType::BF16, c, w, b, op, "residual");
    require(gated, DType::BF16, c, w, b, op, "gated");
    require(normalized, DType::BF16, c, w, b, op, "normalized");
    require(weight, DType::BF16, c, kPleConvTaps, 1, op, "weight");
    if (states.ne[2] <= 0) { throw std::invalid_argument("ple_conv_residual_snapshot: no slots"); }
    require(states, DType::BF16, c, kPleConvState, states.ne[2], op, "states");
    require(initial_state_slots, DType::I32, b, 1, 1, op, "initial_state_slots");
    require(snapshot_base_slots, DType::I32, b, 1, 1, op, "snapshot_base_slots");
    if (valid_columns.data != nullptr) {
        require(valid_columns, DType::I32, b, 1, 1, op, "valid_columns");
    }
    require_disjoint({&gated, &normalized, &weight, &states}, residual, op);
    require_disjoint({&gated, &normalized, &weight, &valid_columns, &initial_state_slots,
                      &snapshot_base_slots},
                     states, op);
    detail::ple_conv_residual_snapshot_launch(gated, normalized, weight, states, valid_columns,
                                              initial_state_slots, snapshot_base_slots, residual,
                                              stream);
}

void ple_conv_residual_record(const Tensor& gated, const Tensor& normalized, const Tensor& weight,
                              const Tensor& states, const Tensor& valid_columns,
                              const Tensor& initial_state_slots, Tensor& record, Tensor& residual,
                              cudaStream_t stream) {
    constexpr const char* op = "ple_conv_residual_record";
    const std::int32_t c     = residual.ne[0];
    const std::int32_t w     = residual.ne[1];
    const std::int32_t b     = residual.ne[2];
    if (c <= 0 || w <= 0 || b <= 0) { throw std::invalid_argument("ple_conv_residual_record: empty"); }
    require(residual, DType::BF16, c, w, b, op, "residual");
    require(gated, DType::BF16, c, w, b, op, "gated");
    require(normalized, DType::BF16, c, w, b, op, "normalized");
    require(record, DType::BF16, c, w, b, op, "record");
    require(weight, DType::BF16, c, kPleConvTaps, 1, op, "weight");
    if (states.ne[2] <= 0) { throw std::invalid_argument("ple_conv_residual_record: no slots"); }
    require(states, DType::BF16, c, kPleConvState, states.ne[2], op, "states");
    require(initial_state_slots, DType::I32, b, 1, 1, op, "initial_state_slots");
    if (valid_columns.data != nullptr) {
        require(valid_columns, DType::I32, b, 1, 1, op, "valid_columns");
    }
    require_disjoint({&gated, &normalized, &weight, &states, &record}, residual, op);
    require_disjoint({&gated, &normalized, &weight, &states}, record, op);
    CUDA_CHECK(cudaMemcpyAsync(record.data, normalized.data, normalized.bytes(),
                               cudaMemcpyDeviceToDevice, stream));
    detail::ple_conv_residual_record_launch(gated, normalized, weight, states, valid_columns,
                                            initial_state_slots, residual, stream);
}

void ple_conv_replay_fold(Tensor& states, const Tensor& record,
                          std::span<const GdnReplayFoldRow> rows, cudaStream_t stream) {
    constexpr const char* op = "ple_conv_replay_fold";
    if (rows.empty()) { return; }
    if (rows.size() > 8) { throw std::invalid_argument("ple_conv_replay_fold: at most 8 rows"); }
    const std::int32_t c = states.ne[0];
    require(states, DType::BF16, c, kPleConvState, states.ne[2], op, "states");
    if (record.dtype != DType::BF16 || record.ne[0] != c || record.ne[3] != 1 ||
        record.ne[2] < static_cast<std::int32_t>(rows.size()) || !record.is_contiguous()) {
        throw std::invalid_argument("ple_conv_replay_fold: invalid record");
    }
    for (const auto& row : rows) {
        if (row.source_state_slot < 0 || row.source_state_slot >= states.ne[2] ||
            row.destination_state_slot < 0 || row.destination_state_slot >= states.ne[2] ||
            row.commit_columns < 0 || row.commit_columns > record.ne[1]) {
            throw std::invalid_argument("ple_conv_replay_fold: invalid row");
        }
    }
    detail::ple_conv_replay_fold_launch(states, record, rows, stream);
}

void causal_conv1d_silu_record(const Tensor& x, const Tensor& weight, const Tensor& conv_states,
                               const Tensor& valid_columns, const Tensor& initial_state_slots,
                               Tensor& record, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "causal_conv1d_silu_record";
    const std::int32_t c     = x.ne[0];
    const std::int32_t w     = x.ne[1];
    const std::int32_t b     = x.ne[2];
    require(x, DType::BF16, c, w, b, op, "x");
    require(out, DType::BF16, c, w, b, op, "out");
    require(record, DType::BF16, c, w, b, op, "record");
    require(weight, DType::BF16, c, 4, 1, op, "weight");
    require(conv_states, DType::BF16, c, 3, conv_states.ne[2], op, "conv_states");
    require(initial_state_slots, DType::I32, b, 1, 1, op, "initial_state_slots");
    if (valid_columns.data != nullptr) {
        require(valid_columns, DType::I32, b, 1, 1, op, "valid_columns");
    }
    require_disjoint({&x, &weight, &conv_states, &record}, out, op);
    CUDA_CHECK(cudaMemcpyAsync(record.data, x.data, x.bytes(), cudaMemcpyDeviceToDevice, stream));
    detail::causal_conv1d_silu_record_launch(x, weight, conv_states, valid_columns,
                                             initial_state_slots, out, stream);
}

} // namespace ninfer::ops
