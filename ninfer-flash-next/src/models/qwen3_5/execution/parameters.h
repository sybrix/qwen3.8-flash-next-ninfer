#pragma once

#include "models/qwen3_5/model.h"
#include "ninfer/ops/weight_input.h"

#include <array>
#include <memory>
#include <limits>
#include <optional>
#include <stdexcept>
#include <variant>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using LinearParameters = ops::SingleProjectionWeight;

[[nodiscard]] inline std::int32_t dimension(std::uint64_t value) {
    if (value > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("model dimension exceeds the Tensor integer domain");
    }
    return static_cast<std::int32_t>(value);
}

struct DenseParameters {
    LinearParameters gate_up;
    LinearParameters down;
};

using FfnParameters = std::variant<DenseParameters, ops::SparseMoeWeights>;

struct AttentionParameters {
    ops::ProjectionWeights projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
};

struct GdnParameters {
    ops::ProjectionWeights projection;
    ops::ProjectionWeights control;
    Tensor a_log, dt_bias, convolution, norm;
    LinearParameters output;
};

// Qwen4Exp (docs/maintainer/qwen4_exp-model.md). Phase-1 execution uses one ordinary BF16
// projection per logical parent; fused Qwen3.5 projection Ops are not shaped for it.
struct GatedResidualParameters {
    Tensor norm;
    LinearParameters down, up;
    std::optional<LinearParameters> inject;
};

struct Qwen4ExpAttentionParameters {
    LinearParameters query, key, gate, value, output;
    Tensor query_norm, key_norm;
    // QSA indexer: [heads*Di,H] query and [Di,H] key projections and their Di norms.
    LinearParameters index_query, index_key;
    Tensor index_query_norm, index_key_norm;
};

struct Qwen4ExpGdnParameters {
    LinearParameters qkv, z, a, b, output;
    Tensor a_log, dt_bias, convolution, norm;
};

struct RoutedMoeParameters {
    LinearParameters router; // routing rows [E,H] followed by the shared-expert score row
    Nvfp4Bank gate, up, down;
    LinearParameters shared_gate_up; // [2*I_shared,H]: gate rows, then up rows
    LinearParameters shared_down;
    std::int32_t top_k = 0;
};

struct PleParameters {
    const PleLayerConfig* config = nullptr;
    LinearParameters key_projection, value_projection;
    Tensor key_norm, query_norm, conv_norm, convolution, table_scale;
    std::vector<const std::uint8_t*> table; // mapped FP8 row shards
    std::uint64_t shard_rows = 0;
    std::uint32_t row_bytes  = 0; // one head's row
};

struct Qwen4ExpBlockParameters {
    GatedResidualParameters mixer_residual, ffn_residual;
    std::variant<Qwen4ExpAttentionParameters, Qwen4ExpGdnParameters> mixer;
    RoutedMoeParameters moe;
    std::optional<PleParameters> ple;
};

struct BlockParameters {
    Tensor input_norm, post_attention_norm;
    std::variant<AttentionParameters, GdnParameters> mixer;
    FfnParameters ffn;
    ops::SparseMoeHints projection_prefetch;
    std::optional<Qwen4ExpBlockParameters> qwen4_exp;
};

struct TextParameters {
    Weight token_embedding;
    LinearParameters output_head;
    Tensor final_norm;
    std::vector<BlockParameters> layers;
    std::optional<GatedResidualParameters> final_residual; // Qwen4Exp
};

struct MtpProjectionParameters {
    LinearParameters packed;
    // Dense MTP projects K/V and Q/gate independently in its incremental path.
    // MoE MTP uses its existing complete-parent Attention projection.
    std::optional<std::array<LinearParameters, 4>> rows;
};

struct Qwen4ExpMtpParameters {
    Tensor embedding_norm, hidden_norm;
    LinearParameters embedding_projection, hidden_projection;
    Qwen4ExpBlockParameters layer;
    GatedResidualParameters final_residual;
};

struct MtpParameters {
    std::optional<Qwen4ExpMtpParameters> qwen4_exp;
    LinearParameters input_projection;
    Tensor embedding_norm, hidden_norm, input_norm, post_attention_norm, final_norm;
    MtpProjectionParameters projection;
    Tensor query_norm, key_norm;
    LinearParameters output;
    FfnParameters ffn;
    LinearParameters output_head;
};

struct NormParameters {
    Tensor weight, bias;
};

struct VisionBlockParameters {
    NormParameters norm1, norm2;
    LinearParameters qkv;
    Tensor qkv_bias;
    LinearParameters output, fc1, fc2;
    Tensor output_bias, fc1_bias, fc2_bias;
};

struct VisionParameters {
    LinearParameters patch_embedding;
    Tensor patch_embedding_bias, position_embedding;
    std::vector<VisionBlockParameters> layers;
    NormParameters merger_norm;
    LinearParameters merger_fc1, merger_fc2;
    Tensor merger_fc1_bias, merger_fc2_bias;
};

struct DynamicConvParameters {
    Tensor base_kernel;
    LinearParameters kernel_projection;
};

struct DraftBlockParameters {
    Tensor input_norm, post_attention_norm;
    LinearParameters query_key_value, context_key, context_value;
    Tensor query_norm, key_norm;
    LinearParameters output;
    DenseParameters mlp;
    std::optional<DynamicConvParameters> attention_conv, mlp_conv;
};

struct SelectorParameters {
    LinearParameters hidden_projection;
    Tensor predecessor_codebook, successor_codebook;
};

struct DraftParameters {
    LinearParameters feature_projection;
    Tensor context_norm, final_norm;
    std::vector<DraftBlockParameters> layers;
    std::optional<SelectorParameters> selector;
    LinearParameters output_head;
};

struct ProposalParameters {
    LinearParameters head;
    std::optional<Tensor> token_ids;
    std::uint32_t rows = 0;
};

// Cold native preparation for the fixed model implementation. This owner is stable before
// startup sizing, execution, or Graph capture; all weight addresses borrow the source Model.
// Shape-dependent kernel selection and scratch remain with the calling implementation and Op.
class Parameters {
public:
    explicit Parameters(const Model& source);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;
    Parameters(Parameters&&)                 = delete;
    Parameters& operator=(Parameters&&)      = delete;

    const Model& model;
    TextParameters text;
    std::optional<MtpParameters> mtp;
    std::optional<VisionParameters> vision;
    std::optional<DraftParameters> draft;
    std::optional<ProposalParameters> proposal;
};

} // namespace ninfer::models::qwen3_5::execution
