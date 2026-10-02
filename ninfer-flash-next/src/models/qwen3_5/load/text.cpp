#include "models/qwen3_5/load/bindings.h"

#include <algorithm>
#include <limits>
#include <set>

namespace ninfer::models::qwen3_5::loading {

AttentionWeights bind_attention(Bindings& b, const TextConfig& config, const std::string& p) {
    const auto& a = config.attention.value();
    const auto h  = config.hidden_size;
    const auto q  = a.query_width();
    const auto k  = a.key_width();
    AttentionWeights out;
    out.query      = b.parameter(p + "attention/query", {q, h}, {p + "mixer_input"});
    out.key        = b.parameter(p + "attention/key", {k, h}, {p + "mixer_input"});
    out.gate       = b.parameter(p + "attention/gate", {q, h}, {p + "mixer_input"});
    out.value      = b.parameter(p + "attention/value", {k, h}, {p + "mixer_input"});
    out.query_norm = b.direct(p + "attention/query_norm", {a.head_dim});
    out.key_norm   = b.direct(p + "attention/key_norm", {a.head_dim});
    out.output     = b.parameter(p + "attention/output", {h, q}, {p + "attention/gated_output"});
    if (config.qsa) {
        const auto d = config.qsa->index_head_dim;
        out.indexer  = IndexerWeights{
            b.parameter(p + "attention/index_query", {config.qsa->index_heads * d, h},
                         {p + "mixer_input"}),
            b.parameter(p + "attention/index_key", {d, h}, {p + "mixer_input"}),
            b.direct(p + "attention/index_query_norm", {d}),
            b.direct(p + "attention/index_key_norm", {d})};
    }
    return out;
}

DenseWeights bind_dense(Bindings& b, std::uint64_t h, std::uint64_t intermediate,
                        const std::string& p, bool draft) {
    const auto input = p + (draft ? "mlp_input" : "ffn_input");
    return {b.parameter(p + "mlp/gate", {intermediate, h}, {input}),
            b.parameter(p + "mlp/up", {intermediate, h}, {input}),
            b.parameter(p + "mlp/down", {h, intermediate},
                        {p + (draft ? "mlp_product" : "mlp/product")})};
}

namespace {

GdnWeights bind_gdn(Bindings& b, const TextConfig& text, const std::string& p) {
    const auto& g    = text.gdn.value();
    const auto h     = text.hidden_size;
    const auto k     = g.key_width();
    const auto v     = g.value_width();
    const auto heads = g.linear_num_value_heads;
    GdnWeights out;
    out.query        = b.parameter(p + "gdn/query", {k, h}, {p + "mixer_input"});
    out.key          = b.parameter(p + "gdn/key", {k, h}, {p + "mixer_input"});
    out.value        = b.parameter(p + "gdn/value", {v, h}, {p + "mixer_input"});
    out.z            = b.parameter(p + "gdn/z", {v, h}, {p + "mixer_input"});
    out.a_projection = b.parameter(p + "gdn/a_projection", {heads, h}, {p + "mixer_input"});
    out.b_projection = b.parameter(p + "gdn/b_projection", {heads, h}, {p + "mixer_input"});
    out.a_log        = b.direct(p + "gdn/a_log", {heads}, QType::FP32);
    out.dt_bias      = b.direct(p + "gdn/dt_bias", {heads}, QType::FP32);
    out.convolution =
        b.direct(p + "gdn/convolution", {g.linear_conv_kernel_dim, g.conv_channels()});
    out.norm   = b.direct(p + "gdn/norm", {g.linear_value_head_dim});
    out.output = b.parameter(p + "gdn/output", {h, v}, {p + "gdn/gated_output"});
    return out;
}

MoeWeights bind_moe(Bindings& b, const TextConfig& config, const std::string& prefix) {
    const auto& moe   = std::get<MoeConfig>(config.ffn);
    const auto p      = prefix + "moe/";
    const auto input  = prefix + "ffn_input";
    const auto h      = config.hidden_size;
    const auto ir     = moe.moe_intermediate_size;
    const auto shared = moe.shared_expert_intermediate_size;
    MoeWeights out;
    out.router       = b.parameter(p + "router", {moe.num_experts, h}, {input});
    out.shared_score = b.parameter(p + "shared_score", {1, h}, {input});
    out.experts.reserve(moe.num_experts);
    for (std::uint32_t e = 0; e < moe.num_experts; ++e) {
        const auto ep = p + "experts/" + std::to_string(e) + "/";
        out.experts.push_back({b.parameter(ep + "gate", {ir, h}, {input}),
                               b.parameter(ep + "up", {ir, h}, {input}),
                               b.parameter(ep + "down", {h, ir}, {ep + "product"})});
    }
    out.shared = {b.parameter(p + "shared/gate", {shared, h}, {input}),
                  b.parameter(p + "shared/up", {shared, h}, {input}),
                  b.parameter(p + "shared/down", {h, shared}, {p + "shared/product"})};
    return out;
}

} // namespace

GatedResidualWeights bind_gated_residual(Bindings& b, const TextConfig& config,
                                         const std::string& p, bool inject) {
    const auto& hc    = config.hyper_connection.value();
    const auto width  = std::uint64_t(hc.streams) * config.hidden_size;
    GatedResidualWeights out;
    out.norm = b.direct(p + "norm", {width});
    out.down = b.parameter(p + "down", {hc.lowrank, width}, {p + "normalized_input"});
    out.up   = b.parameter(p + "up", {width, hc.lowrank}, {p + "mix_hidden"});
    if (inject) { out.inject = b.parameter(p + "inject", {hc.streams, width}, {p + "normalized_input"}); }
    return out;
}

namespace {

PleWeights bind_ple(Bindings& b, const TextConfig& config, const PleLayerConfig& layer,
                    const std::string& p) {
    const auto& ple   = config.ple.value();
    const auto width  = std::uint64_t(config.hyper_connection->streams) * config.hidden_size;
    const auto embed  = ple.embed_dim;
    PleWeights out;
    out.key_projection   = b.parameter(p + "key_projection", {width, embed}, {p + "embedding"});
    out.value_projection = b.parameter(p + "value_projection", {config.hidden_size, embed},
                                       {p + "embedding"});
    out.key_norm         = b.direct(p + "key_norm", {width});
    out.query_norm       = b.direct(p + "query_norm", {width});
    out.conv_norm        = b.direct(p + "conv_norm", {width});
    out.convolution      = b.direct(p + "convolution", {ple.conv_kernel, width});
    out.table_scale      = b.direct(p + "table_scale", {1});
    const auto rows      = (layer.table_rows + layer.table_shards - 1) / layer.table_shards;
    std::uint64_t covered = 0;
    for (std::uint32_t shard = 0; shard < layer.table_shards; ++shard) {
        const auto count = std::min<std::uint64_t>(rows, layer.table_rows - covered);
        out.table.push_back(b.mapped(p + "table/" + std::to_string(shard), {count, ple.head_dim()},
                                     QType::FP8_E4M3FN));
        covered += count;
    }
    return out;
}

} // namespace

BlockWeights bind_block(Bindings& b, const TextConfig& config, const std::string& p,
                        MixerKind mixer) {
    BlockWeights out;
    if (config.hyper_connection) {
        out.mixer_residual = bind_gated_residual(b, config, p + "mixer_residual/", true);
        out.ffn_residual   = bind_gated_residual(b, config, p + "ffn_residual/", true);
    } else {
        out.input_norm          = b.direct(p + "input_norm", {config.hidden_size});
        out.post_attention_norm = b.direct(p + "post_attention_norm", {config.hidden_size});
    }
    if (mixer == MixerKind::FullAttention) {
        out.mixer = bind_attention(b, config, p);
    } else {
        out.mixer = bind_gdn(b, config, p);
    }
    if (const auto* dense = std::get_if<DenseConfig>(&config.ffn)) {
        out.ffn = bind_dense(b, config.hidden_size, dense->intermediate_size, p);
    } else {
        out.ffn = bind_moe(b, config, p);
    }
    return out;
}

TextWeights bind_text(Bindings& b, const TextConfig& config, const LoadOptions& options) {
    TextWeights out;
    out.token_embedding =
        b.parameter("text/token_embedding", {config.vocab_size, config.hidden_size});
    std::vector<std::string> head_inputs{"text/final_hidden"};
    if (options.speculative != SpeculativeBackend::None && !options.proposal_enabled()) {
        head_inputs.push_back(std::string(options.speculative_component()) + "/final_hidden");
    }
    out.output_head = b.parameter("text/output_head", {config.vocab_size, config.hidden_size},
                                  std::move(head_inputs));
    if (config.hyper_connection) {
        out.final_residual = bind_gated_residual(b, config, "text/final_residual/", false);
    } else {
        out.final_norm = b.direct("text/final_norm", {config.hidden_size});
    }
    out.layers.reserve(config.num_hidden_layers);
    for (std::uint32_t i = 0; i < config.num_hidden_layers; ++i) {
        const auto prefix = "text/layers/" + std::to_string(i) + "/";
        out.layers.push_back(bind_block(b, config, prefix, config.layer_types[i]));
        if (const auto* ple = config.ple_layer(i)) {
            out.layers.back().ple = bind_ple(b, config, *ple, prefix + "ple/");
        }
    }
    return out;
}

ProposalWeights bind_proposal(Bindings& b, const artifact::Proposal& proposal,
                              const TextConfig& target, const LoadOptions& options,
                              std::uint32_t public_tokens) {
    const auto rows = proposal.indexed ? proposal.rows : target.vocab_size;
    if (rows > std::numeric_limits<std::uint32_t>::max() ||
        (proposal.indexed && rows > public_tokens)) {
        throw artifact::ArtifactError("proposal rows exceed the output domain");
    }
    ProposalWeights out;
    out.rows = static_cast<std::uint32_t>(rows);
    out.head = b.parameter("proposal/head", {rows, target.hidden_size},
                           {std::string(options.speculative_component()) + "/final_hidden"});
    if (proposal.indexed) {
        out.token_ids = b.direct("proposal/token_ids", {rows}, QType::INT32);
        out.global_token_ids =
            b.binder.values(b.at(*out.token_ids).reference.binding, QType::INT32).integers();
        std::set<std::int32_t> unique;
        for (const auto id : out.global_token_ids) {
            if (id < 0 || std::uint32_t(id) >= public_tokens || !unique.insert(id).second) {
                throw artifact::ArtifactError(
                    "proposal token IDs must be unique and in the public domain");
            }
        }
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
