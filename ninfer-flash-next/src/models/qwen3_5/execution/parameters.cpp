#include "models/qwen3_5/execution/parameters.h"

#include "core/weight_view.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

template <class Function>
auto with_context(const std::string& context, Function&& function) {
    try {
        return function();
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(context + ": " + error.what());
    }
}

class Prepare {
public:
    explicit Prepare(const Model& model) : model_(model) {}

    LinearParameters linear(WeightId id) const {
        return with_context(model_.weight(id).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    LinearParameters linear(WeightUseId id) const {
        return with_context(model_.weight(id.parameter).name,
                            [&] { return ops::prepare_linear_weight(model_.input(id)); });
    }

    Tensor tensor(WeightId id) const {
        const auto& bound = model_.weight(id);
        return with_context(bound.name, [&] {
            const auto& view = bound.view;
            if (view.shape.size() > 4) {
                throw std::invalid_argument("direct parameter exceeds Tensor rank");
            }
            std::array<std::int32_t, 4> axes{1, 1, 1, 1};
            for (std::size_t i = 0; i < view.shape.size(); ++i) {
                const auto extent = view.shape[view.shape.size() - 1 - i];
                if (extent > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
                    throw std::invalid_argument("direct parameter exceeds Tensor extent");
                }
                axes[i] = static_cast<std::int32_t>(extent);
            }
            return weight_tensor(view, {axes[0], axes[1], axes[2], axes[3]});
        });
    }

    DenseParameters dense(const DenseWeights& w) const {
        return {with_context(model_.weight(w.gate).name,
                             [&] {
                                 return ops::prepare_linear_swiglu_weight(model_.input(w.gate),
                                                                          model_.input(w.up));
                             }),
                linear(w.down)};
    }

    FfnParameters ffn(const BlockWeights& w) const {
        if (const auto* d = std::get_if<DenseWeights>(&w.ffn)) { return dense(*d); }
        const auto& moe = std::get<MoeWeights>(w.ffn);
        if (std::get<MoeConfig>(model_.config().text.ffn).num_experts_per_tok != 8) {
            throw std::invalid_argument("SparseMoe implements top-8 routing");
        }
        std::vector<ops::WeightInput> gate_up, down;
        gate_up.reserve(2 * moe.experts.size());
        down.reserve(moe.experts.size());
        for (const auto& expert : moe.experts) {
            gate_up.push_back(model_.input(expert.gate));
            gate_up.push_back(model_.input(expert.up));
            down.push_back(model_.input(expert.down));
        }
        return with_context(model_.weight(moe.router).name, [&] {
            return ops::prepare_sparse_moe_weights(
                model_.input(moe.router), model_.input(moe.shared_score), gate_up, down,
                model_.input(moe.shared.gate), model_.input(moe.shared.up),
                model_.input(moe.shared.down));
        });
    }

    LinearParameters rows(std::initializer_list<WeightId> ids) const {
        std::vector<ops::WeightInput> inputs;
        for (const auto id : ids) { inputs.push_back(model_.input(id)); }
        return with_context(model_.weight(*ids.begin()).name,
                            [&] { return ops::prepare_linear_weight(inputs); });
    }

    GatedResidualParameters gated_residual(const GatedResidualWeights& w) const {
        GatedResidualParameters out{tensor(w.norm), linear(w.down), linear(w.up), std::nullopt};
        if (w.inject) { out.inject = linear(*w.inject); }
        return out;
    }

    // Every expert's logical matrix must be exactly the e-th [N,K] slice of one bank parent.
    Nvfp4Bank bank(const MoeWeights& moe, WeightId DenseWeights::*role) const {
        const auto& first = model_.weight(moe.experts.front().*role).view;
        const auto* parent = first.parts.front().parent;
        const auto slice   = weight_element_count(first.shape);
        for (std::size_t e = 0; e < moe.experts.size(); ++e) {
            const auto& view = model_.weight(moe.experts[e].*role).view;
            if (view.parts.size() != 1 || view.parts[0].parent != parent ||
                view.parts[0].begin != e * slice || view.parts[0].end != (e + 1) * slice) {
                throw std::invalid_argument(model_.weight(moe.experts[e].*role).name +
                                            ": expert is not its bank slice");
            }
        }
        WeightView whole;
        whole.shape = parent->geometry.shape;
        whole.parts = {{parent, 0, parent->geometry.elements}};
        auto out    = native_nvfp4_bank(whole);
        if (static_cast<std::size_t>(out.experts) != moe.experts.size()) {
            throw std::invalid_argument("NVFP4 bank expert count differs from the router");
        }
        return out;
    }

    PleParameters ple(const PleWeights& w, const PleLayerConfig& config) const {
        PleParameters out;
        out.config           = &config;
        out.key_projection   = linear(w.key_projection);
        out.value_projection = linear(w.value_projection);
        out.key_norm         = tensor(w.key_norm);
        out.query_norm       = tensor(w.query_norm);
        out.conv_norm        = tensor(w.conv_norm);
        out.convolution      = tensor(w.convolution);
        out.table_scale      = tensor(w.table_scale);
        for (const auto id : w.table) {
            const auto& view = model_.weight(id).view;
            const auto region = contiguous_weight_region(view);
            out.table.push_back(reinterpret_cast<const std::uint8_t*>(region.parent->data) +
                                region.begin);
            if (out.table.size() == 1) {
                out.shard_rows = view.shape[0];
                out.row_bytes  = static_cast<std::uint32_t>(view.shape[1]);
            }
        }
        return out;
    }

    Qwen4ExpBlockParameters qwen4_exp_block(const BlockWeights& w, std::uint32_t layer) const {
        const auto& text = model_.config().text;
        Qwen4ExpBlockParameters out;
        out.mixer_residual = gated_residual(*w.mixer_residual);
        out.ffn_residual   = gated_residual(*w.ffn_residual);
        if (const auto* a = std::get_if<AttentionWeights>(&w.mixer)) {
            const auto& index = a->indexer.value();
            out.mixer = Qwen4ExpAttentionParameters{linear(a->query),     linear(a->key),
                                                    linear(a->gate),      linear(a->value),
                                                    linear(a->output),    tensor(a->query_norm),
                                                    tensor(a->key_norm),  linear(index.query),
                                                    linear(index.key),    tensor(index.query_norm),
                                                    tensor(index.key_norm)};
        } else {
            const auto& g = std::get<GdnWeights>(w.mixer);
            out.mixer     = Qwen4ExpGdnParameters{rows({g.query, g.key, g.value}),
                                                  linear(g.z),
                                                  linear(g.a_projection),
                                                  linear(g.b_projection),
                                                  linear(g.output),
                                                  tensor(g.a_log),
                                                  tensor(g.dt_bias),
                                                  tensor(g.convolution),
                                                  tensor(g.norm)};
        }
        const auto& moe = std::get<MoeWeights>(w.ffn);
        out.moe.router  = rows({moe.router, moe.shared_score});
        out.moe.gate    = bank(moe, &DenseWeights::gate);
        out.moe.up      = bank(moe, &DenseWeights::up);
        out.moe.down    = bank(moe, &DenseWeights::down);
        out.moe.shared_gate_up = rows({moe.shared.gate, moe.shared.up});
        out.moe.shared_down    = linear(moe.shared.down);
        out.moe.top_k = static_cast<std::int32_t>(std::get<MoeConfig>(text.ffn).num_experts_per_tok);
        if (w.ple) { out.ple = ple(*w.ple, *text.ple_layer(layer)); }
        return out;
    }

    BlockParameters block(const BlockWeights& w, std::uint32_t layer) const {
        if (model_.config().text.architecture == Architecture::Qwen4Exp) {
            BlockParameters out;
            out.qwen4_exp = qwen4_exp_block(w, layer);
            return out;
        }
        BlockParameters out;
        out.input_norm          = tensor(w.input_norm);
        out.post_attention_norm = tensor(w.post_attention_norm);
        out.ffn                 = ffn(w);
        if (const auto* a = std::get_if<AttentionWeights>(&w.mixer)) {
            out.mixer = AttentionParameters{
                ops::prepare_attn_input_proj_weights(model_.input(a->query), model_.input(a->key),
                                                     model_.input(a->gate), model_.input(a->value)),
                tensor(a->query_norm), tensor(a->key_norm), linear(a->output)};
            out.projection_prefetch =
                prefetch(std::get<AttentionParameters>(out.mixer).projection, a->query);
        } else {
            const auto& g = std::get<GdnWeights>(w.mixer);
            out.mixer     = GdnParameters{
                ops::prepare_gdn_input_proj_weights(model_.input(g.query), model_.input(g.key),
                                                        model_.input(g.value), model_.input(g.z)),
                ops::prepare_gdn_gating_proj_weights(model_.input(g.a_projection),
                                                         model_.input(g.b_projection)),
                tensor(g.a_log),
                tensor(g.dt_bias),
                tensor(g.convolution),
                tensor(g.norm),
                linear(g.output)};
            out.projection_prefetch =
                prefetch(std::get<GdnParameters>(out.mixer).projection, g.query);
        }
        return out;
    }

    ops::SparseMoeHints prefetch(const ops::ProjectionWeights& projection, WeightId query) const {
        const auto* single = std::get_if<LinearParameters>(&projection);
        const auto& weight =
            single ? single->weight : std::get<ops::PairedProjectionWeights>(projection).first;
        const auto& geometry = model_.weight(query).view.parts.front().parent->geometry;
        const auto row_bytes = geometry.layout == QuantLayout::Contiguous
                                   ? std::uint64_t(weight.k) * dtype_size(DType::BF16)
                                   : geometry.code_bytes_per_row;
        return {weight.qdata, static_cast<std::size_t>(row_bytes * weight.n)};
    }

    MtpParameters mtp(const MtpWeights& w) const {
        if (model_.config().text.architecture == Architecture::Qwen4Exp) {
            MtpParameters out;
            out.qwen4_exp = Qwen4ExpMtpParameters{tensor(w.embedding_norm),
                                                  tensor(w.hidden_norm),
                                                  linear(w.embedding_projection),
                                                  linear(w.hidden_projection),
                                                  qwen4_exp_block(w.layer, 0),
                                                  gated_residual(*w.final_residual)};
            out.output_head = linear(w.output_head_use);
            return out;
        }
        const auto& a = std::get<AttentionWeights>(w.layer.mixer);
        const std::array inputs{model_.input(a.query), model_.input(a.key), model_.input(a.gate),
                                model_.input(a.value)};
        MtpParameters out;
        out.input_projection    = linear(w.input_projection);
        out.embedding_norm      = tensor(w.embedding_norm);
        out.hidden_norm         = tensor(w.hidden_norm);
        out.input_norm          = tensor(w.layer.input_norm);
        out.post_attention_norm = tensor(w.layer.post_attention_norm);
        out.final_norm          = tensor(w.final_norm);
        out.projection.packed   = ops::prepare_linear_weight(inputs);
        if (model_.config().text.architecture == Architecture::Qwen3_5) {
            out.projection.rows = {linear(a.query), linear(a.key), linear(a.gate), linear(a.value)};
        }
        out.query_norm  = tensor(a.query_norm);
        out.key_norm    = tensor(a.key_norm);
        out.output      = linear(a.output);
        out.ffn         = ffn(w.layer);
        out.output_head = linear(w.output_head_use);
        return out;
    }

    NormParameters norm(const NormWeights& w) const { return {tensor(w.weight), tensor(w.bias)}; }

    std::optional<Tensor> joined_bias(const std::array<WeightId, 3>& ids) const {
        WeightView view;
        std::uint64_t count = 0;
        for (const auto id : ids) {
            const auto& input = model_.weight(id).view;
            count += weight_element_count(input.shape);
            for (const auto& part : input.parts) {
                if (!view.parts.empty() && (view.parts.back().parent != part.parent ||
                                            view.parts.back().end != part.begin)) {
                    return std::nullopt;
                }
                view.parts.push_back(part);
            }
        }
        if (count > std::uint64_t(std::numeric_limits<std::int32_t>::max())) {
            throw std::invalid_argument("Vision bias exceeds Tensor extent");
        }
        view.shape = {count};
        return weight_tensor(view, {static_cast<std::int32_t>(count)});
    }

    VisionParameters vision(const VisionWeights& w) const {
        VisionParameters out;
        out.patch_embedding      = linear(w.patch_embedding);
        out.patch_embedding_bias = tensor(w.patch_embedding_bias);
        out.position_embedding   = tensor(w.position_embedding);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context("vision/layers/" + std::to_string(i), [&] {
                const auto& layer = w.layers[i];
                const std::array qkv{model_.input(layer.query), model_.input(layer.key),
                                     model_.input(layer.value)};
                const std::array ids{layer.query_bias, layer.key_bias, layer.value_bias};
                const auto bias = joined_bias(ids);
                if (!bias) {
                    throw std::invalid_argument(
                        "Vision QKV bias: this fixed call requires a contiguous bias bank");
                }
                return VisionBlockParameters{norm(layer.norm1),
                                             norm(layer.norm2),
                                             ops::prepare_linear_weight(qkv),
                                             *bias,
                                             linear(layer.output),
                                             linear(layer.fc1),
                                             linear(layer.fc2),
                                             tensor(layer.output_bias),
                                             tensor(layer.fc1_bias),
                                             tensor(layer.fc2_bias)};
            }));
        }
        out.merger_norm     = norm(w.merger_norm);
        out.merger_fc1      = linear(w.merger_fc1);
        out.merger_fc2      = linear(w.merger_fc2);
        out.merger_fc1_bias = tensor(w.merger_fc1_bias);
        out.merger_fc2_bias = tensor(w.merger_fc2_bias);
        return out;
    }

    DynamicConvParameters convolution(const DynamicConvWeights& w) const {
        return {tensor(w.base_kernel), linear(w.kernel_projection)};
    }

    DraftParameters draft(const DraftWeights& w) const {
        DraftParameters out;
        out.feature_projection = linear(w.feature_projection);
        out.context_norm       = tensor(w.context_norm);
        out.final_norm         = tensor(w.final_norm);
        out.output_head        = linear(w.output_head_use);
        out.layers.reserve(w.layers.size());
        for (std::size_t i = 0; i < w.layers.size(); ++i) {
            out.layers.push_back(with_context(
                std::string(model_.options().speculative_component()) + "/layers/" +
                    std::to_string(i),
                [&] {
                    const auto& layer = w.layers[i];
                    const auto& a     = layer.attention;
                    DraftBlockParameters result;
                    result.input_norm          = tensor(layer.input_norm);
                    result.post_attention_norm = tensor(layer.post_attention_norm);
                    result.query_key_value     = ops::prepare_attn_input_proj_weights(
                        model_.input(a.query), model_.input(a.key), model_.input(a.value));
                    result.context_key   = linear(a.context_key);
                    result.context_value = linear(a.context_value);
                    result.query_norm    = tensor(a.query_norm);
                    result.key_norm      = tensor(a.key_norm);
                    result.output        = linear(a.output);
                    result.mlp           = dense(layer.mlp);
                    if (layer.attention_conv) {
                        result.attention_conv = convolution(*layer.attention_conv);
                    }
                    if (layer.mlp_conv) { result.mlp_conv = convolution(*layer.mlp_conv); }
                    return result;
                }));
        }
        if (w.selector) {
            out.selector = SelectorParameters{linear(w.selector->hidden_projection),
                                              tensor(w.selector->predecessor_codebook),
                                              tensor(w.selector->successor_codebook)};
        }
        return out;
    }

private:
    const Model& model_;
};

} // namespace

Parameters::Parameters(const Model& source) : model(source) {
    const Prepare prepare(model);
    const auto& w        = model.weights();
    text.token_embedding = native_weight(model.weight(w.text.token_embedding).view);
    text.output_head     = prepare.linear(w.text.output_head_use);
    if (w.text.final_residual) {
        text.final_residual = prepare.gated_residual(*w.text.final_residual);
    } else {
        text.final_norm = prepare.tensor(w.text.final_norm);
    }
    text.layers.reserve(w.text.layers.size());
    for (std::size_t i = 0; i < w.text.layers.size(); ++i) {
        text.layers.push_back(with_context("text/layers/" + std::to_string(i), [&] {
            return prepare.block(w.text.layers[i], static_cast<std::uint32_t>(i));
        }));
    }
    if (w.mtp) {
        mtp = with_context("mtp", [&] { return prepare.mtp(*w.mtp); });
    }
    if (w.vision) {
        vision = with_context("vision", [&] { return prepare.vision(*w.vision); });
    }
    if (w.draft) {
        draft = with_context(std::string(model.options().speculative_component()),
                             [&] { return prepare.draft(*w.draft); });
    }
    if (w.proposal) {
        proposal =
            ProposalParameters{prepare.linear(w.proposal->head), std::nullopt, w.proposal->rows};
        if (w.proposal->token_ids) { proposal->token_ids = prepare.tensor(*w.proposal->token_ids); }
    }
}

} // namespace ninfer::models::qwen3_5::execution
