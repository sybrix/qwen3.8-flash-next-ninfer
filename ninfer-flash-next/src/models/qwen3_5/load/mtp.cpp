#include "models/qwen3_5/load/bindings.h"

namespace ninfer::models::qwen3_5::loading {

MtpWeights bind_mtp(Bindings& b, const TextConfig& config, const TextWeights& target) {
    const auto h = config.hidden_size;
    MtpWeights out;
    if (config.hyper_connection) {
        const auto width          = std::uint64_t(config.hyper_connection->streams) * h;
        out.embedding_norm        = b.direct("mtp/embedding_norm", {h});
        out.hidden_norm           = b.direct("mtp/hidden_norm", {width});
        out.embedding_projection  = b.parameter("mtp/embedding_projection", {h, h},
                                                {"mtp/normalized_embedding"});
        out.hidden_projection     = b.parameter("mtp/hidden_projection", {h, h},
                                                {"mtp/normalized_hidden"});
        out.layer                 = bind_block(b, config, "mtp/layers/0/", MixerKind::FullAttention);
        out.final_residual        = bind_gated_residual(b, config, "mtp/final_residual/", false);
        out.token_embedding       = target.token_embedding;
        out.output_head           = target.output_head;
        return out;
    }
    out.input_projection = b.parameter("mtp/input_projection", {h, 2ULL * h}, {"mtp/stem_input"});
    out.embedding_norm   = b.direct("mtp/embedding_norm", {h});
    out.hidden_norm      = b.direct("mtp/hidden_norm", {h});
    out.final_norm       = b.direct("mtp/final_norm", {h});
    out.layer            = bind_block(b, config, "mtp/layers/0/", MixerKind::FullAttention);
    out.token_embedding  = target.token_embedding;
    out.output_head      = target.output_head;
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
