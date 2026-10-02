#pragma once
#include "models/qwen3_5/program/internal.h"


#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "core/weight.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/softmax_attention.h"
#include "ninfer/ops/sparse_moe.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/program/round_buffers.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

using Phase = qwen3_5::TextPhase;

enum class GdnStateAction : std::uint8_t {
    UpdateInPlace,
    RecordForReplay,
};

struct NullTap {
    static constexpr bool enabled = false;
};

struct PrefillChunkResult {
    std::uint32_t processed_tokens = 0;
    bool finalized                 = false;
    runtime::ExecutionTiming timing;
};

struct DFlashFeatureSink {
    static constexpr bool enabled = true;
    using PrefillConsumer         = std::function<void(const Tensor&, const Tensor&, bool)>;

    Tensor* features                  = nullptr;
    Tensor* positions                 = nullptr;
    Tensor* batch_features            = nullptr;
    const Tensor* batch_lanes         = nullptr;
    const Tensor* batch_valid_columns = nullptr;
    std::int32_t batch_width          = 0;
    std::int32_t batch_size           = 0;
    std::span<const std::uint32_t> layers;
    PrefillConsumer consume_prefill;
    std::uint32_t captured_mask = 0;
    std::int32_t active_tokens  = 0;

    void begin(const Tensor& value);
    void capture_layer(int layer, const Tensor& value, cudaStream_t stream);
    void capture_positions(const Tensor& source, cudaStream_t stream);
    void consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint);
};

// Qwen4Exp program-owned runtime inputs: PLE convolution state pools (one BF16 [C,9,slots]
// tensor per PLE layer) and pinned staging for host-gathered Prefill rows.
struct Qwen4ExpRuntime {
    std::vector<Tensor> ple_states;
    std::uint8_t* ple_staging       = nullptr;
    std::size_t ple_staging_bytes   = 0;
    // BF16 [S*H, prefill_chunk]: pre-mixer streams of the latest Prefill chunk (carried hidden).
    Tensor prefill_streams;
    // FP32 [S, columns] of ones: broadcasts an [H,T] term into every stream via
    // hyper_connection_inject.
    Tensor stream_ones;
    // MTP verify: per PLE layer BF16 [S*H, K+1, batch_capacity] ReplaySSM-style conv records.
    std::vector<Tensor> ple_records;
};

class VisionPrefillSession;

class TextContext {
public:
    TextContext(DeviceContext& ctx, const execution::Parameters& weights, WorkspaceArena& work,
                qwen3_5::PagedKVCacheView kv, LinearAttentionStatePool& state,
                qwen3_5::RoundState& io, Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                std::uint32_t text_kv_base,
                qwen3_5::PagedKVCacheView mtp_kv           = qwen3_5::PagedKVCacheView(),
                const qwen3_5::PagedKVCache* batch_text_kv = nullptr,
                const qwen3_5::PagedKVCache* batch_mtp_kv  = nullptr);
    ~TextContext();

    TextContext(const TextContext&)            = delete;
    TextContext& operator=(const TextContext&) = delete;

    void set_proposal_head(const LinearParameters* weight, const std::int32_t* ids,
                           int count) noexcept {
        proposal_head_     = weight;
        proposal_head_ids_ = ids;
        proposal_head_n_   = count;
    }

    void set_sampling(const ops::SamplingConfig* config) noexcept { sampling_config_ = config; }

    void set_prefill_split_frontier(std::int64_t position) noexcept {
        prefill_split_frontier_ = position;
    }

    void set_rewrite_checkpoint_hidden_output(Tensor* output) noexcept {
        rewrite_checkpoint_hidden_output_ = output;
    }

    void set_mtp_proposal_extent(std::uint32_t extent) noexcept { mtp_proposal_extent_ = extent; }

    void set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot);

    // Qwen4Exp: PLE convolution state pools, one BF16 [C,9,slots] tensor per PLE layer, in PLE
    // layer order; and, for an ordinary decode unit, the host-gathered FP8 rows [E,B] already on
    // the device. Prefill gathers its own rows through the pinned staging buffer.
    void set_ple_states(std::span<const Tensor> states) noexcept { ple_states_ = states; }
    // Output-head logits [V,T] from carried hidden [carried,T] (Qwen4Exp mixes its streams first).
    void logits_from_carried(const Tensor& carried, Tensor& logits);
    void set_qwen4_exp_runtime(const Qwen4ExpRuntime& runtime) noexcept {
        qwen4_runtime_ = &runtime;
        ple_states_    = runtime.ple_states;
    }
    void set_ple_rows(const Tensor* rows) noexcept { decode_ple_rows_ = rows; }
    void set_ple_staging(std::uint8_t* pinned, std::size_t bytes) noexcept {
        ple_staging_       = pinned;
        ple_staging_bytes_ = bytes;
    }
    void set_gdn_state_action(GdnStateAction action, const GdnReplayRecords* replay_records);

    [[nodiscard]] const LinearParameters* proposal_head() const noexcept { return proposal_head_; }

    [[nodiscard]] const std::int32_t* proposal_head_ids() const noexcept {
        return proposal_head_ids_;
    }

    [[nodiscard]] int proposal_head_n() const noexcept { return proposal_head_n_; }

    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(std::span<const int> full_ids,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    [[nodiscard]] PrefillChunkResult
    prefill_chunk(const qwen3_5::PreparedPromptData& input, std::uint32_t begin,
                  std::uint32_t nominal_length, VisionPrefillSession& vision, bool finalize_at_end);
    [[nodiscard]] PrefillChunkResult prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                                   std::uint32_t begin,
                                                   std::uint32_t nominal_length,
                                                   VisionPrefillSession& vision,
                                                   bool finalize_at_end, DFlashFeatureSink& sink);
    void ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                               const Tensor& rope_positions, const Tensor& kv_table_rows,
                               const Tensor& linear_state_source_slots,
                               const Tensor& linear_state_destination_slots,
                               ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                               Tensor& logits);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens);
    void target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                             const Tensor& rope_positions, const Tensor& valid_columns,
                             const Tensor& kv_table_rows, const Tensor& linear_state_source_slots,
                             ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                             Tensor& logits, Tensor& target_tokens, DFlashFeatureSink& sink);
    void mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                  const Tensor& cache_positions, const Tensor& rope_positions,
                                  const Tensor& valid_columns, const Tensor& kv_table_rows,
                                  ops::CausalAttentionExecutionEnvelope envelope,
                                  Tensor& mtp_hidden);
    void mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens);
    void mtp_forward_batch(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                           ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                           int logits_column, Tensor* logits, Tensor* draft_token,
                           const Tensor* explicit_rope_positions = nullptr,
                           const Tensor* input_embeddings        = nullptr);
    void mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                             const Tensor& position, ops::CausalAttentionExecutionEnvelope envelope,
                             Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token);
private:
    [[nodiscard]] bool mtp_enabled() const noexcept {
        return mtp_kv_.valid() || batch_mtp_kv_ != nullptr;
    }

    void attn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase);
    // Qwen4Exp hyper-connection path; `x` is the [S*H,T] residual stream.
    void qwen4_exp_embed(const Tensor& ids, Tensor& x);
    void qwen4_exp_layers(Tensor& x, Phase phase);
    void qwen4_exp_gated_residual(const GatedResidualParameters& p, const Tensor& x, Tensor& mixed,
                                  Tensor* inject);
    void qwen4_exp_attention(const Qwen4ExpAttentionParameters& p, const Tensor& u, Tensor& y,
                             const qwen3_5::PagedKVCache& cache, std::uint32_t layer,
                             const Tensor& kv_table_rows);
    // One hyper-connection block; attention layers use `cache` layer `layer` through
    // `kv_table_rows`. `ple_index` advances past a PLE stage.
    void qwen4_exp_block(const Qwen4ExpBlockParameters& block, Tensor& x, Phase phase, int compact,
                         const qwen3_5::PagedKVCache& cache, const Tensor& kv_table_rows,
                         std::uint32_t& ple_index);
    // MTP draft: stem + block + draft streams written to `streams` [S*H,T].
    void qwen4_exp_mtp_core(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                            Tensor& streams);
    // Collapses [S*H,T] streams to [H,T] with a stream mixer (no injection).
    void qwen4_exp_mix(const GatedResidualParameters& p, const Tensor& streams, Tensor& hidden);
    [[nodiscard]] std::int32_t carried_width() const noexcept {
        return static_cast<std::int32_t>(config_.carried_hidden_width());
    }
    void qwen4_exp_gdn(const Qwen4ExpGdnParameters& p, const Tensor& u, Tensor& y, int index,
                       Phase phase);
    void qwen4_exp_moe(const RoutedMoeParameters& p, const Tensor& u, Tensor& y);
    void qwen4_exp_ple(const PleParameters& p, std::uint32_t ple_index, Tensor& x, Phase phase);
    void qwen4_exp_final(const Tensor& x, Tensor& hidden);
    [[nodiscard]] bool qwen4_exp() const noexcept { return config_.hyper_connection.has_value(); }
    void gdn_mix(const BlockParameters& weights, Tensor& x, int index, Phase phase);
    void mlp_tail(const BlockParameters& weights, Tensor& x, Phase phase,
                  const ops::SparseMoeHints& hints);
    [[nodiscard]] ops::SparseMoeHints next_projection_hints(int layer) const;
    void run_layers(Tensor& x, Phase phase);
    template <class Tap>
    void run_layers(Tensor& x, Phase phase, Tap& tap);
    template <class Tap>
    void target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                  const Tensor& rope_positions, const Tensor& valid_columns,
                                  const Tensor& kv_table_rows,
                                  const Tensor& linear_state_source_slots,
                                  ops::CausalAttentionExecutionEnvelope envelope, Tensor& hidden,
                                  Tensor& logits, Tensor& target_tokens, Tap& tap);

    void mtp_forward_stem(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                          Tensor& x, Tensor& ah);
    void mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden);
    void mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                          const Tensor& rope_positions,
                          ops::CausalAttentionExecutionEnvelope envelope, Tensor& mtp_hidden,
                          const Tensor* input_embeddings);
    void mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden, const Tensor* input_embeddings,
                           const Tensor& positions, const Tensor& rope_positions,
                           ops::CausalAttentionExecutionEnvelope envelope, bool final_chunk,
                           Tensor* final_hidden, Tensor* logits, Tensor* draft_token);
    void proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens);

    struct MultimodalPrefill {
        std::span<const int> token_ids;
        std::span<const std::int32_t> positions;
        VisionPrefillSession* vision = nullptr;
        std::uint32_t begin          = 0;
        std::int32_t rope_delta      = 0;
    };

    struct TextPrefill {
        std::span<const int> token_ids;
        std::uint32_t begin = 0;
    };

    template <class Tap>
    [[nodiscard]] PrefillChunkResult
    prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                 const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end);
    DeviceContext& ctx_;
    const Parameters& parameters_;
    const TextConfig& config_;
    WorkspaceArena& work_;
    qwen3_5::PagedKVCacheView kv_;
    qwen3_5::PagedKVCacheView mtp_kv_;
    const qwen3_5::PagedKVCache* batch_text_kv_ = nullptr;
    const qwen3_5::PagedKVCache* batch_mtp_kv_  = nullptr;
    LinearAttentionStatePool& state_;
    qwen3_5::RoundState& io_;
    Tensor& prefill_hidden_;
    std::uint32_t prefill_chunk_;
    std::uint32_t text_kv_base_;
    const Tensor* active_cache_positions_                                          = nullptr;
    const Tensor* active_rope_positions_                                           = nullptr;
    const Tensor* active_kv_table_rows_                                            = nullptr;
    const Tensor* active_linear_state_source_slots_                                = nullptr;
    const Tensor* active_linear_state_destination_slots_                           = nullptr;
    const Tensor* active_valid_columns_                                            = nullptr;
    const Tensor* active_backend_kv_table_rows_                                    = nullptr;
    const ops::CausalAttentionExecutionEnvelope* active_causal_attention_envelope_ = nullptr;
    std::int32_t active_sequence_batch_                                            = 0;
    std::int32_t active_sequence_width_                                            = 0;
    std::int32_t rope_delta_                                                       = 0;
    std::int32_t linear_state_source_slot_                                         = 0;
    std::int32_t linear_state_destination_slot_                                    = 0;
    GdnStateAction gdn_state_action_          = GdnStateAction::UpdateInPlace;
    const GdnReplayRecords* replay_records_   = nullptr;
    std::int64_t prefill_split_frontier_      = -1;
    Tensor* rewrite_checkpoint_hidden_output_ = nullptr;
    std::uint32_t mtp_proposal_extent_        = 0;

    const Weight* embed_                        = nullptr;
    const Tensor* final_norm_                   = nullptr;
    const LinearParameters* lm_head_            = nullptr;
    const LinearParameters* proposal_head_      = nullptr;
    const std::int32_t* proposal_head_ids_      = nullptr;
    int proposal_head_n_                        = 0;
    const ops::SamplingConfig* sampling_config_ = nullptr;
    const MtpParameters* mtp_                   = nullptr;
    std::span<const Tensor> ple_states_;
    const Qwen4ExpRuntime* qwen4_runtime_ = nullptr;
    const Tensor* decode_ple_rows_  = nullptr;
    const Tensor* active_ple_rows_  = nullptr;
    std::uint8_t* ple_staging_      = nullptr;
    std::size_t ple_staging_bytes_  = 0;
};

} // namespace ninfer::models::qwen3_5::execution
