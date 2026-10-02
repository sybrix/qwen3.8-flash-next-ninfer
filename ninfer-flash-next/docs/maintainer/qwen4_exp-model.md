# Qwen4Exp (Qwen3.8-Flash-Next) model reference

This reference describes the Qwen4Exp Text mathematics targeted by NInfer's `flash-next` work:
hyper-connection residual streams, query-sparse attention (QSA), a per-layer n-gram embedding
(PLE), Gated DeltaNet and a 512-expert routed MoE. The checkpoint
`Qwen/Qwen3.8-Flash-Next` and its NVIDIA ModelOpt NVFP4 export are instances of this architecture.
The authoritative source mathematics is transformers'
`models/qwen4_exp/modeling_qwen4_exp.py`; every formula below was transcribed from it.

[Qwen3.5 model reference](qwen3_5-model.md) defines the GDN recurrence, gated GQA, interleaved
partial MRoPE and routed-expert SwiGLU that Qwen4Exp reuses unchanged unless stated here.

## Architecture and configuration

| `architectures[0]` | `model_type` |
|---|---|
| `Qwen4ExpForConditionalGeneration` | `qwen4_exp` (Text: `qwen4_exp_text`) |

Released geometry (Qwen3.8-Flash-Next):

| Quantity | Value |
|---|---:|
| Hidden width H | 2560 |
| Hyper-connection streams S (`hc_count`) / mixer rank R (`hc_lowrank`) | 4 / 320 |
| Text layers | 48 |
| Full attention (QSA) / GDN layers | 12 / 36, full at indices `3,7,...,47` |
| Q heads / KV heads / head dimension | 24 / 2 / 256 |
| GDN key heads / value heads / head dimension | 16 / 48 / 128 |
| GDN convolution width | 4 |
| Routed experts / selected per token | 512 / 10 |
| Routed / shared intermediate width | 640 / 640 |
| QSA indexer heads / KV heads / head dimension | 4 / 1 / 128 |
| QSA compress ratio / token budget | 4 / 2048 |
| PLE layers (`ple_layer_ids`, 1-based) | `[2]` → zero-based block 1 |
| PLE n-gram order / heads per order / per-head width | 3 / 8 / 160 |
| PLE convolution width / dilation | 4 / 3 |
| Embedding/output matrix rows | 248320, untied |
| `rms_norm_eps`, `rope_theta`, partial rotary, MRoPE sections | 1e-6, 1e7, 0.25, `[11,11,10]` |
| GDN output-gate activation (`output_gate_type`) | sigmoid |

## Norm conventions

`RMSNorm(x; w)` over width D is zero-centred: `x / sqrt(mean(x²) + eps) * (1 + w)`, computed in
FP32 and rounded to the activation dtype. A *grouped* RMSNorm of width `S·H` with group size H
normalizes each of the S contiguous H-wide groups independently, then applies the full `S·H`
weight vector.

GDN's output norm is the *non*-centred gated form `w * RMS(o) * act(z)` with `act = sigmoid`
for Qwen4Exp (Qwen3.5 uses SiLU).

## Residual topology: hyper-connections

The residual stream between blocks is `X ∈ R^{S·H}` per token (S=4 streams of H). There is no
per-block input RMSNorm, no post-attention RMSNorm and no final RMSNorm; hyper-connection mixers
replace all three.

Embedding: `X_0 = repeat(E[token], S)` (each stream equals the token embedding).

A *gated residual* `GR` with weights `(n ∈ R^{SH}, D ∈ R^{R×SH}, U ∈ R^{SH×R}, B ∈ R^{S×SH})`:

```text
N   = GroupedRMSNorm(X; n)                        # S groups of H
m   = sigmoid( U · silu( (D · N) / S ) )          # R^{SH}
u   = mean_s ( m[s] ⊙ N[s] )                      # R^H, the block input
c   = 2 · sigmoid( (B · N) / S )                  # R^S, injection weights (absent on the final mixer)
```

Each block applies two of them:

```text
(u, c) = GR_attn(X);  y = Mixer(u);  X ← X + c ⊗ y      # X[s] += c[s] · y for every stream
(u, c) = GR_mlp(X);   y = MoE(u);    X ← X + c ⊗ y
```

After the last block, `u = GR_final(X)` (no `B`) feeds the output head directly:
`logits = W_out · u`.

## PLE: per-layer n-gram embedding

Block 1 first adds `X ← X + PLE(X, tokens)`, before its attention gated residual.

**Hashing.** With unigram vocabulary V=248320, n-gram order G=3, `heads_per_ngram`=8 and the
checkpoint buffers `multipliers[0..2]`, `head_vocab_sizes[0..15]`, `head_offsets[0..15]`
(all int64; derived deterministically from config `seed`, but read from the checkpoint):

```text
t_j      = token at position p - j, or EOS if p - j precedes the start of the token's EOS-delimited
           segment (EOS = 248044; a position equal to EOS starts a new segment *after* itself)
mix_g    = XOR_{j=0}^{g-1} ( t_j · multipliers[j] )        # int64, wrapping multiply, g ∈ {2,3}
id[h]    = (mix_g mod head_vocab_sizes[h]) + head_offsets[h]   # h ∈ heads of order g, mod is floor
```

Heads 0–7 use order 2, heads 8–15 use order 3. `mod` follows `torch.remainder` (non-negative
result for positive divisors). Each request keeps the last `G-1=2` tokens as state; a new request
starts from `[EOS, EOS]`.

**Embedding.** `e = concat_h Table[id[h]] ∈ R^{16·160 = 2560}` where `Table` is the
`[320001536, 160]` matrix stored as 128 FP8-E4M3 row shards with one per-tensor BF16 scale.

**Gated injection.**

```text
K   = GroupedRMSNorm(W_k · e; n_k)        # W_k ∈ R^{SH×H} → S keys of H
v   = W_v · e                              # W_v ∈ R^{H×H}
Q   = GroupedRMSNorm(X; n_q)
γ_s = <K[s], Q[s]> / sqrt(H)
γ_s = sign(γ_s) · sqrt(max(|γ_s|, 1e-6))
g   = concat_s sigmoid(γ_s) · v            # R^{SH}
gn  = GroupedRMSNorm(g; n_c)
PLE = g + silu( DepthwiseCausalConv(gn; w_conv, kernel 4, dilation 3) )
```

The convolution reads positions `p, p-3, p-6, p-9`; each request keeps the last 9 `gn` vectors
(9·S·H values) as state, zero for a new request.

## Query-sparse attention (QSA)

QSA wraps the Qwen3.5 gated GQA block. Before attention each full-attention layer computes, from
its block input u:

```text
[q_I(4×128) | k_I(1×128)] = W_I · u
q_I = RoPE(RMSNorm_128(q_I; n_qI))                       # same partial MRoPE as Text
cache k_I per token (pre-norm, pre-RoPE)
```

For query position p with V visible tokens, blocks of 4 consecutive visible tokens (only complete
blocks) are pooled: `k̄_b = RoPE_{pos(first token of b)}(RMSNorm_128(mean(k_I in b); n_kI))`.
Block score `σ_b = sum_heads relu(<q_I, k̄_b>) / sqrt(128)`. The top `2048/4 = 512` blocks plus the
`V mod 4` tail tokens form the attended set; attention is otherwise the Qwen3.5 gated GQA.

When `V ≤ 2048 + 3` every complete block is selected, so QSA equals dense causal attention. The
phase-1 implementation relies on this and is exact only for contexts up to 2051 tokens.

## Sparse MoE

Identical to Qwen3.5 MoE (softmax over all 512 router logits, top-10, renormalize, SwiGLU experts,
sigmoid-gated shared expert) with these widths. In the NVIDIA ModelOpt export, routed experts are
NVFP4 with one FP32 global scale (`weight_scale_2`) *per expert projection*; every other Text
weight is BF16.

## State per request

| State | Shape | Notes |
|---|---|---|
| GDN recurrent | 36 × 48 × 128 × 128 FP32 | as Qwen3.5 |
| GDN convolution | 36 × 3 × 10240 | as Qwen3.5 |
| PLE convolution | 9 × 10240 | last 9 normalized gated values |
| PLE token history | 2 tokens | last two tokens of the sequence |
| Main KV | 12 layers × 2 heads × 256 (K and V) per token | as Qwen3.5 |
| QSA index keys | 12 layers × 128 per token | phase 2 |

## Implementation status

Phase 1 runs Text Prefill and ordinary decode from the NVIDIA ModelOpt NVFP4 export
(`tools/convert` recipe `qwen3_8_flash_next_nvfp4`):

- Routed experts execute from rank-3 NVFP4 banks with per-expert divisors (`routed_moe`, A16);
  every other Text weight is BF16 and runs through ordinary `linear` projections. Calls with
  fewer than 32 tokens use the memory-bound CUDA-core route; larger calls build a work list of
  64-pair tiles per active expert and run BF16 tensor-core MMA on weights decoded in shared
  memory (e2m1 x E4M3 is exact in BF16; the expert divisor applies to the FP32 accumulators).
- PLE hashing and row gathering run on the host from the Mapped FP8 table: Prefill uploads rows
  through pinned staging, ordinary decode carries them in the round ingress. The PLE convolution
  window is part of the StateImage.
- QSA is dense causal attention while every visible extent is within the dense-equivalent
  extent `budget + compress_ratio - 1` (2051); beyond it the layer selects tokens with
  `qsa_select` and attends with `qsa_sparse_attention` (BF16 tensor-core scores, FP16 P x V).
  Index keys are a BF16 plane per layer in the Main paged pool, so `max_context` above 2051
  requires BF16 or Fp8E4M3Row256 KV storage. Decode graph tiers past the extent form their own
  topology class.
- MTP (`--spec mtp`, converter `--components text,mtp`) follows SGLang's Qwen4ExpForCausalLMMTP:
  each target stream s of the pre-mixer state is fused as
  `hidden_projection(hidden_norm(streams))_s + embedding_projection(embedding_norm(e))`, then one
  full-attention hyper-connection block (QSA indexer, routed MoE) with its own paged KV (with an
  index plane) and the draft's own final stream mixer feed the shared output head. Chained drafts
  feed back the draft's pre-mixer streams. The checkpoint's FP8_PB_WO draft experts are
  re-encoded as NVFP4 banks at conversion.
- The carried hidden (continuation hidden, StateImage, MTP stem input) is the [S*H] pre-mixer
  stream state (`TextConfig::carried_hidden_width`); logits from it apply the final mixer first.
- Speculative verify records instead of updating state: GDN layers use
  `causal_conv1d_silu_record` + `gated_delta_net_replay_record` (folded by `gdn_replay_fold`,
  registered for 36 layers), and the PLE convolution uses `ple_conv_residual_record` +
  `ple_conv_replay_fold`. Verify columns' PLE rows are hashed on the host from the ledger and the
  host-held drafts and travel in the MTP round ingress.
- Vision and DFlash are rejected for this architecture.
