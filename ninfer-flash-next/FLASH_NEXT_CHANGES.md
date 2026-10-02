# Qwen3.8-Flash-Next changes to NInfer

This directory is a modified copy of [NInfer](https://github.com/Neroued/ninfer), based on
upstream commit `d44ab58`. NInfer is licensed under the Apache License 2.0 (see `LICENSE`). This
file is the notice of modifications required by section 4(b).

The changes add the `qwen4_exp` architecture (Qwen3.8-Flash-Next) to the Qwen3.5 model family
runtime. [`docs/maintainer/qwen4_exp-model.md`](docs/maintainer/qwen4_exp-model.md) gives the
model mathematics and the implementation status.

## Phases

| Phase | Adds |
|---|---|
| 1 | Text model: 4-stream hyper-connections, Gated DeltaNet, 24/2-head gated attention, 512-expert NVFP4 routed MoE, the host-side per-layer n-gram embedding, and the ModelOpt NVFP4 converter recipe. |
| 2 | Query-sparse attention: an index-key KV plane plus selection and sparse-attention kernels, which unlock 262K context. |
| 3 | Tensor-core MoE prefill and tensor-core sparse attention (~9× prompt throughput). |
| MTP | The model's MTP draft head for speculative decoding, with the GDN and PLE state record/fold rollback it needs. |
| Prefill I/O | The n-gram table gather is parallelized so cold page-cache prompts aren't disk-latency bound. |
| Vision | Image input through the Qwen3.5 vision tower (Q8, MLP zero-padded 4304 → 4352), 3-axis RoPE positions recorded in the KV cache for QSA. |

## Commit log (oldest first)

### docs(qwen4_exp): add Qwen3.8-Flash-Next model reference


### feat(ops): add sigmoid gate to gated_rmsnorm

Qwen4Exp GDN applies its output gate with sigmoid instead of SiLU.


### feat(ops): add Qwen4Exp hyper-connection Ops

Grouped zero-centred RMSNorm, sigmoid-weighted stream mixing, gate
preparation and in-place stream injection for the 4-stream residual.


### feat(ops): add Qwen4Exp PLE Ops

FP8 n-gram row dequantization, the stream-gated value and the dilated
causal convolution residual with sequence and slot-snapshot state forms.


### feat(artifact): add raw FP8 words, NVFP4 expert banks and the Qwen4Exp converter

fp8_e4m3fn stores raw E4M3FN words with a separately bound multiplier.
block_scale_k16_m128x4_bank_v1 stores a rank-3 [E,N,K] NVFP4 bank in
which every expert keeps its own weight divisor, as ModelOpt exports it.
The qwen4_exp adapter and qwen3_8_flash_next_nvfp4 recipe import the
NVIDIA NVFP4 checkpoint: experts as banks, the PLE table as raw FP8,
everything else BF16.


### feat(ops): add routed_moe for NVFP4 expert banks

Top-K softmax routing over 512 experts, counting-sort grouping by expert,
A16 NVFP4 gate/up and down passes with per-expert divisors, and a
fixed-order combine with the sigmoid-gated shared-expert output.


### feat(linear): register Qwen3.8-Flash-Next BF16 shapes

Add BF16 A16 shape-closed ladders for the hidden-2560 model problems
[10240,2560], [6144,2560], [2560,2560], [1280,2560], [640,2560],
[513,2560], [512,2560], [96,2560], [48,2560], [248320,2560],
[2560,6144], [2560,640], [320,10240], [10240,320] and [4,10240], each
at every positive T.

K=320 admits only 64-wide MMA K tiles, N=4 only SIMT/GEMV row blocks,
and N=513 runs SIMT row blocks of three for T<=4 and otherwise a
[0,512) MMA head plus a one-row SIMT tail into the same strided output.

The BF16 linear test qualifies every new problem against the FP64
oracle across decode, selector boundaries and prefill through T=8192,
plus [48,2560] executed on both row halves of a [96,2560] parent.


### feat(qwen4_exp): run Qwen3.8-Flash-Next text through the Qwen3.5 family runtime

Adds the Qwen4ExpForCausalLM architecture: config parsing, bindings for
hyper-connection mixers, QSA indexer projections and PLE, native
parameters with NVFP4 expert banks, and the hyper-connection layer loop
for Prefill and ordinary decode. QSA runs as dense attention, which is
exact up to the 2051-token dense-equivalent extent.

PLE n-gram rows are hashed and gathered on the host from the mapped FP8
table: Prefill stages them through pinned memory, ordinary decode carries
them in its round ingress. The PLE convolution state lives in the
StateImage so checkpoints and prefix reuse preserve it. Artifacts gain a
Mapped residency so the 47 GiB table stays in the page cache.


### feat(qwen4_exp): bound Phase-1 contexts to the dense-equivalent QSA extent


### bench(ops): time routed_moe; fix(convert): bind PLE table shards as raw FP8

The PLE shards are FP8 matrices without row scales, so the generic matrix
resolver must not treat them as fp8_e4m3fn_row_bf16 sources.


### feat(rope): add fixed text path for 24/2 head geometry

Register D256/R64 Text 1-D and MRoPE fixed launches for 24 query and
2 KV heads (previously served by the generic kernel) and qualify them
against the FP64 oracle at decode, prefill and padded MRoPE shapes.


### feat(attention): register 24/2 causal softmax attention geometry

Admit [D,Hq,Hkv]=[256,24,2] (group 12) in causal_softmax_attention and
causal_softmax_attention_cached for BF16, INT8-G64, FP8-row, NVFP4-G16
and K8V4 caches across every route (fused grouped append, parallel
grouped, tiled prefill, batched verify/decode, split-KV merge).

Geometry dispatch now goes through one registered table keyed by both
head counts; plans take kv_heads explicitly instead of inferring it
from the query head count. Quantized grouped kernels pack at most 64
query rows, so the grouped/parallel token tile becomes
min(8, 64/group): unchanged (8) for groups 6 and 8, five for group 12.
NVFP4/K8V4 halved parallel tiles extend to 3..5 for group 12.

Tests add the geometry to every per-geometry loop and mirror the
16/2-specific cases, plus group-12 grouped/parallel width boundaries
(W=5/6/10/11) for single and batched rows. The op bench accepts
--geometry d256-h24-kv2.


### fix(qwen4_exp): bind the Qwen4Exp runtime on every Program execution core

Prefill and the speculative decode cores omitted the runtime, so Prefill
had no PLE staging.


### feat(ops): add Qwen4Exp query-sparse attention Ops and index-key KV plane

qsa_prepare_query, qsa_index_append, qsa_select (pooled block scoring and
top-budget radix selection) and qsa_sparse_attention over BF16/FP8 paged
KV. Raw index keys are one BF16 plane per layer in the Main KV pool, so
they share its frontier, paging, replicas and prefix reuse.


### feat(qwen4_exp): run query-sparse attention beyond the dense-equivalent extent

Full-attention layers always project and store QSA index keys; when a
unit's visible extent exceeds budget + ratio - 1 they append K/V, select
the top block budget per column and attend only to the selected tokens.
Decode graph tiers past that extent form their own topology. Contexts
beyond it require BF16 or FP8 Main KV. Adds the batched kv_cache_append
entry and the 128x2560 BF16 linear shape.


### perf(qwen4_exp): tensor-core routed MoE prefill and QSA sparse attention

routed_moe calls with at least 32 tokens build a work list of 64-pair tiles per active
expert and run BF16 WMMA on NVFP4 weights decoded in shared memory (e2m1 x E4M3 is exact
in BF16; the per-expert divisor applies to FP32 accumulators). qsa_sparse_attention scores
with BF16 tensor cores and accumulates P x V in FP16, with an FP32 online softmax.

Real model: prefill 3.6k tok/s on a 7.7K prompt (was ~400); perplexity unchanged
(4K 1.7992 vs reference 1.8019, 8K 1.7757 vs 1.7752). All 131 tests pass.


### feat(qwen4_exp): MTP speculative decoding

Implements the Qwen3.8-Flash-Next draft head (SGLang Qwen4ExpForCausalLMMTP semantics):
per-stream fusion of the target's pre-mixer hyper-connection state with the next token's
embedding, one full-attention hyper-connection block (QSA indexer, routed MoE) with its own
paged KV and index plane, the draft's final stream mixer, and the shared output head.

- Converter: --components text,mtp; FP8_PB_WO draft experts re-encoded as NVFP4 banks.
- The carried hidden (continuation, StateImage, MTP stem) is the [S*H] pre-mixer state.
- Verify records instead of mutating state: new causal_conv1d_silu_record,
  ple_conv_residual_record and ple_conv_replay_fold ops (bit-exact vs snapshot forms);
  gdn_replay_fold registered for the 36-layer geometry. Verify-column PLE rows are hashed
  on the host and carried in the MTP ingress.
- MTP CUDA graph tiers get dense/sparse/straddling topology classes; the stream broadcast
  in qwen4_exp_embed is a kernel so graph updates stay legal.

Real model (fp8 KV): decode 77 -> 170 tok/s at K=3 (87.7% acceptance), 54 -> 90 tok/s at
~8K context. Teacher-forced vs the PyTorch reference: 265/270, all mismatches near-ties.


### perf(qwen4_exp): gather Prefill PLE rows on parallel threads

Each n-gram row is a random read from the memory-mapped 53.7 GB table. With a cold page cache
every read is a synchronous NVMe page fault (~60 us), so a 7.7K-token prompt spent ~7.6 s in
the serial host gather (server TTFT 8.3 s after the cache had been evicted). Gathers of 128+
columns now split across up to 16 threads, overlapping the faults (~8x measured on the table:
~0.9 s cold for 7.7K tokens). Output is byte-identical; decode/verify gathers stay serial.


### feat(qwen4_exp): image input via the Qwen3.5 vision tower

Qwen3.8-Flash-Next's vision tower, image placeholders and 3-axis Text MRoPE are identical to
Qwen3.5's (transformers qwen4_exp vs qwen3_5), so the existing Vision runtime is reused:

- Converter: --components text,mtp,vision. The vision MLP is zero-padded 4304 -> 4352 (exact)
  so every vision projection runs at Q8; new Q8 shapes 3456x1152, 1152x1152, 4352x1152,
  1152x4352, 1152x1536, 2560x4608 (column extent up to 131072 patches). The Qwen3.5 Q4/Q5
  vision mix moved real image embeddings ~35% from BF16; Q8 ~6%.
- Prefill scatters vision embeddings into the [H,T] token embedding before the stream
  broadcast; PLE hashes the multimodal prompt ids.
- QSA: index queries rotate with the columns' RoPE positions ([T] or [T,3]); a per-token RoPE
  position plane (I32 [4,64,1,pages]) in the Main and MTP paged pools lets pooled blocks rotate
  with their first token's recorded multimodal position. New qsa overloads; tests cover 3-axis
  positions and fail without the plane.

Real model vs the PyTorch reference (teacher-forced): chart image 121/128, image + 3K-token text
(sparse QSA) 88/96, mismatches near-ties. Decode with MTP K=3: 110 tok/s (short), 91 tok/s (3.4K).


