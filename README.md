# Qwen3.8-Flash-Next on NInfer

This repository runs **Qwen3.8-Flash-Next** — Qwen's hybrid-attention mixture-of-experts model,
architecture `qwen4_exp` — on [NInfer](https://github.com/Neroued/ninfer). NInfer is a native
CUDA inference engine and OpenAI/Anthropic-compatible server. The target is a single NVIDIA
RTX PRO 6000 Blackwell (96 GB, sm_120), using the NVIDIA ModelOpt NVFP4 checkpoint
[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4).

Upstream NInfer did not support this architecture. The port adds every feature the model needs:

- **4-stream hyper-connections:** the residual is carried as 4 streams instead of one.
- **Gated DeltaNet linear attention:** used in 36 of the 48 layers.
- **Query-sparse attention (QSA):** a learned block selector lets the 12 full-attention layers
  reach the model's native **262,144-token** context.
- **512-expert NVFP4 MoE:** top-10 routing plus a shared expert.
- **Per-layer n-gram embedding (PLE):** a 53.7 GB FP8 hash table, memory-mapped from disk.
- **The model's own MTP head:** used for speculative decoding.

## Results

Measured on one RTX PRO 6000 Blackwell Max-Q, with FP8 KV cache:

| | Result |
|---|---|
| Accuracy vs. the PyTorch reference (perplexity, 4K / 8K context) | 1.7992 vs 1.8019 / 1.7757 vs 1.7752 (≤ 0.15%) |
| Prompt processing (7.7K-token prompt) | ~3,500 tok/s |
| Decode, no speculation | 77 tok/s (54 tok/s at 8K context) |
| Decode, MTP with 3 draft tokens, greedy | **170 tok/s**; 88% of drafts accepted |
| Decode, MTP, ~8K context | 90 tok/s |
| GPU memory | ~80 GiB with 262K context, 2 concurrent requests |

**Correctness:** greedy MTP output matches non-speculative output. Teacher-forced through the
reference, it agreed on 265 of 270 tokens, and every mismatch was a near-tie (logit gap ≤ 0.5).

## Repository layout

| Folder | What it is |
|---|---|
| [`ninfer-flash-next/`](ninfer-flash-next/) | NInfer with the Qwen3.8-Flash-Next port: model runtime, CUDA kernels, converter. A snapshot of the `flash-next` branch, based on upstream NInfer `d44ab58`. See [`FLASH_NEXT_CHANGES.md`](ninfer-flash-next/FLASH_NEXT_CHANGES.md). |
| [`flash-next-reference/`](flash-next-reference/) | PyTorch reference and validation tools: a layer-streaming reference, teacher-forced comparison, a synthetic-checkpoint generator, server tests and real-model test scripts. |
| [`serving-scripts/`](serving-scripts/) | Scripts that serve the model and swap a single GPU between Flash-Next and another NInfer model (Qwen3.8-27B) on port 8000. Includes a systemd unit. |

## Quick start

```bash
# 1. Build the engine (CUDA 13.1 validated; the build targets sm_120a only)
cd ninfer-flash-next
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# 2. Convert the checkpoint to a NInfer artifact (~30 min, ~131 GB, needs ~131 GB free disk).
#    --components text,mtp includes the MTP draft head.
#    --max-file-bytes keeps it one file; the memory-mapped n-gram table must not span files.
python -m tools.convert --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
    --recipe qwen3_8_flash_next_nvfp4 --components text,mtp \
    --name qwen3.8-flash-next --max-file-bytes 200000000000 \
    --out models/qwen3_8_flash_next_nvfp4_mtp.ninfer

# 3. Serve it (OpenAI + Anthropic APIs on :8000)
../serving-scripts/start-flash-next.sh
```

The serving scripts default to `$HOME/Projects/ninfer-flash-next`. Override this with
`FLASH_NEXT_DIR=/path/to/ninfer-flash-next`.

## Requirements and limits

- **GPU:** a 96 GB Blackwell GPU. Weights use about 72.5 GiB.
- **RAM and disk:** about 64 GB of RAM and a fast NVMe. The 53.7 GB n-gram table stays on disk
  and is read through the page cache, so a cold cache adds about 0.4 s to an 8K prompt.
- **Text only:** vision input and DFlash speculation are not implemented for this architecture.
- **KV cache format:** contexts beyond 2,051 tokens use sparse attention, which needs a BF16 or
  FP8 KV cache.

## How the port was validated

Every phase was checked against a layer-streaming PyTorch reference built from the official
`transformers` `qwen4_exp` modules. The MTP head had to be built from SGLang's implementation,
because `transformers` does not load the MTP weights. Validation used:

- teacher-forced logits and greedy-token agreement;
- perplexity at 4K and 8K context;
- concurrency, prefix-reuse and multi-turn tests on a synthetic 4-layer checkpoint with the
  real per-layer shapes;
- bit-exact unit tests for every new kernel.

[`ninfer-flash-next/docs/maintainer/qwen4_exp-model.md`](ninfer-flash-next/docs/maintainer/qwen4_exp-model.md)
describes the model's mathematics and how the implementation maps onto NInfer.

## License

`ninfer-flash-next/` is a modified copy of NInfer and stays under the
[Apache License 2.0](ninfer-flash-next/LICENSE). Vendored third-party code keeps its own license.
The other folders are also released under Apache-2.0.

Model weights are not included. See the model card on Hugging Face for its license.
