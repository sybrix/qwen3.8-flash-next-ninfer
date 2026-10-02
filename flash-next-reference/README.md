# Flash-Next reference and validation tools

These tools checked the NInfer port of Qwen3.8-Flash-Next against an independent PyTorch
implementation. They also exercised the real server.

## Reference model

| Script | Purpose |
|---|---|
| `flashnext_ref.py` | Layer-streaming PyTorch reference. It runs the official `transformers` `qwen4_exp` modules one decoder layer at a time from the NVFP4 checkpoint, so the 130 GB model never has to be resident. Routed experts are dequantized to BF16, and the n-gram table is gathered row by row from the memory-mapped FP8 shards. Outputs per-layer streams, logits, NLL and argmax. `--nll-only` runs long contexts in bounded memory; `--mtp` also runs the MTP draft head teacher-forced. That head follows SGLang's `Qwen4ExpForCausalLMMTP`, because `transformers` skips the MTP weights. `--messages` takes CLI-style messages with images. The HF processor renders them, the checkpoint's vision tower embeds the images, and `get_rope_index` supplies 3-axis positions. `--generated` appends NInfer's tokens for a teacher-forced check. |
| `compare_decode.py` | Teacher-forces a NInfer generation through the reference. For each step it reports whether NInfer's token is the reference argmax, and the reference margin, so real errors can be told apart from near-ties. |
| `e2e_decode.sh` | Greedy-decodes with the NInfer CLI, then runs `compare_decode.py` on the result. |
| `gen_synthetic.py` | Writes a synthetic 4-layer checkpoint with the real per-layer shapes: 512 NVFP4 experts, hyper-connections, the PLE table, with `--mtp` an FP8-block MTP head, and with `--vision` the full-size vision tower. Fast end-to-end tests run on it without loading the full model. |

## Server and integration tests

| Script | Purpose |
|---|---|
| `serve_test.py` | Sends the same prompts sequentially and concurrently (only the decode batch size differs), then a two-turn Responses chain to check prefix reuse. |
| `turn_test.py`, `turn_compare.py`, `check_text.py` | Multi-turn continuation checks and output comparison helpers. |
| `mtp_real_test.sh` | Real-model MTP benchmark: perplexity sanity check, then decode speed and acceptance at K=1/2/3 at short and ~8K context, then greedy agreement with non-speculative output. It stops the production server and always restarts it on exit, and refuses to start unless the artifact is a single complete file. |
| `swap_mtp_test.sh` | Swaps the server to Flash-Next, sends OpenAI, Anthropic, long-context and concurrent requests, then swaps back. |
| `cold_prefill_test.sh` | Drops the OS page cache and measures cold-cache prompt latency on the server. |
| `vision_real_test.sh` | Real-model vision benchmark: image prompts (a chart, and a photo plus ~3K tokens of text) with and without MTP, then teacher-forced reference checks. It restores whichever model was serving. |
| `vision_quant_check.py` | Measures how far a vision-weight quantization recipe moves real image embeddings from BF16, for any Qwen3.5-family checkpoint. |

## Usage notes

- **Python environment:** `torch`, `safetensors`, and a `transformers` build that includes
  `qwen4_exp`. For example: `uv venv && uv pip install torch safetensors git+https://github.com/huggingface/transformers`.
- **Checkpoint:** point `--checkpoint` at a local snapshot of `nvidia/Qwen3.8-Flash-Next-NVFP4`.
- **Default paths:** the shell scripts assume the engine is at `$HOME/Projects/ninfer-flash-next`
  and these tools are at `$HOME/Projects/flash-next-ref`. The serving scripts default to the
  sibling `serving-scripts/`; override with `SERVING_SCRIPTS=...`.
- **Test texts:** the texts (`pg19_4096.txt`, `pg19_8192.txt`, `long_prompt.txt`) are not
  included. They are prefixes of the PG-19 streams in
  `ninfer-flash-next/eval/corpora/perplexity-1m`, re-tokenized to exact lengths with the model
  tokenizer.
- **Long jobs:** the real-model scripts run long. Start them detached
  (`setsid nohup ./mtp_real_test.sh >/dev/null 2>&1 < /dev/null &`) so they finish, and restore
  the production server, even if your shell goes away.
