#!/bin/bash
# start-flash-next.sh
# Qwen3.8-Flash-Next (qwen4_exp) NVFP4 served by NInfer on :8000.
#
# This is the experimental port on branch `flash-next` in the worktree
# ~/Projects/ninfer-flash-next — NOT the production checkout ~/Projects/ninfer
# that start-ninfer.sh / ninfer.service run. Text + images (--vision) with MTP
# speculative decoding; DFlash is rejected for this architecture.
#
# Normally you do not run this directly; use ./swap-model.sh flash-next, which
# stops the 27B service first and waits for this server to come up.
#
# ── VRAM BUDGET (96 GiB card) ─────────────────────────────────────────────────
#   BF16 + NVFP4 weights                ~72.5 GiB   (27B server must be stopped)
#   KV pool at 262,144 tokens, fp8       ~4 GiB     (12 attention layers only;
#                                                    the 36 GDN layers keep a
#                                                    fixed state per lane)
#   -> nothing else large (ComfyUI) can share the card while this runs.
#
# ── HOST RAM ──────────────────────────────────────────────────────────────────
# The 53.7 GB PLE n-gram table stays on disk and is memory-mapped; every token
# reads rows from it through the page cache. Pinned Host KV comes out of the
# same 62 GB, so it is kept at 2 GiB here (27B uses 8 GiB) to leave the page
# cache for the PLE table.
#
# ── FLAG NOTES ────────────────────────────────────────────────────────────────
# --max-context / --kv-capacity 262144
#     The model's native window. Beyond 2,051 tokens attention is query-sparse
#     (QSA), which requires bf16 or fp8 KV.
# --kv-dtype fp8
#     Same as the 27B config. Validated: perplexity within 0.1% of bf16 at 8K.
# --vision
#     Image input (the Qwen3.5 vision tower; artifact converted with --components
#     text,mtp,vision, vision projections at Q8). Like the 27B, residency is fixed at
#     startup: without this flag every image request fails with HTTP 400
#     "vision_disabled". Validated 2026-10-01 against the PyTorch reference (chart 121/128,
#     image + 3K text 88/96 teacher-forced tokens, mismatches near-ties).
# --spec mtp --draft-tokens 3
#     The model's own one-layer MTP draft head (artifact converted with --components text,mtp;
#     its FP8 experts re-encoded as NVFP4). Measured 2026-10-01: decode 77 -> 170 tok/s on a
#     512-token coding answer (87.7% acceptance, 3.6 tok/round); 54 -> 90 tok/s at ~8K context.
#     K=1: 124 tok/s, K=2: 150 tok/s. Greedy output matches non-speculative decoding except
#     at near-ties.
# --model-id qwen3.8-flash-next
#     The name /v1/models reports. Requests are not rejected for naming another
#     model (a client still configured for qwen3.8-27b gets Flash-Next), so
#     check ./swap-model.sh status to see which model is answering.
#
# ── USAGE ─────────────────────────────────────────────────────────────────────
#   ./start-flash-next.sh               # foreground (Ctrl-C stops)
#   ./start-flash-next.sh --background  # detach, append to serve.log

set -u

NINFER_DIR="${FLASH_NEXT_DIR:-$HOME/Projects/ninfer-flash-next}"
SERVE_BIN="${NINFER_DIR}/build/apps/ninfer-serve"
MODEL="${NINFER_DIR}/models/qwen3_8_flash_next_nvfp4_mtp_vision.ninfer"
MODEL_ID="qwen3.8-flash-next"
HOST="0.0.0.0"
PORT="8000"
LOG="${NINFER_DIR}/serve.log"

BACKGROUND=0
case "${1:-}" in
    --background) BACKGROUND=1 ;;
    "")           ;;
    *)            echo "usage: $0 [--background]" >&2; exit 2 ;;
esac

if [ ! -x "$SERVE_BIN" ]; then
    echo "ERROR: ${SERVE_BIN} is missing or not executable."
    echo "  Build it:  PATH=/usr/local/cuda-13.1/bin:\$PATH cmake --build ${NINFER_DIR}/build -j"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "ERROR: artifact not found: ${MODEL}"
    exit 1
fi

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    echo "ERROR: something is already listening on :${PORT}."
    echo "  Use ./swap-model.sh flash-next, which stops the current server first."
    exit 1
fi

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
if [ "${USED:-0}" -gt 8000 ] 2>/dev/null; then
    echo "WARNING: ${USED} MiB of VRAM is already in use; this model needs ~77 GiB. Occupants:"
    nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader | sed 's/^/    /'
fi

cd "$NINFER_DIR" || exit 1

CMD=("$SERVE_BIN" "$MODEL"
     --host "$HOST" --port "$PORT"
     --model-id "$MODEL_ID"
     --vision
     --max-context 262144
     --kv-capacity 262144
     --max-concurrency 2
     --kv-dtype fp8
     --device-state-slots 2
     --host-state-slots 4
     --host-kv-mib 2048
     --spec mtp
     --draft-tokens 3
     --preserve-thinking)

if [ "$BACKGROUND" = 1 ]; then
    nohup "${CMD[@]}" >> "$LOG" 2>&1 &
    echo "ninfer-serve (Flash-Next) started in the background, pid $!"
    echo "  log:  tail -f ${LOG}"
else
    exec "${CMD[@]}"
fi
