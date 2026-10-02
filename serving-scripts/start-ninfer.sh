#!/bin/bash
# start-qwen3.8-27b-nvfp4-ninfer.sh
# Qwen3.8-27B NVFP4 served by NInfer (native CUDA engine), NOT vLLM, on :8000.
#
# Same weights as start-qwen3.8-27b-nvfp4.sh, but converted to NInfer's own
# single-file container format (models/qwen3_8_27b_nvfp4.ninfer) and served by
# ./build/apps/ninfer-serve. NInfer speaks the OpenAI Chat, OpenAI Responses,
# AND Anthropic Messages wire protocols from one process, which is why Claude
# Code can point at it directly with no translating proxy — unlike vLLM, which
# needs claude-gemma-proxy.py to lift mid-conversation system messages.
#
# ── WHY --vision IS NOT OPTIONAL HERE ─────────────────────────────────────────
# NInfer freezes capability residency at startup. Without --vision the Vision
# tower weights and the Vision-specific unified-workspace extent are never
# allocated, and EVERY media request is rejected during prepare with
#   HTTP 400 | vision disabled
# There is no per-request opt-in and no lazy load — docs/serving.md is explicit:
# "A later request cannot enable a capability omitted at startup." A Claude Code
# session that pastes one screenshot into a server started without this flag
# gets a 400 and the image stays poisoned in context until /clear.
#
# This bit the box on 2026-09-06 (serve.log req#117, req#118). The flag costs
# ~1.4 GiB of VRAM. Leave it on.
#
# ── VRAM BUDGET (measured, --vision on, 96 GiB card) ──────────────────────────
#   NVFP4 weights                       20.0 GiB   ("weights ready" line)
#   runtime: KV pool + workspace         9.4 GiB   ("capacity" line)
#   Vision + media staging               ~1.4 GiB   (cache 1.0 + live 2.0 caps)
#   ------------------------------------------------
#   ninfer-serve RSS on device          29.8 GiB   (nvidia-smi, steady state)
#   -> leaves ~61 GiB for ComfyUI and the desktop.
#
# Host RAM is separate and also pinned up front: 1.15 GiB of Host StateImages
# plus 8.00 GiB of Host KV (--host-kv-mib 8192).
#
# ── FLAG NOTES ────────────────────────────────────────────────────────────────
# --max-context / --kv-capacity 262144
#     262,144 — the artifact's native window (the model card benchmarks at exactly
#     this limit). Do not raise it further: serve_options.cpp validates only
#     "> 0" and the KV range, so a larger value starts fine and silently
#     extrapolates RoPE past training. --kv-capacity is the explicit
#     shared Main Text KV pool; passing both pins the pool instead of letting it
#     size itself from leftover memory, so startup fails loudly rather than
#     silently shrinking the usable context. Logs "pages 4,096/8,192".
# --max-concurrency 2
#     Valid range is 1..8. Two active lanes is right for a single-user box; each
#     lane carries its own StateImage guarantee.
# --kv-dtype fp8
#     Halves the KV pool against the bf16 default. This is what makes 256K fit
#     alongside 20 GiB of weights.
# --device-state-slots 2 --host-state-slots 8 --host-kv-mib 8192
#     Context cache. Total Device StateImage capacity is max-concurrency +
#     device-state-slots = 4: two active guarantees plus a two-slot checkpoint
#     pool. The eight pinned Host slots and 8 GiB of pinned Host KV retain
#     inactive continuations when the device is under pressure, so switching
#     between conversations does not re-prefill from scratch.
# --spec dflash2 --draft-tokens 7 --lm-head-draft
#     DFlash2 masked-block drafter (needs the v3 artifact with DFlash2 companion
#     weights, downloaded 2026-10-01; ~1.65 GiB more weights than MTP3). K=7 is
#     the checkpoint recommendation. Upstream nvfp4 C=1 corpus decode: 208 tok/s
#     vs 169 for MTP3; no gain on free-form story text. Previous setting was
#     --spec mtp --draft-tokens 3 (acceptance ~50%). --lm-head-draft loads the
#     optimized proposal head. Speculative residency is frozen at startup
#     exactly like Vision is.
# --preserve-thinking
#     Keeps closed-turn assistant reasoning across turns. Claude Code replays
#     whole conversations, so dropping it would discard prior thinking blocks.
#
# ── USAGE ─────────────────────────────────────────────────────────────────────
#   ./start-qwen3.8-27b-nvfp4-ninfer.sh              # foreground (Ctrl-C stops)
#   ./start-qwen3.8-27b-nvfp4-ninfer.sh --background  # detach, append to serve.log
#
# Verify vision is actually live after start (not just that /health says ok):
#   ./start-qwen3.8-27b-nvfp4-ninfer.sh --check
# A plain /health probe passes on a vision-disabled server, so it proves nothing
# about images. --check sends a real 1x1 PNG and fails if the server 400s.

set -u

NINFER_DIR="${NINFER_DIR:-$HOME/Projects/ninfer}"
SERVE_BIN="${NINFER_DIR}/build/apps/ninfer-serve"
MODEL="${NINFER_DIR}/models/qwen3_8_27b_nvfp4.ninfer"
HOST="0.0.0.0"
PORT="8000"
LOG="${NINFER_DIR}/serve.log"

BACKGROUND=0
case "${1:-}" in
    --background) BACKGROUND=1 ;;
    --check)      CHECK_ONLY=1 ;;
    "")           ;;
    *)            echo "usage: $0 [--background|--check]" >&2; exit 2 ;;
esac

# ── --check: prove the running server actually accepts an image ───────────────
if [ "${CHECK_ONLY:-0}" = 1 ]; then
    python3 - "$PORT" <<'PY'
import json, sys, urllib.error, urllib.request
port = sys.argv[1]
base = f"http://127.0.0.1:{port}"
# A 1x1 black PNG. The Vision frontend still does its normal resize/patch expansion.
png = ("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUB"
       "AScY42YAAAAASUVORK5CYII=")
try:
    with urllib.request.urlopen(base + "/health", timeout=5) as r:
        print("health:", r.read().decode().strip())
except OSError as e:
    sys.exit(f"FAIL: no server on {base}: {e}")
body = json.dumps({
    "model": "qwen3.8-27b", "max_tokens": 8,
    "messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64",
                                     "media_type": "image/png", "data": png}},
        {"type": "text", "text": "Reply with one word."}]}],
}).encode()
req = urllib.request.Request(base + "/v1/messages", data=body,
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        usage = json.load(r).get("usage", {})
        # input_tokens excludes anything served from the prefix cache, so a
        # small number here after a repeat run is cache reuse, not a short prompt.
        print(f"vision: OK (input_tokens={usage.get('input_tokens')},"
              f" cache_read={usage.get('cache_read_input_tokens')})")
except urllib.error.HTTPError as e:
    detail = e.read().decode(errors="replace")[:300]
    if "vision" in detail:
        sys.exit(f"FAIL: server is running WITHOUT --vision (HTTP {e.code}): {detail}")
    sys.exit(f"FAIL: image request returned HTTP {e.code}: {detail}")
PY
    exit $?
fi

# ── The binary and artifact must exist ────────────────────────────────────────
if [ ! -x "$SERVE_BIN" ]; then
    echo "ERROR: ${SERVE_BIN} is missing or not executable."
    echo "  Build it:  PATH=/usr/local/cuda-13.1/bin:\$PATH cmake --build ${NINFER_DIR}/build -j"
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "ERROR: artifact not found: ${MODEL}"
    exit 1
fi

# ── Verify performance settings are active ────────────────────────────────────
SM0=$(nvidia-smi -i 0 --query-gpu=clocks.current.sm --format=csv,noheader 2>/dev/null | tr -d ' MHz')
GOV=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null)

if [ "${SM0:-0}" -lt 1000 ] 2>/dev/null; then
    echo "WARNING: GPU 0 SM clock is ${SM0} MHz (expected ~3090 MHz)"
    echo "  Run: sudo systemctl start vllm-gpu-performance"
fi

if [ "$GOV" != "performance" ]; then
    echo "WARNING: CPU governor is '$GOV' (expected: performance)"
    echo "  Run: sudo cpupower frequency-set -g performance"
fi

# ── :8000 is shared with the vLLM units — only one server can own it ──────────
# At boot ninfer.service runs this script and owns :8000 (vllm-qwen3.8.service is
# disabled; the two units Conflict, so starting either stops the other).
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    echo "WARNING: something is already listening on :${PORT}."
    if [ -n "$(docker ps -q --filter 'name=^vllm$' 2>/dev/null)" ]; then
        echo "  It is the vLLM container. Run: sudo systemctl stop vllm-qwen3.8"
    fi
    if pgrep -f 'ninfer-serve' >/dev/null 2>&1; then
        echo "  An ninfer-serve is already running:"
        pgrep -af 'ninfer-serve' | sed 's/^/    /'
        echo "  Stop it first: pkill -TERM -f ninfer-serve"
    fi
    exit 1
fi

# ── Warn if something else already holds the VRAM we budgeted for ─────────────
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
if [ "${USED:-0}" -gt 8000 ] 2>/dev/null; then
    echo "WARNING: ${USED} MiB of VRAM is already in use before NInfer starts."
    echo "  This config needs ~30 GiB. Occupants:"
    nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader | sed 's/^/    /'
fi

echo "GPU SM clocks:  $(nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader | paste -sd' / ') MHz"
echo "CPU governor:   ${GOV}"
echo "Artifact:       ${MODEL}"
echo "Vision:         enabled (--vision)"
echo

# ── Start NInfer ──────────────────────────────────────────────────────────────
cd "$NINFER_DIR" || exit 1

CMD=("$SERVE_BIN" "$MODEL"
     --host "$HOST" --port "$PORT"
     --vision
     --max-context 262144
     --kv-capacity 262144
     --max-concurrency 2
     --kv-dtype fp8
     --device-state-slots 2
     --host-state-slots 8
     --host-kv-mib 8192
     --spec dflash2
     --draft-tokens 7
     --lm-head-draft
     --preserve-thinking)

if [ "$BACKGROUND" = 1 ]; then
    nohup "${CMD[@]}" >> "$LOG" 2>&1 &
    echo "ninfer-serve started in the background, pid $!"
    echo "  log:    tail -f ${LOG}"
    echo "  verify: $0 --check"
    echo "  stop:   pkill -TERM -f ninfer-serve"
else
    exec "${CMD[@]}"
fi
