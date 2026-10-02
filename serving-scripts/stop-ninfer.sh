#!/bin/bash
# stop-ninfer.sh
# Stop the NInfer server started by start-ninfer.sh.
#
# At boot ninfer runs under ninfer.service; prefer `sudo systemctl stop ninfer`
# for that instance. This script also works on it (and on manual starts):
# ninfer-serve writes no PID file, so the process is found by cmdline match — the same
# `pkill -TERM -f ninfer-serve` that start-ninfer.sh prints, plus the waiting
# and verification that a bare pkill does not do.
#
# ── WHAT SIGTERM ACTUALLY DOES ────────────────────────────────────────────────
# apps/serve/main.cpp installs the SIGINT/SIGTERM handler only AFTER weights are
# loaded and warmup finishes; the handler calls HttpServer::stop(), which stops
# the cpp-httplib listen loop. Two consequences:
#   * A TERM sent during load/warmup hits the default disposition and kills the
#     process outright. That is safe here (nothing is persisted), just abrupt.
#   * Shutdown is not a drain: in-flight and streaming requests are cut off when
#     the socket closes. Let generations finish before stopping if that matters.
# Freeing ~30 GiB of VRAM takes a moment, so this script waits for the process
# to actually exit and for :8000 to come free instead of returning immediately.
#
# ── USAGE ─────────────────────────────────────────────────────────────────────
#   ./stop-ninfer.sh                # TERM, wait up to 30s, then KILL
#   ./stop-ninfer.sh --timeout 60   # allow longer before escalating
#   ./stop-ninfer.sh --force        # KILL immediately, no TERM
#   ./stop-ninfer.sh --status       # report only, change nothing
#
# This stops NInfer only. If :8000 is held by vLLM instead, that is a systemd
# unit: sudo systemctl stop vllm-qwen3.8

set -u

PATTERN="ninfer-serve"
SERVE_BIN="${NINFER_DIR:-$HOME/Projects/ninfer}/build/apps/ninfer-serve"
PORT="8000"
TIMEOUT=30
FORCE=0
STATUS_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --force)   FORCE=1 ;;
        --status)  STATUS_ONLY=1 ;;
        --timeout) shift; TIMEOUT="${1:-}" ;;
        "")        ;;
        *)         echo "usage: $0 [--force|--status] [--timeout SECONDS]" >&2; exit 2 ;;
    esac
    shift
done

case "$TIMEOUT" in
    ''|*[!0-9]*) echo "ERROR: --timeout wants whole seconds, got '${TIMEOUT}'" >&2; exit 2 ;;
esac

# A cmdline match alone is NOT enough to kill on: `pgrep -f ninfer-serve` also
# hits this script's own shell, an editor with the start script open, a
# `tail`/`grep` mentioning it — anything whose argv happens to contain the
# string. So every candidate is confirmed by its /proc/<pid>/exe, which is the
# real binary behind the process and cannot be faked by argv.
pids() {
    local pid exe
    for pid in $(pgrep -f "$PATTERN" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        exe=$(readlink -f "/proc/${pid}/exe" 2>/dev/null) || continue
        # A server started before a rebuild runs a replaced binary: "<path> (deleted)".
        exe=${exe% (deleted)}
        # Accept the configured build, or any binary actually named ninfer-serve
        # (a differently-located build still counts).
        case "$exe" in
            "$SERVE_BIN"|*/ninfer-serve) echo "$pid" ;;
        esac
    done
}

# Same list, with the cmdline, for display.
pids_verbose() {
    local pid
    for pid in $(pids); do
        echo "  $pid $(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null)"
    done
}

port_holder() {
    ss -ltnp 2>/dev/null | grep ":${PORT} " || true
}

PIDS=$(pids)

if [ -z "$PIDS" ]; then
    echo "No ninfer-serve process is running."
    HOLDER=$(port_holder)
    if [ -n "$HOLDER" ]; then
        echo "NOTE: something else is still listening on :${PORT}:"
        echo "$HOLDER" | sed 's/^/    /'
        if [ -n "$(docker ps -q --filter 'name=^vllm$' 2>/dev/null)" ]; then
            echo "  It is the vLLM container. Stop it with: sudo systemctl stop vllm-qwen3.8"
        fi
    fi
    exit 0
fi

echo "ninfer-serve running:"
pids_verbose | sed 's/^/  /'

if [ "$STATUS_ONLY" = 1 ]; then
    nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader 2>/dev/null \
        | sed 's/^/    gpu: /'
    exit 0
fi

# ── TERM, then wait for the process to go ─────────────────────────────────────
if [ "$FORCE" = 1 ]; then
    echo "Sending SIGKILL (--force)..."
    # shellcheck disable=SC2086
    kill -KILL $PIDS 2>/dev/null
else
    echo "Sending SIGTERM, waiting up to ${TIMEOUT}s..."
    # shellcheck disable=SC2086
    kill -TERM $PIDS 2>/dev/null
fi

WAITED=0
while [ -n "$(pids)" ] && [ "$WAITED" -lt "$TIMEOUT" ]; do
    sleep 1
    WAITED=$((WAITED + 1))
done

# ── Escalate if it ignored the TERM ───────────────────────────────────────────
LEFT=$(pids)
if [ -n "$LEFT" ]; then
    echo "Still alive after ${TIMEOUT}s, sending SIGKILL:"
    pids_verbose | sed 's/^/  /'
    # shellcheck disable=SC2086
    kill -KILL $LEFT 2>/dev/null
    WAITED=0
    while [ -n "$(pids)" ] && [ "$WAITED" -lt 10 ]; do
        sleep 1
        WAITED=$((WAITED + 1))
    done
    if [ -n "$(pids)" ]; then
        echo "ERROR: ninfer-serve survived SIGKILL — likely stuck in an uninterruptible"
        echo "  CUDA call. Check: nvidia-smi ; dmesg | tail"
        exit 1
    fi
fi

echo "ninfer-serve stopped."

# ── Confirm the port and the VRAM actually came back ──────────────────────────
WAITED=0
while [ -n "$(port_holder)" ] && [ "$WAITED" -lt 10 ]; do
    sleep 1
    WAITED=$((WAITED + 1))
done

HOLDER=$(port_holder)
if [ -n "$HOLDER" ]; then
    echo "NOTE: :${PORT} is still held by something else:"
    echo "$HOLDER" | sed 's/^/    /'
else
    echo "  :${PORT} is free."
fi

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
if [ -n "${USED:-}" ]; then
    echo "  VRAM in use now: ${USED} MiB"
    if [ "${USED:-0}" -gt 8000 ] 2>/dev/null; then
        nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader \
            | sed 's/^/    /'
    fi
fi
