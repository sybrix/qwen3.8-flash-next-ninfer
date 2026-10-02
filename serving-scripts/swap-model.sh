#!/bin/bash
# swap-model.sh
# Switch the :8000 NInfer server between Qwen3.8-27B and Qwen3.8-Flash-Next.
#
# The two cannot run together: Flash-Next needs ~77 GiB of the 96 GiB card and
# the 27B server holds ~30 GiB.
#
#   27b         ninfer.service (production checkout, start-ninfer.sh). This is
#               also what comes back after a reboot.
#   flash-next  start-flash-next.sh from the ~/Projects/ninfer-flash-next
#               worktree, run in the background (not a systemd unit).
#
# ── USAGE ─────────────────────────────────────────────────────────────────────
#   ./swap-model.sh flash-next   # stop 27B, start Flash-Next, wait until healthy
#   ./swap-model.sh 27b          # stop Flash-Next, start ninfer.service
#   ./swap-model.sh status       # which model is serving :8000
#
# Swapping cuts off in-flight requests (stop-ninfer.sh explains why).

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
PORT="8000"
NINFER_DIR="${NINFER_DIR:-$HOME/Projects/ninfer}"                       # production 27B checkout
FLASH_NEXT_DIR="${FLASH_NEXT_DIR:-$HOME/Projects/ninfer-flash-next}"  # this port
PROD_BIN="${NINFER_DIR}/build/apps/ninfer-serve"
FLASH_BIN="${FLASH_NEXT_DIR}/build/apps/ninfer-serve"
FLASH_LOG="${FLASH_NEXT_DIR}/serve.log"
# Flash-Next reads a 129 GB artifact; a cold page cache makes the first load slow.
HEALTH_TIMEOUT=600

usage() { echo "usage: $0 flash-next|27b|status" >&2; exit 2; }

# Which build owns the running ninfer-serve, by its real executable.
serving() {
    local pid exe
    for pid in $(pgrep -x ninfer-serve 2>/dev/null); do
        exe=$(readlink -f "/proc/${pid}/exe" 2>/dev/null) || continue
        case "$exe" in
            "$PROD_BIN")  echo "27b" ; return ;;
            "$FLASH_BIN") echo "flash-next" ; return ;;
            *)            echo "other:${exe}" ; return ;;
        esac
    done
    echo "none"
}

served_id() {
    curl -sf --max-time 5 "127.0.0.1:${PORT}/v1/models" 2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null
}

wait_healthy() {
    local pid=$1 waited=0
    printf "Waiting for :%s to report healthy" "$PORT"
    while [ "$waited" -lt "$HEALTH_TIMEOUT" ]; do
        if curl -sf --max-time 2 "127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            echo " ok (${waited}s)"
            return 0
        fi
        if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
            echo " FAILED: the server exited during startup."
            return 1
        fi
        sleep 3
        waited=$((waited + 3))
        printf "."
    done
    echo " FAILED: not healthy after ${HEALTH_TIMEOUT}s."
    return 1
}

# Stops whatever NInfer owns :8000, via systemd for the 27B service so systemd
# does not consider it crashed.
stop_current() {
    if systemctl is-active --quiet ninfer.service; then
        echo "Stopping ninfer.service (27B)..."
        sudo systemctl stop ninfer.service || return 1
    fi
    if [ "$(serving)" != "none" ]; then
        "$HERE/stop-ninfer.sh" || return 1
    fi
    if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
        echo "ERROR: :${PORT} is still held by something that is not NInfer:"
        ss -ltnp 2>/dev/null | grep ":${PORT} " | sed 's/^/    /'
        if [ -n "$(docker ps -q --filter 'name=^vllm$' 2>/dev/null)" ]; then
            echo "  It is the vLLM container. Run: sudo systemctl stop vllm-qwen3.8"
        fi
        return 1
    fi
}

case "${1:-}" in
    status)
        current=$(serving)
        echo "serving: ${current}"
        [ "$current" != "none" ] && echo "model id: $(served_id)"
        systemctl is-active --quiet ninfer.service && echo "ninfer.service: active" \
            || echo "ninfer.service: inactive"
        ;;

    flash-next)
        if [ "$(serving)" = "flash-next" ]; then
            echo "Flash-Next is already serving :${PORT} (model id $(served_id))."
            exit 0
        fi
        stop_current || exit 1
        log_start=$(stat -c %s "$FLASH_LOG" 2>/dev/null || echo 0)
        "$HERE/start-flash-next.sh" --background || exit 1
        pid=$(pgrep -nx ninfer-serve)
        if ! wait_healthy "$pid"; then
            echo "Last log lines (${FLASH_LOG}):"
            tail -c +"$((log_start + 1))" "$FLASH_LOG" | tail -15 | sed 's/^/    /'
            echo "Restoring the 27B server..."
            "$HERE/stop-ninfer.sh" >/dev/null 2>&1
            sudo systemctl start ninfer.service && wait_healthy ""
            exit 1
        fi
        echo "Flash-Next is serving :${PORT} (model id $(served_id); requests naming any model reach it)."
        echo "  log:          tail -f ${FLASH_LOG}"
        echo "  switch back:  $0 27b"
        ;;

    27b)
        if [ "$(serving)" = "27b" ]; then
            echo "The 27B server is already serving :${PORT} (model id $(served_id))."
            exit 0
        fi
        stop_current || exit 1
        echo "Starting ninfer.service (27B)..."
        sudo systemctl start ninfer.service || exit 1
        wait_healthy "" || { echo "  check: journalctl -u ninfer.service ; tail ~/Projects/ninfer/serve.log"; exit 1; }
        echo "27B is serving :${PORT}. Model id: $(served_id)"
        ;;

    *) usage ;;
esac
