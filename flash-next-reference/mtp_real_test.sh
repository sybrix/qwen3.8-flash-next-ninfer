#!/usr/bin/env bash
# Real-model MTP validation for Qwen3.8-Flash-Next. Stops the 27B NInfer service, runs the
# tests, and ALWAYS restarts the service on exit (trap), even if a test fails.
#   setsid nohup ./mtp_real_test.sh >/dev/null 2>&1 < /dev/null &
# Progress: mtp_real/log.txt   Finished: mtp_real/done
set -u
# Directory holding swap-model.sh / stop-ninfer.sh (the repo's serving-scripts/ by default).
SERVING_SCRIPTS="${SERVING_SCRIPTS:-$(cd "$(dirname "$0")/../serving-scripts" && pwd)}"
here=$(cd "$(dirname "$0")" && pwd)
out="$here/mtp_real"
mkdir -p "$out"
rm -f "$out/done"
exec > "$out/log.txt" 2>&1

A=$HOME/Projects/ninfer-flash-next/models/qwen3_8_flash_next_nvfp4_mtp.ninfer
# Pre-flight, before the 27B is touched: Mapped PLE shards must not straddle container files,
# so the artifact must be one complete file.
if [ ! -f "$A" ] || compgen -G "$A.part-*" >/dev/null || compgen -G "$(dirname "$A")/.*.tmp" >/dev/null; then
    echo "ABORT: $A is missing, split into parts, or still being written; 27B left running"
    echo "aborted $(date +%T)" > "$out/done"
    exit 1
fi

restore() {
    echo "== restoring ninfer.service $(date +%T)"
    sudo -n systemctl start ninfer.service
    for _ in $(seq 1 100); do curl -sf 127.0.0.1:8000/health >/dev/null && break; sleep 3; done
    echo "health: $(curl -s 127.0.0.1:8000/health)"
    echo "finished $(date +%T)" > "$out/done"
}
trap restore EXIT

N=$HOME/Projects/ninfer-flash-next/build/apps/ninfer
P=$HOME/Projects/ninfer-flash-next/build/apps/ninfer-perplexity
CODE_PROMPT="Write a Python function that parses an ISO-8601 date string without using datetime, with input validation, then write five unit tests for it. Explain the edge cases briefly."
LONG_PROMPT="$(head -c 30000 "$here/pg19_8192.txt")

Summarize the passage above in two sentences."

echo "== stopping 27B $(date +%T)"
"$SERVING_SCRIPTS/stop-ninfer.sh"

echo "== perplexity 4K (MTP artifact, no speculation)"
d=$(mktemp -d)
timeout 900 "$P" "$A" --text "$here/pg19_4096.txt" --context 4096 --stride 4095 --kv-dtype fp8 \
    --output "$d" > "$out/ppl.txt" 2>&1
grep -E "custom |score rate|error" "$out/ppl.txt"
if ! grep -q "^custom " "$out/ppl.txt"; then
    echo "ABORT: the artifact does not load or score; skipping the decode runs"
    exit 1
fi

run() {
    local name=$1 prompt=$2 max_new=$3
    shift 3
    echo "== $name $(date +%T)"
    timeout 1200 "$N" "$A" --prompt "$prompt" --max-context 16384 --kv-capacity 16384 \
        --max-new "$max_new" --greedy --no-thinking --print-token-ids --kv-dtype fp8 "$@" \
        > "$out/$name.out" 2> "$out/$name.err"
    echo "exit=$?"
    grep -E "prompt tokens|prefill speed|decode speed|acceptance|accepted by pos|^error" \
        "$out/$name.err"
}

run code_base "$CODE_PROMPT" 512
run code_mtp1 "$CODE_PROMPT" 512 --spec mtp --draft-tokens 1
run code_mtp2 "$CODE_PROMPT" 512 --spec mtp --draft-tokens 2
run code_mtp3 "$CODE_PROMPT" 512 --spec mtp --draft-tokens 3
run long_base "$LONG_PROMPT" 256
run long_mtp3 "$LONG_PROMPT" 256 --spec mtp --draft-tokens 3

echo "== greedy agreement with the non-speculative output"
python3 - "$out" <<'PY'
import re, sys
out = sys.argv[1]
def ids(name):
    for line in open(f"{out}/{name}.err"):
        if re.match(r"^tokens +generated ids", line):
            return line.split("ids", 1)[1].replace(",", " ").split()
    return []
for base, others in (("code_base", ("code_mtp1", "code_mtp2", "code_mtp3")),
                     ("long_base", ("long_mtp3",))):
    a = ids(base)
    for other in others:
        b = ids(other)
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        print(f"{other}: {len(b)} tokens, identical prefix "
              f"{first if first is not None else min(len(a), len(b))} of {len(a)}")
PY
