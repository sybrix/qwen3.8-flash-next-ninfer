#!/usr/bin/env bash
# Real-model vision validation for Qwen3.8-Flash-Next on NInfer.
#   1. stops the current :8000 server, runs NInfer with images (short chart, long image+text past
#      the dense-equivalent QSA extent), each without and with MTP;
#   2. starts the 27B (it leaves GPU room for the PyTorch reference) and teacher-forces the
#      non-speculative outputs through the reference;
#   3. ALWAYS restores whichever model was serving before (trap).
#   setsid nohup ./vision_real_test.sh >/dev/null 2>&1 < /dev/null &
# Progress: vision_real/log.txt   Finished: vision_real/done
set -u
SERVING_SCRIPTS="${SERVING_SCRIPTS:-$(cd "$(dirname "$0")/../serving-scripts" && pwd)}"
here=$(cd "$(dirname "$0")" && pwd)
out="$here/vision_real"
mkdir -p "$out"
rm -f "$out/done"
exec > "$out/log.txt" 2>&1

A=$HOME/Projects/ninfer-flash-next/models/qwen3_8_flash_next_nvfp4_mtp_vision.ninfer
if [ ! -f "$A" ] || compgen -G "$A.part-*" >/dev/null || compgen -G "$(dirname "$A")/.*.tmp" >/dev/null; then
    echo "ABORT: $A is missing, split into parts, or still being written; server left running"
    echo "aborted $(date +%T)" > "$out/done"
    exit 1
fi

previous=$("$SERVING_SCRIPTS/swap-model.sh" status | awk '/^serving:/ {print $2}')
echo "== previously serving: $previous ($(date +%T))"
restore() {
    echo "== restoring $previous $(date +%T)"
    case "$previous" in
        flash-next) "$SERVING_SCRIPTS/swap-model.sh" flash-next ;;
        *)          "$SERVING_SCRIPTS/swap-model.sh" 27b ;;
    esac
    "$SERVING_SCRIPTS/swap-model.sh" status
    echo "finished $(date +%T)" > "$out/done"
}
trap restore EXIT

N=$HOME/Projects/ninfer-flash-next/build/apps/ninfer
MEDIA=$HOME/Projects/ninfer-flash-next/examples/cli/media
python3 - "$MEDIA" "$out" "$here/long_prompt.txt" <<'PY'
import json, sys
media, out, long_path = sys.argv[1:4]
json.dump([{"role": "user", "content": [
    {"type": "image", "image": f"{media}/visual_chart.png"},
    {"type": "text", "text": "Describe this chart: what it shows, the axes, and the main trend."}]}],
    open(f"{out}/short.json", "w"))
json.dump([{"role": "user", "content": [
    {"type": "image", "image": f"{media}/natural_scene.png"},
    {"type": "text", "text": open(long_path).read() +
     "\n\nFirst describe the image in two sentences, then summarize the text in two sentences."}]}],
    open(f"{out}/long.json", "w"))
PY

echo "== stopping the current server $(date +%T)"
if systemctl is-active --quiet ninfer.service; then sudo -n systemctl stop ninfer.service; fi
"$SERVING_SCRIPTS/stop-ninfer.sh"

run() {
    local name=$1 messages=$2 max_new=$3
    shift 3
    echo "== $name $(date +%T)"
    timeout 1200 "$N" "$A" --messages "$messages" --vision --max-context 16384 \
        --kv-capacity 16384 --max-new "$max_new" --greedy --no-thinking --print-token-ids \
        --kv-dtype fp8 "$@" > "$out/$name.out" 2> "$out/$name.err"
    echo "exit=$?"
    grep -E "prompt tokens|prefill speed|decode speed|acceptance rate|^error" "$out/$name.err"
    echo "text: $(head -c 400 "$out/$name.out")"
}
run short_base "$out/short.json" 128
if ! grep -qE "^tokens +generated ids" "$out/short_base.err"; then
    echo "ABORT: the vision artifact does not run; skipping the rest"
    exit 1
fi
run short_mtp "$out/short.json" 128 --spec mtp --draft-tokens 3
run long_base "$out/long.json" 96
run long_mtp "$out/long.json" 96 --spec mtp --draft-tokens 3

echo "== greedy agreement, MTP vs non-speculative"
python3 - "$out" <<'PY'
import re, sys
out = sys.argv[1]
def ids(name):
    for line in open(f"{out}/{name}.err"):
        if re.match(r"^tokens +generated ids", line):
            return line.split("ids", 1)[1].replace(",", " ").split()
    return []
for base, other in (("short_base", "short_mtp"), ("long_base", "long_mtp")):
    a, b = ids(base), ids(other)
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    print(f"{other}: identical prefix {first if first is not None else min(len(a), len(b))} of {len(a)}")
PY

echo "== starting the 27B for the reference phase $(date +%T)"
"$SERVING_SCRIPTS/swap-model.sh" 27b

reference() {
    local name=$1 messages=$2
    echo "== reference $name $(date +%T)"
    local generated
    generated=$(grep -E '^tokens +generated ids' "$out/$name.err" | sed -E 's/^tokens +generated ids *//' | tr ',' ' ')
    "$here/.venv/bin/python" "$here/flashnext_ref.py" --checkpoint "$here/ckpt" \
        --messages "$messages" --generated "$generated" --out "$out/ref_$name.pt" \
        > "$out/ref_$name.log" 2>&1
    python3 - "$out/ref_$name.log" <<'PY'
import json, sys
t = open(sys.argv[1]).read()
if "\n[\n" not in t:
    print("reference failed:", t[-800:]); raise SystemExit
a = t.index("\n[\n") + 1; b = t.index("\n]\n", a) + 2
rows = json.loads(t[a:b])
print([l for l in t.splitlines() if l.startswith(("multimodal", "prompt_tokens"))])
print("mismatches (step, ref margin, ninfer gap):",
      [(r["step"], r["margin"], r["ninfer_logit_gap"]) for r in rows if r["ninfer"] != r["reference"]])
PY
}
reference short_base "$out/short.json"
reference long_base "$out/long.json"
