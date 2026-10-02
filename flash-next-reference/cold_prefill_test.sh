#!/usr/bin/env bash
# Cold-page-cache prefill check on the real Flash-Next server; ALWAYS swaps back to the 27B.
set -u
# Directory holding swap-model.sh / stop-ninfer.sh (the repo's serving-scripts/ by default).
SERVING_SCRIPTS="${SERVING_SCRIPTS:-$(cd "$(dirname "$0")/../serving-scripts" && pwd)}"
out=$HOME/Projects/flash-next-ref/cold_test
rm -f "$out/done"
exec > "$out/log.txt" 2>&1
trap 'echo "== swap back $(date +%T)"; $SERVING_SCRIPTS/swap-model.sh 27b; echo "finished $(date +%T)" > "$out/done"' EXIT
echo "== swap to flash-next $(date +%T)"
$SERVING_SCRIPTS/swap-model.sh flash-next || exit 1
echo "== drop page cache $(date +%T)"
sync; sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
python3 - <<'PY'
import json, os, time, urllib.request
def ask(text, label):
    body = {"model": "qwen3.8-flash-next", "max_tokens": 60, "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": text + "\n\nWho is the main character? One sentence."}]}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8000/v1/messages", json.dumps(body).encode(),
        {"Content-Type": "application/json"}), timeout=600))
    print(f"{label}: {time.time()-t:.1f} s, {r['usage']['input_tokens']} input tokens: {r['content'][0]['text'][:150]!r}")
corpus = os.path.expanduser("~/Projects/ninfer-flash-next/eval/corpora/perplexity-1m/data/pg19/")
ask(open(corpus + "01.txt").read()[:30000], "cold (book 01)")
ask(open(corpus + "02.txt").read()[:30000], "cold (book 02)")
PY
grep -E "req#[0-9]+ done" $HOME/Projects/ninfer-flash-next/serve.log | tail -2
