#!/usr/bin/env bash
# Swap to Flash-Next (MTP on), exercise the server, ALWAYS swap back to the 27B on exit.
set -u
# Directory holding swap-model.sh / stop-ninfer.sh (the repo's serving-scripts/ by default).
SERVING_SCRIPTS="${SERVING_SCRIPTS:-$(cd "$(dirname "$0")/../serving-scripts" && pwd)}"
out=$HOME/Projects/flash-next-ref/swap_mtp
rm -f "$out/done"
exec > "$out/log.txt" 2>&1
trap 'echo "== swap back $(date +%T)"; $SERVING_SCRIPTS/swap-model.sh 27b; echo "finished $(date +%T)" > "$out/done"' EXIT
echo "== swap to flash-next $(date +%T)"
$SERVING_SCRIPTS/swap-model.sh flash-next || exit 1
$SERVING_SCRIPTS/swap-model.sh status
python3 - <<'PY'
import json, os, time, urllib.request, concurrent.futures
base = "http://127.0.0.1:8000"
def post(path, body):
    t = time.time()
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r), time.time() - t
r, dt = post("/v1/chat/completions", {"model": "qwen3.8-flash-next", "max_tokens": 60,
    "messages": [{"role": "user", "content": "What is the capital of Australia? One sentence."}],
    "chat_template_kwargs": {"enable_thinking": False}})
print("openai:", round(dt, 1), "s", repr(r["choices"][0]["message"]["content"]))
r, dt = post("/v1/messages", {"model": "qwen3.8-flash-next", "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Write a Python function that checks whether a string is a palindrome, ignoring case and punctuation, with two tests."}]})
print("anthropic+thinking:", round(dt, 1), "s", [b["type"] for b in r["content"]], r["usage"].get("output_tokens"))
print(r["content"][-1].get("text", "")[:300])
long = open(os.path.expanduser("~/Projects/flash-next-ref/pg19_8192.txt")).read()[:30000]
r, dt = post("/v1/messages", {"model": "qwen3.8-flash-next", "max_tokens": 200, "thinking": {"type": "disabled"},
    "messages": [{"role": "user", "content": long + "\n\nWho is the main character and what are the soldiers trying to capture? Two sentences."}]})
print("long:", round(dt, 1), "s", r["usage"].get("input_tokens"), repr(r["content"][0]["text"][:300]))
def one(i):
    return post("/v1/chat/completions", {"model": "qwen3.8-flash-next", "max_tokens": 120,
        "messages": [{"role": "user", "content": f"Give {i+2} facts about octopuses."}],
        "chat_template_kwargs": {"enable_thinking": False}})
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    for r, dt in pool.map(one, range(2)):
        print("concurrent:", round(dt, 1), "s", repr(r["choices"][0]["message"]["content"][:120]))
PY
echo "== server log"
grep -E "req#[0-9]+ done|ERROR" $HOME/Projects/ninfer-flash-next/serve.log | tail -6
