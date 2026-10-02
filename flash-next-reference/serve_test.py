"""Exercise ninfer-serve on a Qwen4Exp artifact with controlled comparisons.

  serve_test.py URL OUT.json

1. Concurrency: the same prompts sent one at a time, then all at once (same server, so only the
   decode batch size differs).
2. Continuation: a Responses chain (turn 1, then turn 2 via previous_response_id). Run this on a
   server with prefix reuse and one without, at identical capacity; turn 2 must match and the
   reuse server must report cached input tokens.
"""

from __future__ import annotations

import concurrent.futures
import json
import sys
import urllib.request

PROMPTS = [
    "Write a short poem about the sea.",
    "List three prime numbers and explain why they are prime.",
    "Translate 'good morning' into French, German and Spanish.",
    "What is the capital of Australia?",
]
MAX_TOKENS = 24


def post(url, path, body):
    request = urllib.request.Request(url + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def chat(url, prompt):
    data = post(url, "/v1/chat/completions", {
        "model": "test", "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS, "temperature": 0})
    return data["choices"][0]["message"]["content"]


def respond(url, text, previous=None):
    body = {"model": "test", "input": text, "max_output_tokens": MAX_TOKENS, "temperature": 0,
            "store": True}
    if previous:
        body["previous_response_id"] = previous
    data = post(url, "/v1/responses", body)
    out = "".join(part.get("text", "") for item in data.get("output", [])
                  if item.get("type") == "message" for part in item.get("content", []))
    return data["id"], out, data.get("usage", {})


def main():
    url, out = sys.argv[1], sys.argv[2]
    result = {"sequential": [chat(url, p) for p in PROMPTS]}
    with concurrent.futures.ThreadPoolExecutor(len(PROMPTS)) as pool:
        result["concurrent"] = list(pool.map(lambda p: chat(url, p), PROMPTS))
    first, text1, usage1 = respond(url, "Describe a lighthouse in two sentences.")
    _, text2, usage2 = respond(url, "Now describe it at night, in one sentence.", first)
    result.update(turn1=text1, turn2=text2, turn1_usage=usage1, turn2_usage=usage2)
    json.dump(result, open(out, "w"), indent=1, ensure_ascii=False)
    for i, (a, b) in enumerate(zip(result["sequential"], result["concurrent"])):
        print(f"prompt {i}: sequential vs concurrent {'MATCH' if a == b else 'DIFF'}")
    print("turn2 usage:", json.dumps(usage2))


if __name__ == "__main__":
    main()
