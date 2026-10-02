import json, sys, urllib.request
url, out = sys.argv[1], sys.argv[2]
def chat(messages):
    body = {"model": "test", "messages": messages, "max_tokens": 24, "temperature": 0}
    r = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=600).read())
    m = d["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or "", "usage": d["usage"]}
t1 = [{"role": "user", "content": "Describe a lighthouse."}]
a = chat(t1)
t2 = t1 + [{"role": "assistant", "content": a["content"], "reasoning_content": a["reasoning"]},
           {"role": "user", "content": "And at night?"}]
b = chat(t2)
json.dump({"turn1": a, "turn2": b}, open(out, "w"), ensure_ascii=False, indent=1)
print(out, "turn2 cached", b["usage"].get("prompt_tokens_details"), "of", b["usage"]["prompt_tokens"])
