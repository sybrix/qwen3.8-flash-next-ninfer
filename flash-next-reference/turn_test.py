import json, sys, urllib.request
url = sys.argv[1]
def chat(messages, extra):
    body = {"model": "test", "messages": messages, "max_tokens": 24, "temperature": 0, **extra}
    r = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(r, timeout=600).read())
    m = d["choices"][0]["message"]
    return m.get("content") or "", m.get("reasoning_content"), d["usage"]
for label, extra in (("thinking off", {"chat_template_kwargs": {"enable_thinking": False}}), ("thinking on", {})):
    t1 = [{"role": "user", "content": f"Describe a lighthouse ({label})."}]
    c, r, u = chat(t1, extra)
    t2 = t1 + [{"role": "assistant", "content": c}, {"role": "user", "content": "And at night?"}]
    _, _, u2 = chat(t2, extra)
    print(label, "turn2 prompt", u2["prompt_tokens"], "cached", u2.get("prompt_tokens_details"))
