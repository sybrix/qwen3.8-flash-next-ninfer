"""Teacher-force a server output text (re-tokenized) through the reference; report non-tie mismatches."""
import json, subprocess, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer
ckpt, template, prompt, text, out = sys.argv[1:6]
tok = AutoTokenizer.from_pretrained(ckpt)
p = tok.apply_chat_template([{"role": "user", "content": prompt}], chat_template=Path(template).read_text(),
                            add_generation_prompt=True, enable_thinking=False, tokenize=True)
p = list(p["input_ids"] if hasattr(p, "keys") else p)
g = tok.encode(text, add_special_tokens=False)
subprocess.run([sys.executable, "flashnext_ref.py", "--checkpoint", ckpt, "--ids", ",".join(map(str, p + g)),
                "--out", out], check=True, capture_output=True)
l = torch.load(out)["logits"]
bad = []
for i, t in enumerate(g):
    row = l[len(p) - 1 + i]; v, ix = row.topk(2)
    if int(ix[0]) != t:
        bad.append((i, round(float(v[0] - v[1]), 3), round(float(v[0] - row[t]), 3)))
print(f"{len(g)} tokens, mismatches (step, ref margin, ninfer gap): {bad}")
