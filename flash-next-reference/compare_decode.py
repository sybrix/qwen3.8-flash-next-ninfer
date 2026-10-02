"""Check NInfer greedy decode against the layer-streaming reference by teacher forcing.

Builds the chat-templated prompt ids exactly as NInfer's frontend does (same template, thinking
disabled), appends NInfer's generated ids, runs one reference Prefill over the whole sequence and
reports, for every generated position, whether NInfer's token is the reference argmax and the
reference logit margin to its runner-up.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--template", required=True, help="chat template used by the artifact")
    p.add_argument("--prompt", required=True)
    p.add_argument("--generated", required=True, help="space-separated NInfer generated ids")
    p.add_argument("--out", default="decode_ref.pt")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        chat_template=Path(args.template).read_text(),
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
    )
    if hasattr(prompt_ids, "keys"):
        prompt_ids = prompt_ids["input_ids"]
    generated = [int(x) for x in args.generated.split()]
    ids = list(prompt_ids) + generated
    here = Path(__file__).parent
    subprocess.run(
        [sys.executable, str(here / "flashnext_ref.py"), "--checkpoint", args.checkpoint,
         "--ids", ",".join(map(str, ids)), "--out", args.out],
        check=True,
    )
    record = torch.load(args.out)
    logits = record["logits"]
    start = len(prompt_ids) - 1
    agree = 0
    rows = []
    for i, token in enumerate(generated):
        row = logits[start + i]
        top2 = row.topk(2)
        best = int(top2.indices[0])
        margin = float(top2.values[0] - top2.values[1])
        agree += best == token
        rows.append({"step": i, "ninfer": token, "reference": best, "margin": round(margin, 4),
                     "ninfer_logit_gap": round(float(top2.values[0] - row[token]), 4)})
    print(json.dumps(rows, indent=1))
    print(f"prompt_tokens={len(prompt_ids)} generated={len(generated)} "
          f"argmax_agreement={agree}/{len(generated)}")
    print("NInfer text:", repr(tok.decode(generated)))


if __name__ == "__main__":
    main()
