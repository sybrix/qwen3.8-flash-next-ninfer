"""How much does NInfer's vision quantization (Q4/Q5/Q6/Q8, as the Qwen3.5 recipes) move the
real image embeddings? Runs the checkpoint vision tower twice: BF16 and NInfer-quantized weights."""
import json, sys, os
from pathlib import Path
import torch
sys.path.insert(0, os.path.expanduser("~/Projects/ninfer-flash-next"))
from tools.convert.quantization.groupwise import quantize_matrix
from tools.convert.official_recipes import _vision_format
from tools.artifact.formats import get_format
from transformers import AutoProcessor
from transformers import AutoConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from flashnext_ref import Checkpoint, restore_buffers

# argv: messages.json [checkpoint]. The processor comes from the Flash-Next snapshot (same
# Qwen3.5 image preprocessing); weights and vision config from `checkpoint`.
root = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("ckpt"); device = torch.device("cuda")
config = AutoConfig.from_pretrained(root); ckpt = Checkpoint(root)
VisionModel = Qwen4ExpVisionModel if config.model_type == "qwen4_exp" else Qwen3_5VisionModel
processor = AutoProcessor.from_pretrained("ckpt")
messages = json.loads(Path(sys.argv[1]).read_text())
for m in messages:
    for part in m["content"] if isinstance(m["content"], list) else []:
        if part.get("type") == "image": part["path"] = part.pop("image")
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt", enable_thinking=False)

def ninfer_name(hf):  # HF vision parameter -> NInfer logical projection, or None (kept BF16)
    if hf == "patch_embed.proj.weight": return "vision/patch_embedding"
    if hf.startswith("merger.linear_fc"): return "vision/merger/fc" + hf[len("merger.linear_fc")]
    if hf.startswith("blocks.") and hf.endswith(".weight"):
        i, rest = hf.split(".")[1], ".".join(hf.split(".")[2:-1])
        return {"attn.qkv": f"vision/layers/{i}/attention/query", "attn.proj": f"vision/layers/{i}/attention/output",
                "mlp.linear_fc1": f"vision/layers/{i}/mlp/fc1", "mlp.linear_fc2": f"vision/layers/{i}/mlp/fc2"}.get(rest)
    return None

def dequant(w, fmt):
    spec = get_format(fmt)
    q = quantize_matrix(w.reshape(w.shape[0], -1).float(), spec, device="cuda")
    groups = q.codes.float() * q.scales.float().unsqueeze(-1)
    return groups.reshape(w.shape[0], -1)[:, : w[0].numel()].reshape(w.shape).to(torch.bfloat16)

def tower(quantized, override=None):
    with torch.device("meta"):
        v = VisionModel._from_config(config.vision_config)
    v.to_empty(device=device); v.to(torch.bfloat16)
    restore_buffers(v, VisionModel, config.vision_config)
    state = {}
    for name, _ in v.named_parameters():
        w = ckpt.get("model.visual." + name).to(device, torch.bfloat16)
        logical = ninfer_name(name)
        if quantized and logical is not None:
            w = dequant(w, override(logical) if callable(override) else (override or _vision_format(logical)))
        state[name] = w
    v.load_state_dict(state); v.eval()
    with torch.no_grad():
        out = v(inputs["pixel_values"].to(device, torch.bfloat16), grid_thw=inputs["image_grid_thw"].to(device))
    e = out.pooler_output if hasattr(out, "pooler_output") else out
    return torch.cat(list(e), 0).float() if isinstance(e, (list, tuple)) else e.float()

print(f"checkpoint {root}: vision out width {config.vision_config.out_hidden_size}")
a = tower(False)
q8_but_fc2 = lambda name: "q5_g64_fp16" if name.endswith("/mlp/fc2") else "q8_g32_fp16"
for label, override in (("recipe (Q4/Q5/Q6/Q8)", None), ("all Q8", "q8_g32_fp16"),
                        ("Q8, fc2 Q5", q8_but_fc2)):
    b = tower(True, override)
    rel = ((a - b).norm(dim=-1) / a.norm(dim=-1))
    print(f"{label}: embedding relative error mean {rel.mean():.4f} median {rel.median():.4f} "
          f"max {rel.max():.4f}; cosine {torch.nn.functional.cosine_similarity(a, b, dim=-1).mean():.5f}")
