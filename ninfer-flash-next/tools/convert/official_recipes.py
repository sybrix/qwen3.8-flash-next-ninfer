"""Official representation recipes built from the same public conversion functions."""

from __future__ import annotations

from .methods import (
    cast_direct,
    fp8_row_maxabs,
    grouped_absmax,
    import_encoded,
    import_nvfp4_bank,
)
from .sources.modelopt import fp8_block_as_nvfp4_source, modelopt_nvfp4_source

Q4 = "q4_g64_fp16"
Q5 = "q5_g64_fp16"
Q6 = "q6_g64_fp16"
Q8 = "q8_g32_fp16"
FP8 = "fp8_e4m3fn_row_bf16"


def _assign(recipe, name, format, *, source=None):
    method = grouped_absmax if format in (Q4, Q5, Q6, Q8) else cast_direct
    recipe.assign(name, format=format, method=method, source=source)


def _vision_format(name):
    if name == "vision/patch_embedding":
        return Q6
    if name.startswith("vision/merger/"):
        return Q8
    if name.endswith(("/attention/query", "/attention/key", "/attention/value", "/mlp/fc1")):
        return Q4
    return Q5


def _vision(model, recipe):
    for name, parameter in model.parameters.items():
        if parameter.projection and name.startswith("vision/"):
            _assign(recipe, name, _vision_format(name))


def _optional(model, recipe):
    _vision(model, recipe)
    for name, parameter in model.parameters.items():
        if not parameter.projection:
            continue
        if name.startswith("vision/"):
            continue
        elif name.startswith(("mtp/", "dflash/", "dflash2/")):
            if name.endswith(
                (
                    "/moe/router",
                    "/moe/shared_score",
                    "/attention_conv/kernel_projection",
                    "/mlp_conv/kernel_projection",
                    "/candidate_selector/hidden_projection",
                )
            ):
                continue
            _assign(recipe, name, Q8)
    for backend in ("dflash", "dflash2"):
        if backend not in model.components:
            continue
        layers = model.components[backend]["config"]["num_hidden_layers"]
        for layer in range(layers):
            prefix = f"{backend}/layers/{layer}/attention/"
            for role in ("key", "value"):
                recipe.share(prefix + "context_" + role, prefix + role)


def _dense_groupwise(model, recipe, vocabulary):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", vocabulary)
    _assign(recipe, "text/output_head", vocabulary)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        if name.endswith(
            (
                "/attention/query",
                "/attention/key",
                "/gdn/query",
                "/gdn/key",
                "/mlp/gate",
                "/mlp/up",
            )
        ):
            format = Q4
        else:
            format = Q5
        _assign(recipe, name, format)


def qwen3_6_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)


def qwen3_8_27b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q8)


def qwen3_6_35b_a3b(model, recipe, sources):
    if "num_experts" not in model.config:
        raise ValueError("this official recipe requires Qwen3.5 MoE mathematics")
    _optional(model, recipe)
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q6)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        if name.endswith(
            (
                "/gdn/a_projection",
                "/gdn/b_projection",
                "/moe/router",
                "/moe/shared_score",
            )
        ):
            continue
        if "/moe/experts/" in name:
            layer = int(name.split("/")[2])
            format = (
                (Q6 if layer in (34, 38, 39) else Q5) if name.endswith("/down") else Q4
            )
        else:
            format = Q8
        _assign(recipe, name, format)


def qwen3_6_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    _assign(recipe, "text/token_embedding", Q8)
    _assign(recipe, "text/output_head", Q8)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/layers/") or not parameter.projection:
            continue
        layer = int(name.split("/")[2])
        if name.endswith(("/gdn/a_projection", "/gdn/b_projection")):
            recipe.separate(name)
            continue
        direct = (
            ("/attention/" in name and not name.endswith("/output") and layer < 24)
            or (name.endswith("/attention/output") and layer in (3, 7))
            or (name.endswith("/gdn/output") and layer == 4)
        )
        if direct:
            continue
        recipe.assign(
            name,
            format="nvfp4",
            method=import_encoded,
            source=model.source(name, quantized, "nvfp4"),
            activation_policy="AllowA4",
        )


def qwen3_8_27b_nvfp4(model, recipe, sources):
    if "num_experts" in model.config:
        raise ValueError("this official recipe requires Qwen3.5 Dense mathematics")
    _optional(model, recipe)
    quantized = sources["quantized"]
    recipe.assign("text/token_embedding", format=FP8, method=fp8_row_maxabs)
    for name, parameter in model.parameters.items():
        if not name.startswith("text/") or name == "text/token_embedding":
            continue
        source = model.source(name, quantized)
        if not parameter.projection or name.endswith(
            ("/gdn/a_projection", "/gdn/b_projection")
        ):
            recipe.assign(name, source=source)
            continue
        layer = int(name.split("/")[2]) if name.startswith("text/layers/") else -1
        format = "nvfp4" if "/mlp/" in name and layer < 56 else FP8
        recipe.assign(
            name,
            format=format,
            method=import_encoded,
            source=model.source(name, quantized, format),
            activation_policy="AllowA4" if format == "nvfp4" else "AllowA8",
        )


def qwen3_8_flash_next_nvfp4(model, recipe, sources):
    """NVIDIA ModelOpt NVFP4 export: experts keep their NVFP4 words, all else stays BF16."""
    if model.config.get("architectures") != ["Qwen4ExpForCausalLM"]:
        raise ValueError("this official recipe requires Qwen4Exp mathematics")
    config = model.config
    base = sources["base"]
    experts, inter, hidden = (
        config["num_experts"],
        config["moe_intermediate_size"],
        config["hidden_size"],
    )
    for layer in range(config["num_hidden_layers"]):
        prefix = f"text/layers/{layer}/moe/experts/"
        source_prefix = f"model.language_model.layers.{layer}.mlp.experts."
        for role, shape in (
            ("gate", (inter, hidden)),
            ("up", (inter, hidden)),
            ("down", (hidden, inter)),
        ):
            names = [f"{prefix}{e}/{role}" for e in range(experts)]
            for e, name in enumerate(names):
                recipe.assign(
                    name,
                    format="nvfp4",
                    layout="block_scale_k16_m128x4_bank_v1",
                    method=import_nvfp4_bank,
                    source=modelopt_nvfp4_source(
                        base, f"{source_prefix}{e}.{role}_proj", shape
                    ),
                )
            recipe.group(names, shape=(experts, *shape))
    # The vision tower is the Qwen3.5 one with its MLP zero-padded to 4352 (see qwen4_exp.py); every
    # Vision projection runs at Q8 (the Qwen3.5 Q4/Q5 mix moves image embeddings ~35%).
    if "vision" in model.components:
        for name, parameter in model.parameters.items():
            if parameter.projection and name.startswith("vision/"):
                _assign(recipe, name, Q8)
    if "mtp" not in model.components:
        return
    # The MTP experts ship as FP8_PB_WO (128x128 block multipliers); re-encode them as
    # NVFP4 so the draft layer runs the same expert-bank kernels as the target.
    prefix, source_prefix = "mtp/layers/0/moe/experts/", "mtp.layers.0.mlp.experts."
    for role, shape in (("gate", (inter, hidden)), ("up", (inter, hidden)), ("down", (hidden, inter))):
        names = [f"{prefix}{e}/{role}" for e in range(experts)]
        for e, name in enumerate(names):
            recipe.assign(
                name,
                format="nvfp4",
                layout="block_scale_k16_m128x4_bank_v1",
                method=import_nvfp4_bank,
                source=fp8_block_as_nvfp4_source(base, f"{source_prefix}{e}.{role}_proj", shape),
            )
        recipe.group(names, shape=(experts, *shape))


RECIPES = {
    "qwen3_6_27b": qwen3_6_27b,
    "qwen3_6_27b_nvfp4": qwen3_6_27b_nvfp4,
    "qwen3_8_27b": qwen3_8_27b,
    "qwen3_8_27b_nvfp4": qwen3_8_27b_nvfp4,
    "qwen3_6_35b_a3b": qwen3_6_35b_a3b,
    "qwen3_8_flash_next_nvfp4": qwen3_8_flash_next_nvfp4,
}
