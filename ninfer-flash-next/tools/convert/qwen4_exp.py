"""Qwen4Exp (Qwen3.8-Flash-Next) adapter: mathematical config and logical parameters.

Text, plus the optional MTP draft head. Hyper-connection mixers, query-sparse attention (QSA) indexers and the per-layer
n-gram embedding (PLE) are mapped alongside the Qwen3.5-shaped GDN, gated attention and MoE.
See docs/maintainer/qwen4_exp-model.md for the mathematics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from .model import Model, Parameter
from .qwen3_5 import _Builder, _fixed, _positive, text_config as qwen3_5_text_config
from .resources import load_resources
from .sources.safetensors import SafetensorsSource, tensor_source

ARCHITECTURE = "Qwen4ExpForConditionalGeneration"
TEXT_PREFIX = "model.language_model."

# Splitmix/prime construction of the PLE hash constants (transformers modeling_qwen4_exp).
_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15


def _splitmix64(value: int) -> int:
    value = (value + _GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % d for d in range(3, math.isqrt(value) + 1, 2))


def ple_hash_constants(raw: dict, ple_index: int) -> dict:
    vocab, order, heads = raw["vocab_size"], raw["ngram_size"], raw["heads_per_ngram"]
    bound = max(1, ((1 << 63) - 1) // max(vocab, 1) // 2)
    base_seed = raw["seed"] + 10007 * ple_index
    multipliers = [
        2 * (_splitmix64((base_seed + _GAMMA * (i + 1)) & _MASK64) % bound) + 1
        for i in range(order)
    ]
    ngram_heads = (order - 1) * heads
    sizes, offsets, total, prime = [], [], 0, raw["ngram_vocab_size_base"] - 1
    for _ in range(ple_index * ngram_heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    for _ in range(ngram_heads):
        prime += 1
        while not _is_prime(prime):
            prime += 1
        sizes.append(prime)
        offsets.append(total)
        total += prime
    divisor = raw["make_ngram_vocab_size_divisible_by"]
    return {
        "multipliers": multipliers,
        "head_vocab_sizes": sizes,
        "head_offsets": offsets,
        "table_rows": -(-total // divisor) * divisor,
    }


def text_config(source: dict) -> dict:
    architectures = source.get("architectures")
    if architectures != [ARCHITECTURE]:
        raise ValueError(f"unsupported Qwen4Exp architecture {architectures!r}")
    raw = source.get("text_config")
    if not isinstance(raw, dict):
        raise ValueError("Qwen4Exp requires text_config")
    _fixed(raw, "output_gate_type", "sigmoid", "text")
    _fixed(raw, "norm_topk_prob", True, "text")
    _fixed(raw, "mtp_use_dedicated_embeddings", False, "text")
    # Reuse the Qwen3.5 parser for the shared GQA/GDN/MoE/RoPE fields.
    shared = dict(source, architectures=["Qwen3_5MoeForConditionalGeneration"])
    shared["text_config"] = {k: v for k, v in raw.items() if k != "attn_output_gate"}
    result = qwen3_5_text_config(shared, mtp=False)
    result["architectures"] = ["Qwen4ExpForCausalLM"]
    result["model_type"] = "qwen4_exp_text"
    for key in (
        "hc_count",
        "hc_lowrank",
        "indexer_n_heads",
        "indexer_kv_heads",
        "indexer_head_dim",
        "indexer_compress_ratio",
        "indexer_budget",
        "ngram_size",
        "heads_per_ngram",
        "ple_embed_dim",
        "ple_conv_kernel_size",
    ):
        result[key] = _positive(raw.get(key), "text." + key)
    if result["indexer_kv_heads"] != 1 or result["indexer_budget"] % result[
        "indexer_compress_ratio"
    ]:
        raise ValueError("QSA requires one index KV head and a whole block budget")
    eos = raw.get("eos_token_id")
    if type(eos) is not int or not 0 <= eos < result["vocab_size"]:
        raise ValueError("Qwen4Exp PLE requires one integer eos_token_id")
    result["eos_token_id"] = eos
    layers = raw.get("ple_layer_ids")
    if (
        not isinstance(layers, list)
        or any(type(i) is not int or not 1 <= i <= result["num_hidden_layers"] for i in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError("ple_layer_ids must name distinct 1-based Text blocks")
    ngram_heads = (result["ngram_size"] - 1) * result["heads_per_ngram"]
    if result["ple_embed_dim"] % ngram_heads:
        raise ValueError("ple_embed_dim must split evenly across n-gram heads")
    for key in ("seed", "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by"):
        if type(raw.get(key)) is not int or raw[key] < 0:
            raise ValueError(f"text.{key}: expected a nonnegative integer")
    result["ple_layers"] = [
        {"layer": block - 1, **ple_hash_constants(raw, index)}
        for index, block in enumerate(layers)
    ]
    return result


class _Qwen4ExpBuilder(_Builder):
    def validate_source(self, selected, original, name):
        if selected is not original:
            raise ValueError(f"{name}: Qwen4Exp maps one checkpoint source")

    def gated_residual(self, prefix, source_prefix, store, config, *, inject=True):
        s, h, r = config["hc_count"], config["hidden_size"], config["hc_lowrank"]
        self.add(prefix + "norm", store, source_prefix + "hc_norm.weight", (s * h,))
        self.add(
            prefix + "down",
            store,
            source_prefix + "input_mix_weight_down.weight",
            (r, s * h),
            inputs=(prefix + "normalized_input",),
        )
        self.add(
            prefix + "up",
            store,
            source_prefix + "input_mix_weight_up.weight",
            (s * h, r),
            inputs=(prefix + "mix_hidden",),
        )
        if inject:
            self.add(
                prefix + "inject",
                store,
                source_prefix + "block_inject_weight.weight",
                (s, s * h),
                inputs=(prefix + "normalized_input",),
            )

    def attention(self, prefix, source_prefix, store, config):
        super().attention(prefix, source_prefix, store, config)
        h = config["hidden_size"]
        d, heads = config["indexer_head_dim"], config["indexer_n_heads"]
        index = source_prefix + "self_attn.indexer."
        for role, begin, count in (("index_query", 0, heads * d), ("index_key", heads * d, d)):
            self.add(
                prefix + "attention/" + role,
                store,
                index + "index_qk_proj.weight",
                (count, h),
                source_shape=((heads + 1) * d, h),
                rows=((begin, begin + count),),
                inputs=(prefix + "mixer_input",),
            )
        for role, field in (("index_query_norm", "q_layernorm"), ("index_key_norm", "k_layernorm")):
            self.add(prefix + "attention/" + role, store, index + field + ".weight", (d,))

    def ple(self, prefix, source_prefix, store, config, record):
        s, h, e = config["hc_count"], config["hidden_size"], config["ple_embed_dim"]
        sp = source_prefix + "ple."
        for role, field, shape in (
            ("key_projection", "key_proj", (s * h, e)),
            ("value_projection", "value_proj", (h, e)),
        ):
            self.add(prefix + role, store, sp + field + ".weight", shape, inputs=(prefix + "embedding",))
        for role, field in (("key_norm", "norm_key"), ("query_norm", "norm_query"), ("conv_norm", "norm_conv")):
            self.add(prefix + role, store, sp + field + ".weight", (s * h,))
        taps = config["ple_conv_kernel_size"]
        self.add(
            prefix + "convolution",
            store,
            sp + "conv1d.weight",
            (taps, s * h),
            source_shape=(s * h, 1, taps),
            transpose=(2, 0, 1),
        )
        emb = sp + "ple_embedding."
        for field, key in (
            ("layer_multipliers", "multipliers"),
            ("ngram_heads_vocab_sizes", "head_vocab_sizes"),
            ("ngram_heads_offsets", "head_offsets"),
        ):
            actual = store.read_flat(emb + field).tolist()
            if actual != record[key]:
                raise ValueError(f"{emb + field}: checkpoint differs from config-derived hash")
        self.add(
            prefix + "table_scale",
            store,
            emb + "ngram_embedding.weight_scale",
            (1,),
        )
        width = e // ((config["ngram_size"] - 1) * config["heads_per_ngram"])
        shard, rows = 0, 0
        while store.has(f"{emb}ngram_embedding.shard_{shard}.weight"):
            name = f"{emb}ngram_embedding.shard_{shard}.weight"
            count = store.describe(name).shape[0]
            shape = (count, width)
            # Raw E4M3FN words; the shared multiplier is the separate table_scale parameter, so
            # the per-row FP8 matrix resolver must not interpret these shards.
            self.model.add(
                Parameter(
                    f"{prefix}table/{shard}",
                    shape,
                    tensor_source(store, name, shape),
                    lambda selected, format=None, name=name, shape=shape: tensor_source(
                        selected, name, shape
                    ),
                    (),
                    "fp8_e4m3fn",
                    residency="text",
                )
            )
            rows += count
            shard += 1
        if rows < record["table_rows"]:
            raise ValueError(f"{emb}: n-gram table has {rows} rows, need {record['table_rows']}")
        record["table_shards"] = shard

    def block(self, prefix, source_prefix, store, config, mixer):
        if mixer == "full_attention":
            self.attention(prefix, source_prefix, store, config)
        else:
            self.gdn(prefix, source_prefix, store, config)
        self.moe(prefix, source_prefix, store, config)
        self.gated_residual(prefix + "mixer_residual/", source_prefix + "attn_hyper_connection.", store, config)
        self.gated_residual(prefix + "ffn_residual/", source_prefix + "mlp_hyper_connection.", store, config)


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    selected = tuple(components)
    if selected not in (("text",), ("text", "mtp")):
        raise ValueError("Qwen4Exp conversion provides text or text,mtp")
    config = text_config(base.config)
    records = {"text": {"config": config}}
    mtp = "mtp" in selected
    if mtp:
        raw = base.config["text_config"]
        if raw.get("mtp_num_hidden_layers") != 1 or raw.get("mtp", {}).get("layer_types") != [
            "full_attention"
        ]:
            raise ValueError("Qwen4Exp MTP requires one full-attention layer")
        records["mtp"] = {"config": {"architectures": ["Qwen4ExpMTP"]}, "target": "text"}
    refs, resources, count, special = load_resources(
        base.root,
        vocab_size=config["vocab_size"],
        vision_config=None,
        overrides=resource_overrides,
    )
    for component, resource_refs in refs.items():
        records[component]["resources"] = resource_refs
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    builder = _Qwen4ExpBuilder(model)
    h, r = config["hidden_size"], config["vocab_size"]
    builder.add("text/token_embedding", base, TEXT_PREFIX + "embed_tokens.weight", (r, h))
    head_inputs = ("text/final_hidden",) + (("mtp/final_hidden",) if mtp else ())
    builder.add("text/output_head", base, "lm_head.weight", (r, h), inputs=head_inputs)
    builder.gated_residual(
        "text/final_residual/", TEXT_PREFIX + "hyper_connection_mixer.", base, config, inject=False
    )
    ple = {record["layer"]: record for record in config["ple_layers"]}
    for i, kind in enumerate(config["layer_types"]):
        prefix, source_prefix = f"text/layers/{i}/", TEXT_PREFIX + f"layers.{i}."
        builder.block(prefix, source_prefix, base, config, kind)
        if i in ple:
            builder.ple(prefix + "ple/", source_prefix, base, config, ple[i])
    if mtp:
        # Draft stem (SGLang Qwen4ExpForCausalLMMTP): each of the S target streams is
        # hidden_projection(hidden_norm(streams))_s + embedding_projection(embedding_norm(e)).
        s = config["hc_count"]
        builder.add("mtp/embedding_norm", base, "mtp.pre_fc_norm_embedding.weight", (h,))
        builder.add("mtp/hidden_norm", base, "mtp.pre_fc_norm_hidden.weight", (s * h,))
        builder.add(
            "mtp/embedding_projection",
            base,
            "mtp.fc_embedding.weight",
            (h, h),
            inputs=("mtp/normalized_embedding",),
        )
        builder.add(
            "mtp/hidden_projection",
            base,
            "mtp.fc_hidden.weight",
            (h, h),
            inputs=("mtp/normalized_hidden",),
        )
        builder.block("mtp/layers/0/", "mtp.layers.0.", base, config, "full_attention")
        builder.gated_residual(
            "mtp/final_residual/", "mtp.hyper_connection_mixer.", base, config, inject=False
        )
    return model


def is_qwen4_exp(source: SafetensorsSource) -> bool:
    return source.config.get("architectures") == [ARCHITECTURE]
