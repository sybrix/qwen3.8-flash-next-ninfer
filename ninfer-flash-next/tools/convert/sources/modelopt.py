"""Interpret NVIDIA ModelOpt NVFP4 matrices (``weight``/``weight_scale``/``weight_scale_2``).

ModelOpt stores multipliers: ``W = e2m1 * weight_scale * weight_scale_2`` and quantizes
activations as ``x / input_scale``. NInfer stores divisors, so the weight divisor is the
FP32 rounding of ``1 / weight_scale_2``. That reciprocal is the only change; codes and
block scales are imported unchanged.
"""

from __future__ import annotations

from math import isfinite, prod
import struct

import torch

from tools.artifact.formats import valid_positive_fp32_word
from .logical import EncodedRows, LogicalSource
from .safetensors import SafetensorsSource

_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _reciprocal_word(store: SafetensorsSource, name: str) -> bytes:
    info = store.describe(name)
    if info.dtype != "F32" or prod(info.shape) != 1:
        raise ValueError(f"{name}: expected a source FP32 scalar")
    multiplier = float(store.read_flat(name)[0])
    if not isfinite(multiplier) or multiplier <= 0:
        raise ValueError(f"{name}: multiplier must be finite and positive")
    raw = struct.pack("<f", 1.0 / multiplier)
    if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
        raise ValueError(f"{name}: reciprocal is not a positive finite FP32")
    return raw


def modelopt_nvfp4_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int]
) -> LogicalSource:
    """One ModelOpt NVFP4 projection ``prefix`` with logical ``[N,K]`` shape."""
    n, k = shape
    if k % 16:
        raise ValueError(f"{prefix}: NVFP4 K must be divisible by 16")
    packed, scale = f"{prefix}.weight", f"{prefix}.weight_scale"
    for name, expected, dtype in (
        (packed, (n, k // 2), "U8"),
        (scale, (n, k // 16), "F8_E4M3"),
    ):
        info = store.describe(name)
        if info.shape != expected or info.dtype != dtype:
            raise ValueError(
                f"{name}: expected {dtype}{expected}, got {info.dtype}{info.shape}"
            )

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        codes = store.read_flat(packed, begin * (k // 2), end * (k // 2)).reshape(
            end - begin, k // 2
        )
        scales = (
            store.read_flat(scale, begin * (k // 16), end * (k // 16))
            .view(torch.uint8)
            .reshape(end - begin, k // 16)
        )
        if bool((scales > 0x7E).any()):
            raise ValueError(f"{scale}: expected nonnegative finite E4M3FN scales")
        return EncodedRows(
            "nvfp4", codes, scales, _reciprocal_word(store, f"{prefix}.weight_scale_2")
        )

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        first, last = begin // k, (end + k - 1) // k
        words = encoded(first, last)
        codes = torch.stack((words.codes & 15, words.codes >> 4), dim=-1).reshape(
            last - first, k
        )
        values = _E2M1[codes.long()] * (
            words.scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
        )
        values = values / struct.unpack("<f", words.weight_divisor)[0]
        return values.reshape(-1)[begin - first * k : end - first * k]

    return LogicalSource(
        shape,
        f"{store.path}:{prefix} (modelopt nvfp4)",
        read,
        encoded,
        lambda: _reciprocal_word(store, f"{prefix}.weight_scale_2"),
        lambda: _reciprocal_word(store, f"{prefix}.input_scale"),
    )


_E2M1_MAGNITUDES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def quantize_nvfp4(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """NVFP4-encode an FP32 ``[N,K]`` matrix the way ModelOpt does.

    The global multiplier is ``amax / (6 * 448)``; each 16-column block stores the E4M3FN
    rounding of ``block_amax / 6 / global``; each value takes the nearest E2M1 code of
    ``w / (block * global)`` (ties to the smaller magnitude). Returns packed codes ``[N,K/2]``
    (low nibble = even column), scale bytes ``[N,K/16]`` and the global multiplier.
    """
    n, k = weight.shape
    amax = float(weight.abs().max())
    multiplier = amax / (6.0 * 448.0) if amax > 0 else 1.0
    blocks = weight.reshape(n, k // 16, 16)
    block_scale = (blocks.abs().amax(-1) / 6.0 / multiplier).clamp(max=448.0)
    scales = block_scale.to(torch.float8_e4m3fn)
    denom = scales.float() * multiplier
    scaled = torch.where(denom[..., None] > 0, blocks / denom[..., None], torch.zeros_like(blocks))
    magnitude = scaled.abs().clamp(max=6.0)
    # Nearest code; argmin returns the first (smaller) magnitude on a tie.
    index = (magnitude[..., None] - _E2M1_MAGNITUDES).abs().argmin(-1)
    codes = (index | ((scaled < 0) & (index > 0)).to(torch.int64) << 3).reshape(n, k).to(torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed, scales.view(torch.uint8), multiplier


def fp8_block_as_nvfp4_source(
    store: SafetensorsSource, prefix: str, shape: tuple[int, int], block: int = 128
) -> LogicalSource:
    """A ModelOpt ``FP8_PB_WO`` projection (E4M3 ``weight`` with one ``weight_scale_inv``
    multiplier per ``block`` x ``block`` tile), re-encoded as NVFP4 for an expert bank."""
    n, k = shape
    weight, scale_inv = f"{prefix}.weight", f"{prefix}.weight_scale_inv"
    rows_b, cols_b = -(-n // block), -(-k // block)
    for name, expected, dtype in ((weight, (n, k), "F8_E4M3"), (scale_inv, (rows_b, cols_b), "BF16")):
        info = store.describe(name)
        if info.shape != expected or info.dtype != dtype:
            raise ValueError(f"{name}: expected {dtype}{expected}, got {info.dtype}{info.shape}")
    cache: dict[str, object] = {}

    def encode():
        if not cache:
            values = store.read_flat(weight).float().reshape(n, k)
            scale = store.read_flat(scale_inv).float().reshape(rows_b, cols_b)
            scale = scale.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
            packed, scales, multiplier = quantize_nvfp4(values * scale)
            raw = struct.pack("<f", 1.0 / multiplier)
            if not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
                raise ValueError(f"{prefix}: NVFP4 divisor is not a positive finite FP32")
            cache.update(codes=packed, scales=scales, divisor=raw)
        return cache

    def encoded(begin: int, end: int) -> EncodedRows:
        if not 0 <= begin < end <= n:
            raise ValueError(f"{prefix}: invalid encoded rows [{begin},{end})")
        words = encode()
        return EncodedRows("nvfp4", words["codes"][begin:end], words["scales"][begin:end], words["divisor"])

    def read(begin: int, end: int) -> torch.Tensor:
        if begin == end:
            return torch.empty(0, dtype=torch.float32)
        words = encode()
        codes = torch.stack((words["codes"] & 15, words["codes"] >> 4), dim=-1).reshape(n, k)
        values = _E2M1[codes.long()] * words["scales"].view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=1)
        values = values / struct.unpack("<f", words["divisor"])[0]
        return values.reshape(-1)[begin:end]

    return LogicalSource(
        shape,
        f"{store.path}:{prefix} (fp8 block -> nvfp4)",
        read,
        encoded,
        lambda: encode()["divisor"],
        None,
    )
