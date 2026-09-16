"""Bit-exact numpy codecs for the number formats the pinned SparkInfer loader reads.

Everything here works on raw little-endian bytes so a checkpoint can be built on any CPU box
without torch. Conventions match what `runtime/src/models/qwen35.cpp` decodes at the pinned
commit (see docs/precision_space.md for the line-level references):

* BF16  -- IEEE-754 binary16 "brain float": 1 sign, 8 exponent, 7 mantissa bits.
* FP8   -- float8_e4m3fn weight, one BF16 scale per output row:  W[r, c] = e4m3(w) * scale[r].
* NVFP4 -- ModelOpt layout: `.weight` U8 packed E2M1 nibbles (low nibble = even column),
           `.weight_scale` UE4M3 per 16 weights, `.weight_scale_2` F32 tensor scale:
           W = e2m1(q) * ue4m3(block) * weight_scale_2.
"""

from __future__ import annotations

import numpy as np

FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
NVFP4_GROUP = 16

# --------------------------------------------------------------------------------------------
# BF16


def bf16_to_f32(raw: bytes | np.ndarray) -> np.ndarray:
    """Decode BF16 bytes (or a uint16 array) to float32."""
    u = np.frombuffer(raw, dtype="<u2") if isinstance(raw, (bytes, bytearray, memoryview)) else raw
    u = u.astype(np.uint32)
    sign = (u >> 15) & 0x1
    exp = (u >> 7) & 0xFF
    mant = u & 0x7F
    # Re-pack as float32 bits: same sign and exponent, mantissa left-aligned in 23 bits.
    bits = (sign << 31) | (exp << 23) | (mant << 16)
    return bits.astype(np.uint32).view(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    """Encode float32 to BF16 (uint16 array), rounding the mantissa to nearest (ties up)."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    sign = (bits >> 31) & 0x1
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    m7 = (mant + (1 << 15)) >> 16
    carry = m7 >> 7
    m7 = m7 & 0x7F
    exp = exp + carry
    overflow = exp >= 0xFF
    exp = np.where(overflow, 0xFF, exp)
    m7 = np.where(overflow & ((bits >> 23) & 0xFF != 0xFF), 0, m7)
    out = (sign << 15) | (exp << 7) | m7
    return out.astype("<u2")


# --------------------------------------------------------------------------------------------
# FP8 E4M3FN


def _e4m3_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.uint32)
    sign = np.where(codes & 0x80, -1.0, 1.0)
    exp = (codes >> 3) & 0xF
    mant = codes & 0x7
    val = np.where(exp == 0, (mant / 8.0) * 2.0 ** (1 - 7), (1.0 + mant / 8.0) * 2.0 ** (exp.astype(np.int64) - 7))
    val = sign * val
    val[(codes & 0x7F) == 0x7F] = np.nan  # e4m3fn: only all-ones is NaN
    return val.astype(np.float32)


E4M3_TABLE = _e4m3_table()
_E4M3_POS_CODES = np.arange(0, 0x7F, dtype=np.uint8)  # 0x00..0x7E, monotonically increasing
_E4M3_POS_VALS = E4M3_TABLE[_E4M3_POS_CODES]
_E4M3_MIDS = (_E4M3_POS_VALS[1:] + _E4M3_POS_VALS[:-1]) / 2.0


def e4m3_encode(x: np.ndarray) -> np.ndarray:
    """Round float32 to the nearest float8_e4m3fn code, saturating at +-448."""
    x = np.asarray(x, dtype=np.float32)
    a = np.minimum(np.abs(x), FP8_E4M3_MAX)
    idx = np.searchsorted(_E4M3_MIDS, a, side="left").astype(np.uint8)
    return np.where(x < 0, idx | 0x80, idx).astype(np.uint8)


def e4m3_decode(codes: np.ndarray) -> np.ndarray:
    return E4M3_TABLE[np.asarray(codes, dtype=np.uint8)]


# --------------------------------------------------------------------------------------------
# FP4 E2M1

E2M1_LEVELS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1_TABLE = np.concatenate([E2M1_LEVELS, -E2M1_LEVELS]).astype(np.float32)
_E2M1_MIDS = (E2M1_LEVELS[1:] + E2M1_LEVELS[:-1]) / 2.0


def e2m1_encode(x: np.ndarray) -> np.ndarray:
    """Round to the nearest E2M1 level (ties toward the smaller magnitude); 4-bit codes."""
    x = np.asarray(x, dtype=np.float32)
    a = np.minimum(np.abs(x), FP4_E2M1_MAX)
    idx = np.searchsorted(_E2M1_MIDS, a, side="left").astype(np.uint8)
    return np.where(x < 0, idx | 0x8, idx).astype(np.uint8)


def pack_nibbles(codes: np.ndarray) -> np.ndarray:
    """[rows, cols] 4-bit codes -> [rows, cols/2] bytes, low nibble holds the even column."""
    c = np.asarray(codes, dtype=np.uint8)
    return (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint8)
    out = np.empty(p.shape[:-1] + (p.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = p & 0x0F
    out[..., 1::2] = p >> 4
    return out


# --------------------------------------------------------------------------------------------
# Quantizers. Each returns the tensors exactly as they are stored in the checkpoint.


def quantize_fp8_per_channel(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """float32 [rows, cols] -> (e4m3 codes [rows, cols], BF16 scale [rows, 1] as uint16)."""
    w = np.asarray(w, dtype=np.float32)
    amax = np.abs(w).max(axis=1)
    scale_bf16 = f32_to_bf16(np.where(amax > 0, amax / FP8_E4M3_MAX, 1.0).astype(np.float32))
    # Quantize against the scale the runtime will actually decode, not the unrounded one.
    scale = bf16_to_f32(scale_bf16)
    codes = e4m3_encode(w / scale[:, None])
    return codes, scale_bf16.reshape(-1, 1)


def dequantize_fp8_per_channel(codes: np.ndarray, scale_bf16: np.ndarray) -> np.ndarray:
    scale = bf16_to_f32(np.asarray(scale_bf16, dtype="<u2").reshape(-1))
    return e4m3_decode(codes) * scale[:, None]


def quantize_nvfp4(w: np.ndarray, global_amax: float | None = None) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """Round-to-nearest NVFP4 in the ModelOpt layout.

    float32 [rows, cols] -> (packed U8 [rows, cols/2], UE4M3 block scales [rows, cols/16],
    weight_scale_2 F32). `global_amax` defaults to max|w| (ModelOpt's max calibration).
    """
    w = np.asarray(w, dtype=np.float32)
    rows, cols = w.shape
    if cols % NVFP4_GROUP:
        raise ValueError(f"NVFP4 needs cols divisible by {NVFP4_GROUP}, got {cols}")
    amax = float(np.abs(w).max()) if global_amax is None else float(global_amax)
    ws2 = np.float32(amax / (FP4_E2M1_MAX * FP8_E4M3_MAX)) if amax > 0 else np.float32(1.0)
    blocks = w.reshape(rows, cols // NVFP4_GROUP, NVFP4_GROUP)
    block_amax = np.abs(blocks).max(axis=2)
    block_scale = e4m3_encode(block_amax / FP4_E2M1_MAX / ws2)
    block_scale = np.where(block_scale & 0x80, 0, block_scale).astype(np.uint8)  # unsigned
    s = e4m3_decode(block_scale) * ws2
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.where(s[:, :, None] > 0, blocks / s[:, :, None], 0.0)
    codes = e2m1_encode(scaled).reshape(rows, cols)
    return pack_nibbles(codes), block_scale, ws2


def dequantize_nvfp4(packed: np.ndarray, block_scale: np.ndarray, ws2: float) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    rows = packed.shape[0]
    codes = unpack_nibbles(packed)
    cols = codes.shape[1]
    vals = E2M1_TABLE[codes].reshape(rows, cols // NVFP4_GROUP, NVFP4_GROUP)
    s = e4m3_decode(np.asarray(block_scale, dtype=np.uint8).reshape(rows, -1)) * np.float32(ws2)
    return (vals * s[:, :, None]).reshape(rows, cols)
