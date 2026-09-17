"""NVFP4 with a per-block clipping search: each block's scale is the one that minimises its squared error."""

from __future__ import annotations

import numpy as np

from ..model.qwen38 import Linear, Unit
from ..precision import NVFP4
from ..quant.formats import (
    E2M1_TABLE,
    FP4_E2M1_MAX,
    FP8_E4M3_MAX,
    NVFP4_GROUP,
    e2m1_encode,
    e4m3_decode,
    e4m3_encode,
    pack_nibbles,
)
from .base import Produced, QuantContext, Quantizer, f32_rows

CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8)


def _encode_rows(rows: np.ndarray, ws2: np.float32) -> tuple[np.ndarray, np.ndarray]:
    n, cols = rows.shape
    blocks = rows.reshape(n, cols // NVFP4_GROUP, NVFP4_GROUP)
    block_amax = np.abs(blocks).max(axis=2)
    best_err = best_codes = best_scale = None
    for clip in CLIPS:
        scale = e4m3_encode(block_amax * clip / FP4_E2M1_MAX / ws2)
        scale = np.where(scale & 0x80, 0, scale).astype(np.uint8)
        s = e4m3_decode(scale) * ws2
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.where(s[:, :, None] > 0, blocks / s[:, :, None], 0.0)
        codes = e2m1_encode(scaled)
        err = ((E2M1_TABLE[codes] * s[:, :, None] - blocks) ** 2).sum(axis=2)
        if best_err is None:
            best_err, best_codes, best_scale = err, codes, scale
        else:
            better = err < best_err
            best_err = np.where(better, err, best_err)
            best_codes = np.where(better[:, :, None], codes, best_codes)
            best_scale = np.where(better, scale, best_scale)
    return pack_nibbles(best_codes.reshape(n, cols)), best_scale.astype(np.uint8)


class MSEClip(Quantizer):
    name = "mse_clip"
    version = 1
    formats = (NVFP4,)
    lineage = "regenerable"
    replay_mode = "independent"
    description = "NVFP4, per-block scale chosen among 5 clip ratios by squared reconstruction error"

    def encode(self, ctx: QuantContext, unit: Unit, lin: Linear, fmt: str) -> list[Produced]:
        amax = 0.0
        for _, rows in f32_rows(ctx, lin):
            amax = max(amax, float(np.abs(rows).max()))
        ws2 = np.float32(amax / (FP4_E2M1_MAX * FP8_E4M3_MAX)) if amax > 0 else np.float32(1.0)
        packed = np.empty((lin.rows, lin.cols // 2), np.uint8)
        scales = np.empty((lin.rows, lin.cols // NVFP4_GROUP), np.uint8)
        for r, rows in f32_rows(ctx, lin):
            p, s = _encode_rows(rows, ws2)
            packed[r : r + len(rows)] = p
            scales[r : r + len(rows)] = s
        return [(".weight", "U8", packed.shape, packed), (".weight_scale", "F8_E4M3", scales.shape, scales),
                (".weight_scale_2", "F32", (), np.asarray(ws2, "<f4").reshape(()))]
