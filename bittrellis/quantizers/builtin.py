"""Built-in quantizers for HPC-01."""

from __future__ import annotations

import numpy as np

from ..model.qwen38 import Linear, Unit
from ..precision import FP8, NVFP4, Q4_K
from ..quant.formats import quantize_fp8_per_channel, quantize_nvfp4
from .base import Produced, QuantContext, Quantizer, f32_rows

MODELOPT_SUFFIXES = (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")
CT_SUFFIXES = (".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale")


class Runtime(Quantizer):
    """Q4_K: store the frozen BF16 tensor; SparkInfer fits Q4_K at load. No choices exist here."""

    name = "runtime"
    formats = (Q4_K,)
    lineage = "runtime"
    replay_mode = "none"
    description = "stored BF16 (byte-identical to base); SparkInfer's Lloyd Q4_K fit at load"

    def encode(self, ctx: QuantContext, unit: Unit, lin: Linear, fmt: str) -> list[Produced]:
        ref = ctx.base.get(lin.prefix + ".weight")
        return [(".weight", "BF16", ref.shape, ctx.base.raw(lin.prefix + ".weight"))]


class _Splice(Quantizer):
    """Copy a unit's already-quantized bytes out of a pinned, hash-verified checkpoint."""

    lineage = "attested"
    replay_mode = "none"
    suffix_sets: tuple[tuple[str, ...], ...] = (MODELOPT_SUFFIXES, CT_SUFFIXES)
    marker_by_format = {NVFP4: (".weight_scale_2", ".weight_packed")}

    def _layout(self, src, lin: Linear) -> tuple[str, ...] | None:
        if src.get(lin.prefix + ".weight_scale_2") is not None:
            return MODELOPT_SUFFIXES
        if src.get(lin.prefix + ".weight_packed") is not None:
            return CT_SUFFIXES
        return None

    def available(self, ctx: QuantContext, unit: Unit, fmt: str) -> str | None:
        src = ctx.sources.get(self.source_id)
        if src is None:
            return f"attested source {self.source_id!r} is not available"
        for lin in unit.linears:
            if self._layout(src, lin) is None:
                return f"{self.source_id} has no {fmt} bytes for {lin.prefix}"
        return None

    def encode(self, ctx: QuantContext, unit: Unit, lin: Linear, fmt: str) -> list[Produced]:
        src = ctx.sources[self.source_id]
        out = []
        for suf in self._layout(src, lin):
            t = src.get(lin.prefix + suf)
            if t is not None:
                out.append((suf, t.dtype, t.shape, src.raw(lin.prefix + suf)))
        return out


class Baseline(_Splice):
    name = "baseline"
    formats = (NVFP4,)
    source_id = "gittensor_nvfp4"
    description = "the shipped gittensor NVFP4 bytes (V0), ModelOpt round-to-nearest"


class Unsloth(_Splice):
    name = "unsloth"
    formats = (NVFP4,)
    source_id = "unsloth_nvfp4"
    description = "unsloth's calibrated NVFP4 (llm-compressor, actorder static); MLP layers 0-55 only"


class RTN(Quantizer):
    """Round-to-nearest from BF16. NVFP4 with a max-calibrated tensor scale; FP8 per row."""

    name = "rtn"
    formats = (NVFP4, FP8)
    lineage = "regenerable"
    replay_mode = "independent"
    description = "round-to-nearest from BF16 (NVFP4 max-calibrated, FP8 per-row)"

    def encode(self, ctx: QuantContext, unit: Unit, lin: Linear, fmt: str) -> list[Produced]:
        if fmt == FP8:
            codes = np.empty((lin.rows, lin.cols), np.uint8)
            scale = np.empty((lin.rows, 1), "<u2")
            for r, rows in f32_rows(ctx, lin):
                c, s = quantize_fp8_per_channel(rows)
                codes[r : r + len(rows)] = c
                scale[r : r + len(rows)] = s
            return [(".weight", "F8_E4M3", codes.shape, codes), (".weight_scale", "BF16", scale.shape, scale)]
        amax = 0.0
        for _, rows in f32_rows(ctx, lin):
            amax = max(amax, float(np.abs(rows).max()))
        packed = np.empty((lin.rows, lin.cols // 2), np.uint8)
        scales = np.empty((lin.rows, lin.cols // 16), np.uint8)
        ws2 = np.float32(1.0)
        for r, rows in f32_rows(ctx, lin):
            p, s, ws2 = quantize_nvfp4(rows, global_amax=amax)
            packed[r : r + len(rows)] = p
            scales[r : r + len(rows)] = s
        return [(".weight", "U8", packed.shape, packed), (".weight_scale", "F8_E4M3", scales.shape, scales),
                (".weight_scale_2", "F32", (), np.asarray(ws2, "<f4").reshape(()))]
