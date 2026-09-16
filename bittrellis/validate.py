"""Audit a checkpoint before anything is benchmarked.

`describe()` answers "what will SparkInfer execute?" for any Qwen3.8 checkpoint (including the
external references). `audit()` additionally enforces the HPC-01 rules for candidates:

1. config.json is the baseline config except for `quantization_config`.
2. The tensor set is exactly what the manifest implies -- nothing added, nothing missing.
3. Every non-searchable tensor is byte-identical to the baseline.
4. Every BF16 (Q4_K) Linear is byte-identical to the BF16 base model.
5. Every NVFP4 / FP8 Linear decodes to within a fidelity bound of the base weights, sampled
   over rows, so fine-tuned or substituted weights cannot hide inside a "quantized" tensor.
6. Each unit resolves to the manifest's precision under the pinned loader rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .manifest import Manifest
from .model.qwen38 import Qwen38Arch, Unit
from .precision import FP8, NVFP4, Q4_K, S_FP8, S_NVFP4, resolve_checkpoint, stored_format
from .quant.formats import (
    bf16_to_f32,
    dequantize_fp8_per_channel,
    dequantize_nvfp4,
    quantize_fp8_per_channel,
    quantize_nvfp4,
)
from .safetensors_io import SafeTensorsDir

SAMPLE_ROWS = 24
# A candidate's quantized rows may not reconstruct the base rows worse than this multiple of
# plain round-to-nearest on the same rows (plus a small absolute slack for tiny rows).
FIDELITY_RATIO = {NVFP4: 1.35, FP8: 1.35}
FIDELITY_SLACK = 0.01


@dataclass
class AuditResult:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    runtime: dict[str, str] = field(default_factory=dict)
    fidelity: dict[str, float] = field(default_factory=dict)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.errors.append(msg)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors[:200], "n_errors": len(self.errors),
                "warnings": self.warnings[:50], "runtime": self.runtime, "fidelity": self.fidelity}


def describe(ckpt_dir: str | Path) -> dict[str, str]:
    """Runtime precision label per unit, e.g. {'L0.gdn.qkv': 'FP8', 'L3.attn.q': 'Q4_K(fp8)'}."""
    cfg = json.loads((Path(ckpt_dir) / "config.json").read_text())
    units = Qwen38Arch.from_config(cfg).units()
    with SafeTensorsDir(ckpt_dir) as ck:
        return {uid: r.label for uid, r in resolve_checkpoint(ck, units).items()}


def _rows(n_rows: int) -> np.ndarray:
    return np.unique(np.linspace(0, n_rows - 1, SAMPLE_ROWS).astype(np.int64))


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    nb = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float(np.linalg.norm(a))


def _fidelity(ck: SafeTensorsDir, base: SafeTensorsDir, unit: Unit, precision: str, res: AuditResult) -> None:
    for lin in unit.linears:
        rows = _rows(lin.rows)
        w = bf16_to_f32(base.array(lin.prefix + ".weight", "<u2")[rows].reshape(-1)).reshape(len(rows), lin.cols)
        if precision == NVFP4:
            wp = ck.get(lin.prefix + ".weight_packed")
            if wp is not None:
                packed = ck.array(lin.prefix + ".weight_packed", "u1").reshape(lin.rows, -1)[rows]
                g = float(ck.array(lin.prefix + ".weight_global_scale", "<f4").reshape(-1)[0])
                ws2 = 1.0 / g
            else:
                packed = ck.array(lin.prefix + ".weight", "u1").reshape(lin.rows, -1)[rows]
                ws2 = float(ck.array(lin.prefix + ".weight_scale_2", "<f4").reshape(-1)[0])
            scales = ck.array(lin.prefix + ".weight_scale", "u1").reshape(lin.rows, -1)[rows]
            got = _rel(dequantize_nvfp4(packed, scales, ws2), w)
            # RTN reference with the candidate's own global scale, so a calibrated global scale
            # is compared like for like.
            ref = _rel(dequantize_nvfp4(*quantize_nvfp4(w, global_amax=ws2 * 6.0 * 448.0)), w)
        else:
            codes = ck.array(lin.prefix + ".weight", "u1").reshape(lin.rows, lin.cols)[rows]
            scale = ck.array(lin.prefix + ".weight_scale", "<u2").reshape(-1)[rows]
            got = _rel(dequantize_fp8_per_channel(codes, scale), w)
            ref = _rel(dequantize_fp8_per_channel(*quantize_fp8_per_channel(w)), w)
        res.fidelity[lin.prefix] = round(got, 5)
        if got > FIDELITY_RATIO[precision] * ref + FIDELITY_SLACK:
            res.fail(f"{lin.prefix}: {precision} reconstruction error {got:.4f} exceeds "
                     f"{FIDELITY_RATIO[precision]}x round-to-nearest ({ref:.4f}) -- not a faithful encoding")


def audit(ckpt_dir: str | Path, manifest: Manifest, base_dir: str | Path, baseline_dir: str | Path,
          check_bytes: bool = True, check_fidelity: bool = True) -> AuditResult:
    res = AuditResult()
    ckpt_dir, base_dir, baseline_dir = Path(ckpt_dir), Path(base_dir), Path(baseline_dir)
    cfg = json.loads((ckpt_dir / "config.json").read_text())
    base_cfg = json.loads((baseline_dir / "config.json").read_text())
    if "quantization_config" not in cfg:
        res.fail("config.json has no quantization_config (the loader rejects the directory)")
    strip = lambda c: {k: v for k, v in c.items() if k != "quantization_config"}  # noqa: E731
    if strip(cfg) != strip(base_cfg):
        res.fail("config.json differs from the baseline outside quantization_config")
    arch = Qwen38Arch.from_config(base_cfg)
    units = arch.units()
    try:
        expanded = manifest.expand(units)
    except ValueError as e:
        res.fail(f"manifest: {e}")
        return res

    with SafeTensorsDir(ckpt_dir) as ck, SafeTensorsDir(base_dir) as base, SafeTensorsDir(baseline_dir) as bl:
        resolved = resolve_checkpoint(ck, units)
        res.runtime = {uid: r.label for uid, r in resolved.items()}
        expected_names: set[str] = set()
        searchable = set()
        for u in units:
            want = expanded[u.id]
            got = resolved[u.id]
            if got.precision is None:
                res.fail(f"{u.id}: does not load ({got.note})")
                continue
            if got.label != want:
                res.fail(f"{u.id}: manifest says {want}, runtime would execute {got.label}")
                continue
            for lin in u.linears:
                searchable.add(lin.prefix)
                fmt, _ = stored_format(ck, lin)
                if want == Q4_K:
                    expected_names.add(lin.prefix + ".weight")
                    if check_bytes and ck.sha256(lin.prefix + ".weight") != base.sha256(lin.prefix + ".weight"):
                        res.fail(f"{lin.prefix}.weight: BF16 bytes differ from the base model")
                elif fmt == S_FP8:
                    expected_names.update({lin.prefix + ".weight", lin.prefix + ".weight_scale"})
                elif fmt == S_NVFP4:
                    names = ([".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale"]
                             if ck.get(lin.prefix + ".weight_packed") else
                             [".weight", ".weight_scale", ".weight_scale_2", ".input_scale"])
                    expected_names.update(lin.prefix + s for s in names if ck.get(lin.prefix + s) is not None)
            if check_fidelity and want in (NVFP4, FP8):
                _fidelity(ck, base, u, want, res)

        for name, ref in bl.tensors.items():
            prefix, _, _ = name.rpartition(".")
            if prefix in searchable:
                continue
            expected_names.add(name)
            t = ck.get(name)
            if t is None:
                res.fail(f"{name}: missing (present in baseline)")
            elif (t.dtype, t.shape) != (ref.dtype, ref.shape):
                res.fail(f"{name}: {t.dtype}{list(t.shape)} != baseline {ref.dtype}{list(ref.shape)}")
            elif check_bytes and ck.sha256(name) != bl.sha256(name):
                res.fail(f"{name}: bytes differ from the baseline")
        extra = sorted(set(ck.tensors) - expected_names)
        if extra:
            res.fail(f"{len(extra)} unexpected tensors, e.g. {extra[:5]}")
    return res
