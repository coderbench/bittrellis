"""Build a deployable SparkInfer checkpoint from a precision manifest.

Inputs are the two pinned checkpoints: the BF16 base model and the NVFP4 baseline. Output is a
ModelOpt-layout directory the pinned loader reads directly:

    precision  stored bytes                               source
    ---------  -----------------------------------------  ---------------------------------------
    NVFP4      .weight U8 + .weight_scale + _scale_2       baseline bytes, or RTN from BF16
    FP8        .weight F8_E4M3 + per-row BF16 .weight_scale  RTN from BF16 (GDN only)
    Q4_K       .weight BF16 (byte-identical to base)       SparkInfer fits Q4_K at load

Non-searchable tensors (embeddings, norms, conv1d, in_proj_a/b, A_log, dt_bias, vision tower)
are copied byte-for-byte from the baseline. No GPU is needed; the build is deterministic, so
the same manifest always yields the same shard hashes.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from . import __version__
from .manifest import Manifest, candidate_hash, summarize
from .model.qwen38 import Linear, Qwen38Arch, Unit
from .precision import FP8, NVFP4, Q4_K
from .quant.formats import FP4_E2M1_MAX, FP8_E4M3_MAX, bf16_to_f32, quantize_fp8_per_channel, quantize_nvfp4
from .safetensors_io import SafeTensorsDir, ShardWriter, file_sha256
from .track import Track

NVFP4_SUFFIXES = (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")
ROW_CHUNK = 2048


@dataclass
class PlannedTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    make: Callable[[], object]


def _base_f32_rows(base: SafeTensorsDir, lin: Linear):
    """Yield (row_start, float32 rows) of a BF16 base Linear in bounded-memory chunks."""
    name = lin.prefix + ".weight"
    t = base.get(name)
    if t is None or t.dtype != "BF16" or t.shape != (lin.rows, lin.cols):
        raise ValueError(f"base checkpoint: expected BF16 {name} [{lin.rows}, {lin.cols}], got {t}")
    u = base.array(name, "<u2")
    for r in range(0, lin.rows, ROW_CHUNK):
        yield r, bf16_to_f32(u[r : r + ROW_CHUNK].reshape(-1)).reshape(-1, lin.cols)


def _nvfp4_rtn(base: SafeTensorsDir, lin: Linear) -> tuple[np.ndarray, np.ndarray, np.float32]:
    amax = 0.0
    for _, rows in _base_f32_rows(base, lin):
        amax = max(amax, float(np.abs(rows).max()))
    packed = np.empty((lin.rows, lin.cols // 2), np.uint8)
    scales = np.empty((lin.rows, lin.cols // 16), np.uint8)
    ws2 = np.float32(1.0)
    for r, rows in _base_f32_rows(base, lin):
        p, s, ws2 = quantize_nvfp4(rows, global_amax=amax)
        packed[r : r + len(rows)] = p
        scales[r : r + len(rows)] = s
    return packed, scales, ws2


def _fp8_rtn(base: SafeTensorsDir, lin: Linear) -> tuple[np.ndarray, np.ndarray]:
    codes = np.empty((lin.rows, lin.cols), np.uint8)
    scale = np.empty((lin.rows, 1), "<u2")
    for r, rows in _base_f32_rows(base, lin):
        c, s = quantize_fp8_per_channel(rows)
        codes[r : r + len(rows)] = c
        scale[r : r + len(rows)] = s
    return codes, scale


class _Memo:
    """Compute a multi-tensor encoding once, hand out its parts lazily."""

    def __init__(self, fn):
        self.fn, self.value = fn, None

    def __call__(self):
        if self.value is None:
            self.value = self.fn()
        return self.value


CT_NVFP4_SUFFIXES = (".weight_packed", ".weight_scale", ".weight_global_scale", ".input_global_scale")


def plan_tensors(units: list[Unit], full: dict[str, tuple[str, str]], base: SafeTensorsDir,
                 baseline: SafeTensorsDir, sources: dict[str, SafeTensorsDir] | None = None) -> list[PlannedTensor]:
    sources = sources or {}
    plan: list[PlannedTensor] = []
    searchable: set[str] = set()
    for u in units:
        prec, quantizer = full[u.id]
        for lin in u.linears:
            p = lin.prefix
            searchable.add(p)
            if prec == Q4_K:
                ref = base.get(p + ".weight")
                if ref is None:
                    raise ValueError(f"base checkpoint lacks {p}.weight")
                plan.append(PlannedTensor(p + ".weight", "BF16", ref.shape, lambda n=p + ".weight": base.raw(n)))
            elif prec == FP8:
                memo = _Memo(lambda lin=lin: _fp8_rtn(base, lin))
                plan.append(PlannedTensor(p + ".weight", "F8_E4M3", (lin.rows, lin.cols), lambda m=memo: m()[0]))
                plan.append(PlannedTensor(p + ".weight_scale", "BF16", (lin.rows, 1), lambda m=memo: m()[1]))
            elif prec == NVFP4 and quantizer == "unsloth":
                src = sources.get("unsloth")
                if src is None:
                    raise ValueError("quantizer 'unsloth' needs the pinned R1 checkpoint (--unsloth)")
                if src.get(p + ".weight_packed") is None:
                    raise ValueError(f"unsloth checkpoint has no NVFP4 bytes for {p} (it ships NVFP4 only for MLP layers 0-55)")
                for suf in CT_NVFP4_SUFFIXES:
                    ref = src.get(p + suf)
                    if ref is not None:
                        plan.append(PlannedTensor(p + suf, ref.dtype, ref.shape, lambda n=p + suf, s_=src: s_.raw(n)))
            elif prec == NVFP4 and quantizer == "baseline":
                if baseline.get(p + ".weight_scale_2") is None:
                    raise ValueError(f"baseline has no ModelOpt NVFP4 tensors for {p}; use quantizers.NVFP4: rtn")
                for suf in NVFP4_SUFFIXES:
                    ref = baseline.get(p + suf)
                    if ref is not None:
                        plan.append(PlannedTensor(p + suf, ref.dtype, ref.shape, lambda n=p + suf: baseline.raw(n)))
            elif prec == NVFP4 and quantizer == "rtn":
                memo = _Memo(lambda lin=lin: _nvfp4_rtn(base, lin))
                plan.append(PlannedTensor(p + ".weight", "U8", (lin.rows, lin.cols // 2), lambda m=memo: m()[0]))
                plan.append(PlannedTensor(p + ".weight_scale", "F8_E4M3", (lin.rows, lin.cols // 16), lambda m=memo: m()[1]))
                plan.append(PlannedTensor(p + ".weight_scale_2", "F32", (), lambda m=memo: np.asarray(m()[2], "<f4").reshape(())))
            else:
                raise ValueError(f"{u.id}: unhandled {prec}@{quantizer}")
    linear_suffixes = NVFP4_SUFFIXES + (".weight_packed", ".weight_global_scale", ".input_global_scale")
    for name, ref in baseline.tensors.items():
        prefix, _, suffix = name.rpartition(".")
        if prefix in searchable and "." + suffix in linear_suffixes:
            continue
        plan.append(PlannedTensor(name, ref.dtype, ref.shape, lambda n=name: baseline.raw(n)))
    plan.sort(key=lambda t: t.name)
    names = [t.name for t in plan]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate tensor in build plan")
    return plan


def quant_config(units: list[Unit], full: dict[str, tuple[str, str]], baseline_cfg: dict) -> dict:
    layers: dict[str, dict] = {}
    q4k: list[str] = []
    for u in units:
        for lin in u.linears:
            p, qz = full[u.id]
            if p == NVFP4:
                layers[lin.prefix] = {"quant_algo": "NVFP4", "group_size": 16, "quantizer": qz}
            elif p == FP8:
                layers[lin.prefix] = {"quant_algo": "FP8_PER_CHANNEL"}
            else:
                q4k.append(lin.prefix)
    old = baseline_cfg.get("quantization_config", {})
    ignore = sorted(set(old.get("ignore", [])) | set(q4k))
    return {
        "quant_method": "modelopt",
        "quant_algo": "MIXED_PRECISION",
        "producer": {"name": "bittrellis", "version": __version__},
        "quantized_layers": layers,
        "ignore": ignore,
        "runtime_fit": {"Q4_K": q4k},
        "fp8_scale": {"max": FP8_E4M3_MAX}, "nvfp4_scale": {"max": FP4_E2M1_MAX * FP8_E4M3_MAX},
    }


def build(manifest: Manifest, track: Track, base_dir: Path, baseline_dir: Path, out_dir: Path,
          log: Callable[[str], None] = print, shard_bytes: int = 5 * 1024**3,
          sources: dict[str, Path] | None = None) -> dict:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty")
    if manifest.track != track.id:
        raise ValueError(f"manifest is for {manifest.track}, track is {track.id}")
    baseline_cfg = json.loads((Path(baseline_dir) / "config.json").read_text())
    arch = Qwen38Arch.from_config(baseline_cfg)
    units = arch.units()
    full = manifest.expand_full(units)
    cid = candidate_hash(track.id, full)
    t0 = time.time()
    used = {q for _, q in full.values()}
    opened = {k: SafeTensorsDir(v) for k, v in (sources or {}).items() if k in used}
    with SafeTensorsDir(base_dir) as base, SafeTensorsDir(baseline_dir) as baseline:
        plan = plan_tensors(units, full, base, baseline, opened)
        log(f"[build] {manifest.name} ({cid}): {len(plan)} tensors -> {out_dir}")
        writer = ShardWriter(out_dir, shard_bytes)
        for i, t in enumerate(plan):
            writer.add(t.name, t.dtype, t.shape, t.make())
            if (i + 1) % 250 == 0:
                log(f"[build]   {i + 1}/{len(plan)} tensors, {time.time() - t0:.0f}s")
        writer.close(metadata={"producer": "bittrellis", "candidate_id": cid})
    for src in opened.values():
        src.close()
    cfg = dict(baseline_cfg)
    cfg["quantization_config"] = quant_config(units, full, baseline_cfg)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    hf_quant = {"producer": {"name": "bittrellis", "version": __version__},
                "quantization": {k: v for k, v in cfg["quantization_config"].items() if k != "quant_method"}}
    (out_dir / "hf_quant_config.json").write_text(json.dumps(hf_quant, indent=2) + "\n")
    for f in track["model"]["aux_files"]:
        src = Path(baseline_dir) / f
        if src.exists():
            shutil.copy2(src, out_dir / f)
    (out_dir / "precision_manifest.yaml").write_text(yaml.safe_dump(manifest.to_dict(units), sort_keys=False))
    files = sorted(p.name for p in out_dir.glob("*.safetensors"))
    record = {
        "candidate_id": cid,
        "name": manifest.name,
        "track": track.id,
        "bittrellis_version": __version__,
        "base": track["model"]["base"],
        "baseline": track["model"]["baseline"],
        "quantizers": manifest.quantizers,
        "quantizer_sources": {k: track["model"].get("quantizer_sources", {}).get(k) for k in opened},
        "summary": summarize(full, units),
        "checkpoint_bytes": sum((out_dir / f).stat().st_size for f in files),
        "files": {f: file_sha256(out_dir / f) for f in files},
        "build_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "bittrellis_build.json").write_text(json.dumps(record, indent=2) + "\n")
    log(f"[build] done in {record['build_seconds']}s, {record['checkpoint_bytes'] / 1e9:.2f} GB")
    return record
