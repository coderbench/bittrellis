"""The quantizer contract.

A quantizer turns one searchable Linear into the tensors a checkpoint stores for it. Every
quantizer declares:

* `name`, `version`         identity; both enter the candidate id, so changing the algorithm
                            changes every candidate that uses it
* `formats`                 execution formats it can produce (NVFP4, FP8, Q4_K)
* `lineage`                 how the audit establishes that its bytes are legitimate:
    - "runtime"      nothing stored beyond the frozen BF16 tensor; SparkInfer fits the format
    - "regenerable"  implemented here; the evaluator rebuilds sampled tensors and compares bytes
    - "attested"     copied from a maintainer-approved, hash-pinned checkpoint (sources.lock.json)
* `replay_mode`             for regenerable quantizers:
    - "independent"  a tensor depends only on its own BF16 weight (and the calibration manifest)
    - "sequential"   a tensor may depend on earlier quantized layers (GPTQ-style propagation);
                     the audit replays the pipeline in `pipeline_order` up to each sampled unit

Rules every quantizer must follow (enforced by the audit):
* only the searchable Linear's own tensors may be produced -- never norms, neighbours or scales
  folded into other tensors (no SmoothQuant/AWQ-style migration);
* output must be deterministic for (base weights, version, params, calibration manifest);
* NVFP4 in the ModelOpt or compressed-tensors layout, FP8 as E4M3 with one BF16 scale per row.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..model.qwen38 import Linear, Unit
from ..safetensors_io import SafeTensorsDir

# A produced tensor: (suffix, dtype, shape, data) where data is bytes/memoryview/np.ndarray.
Produced = tuple[str, str, tuple[int, ...], object]


@dataclass
class QuantContext:
    """Everything a quantizer may read. Nothing else is available to it."""

    base: SafeTensorsDir                       # canonical BF16 weights
    sources: dict[str, SafeTensorsDir]         # verified attested sources by source id
    params: dict = field(default_factory=dict)  # per-quantizer parameters from the manifest
    calibration: Path | None = None            # calibration manifest (public calibration split)
    state: dict = field(default_factory=dict)  # scratch space for sequential quantizers


class Quantizer:
    name: str = ""
    version: int = 1
    formats: tuple[str, ...] = ()
    lineage: str = "regenerable"
    replay_mode: str = "independent"
    source_id: str | None = None               # attested quantizers: which pinned source
    description: str = ""

    @property
    def ref(self) -> str:
        """Identity string used in candidate ids: name@vN."""
        return f"{self.name}@v{self.version}"

    def supports(self, unit: Unit, fmt: str) -> bool:
        return fmt in self.formats

    def available(self, ctx: QuantContext, unit: Unit, fmt: str) -> str | None:
        """None if this quantizer can produce `fmt` for `unit` with this context, else a reason."""
        return None

    def begin(self, ctx: QuantContext) -> None:
        """Called once before the first encode() of a build or replay."""

    def encode(self, ctx: QuantContext, unit: Unit, lin: Linear, fmt: str) -> list[Produced]:
        raise NotImplementedError


def f32_rows(ctx: QuantContext, lin: Linear, chunk: int = 2048):
    """Yield (row_start, float32 rows) of the BF16 base weight in bounded-memory chunks."""
    from ..quant.formats import bf16_to_f32

    name = lin.prefix + ".weight"
    t = ctx.base.get(name)
    if t is None or t.dtype != "BF16" or t.shape != (lin.rows, lin.cols):
        raise ValueError(f"base checkpoint: expected BF16 {name} [{lin.rows}, {lin.cols}], got {t}")
    u = ctx.base.array(name, "<u2")
    for r in range(0, lin.rows, chunk):
        yield r, bf16_to_f32(u[r : r + chunk].reshape(-1)).reshape(-1, lin.cols)


def f32_weight(ctx: QuantContext, lin: Linear) -> np.ndarray:
    out = np.empty((lin.rows, lin.cols), np.float32)
    for r, rows in f32_rows(ctx, lin):
        out[r : r + len(rows)] = rows
    return out
