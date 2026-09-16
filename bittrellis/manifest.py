"""Precision manifests: the artifact miners submit.

A manifest is short rules plus optional per-unit overrides; it *expands* to one precision (and
the quantizer that produces its bytes) per searchable unit. The expanded map, not the rules, is
what gets hashed, built and scored, so two manifests that expand identically are the same
candidate.

    schema: bittrellis/precision-manifest@1
    track: HPC-01
    name: gdn-fp8-late
    default: NVFP4
    quantizers:                  # default quantizer per precision (see QUANTIZERS)
      NVFP4: baseline
    rules:                       # applied in order; later rules override earlier ones
      - match: "L*.gdn.*"
        layers: "40-63"
        precision: FP8
      - match: "L*.mlp"
        layers: "0-55"
        precision: NVFP4
        quantizer: unsloth       # optional per-rule quantizer
    modules:                     # explicit per-unit overrides, applied last
      lm_head: Q4_K
      L3.attn.o: {precision: NVFP4, quantizer: rtn}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .model.qwen38 import Unit, select_units
from .precision import PRECISIONS, SPACE

SCHEMA = "bittrellis/precision-manifest@1"
# How the stored bytes of each precision can be produced (implemented in bittrellis/build.py).
QUANTIZERS: dict[str, tuple[str, ...]] = {
    "NVFP4": ("baseline", "rtn", "unsloth"),  # shipped R0 bytes · round-to-nearest · pinned R1 (GPTQ) bytes
    "FP8": ("rtn",),
    "Q4_K": ("runtime",),                     # stored BF16; SparkInfer fits Q4_K at load
}
DEFAULT_QUANTIZERS = {"NVFP4": "baseline", "FP8": "rtn", "Q4_K": "runtime"}


class ManifestError(ValueError):
    pass


@dataclass
class Manifest:
    track: str
    name: str
    default: str
    rules: list[dict] = field(default_factory=list)
    modules: dict[str, str | dict] = field(default_factory=dict)
    quantizers: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_QUANTIZERS))
    description: str = ""
    authors: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> Manifest:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    @classmethod
    def from_dict(cls, d: dict) -> Manifest:
        if not isinstance(d, dict):
            raise ManifestError("manifest must be a mapping")
        if d.get("schema") != SCHEMA:
            raise ManifestError(f"schema must be {SCHEMA!r}")
        unknown = set(d) - {"schema", "track", "name", "default", "rules", "modules", "quantizers",
                            "description", "authors", "expanded"}
        if unknown:
            raise ManifestError(f"unknown manifest keys: {sorted(unknown)}")
        for key in ("track", "name", "default"):
            if not d.get(key):
                raise ManifestError(f"manifest needs {key!r}")
        q = dict(DEFAULT_QUANTIZERS)
        q.update(d.get("quantizers") or {})
        return cls(
            track=str(d["track"]), name=str(d["name"]), default=str(d["default"]),
            rules=list(d.get("rules") or []), modules=dict(d.get("modules") or {}),
            quantizers=q, description=str(d.get("description", "")), authors=list(d.get("authors") or []),
        )

    def expand_full(self, units: list[Unit]) -> dict[str, tuple[str, str]]:
        """Resolve rules into {unit_id: (precision, quantizer)}; raises on anything undeployable."""
        for p, qz in self.quantizers.items():
            if p not in QUANTIZERS or qz not in QUANTIZERS[p]:
                raise ManifestError(f"quantizers.{p}: {qz!r} is not one of {QUANTIZERS.get(p, ())}")
        by_id = {u.id: u for u in units}
        out: dict[str, tuple[str, str]] = {}
        for u in units:
            out[u.id] = self._check(u, self.default, None, "default")
        for i, rule in enumerate(self.rules):
            if not isinstance(rule, dict) or "match" not in rule or "precision" not in rule:
                raise ManifestError(f"rule {i} needs 'match' and 'precision'")
            extra = set(rule) - {"match", "layers", "precision", "quantizer", "note"}
            if extra:
                raise ManifestError(f"rule {i} has unknown keys {sorted(extra)}")
            hits = select_units(units, str(rule["match"]), rule.get("layers"))
            if not hits:
                raise ManifestError(f"rule {i} ({rule['match']!r}, layers={rule.get('layers')}) matches no unit")
            for u in hits:
                out[u.id] = self._check(u, rule["precision"], rule.get("quantizer"), f"rule {i}")
        for uid, spec in self.modules.items():
            if uid not in by_id:
                raise ManifestError(f"modules: unknown unit {uid!r}")
            if isinstance(spec, dict):
                if set(spec) - {"precision", "quantizer"} or "precision" not in spec:
                    raise ManifestError(f"modules.{uid}: use a precision string or {{precision, quantizer}}")
                out[uid] = self._check(by_id[uid], spec["precision"], spec.get("quantizer"), f"modules.{uid}")
            else:
                out[uid] = self._check(by_id[uid], spec, None, f"modules.{uid}")
        return out

    def expand(self, units: list[Unit]) -> dict[str, str]:
        """{unit_id: precision}."""
        return {k: p for k, (p, _) in self.expand_full(units).items()}

    def _check(self, unit: Unit, precision: str, quantizer: str | None, where: str) -> tuple[str, str]:
        if precision not in PRECISIONS or precision not in SPACE[unit.kind]:
            raise ManifestError(
                f"{where}: {unit.id} cannot run {precision!r} on the pinned runtime "
                f"(allowed for {unit.kind}: {', '.join(SPACE[unit.kind])})"
            )
        qz = quantizer or self.quantizers.get(precision, DEFAULT_QUANTIZERS[precision])
        if qz not in QUANTIZERS[precision]:
            raise ManifestError(f"{where}: quantizer {qz!r} cannot produce {precision} (use {QUANTIZERS[precision]})")
        return precision, qz

    def candidate_id(self, units: list[Unit]) -> str:
        return candidate_hash(self.track, self.expand_full(units))

    def to_dict(self, units: list[Unit] | None = None) -> dict:
        d = {
            "schema": SCHEMA, "track": self.track, "name": self.name, "description": self.description,
            "authors": self.authors, "default": self.default, "rules": self.rules, "modules": self.modules,
            "quantizers": self.quantizers,
        }
        if units is not None:
            d["expanded"] = {k: f"{p}@{q}" for k, (p, q) in self.expand_full(units).items()}
        return d


def candidate_hash(track: str, full: dict[str, tuple[str, str]]) -> str:
    """Content address of a candidate: track + every unit's precision and quantizer."""
    payload = {"track": track, "modules": {k: f"{p}@{q}" for k, (p, q) in sorted(full.items())}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def summarize(expanded: dict[str, str] | dict[str, tuple[str, str]], units: list[Unit]) -> dict[str, dict[str, int]]:
    """Counts of units per (kind, precision[@quantizer]) for reports and PR descriptions."""
    by_id = {u.id: u for u in units}
    out: dict[str, dict[str, int]] = {}
    for uid, v in expanded.items():
        if isinstance(v, str):
            label = v
        else:
            label = v[0] if v[1] == DEFAULT_QUANTIZERS[v[0]] else f"{v[0]}@{v[1]}"
        k = by_id[uid].kind
        out.setdefault(k, {})
        out[k][label] = out[k].get(label, 0) + 1
    return out
