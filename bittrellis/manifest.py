"""Precision manifests: the artifact miners submit.

A manifest is short rules plus optional per-unit overrides; it *expands* to one precision per
searchable unit. The expanded map, not the rules, is what gets hashed, built and scored, so two
manifests that expand identically are the same candidate.

    schema: bittrellis/precision-manifest@1
    track: HPC-01
    name: gdn-fp8-late
    default: NVFP4
    rules:                       # applied in order; later rules override earlier ones
      - match: "L*.gdn.*"
        layers: "40-63"
        precision: FP8
    modules:                     # explicit per-unit overrides, applied last
      lm_head: Q4_K
    quantizers:                  # how stored bytes are produced (see bittrellis/build.py)
      NVFP4: baseline
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
NVFP4_QUANTIZERS = ("baseline", "rtn")
FP8_QUANTIZERS = ("rtn",)
DEFAULT_QUANTIZERS = {"NVFP4": "baseline", "FP8": "rtn"}


class ManifestError(ValueError):
    pass


@dataclass
class Manifest:
    track: str
    name: str
    default: str
    rules: list[dict] = field(default_factory=list)
    modules: dict[str, str] = field(default_factory=dict)
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

    def expand(self, units: list[Unit]) -> dict[str, str]:
        """Resolve rules into {unit_id: precision}; raises on anything the runtime cannot run."""
        if self.default not in PRECISIONS:
            raise ManifestError(f"default precision {self.default!r} is not one of {PRECISIONS}")
        by_id = {u.id: u for u in units}
        out: dict[str, str] = {}
        for u in units:
            out[u.id] = self._check(u, self.default, "default")
        for i, rule in enumerate(self.rules):
            if not isinstance(rule, dict) or "match" not in rule or "precision" not in rule:
                raise ManifestError(f"rule {i} needs 'match' and 'precision'")
            extra = set(rule) - {"match", "layers", "precision", "note"}
            if extra:
                raise ManifestError(f"rule {i} has unknown keys {sorted(extra)}")
            hits = select_units(units, str(rule["match"]), rule.get("layers"))
            if not hits:
                raise ManifestError(f"rule {i} ({rule['match']!r}, layers={rule.get('layers')}) matches no unit")
            for u in hits:
                out[u.id] = self._check(u, rule["precision"], f"rule {i}")
        for uid, p in self.modules.items():
            if uid not in by_id:
                raise ManifestError(f"modules: unknown unit {uid!r}")
            out[uid] = self._check(by_id[uid], p, "modules")
        if self.quantizers.get("NVFP4") not in NVFP4_QUANTIZERS:
            raise ManifestError(f"quantizers.NVFP4 must be one of {NVFP4_QUANTIZERS}")
        if self.quantizers.get("FP8") not in FP8_QUANTIZERS:
            raise ManifestError(f"quantizers.FP8 must be one of {FP8_QUANTIZERS}")
        return out

    @staticmethod
    def _check(unit: Unit, precision: str, where: str) -> str:
        if precision not in SPACE[unit.kind]:
            raise ManifestError(
                f"{where}: {unit.id} cannot run {precision!r} on the pinned runtime "
                f"(allowed for {unit.kind}: {', '.join(SPACE[unit.kind])})"
            )
        return precision

    def candidate_id(self, units: list[Unit]) -> str:
        return candidate_hash(self.track, self.expand(units), self.quantizers)

    def to_dict(self, units: list[Unit] | None = None) -> dict:
        d = {
            "schema": SCHEMA, "track": self.track, "name": self.name, "description": self.description,
            "authors": self.authors, "default": self.default, "rules": self.rules, "modules": self.modules,
            "quantizers": self.quantizers,
        }
        if units is not None:
            d["expanded"] = self.expand(units)
        return d


def candidate_hash(track: str, expanded: dict[str, str], quantizers: dict[str, str]) -> str:
    """Content address of a candidate: track + every unit's precision + the quantizers used."""
    used = sorted(set(expanded.values()))
    payload = {"track": track, "modules": dict(sorted(expanded.items())),
               "quantizers": {p: quantizers[p] for p in used if p in quantizers}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def summarize(expanded: dict[str, str], units: list[Unit]) -> dict[str, dict[str, int]]:
    """Counts of units per (kind, precision) -- handy for reports and PR descriptions."""
    by_id = {u.id: u for u in units}
    out: dict[str, dict[str, int]] = {}
    for uid, p in expanded.items():
        k = by_id[uid].kind
        out.setdefault(k, {})
        out[k][p] = out[k].get(p, 0) + 1
    return out
