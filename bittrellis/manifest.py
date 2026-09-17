"""Candidate manifests: the artifact miners submit.

A manifest is a default, ordered rules and per-unit overrides. It *expands* to one assignment per
searchable unit: the stored format, the quantizer that produces its bytes (name@version and
parameters), and the format SparkInfer executes. The expanded assignment, not the rule text, is
hashed, built and scored, so manifests that expand identically are the same candidate.

    schema: bittrellis/manifest@2
    track: HPC-01
    name: gdn-fp8-deep-calibrated-mlp
    default: NVFP4                       # format for every unit not matched below
    quantizers: {NVFP4: baseline}        # default quantizer per format (optional)
    rules:                               # applied in order; later rules win
      - match: "L*.gdn.*"
        layers: "32-63"
        format: FP8
      - match: "L*.mlp"
        layers: "0-55"
        format: NVFP4
        quantizer: unsloth
    modules:                             # exact unit overrides, applied last
      lm_head: Q4_K
      L3.attn.o: {format: NVFP4, quantizer: rtn}

Schema `bittrellis/precision-manifest@1` (key `precision` instead of `format`) is still read.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import quantizers as Q
from .model.qwen38 import Unit, select_units
from .precision import PRECISIONS, SPACE, execution

SCHEMA = "bittrellis/manifest@2"
SCHEMAS = (SCHEMA, "bittrellis/precision-manifest@1")
DEFAULT_QUANTIZERS = dict(Q.DEFAULT_FOR_FORMAT)


class ManifestError(ValueError):
    pass


@dataclass(frozen=True)
class Assignment:
    format: str                      # stored/selected format: NVFP4 | FP8 | Q4_K
    quantizer: str                   # registry name
    params: tuple = ()               # sorted (key, value) pairs

    @property
    def quantizer_ref(self) -> str:
        return Q.get(self.quantizer).ref

    def key(self) -> str:
        """Canonical identity string: FORMAT@name@vN[+params-hash]."""
        s = f"{self.format}@{self.quantizer_ref}"
        if self.params:
            s += "+" + hashlib.sha256(json.dumps(self.params, sort_keys=True).encode()).hexdigest()[:12]
        return s

    def label(self) -> str:
        return self.format if self.quantizer == DEFAULT_QUANTIZERS[self.format] and not self.params else f"{self.format}@{self.quantizer}"


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
        if d.get("schema") not in SCHEMAS:
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

    # ------------------------------------------------------------------ expansion

    def expand_assignments(self, units: list[Unit]) -> dict[str, Assignment]:
        for fmt, qz in self.quantizers.items():
            if fmt not in PRECISIONS:
                raise ManifestError(f"quantizers: unknown format {fmt!r}")
            self._quantizer(fmt, qz, "quantizers")
        by_id = {u.id: u for u in units}
        out = {u.id: self._check(u, self.default, None, None, "default") for u in units}
        for i, rule in enumerate(self.rules):
            if not isinstance(rule, dict) or "match" not in rule or not ({"format", "precision"} & set(rule)):
                raise ManifestError(f"rule {i} needs 'match' and 'format'")
            extra = set(rule) - {"match", "layers", "format", "precision", "quantizer", "params", "note"}
            if extra:
                raise ManifestError(f"rule {i} has unknown keys {sorted(extra)}")
            hits = select_units(units, str(rule["match"]), rule.get("layers"))
            if not hits:
                raise ManifestError(f"rule {i} ({rule['match']!r}, layers={rule.get('layers')}) matches no unit")
            fmt = rule.get("format", rule.get("precision"))
            for u in hits:
                out[u.id] = self._check(u, fmt, rule.get("quantizer"), rule.get("params"), f"rule {i}")
        for uid, spec in self.modules.items():
            if uid not in by_id:
                raise ManifestError(f"modules: unknown unit {uid!r}")
            if isinstance(spec, dict):
                if set(spec) - {"format", "precision", "quantizer", "params"} or not ({"format", "precision"} & set(spec)):
                    raise ManifestError(f"modules.{uid}: use a format string or {{format, quantizer, params}}")
                out[uid] = self._check(by_id[uid], spec.get("format", spec.get("precision")), spec.get("quantizer"),
                                       spec.get("params"), f"modules.{uid}")
            else:
                out[uid] = self._check(by_id[uid], spec, None, None, f"modules.{uid}")
        return out

    def expand(self, units: list[Unit]) -> dict[str, str]:
        """{unit_id: format}."""
        return {k: a.format for k, a in self.expand_assignments(units).items()}

    def _quantizer(self, fmt: str, name: str, where: str) -> Q.Quantizer:
        try:
            qz = Q.get(name)
        except KeyError as e:
            raise ManifestError(f"{where}: {e.args[0]}") from None
        if fmt not in qz.formats:
            raise ManifestError(f"{where}: quantizer {name!r} cannot produce {fmt} (it produces {qz.formats})")
        return qz

    def _check(self, unit: Unit, fmt: str, quantizer: str | None, params: dict | None, where: str) -> Assignment:
        if fmt not in PRECISIONS or fmt not in SPACE[unit.kind]:
            raise ManifestError(
                f"{where}: {unit.id} cannot run {fmt!r} on the pinned runtime "
                f"(allowed for {unit.kind}: {', '.join(SPACE[unit.kind])})"
            )
        name = quantizer or self.quantizers.get(fmt, DEFAULT_QUANTIZERS[fmt])
        qz = self._quantizer(fmt, name, where)
        if not qz.supports(unit, fmt):
            raise ManifestError(f"{where}: quantizer {name!r} does not support {unit.id}")
        if params is not None and not isinstance(params, dict):
            raise ManifestError(f"{where}: params must be a mapping")
        return Assignment(fmt, name, tuple(sorted((params or {}).items())))

    # ------------------------------------------------------------------ identity + serialization

    def candidate_id(self, units: list[Unit]) -> str:
        return candidate_hash(self.track, self.expand_assignments(units))

    def to_dict(self, units: list[Unit] | None = None) -> dict:
        d = {
            "schema": SCHEMA, "track": self.track, "name": self.name, "description": self.description,
            "authors": self.authors, "default": self.default, "rules": self.rules, "modules": self.modules,
            "quantizers": self.quantizers,
        }
        if units is not None:
            by_id = {u.id: u for u in units}
            d["expanded"] = {
                k: {"source_format": a.format, "quantizer": a.quantizer_ref, "params": dict(a.params),
                    "execution": execution(by_id[k].kind, a.format)}
                for k, a in self.expand_assignments(units).items()
            }
        return d


def candidate_hash(track: str, assignments: dict[str, Assignment]) -> str:
    """Content address of a candidate: track + every unit's FORMAT@quantizer@version[+params]."""
    return candidate_hash_keys(track, {k: a.key() for k, a in assignments.items()})


def candidate_hash_keys(track: str, keys: dict[str, str]) -> str:
    """`candidate_hash` from assignment keys alone, without looking any quantizer up."""
    payload = {"track": track, "units": dict(sorted(keys.items()))}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def summarize(assignments: dict[str, Assignment] | dict[str, str], units: list[Unit]) -> dict[str, dict[str, int]]:
    """Counts of units per kind and FORMAT[@quantizer] for reports and PR descriptions."""
    by_id = {u.id: u for u in units}
    out: dict[str, dict[str, int]] = {}
    for uid, a in assignments.items():
        label = a if isinstance(a, str) else a.label()
        kind = by_id[uid].kind
        out.setdefault(kind, {})
        out[kind][label] = out[kind].get(label, 0) + 1
    return out


def distance(a: dict[str, Assignment], b: dict[str, Assignment]) -> int:
    """Number of units whose assignment differs; 0 means the same candidate."""
    return sum(1 for k in a if a[k].key() != b.get(k, Assignment("", "runtime")).key()) if a else 0
