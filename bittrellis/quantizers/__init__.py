"""Quantizer registry.

To add a quantizer: implement `Quantizer` (see base.py for the contract) in a new module here and
add an instance to `_BUILTIN`. Its `ref` (name@vN) becomes part of every candidate id that uses it.
Attested quantizers additionally need a maintainer-approved entry in configs/sources.lock.json.
"""

from __future__ import annotations

from .base import Produced, QuantContext, Quantizer
from .builtin import RTN, Baseline, Runtime, Unsloth

_BUILTIN: list[Quantizer] = [Runtime(), Baseline(), Unsloth(), RTN()]
REGISTRY: dict[str, Quantizer] = {q.name: q for q in _BUILTIN}
DEFAULT_FOR_FORMAT = {"NVFP4": "baseline", "FP8": "rtn", "Q4_K": "runtime"}


def get(name: str) -> Quantizer:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown quantizer {name!r} (known: {sorted(REGISTRY)})") from None


def register(q: Quantizer) -> Quantizer:
    """Register an additional quantizer (tests and experimental plugins)."""
    if q.name in REGISTRY and REGISTRY[q.name] is not q:
        raise ValueError(f"quantizer {q.name!r} already registered")
    if q.lineage not in ("runtime", "regenerable", "attested"):
        raise ValueError(f"{q.name}: bad lineage {q.lineage!r}")
    if q.lineage == "regenerable" and q.replay_mode not in ("independent", "sequential"):
        raise ValueError(f"{q.name}: regenerable quantizers need replay_mode independent|sequential")
    REGISTRY[q.name] = q
    return q


__all__ = ["REGISTRY", "DEFAULT_FOR_FORMAT", "Produced", "QuantContext", "Quantizer", "get", "register"]
