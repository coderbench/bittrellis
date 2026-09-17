"""Gates, noise-aware Pareto frontier and Frontier Gain (FG-2).

Objectives (HPC-01): minimize RP-KL, maximize decode tok/s, maximize 4K prefill tok/s, minimize
peak GPU memory.

Only *internal* rows -- legal BitTrellis manifests on the pinned runtime, starting with the V0
incumbent -- take part in dominance and Frontier Gain. External references are reported beside
the frontier and never move it.

"Materially better" on an objective:
* RP-KL: lower by more than the floor AND the paired block-bootstrap 95% interval of the per-position
  difference excludes zero;
* decode / prefill: higher by more than max(floor, either result's two-run relative spread);
* peak GPU memory: lower by more than the floor (GiB).

A dominates B if A is materially better on at least one objective and materially worse on none.
Pairwise ε-dominance is not guaranteed to be transitive; the frontier is the set of valid internal
rows that no other valid internal row dominates.

FG-2 of a row is the increase in normalized dominated hypervolume it adds to every other valid
internal row (1 = best edge of the track's box on each axis).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

FG_VERSION = "FG-2"
OBJECTIVES = (("rp_kl", "min"), ("decode_tps", "max"), ("prefill_tps", "max"), ("peak_gpu_gib", "min"))


@dataclass
class Row:
    id: str
    name: str
    kind: str                           # "internal" or "external"
    rp_kl: float
    decode_tps: float
    prefill_tps: float
    peak_gpu_gib: float
    decode_spread: float = 0.0
    prefill_spread: float = 0.0
    top1: float | None = None
    needles_by_length: dict | None = None
    correctness_ok: bool | None = None
    audit_ok: bool | None = None
    tasks: dict | None = None
    holdout: str | None = None          # "PASS", "FAIL" or None (not run)
    extra: dict = field(default_factory=dict)
    gate_failures: list[str] = field(default_factory=list)
    frontier: bool = False
    gain: float = 0.0

    @property
    def valid(self) -> bool:
        return not self.gate_failures


QualityCmp = Callable[[Row, Row], int]  # -1: a materially better, +1: a materially worse, 0: not distinguishable


def apply_gates(row: Row, gates: dict, incumbent_tasks: dict | None) -> list[str]:
    fails = []
    if row.rp_kl > gates["rp_kl_max"]:
        fails.append(f"RP-KL {row.rp_kl:.4f} > {gates['rp_kl_max']}")
    if row.top1 is not None and row.top1 < gates["top1_min"]:
        fails.append(f"top-1 {row.top1:.3f} < {gates['top1_min']}")
    guard = gates["long_context_guard"]["required_success"]
    for stream, need in guard.items():
        got = (row.needles_by_length or {}).get(stream)
        if got is None:
            fails.append(f"long-context guard: {stream} not measured")
        elif got["required"] and got["retrieved"] / got["required"] < need:
            fails.append(f"long-context guard: {stream} {got['retrieved']}/{got['required']} needles")
    if row.kind == "internal":
        if row.audit_ok is not True:
            fails.append("checkpoint audit failed or missing")
        if row.correctness_ok is False:
            fails.append("runtime correctness failed")
        if row.holdout == "FAIL":
            fails.append("private holdout FAIL")
    if row.tasks and incumbent_tasks:
        for suite, s in incumbent_tasks["suites"].items():
            got = row.tasks["suites"].get(suite)
            if got is None:
                fails.append(f"task suite {suite} missing")
            elif s["passed"] - got["passed"] > gates["task_max_drop_items"]:
                fails.append(f"task guard {suite}: {got['passed']}/{got['n']} vs incumbent {s['passed']}/{s['n']}")
    row.gate_failures = fails
    return fails


def _cmp_perf(a: Row, b: Row, key: str, floor: float) -> int:
    va, vb = getattr(a, key), getattr(b, key)
    if key in ("decode_tps", "prefill_tps"):
        thr = max(floor, getattr(a, key.replace("_tps", "_spread")), getattr(b, key.replace("_tps", "_spread")))
        rel = (va - vb) / vb if vb else 0.0
        return -1 if rel > thr else (1 if rel < -thr else 0)
    diff = va - vb  # peak_gpu_gib, lower is better
    return -1 if diff < -floor else (1 if diff > floor else 0)


def dominates(a: Row, b: Row, floors: dict, quality_cmp: QualityCmp) -> bool:
    signs = [quality_cmp(a, b)] + [_cmp_perf(a, b, k, floors[k]) for k, _ in OBJECTIVES[1:]]
    return -1 in signs and 1 not in signs


def pareto(rows: list[Row], floors: dict, quality_cmp: QualityCmp) -> list[Row]:
    pool = [r for r in rows if r.kind == "internal" and r.valid]
    return [r for r in pool if not any(dominates(o, r, floors, quality_cmp) for o in pool if o is not r)]


def normalize(row: Row, box: dict) -> tuple[float, ...]:
    out = []
    for k, sense in OBJECTIVES:
        lo, hi = box[k]
        x = (getattr(row, k) - lo) / (hi - lo)
        out.append(min(1.0, max(0.0, 1.0 - x if sense == "min" else x)))
    return tuple(out)


def hypervolume(points: list[tuple[float, ...]]) -> float:
    """Exact volume of the union of boxes [0, p] in [0,1]^d (maximization), by slicing."""
    if not points:
        return 0.0
    if len(points[0]) == 1:
        return max(p[0] for p in points)
    pts = sorted(points, key=lambda p: -p[0])
    vol = 0.0
    for i, p in enumerate(pts):
        nxt = pts[i + 1][0] if i + 1 < len(pts) else 0.0
        if p[0] > nxt:
            vol += (p[0] - nxt) * hypervolume([q[1:] for q in pts[: i + 1]])
    return vol


def frontier_gain(row: Row, incumbents: list[Row], box: dict) -> float:
    """FG-2 of `row` against valid internal incumbents. External or invalid rows gain nothing."""
    if row.kind != "internal" or not row.valid:
        return 0.0
    base = [normalize(r, box) for r in incumbents if r.kind == "internal" and r.valid]
    return max(0.0, hypervolume(base + [normalize(row, box)]) - hypervolume(base))


def rank(rows: list[Row], box: dict, floors: dict, quality_cmp: QualityCmp) -> list[Row]:
    front = {id(r) for r in pareto(rows, floors, quality_cmp)}
    for r in rows:
        r.frontier = id(r) in front
        r.gain = frontier_gain(r, [o for o in rows if o is not r], box) if r.frontier else 0.0
    return rows
