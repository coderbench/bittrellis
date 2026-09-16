"""Quality gates, Pareto frontier and Frontier Gain.

Objectives (HPC-01): minimize KL to BF16, maximize decode tok/s, minimize VRAM.

Frontier Gain (FG-1) is the increase in normalized dominated hypervolume a result adds to the
current frontier. Each objective is mapped to [0, 1] inside the track's fixed box (1 = best edge),
so FG-1 is a share of the box: 0 for a dominated result, and additive progress for everything
else. It measures effect, not effort; a result that is slower but much smaller or much more
accurate earns gain just like a faster one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FG_VERSION = "FG-1"


@dataclass
class Row:
    id: str
    name: str
    kind: str                      # "candidate" or "reference"
    kl: float
    decode_tps: float
    vram_gib: float
    top1: float | None = None
    prefill_tps: float | None = None
    needle_recall: float | None = None
    checkpoint_gib: float | None = None
    tasks: dict | None = None
    audit_ok: bool | None = None
    gate_failures: list[str] = field(default_factory=list)
    frontier: bool = False
    gain: float = 0.0

    @property
    def valid(self) -> bool:
        return not self.gate_failures

    def objectives(self) -> tuple[float, float, float]:
        return (self.kl, self.decode_tps, self.vram_gib)


def apply_gates(row: Row, gates: dict, reference_tasks: dict | None) -> list[str]:
    fails = []
    if row.kl > gates["kl_max"]:
        fails.append(f"KL {row.kl:.4f} > {gates['kl_max']}")
    if row.top1 is not None and row.top1 < gates["top1_min"]:
        fails.append(f"top-1 {row.top1:.3f} < {gates['top1_min']}")
    if row.needle_recall is not None and row.needle_recall < gates["needle_min"]:
        fails.append(f"needle recall {row.needle_recall:.2f} < {gates['needle_min']}")
    if row.audit_ok is False:
        fails.append("checkpoint audit failed")
    if row.tasks and reference_tasks:
        for suite, s in reference_tasks["suites"].items():
            got = row.tasks["suites"].get(suite)
            if got is None:
                fails.append(f"task suite {suite} missing")
            elif s["passed"] - got["passed"] > gates["task_max_drop_items"]:
                fails.append(f"{suite}: {got['passed']}/{got['n']} vs reference {s['passed']}/{s['n']}")
    row.gate_failures = fails
    return fails


def dominates(a: Row, b: Row) -> bool:
    """a dominates b: no worse on every objective, strictly better on one."""
    ge = a.kl <= b.kl and a.decode_tps >= b.decode_tps and a.vram_gib <= b.vram_gib
    gt = a.kl < b.kl or a.decode_tps > b.decode_tps or a.vram_gib < b.vram_gib
    return ge and gt


def pareto(rows: list[Row]) -> list[Row]:
    valid = [r for r in rows if r.valid]
    return [r for r in valid if not any(dominates(o, r) for o in valid if o is not r)]


def normalize(row: Row, box: dict) -> tuple[float, float, float]:
    def clip(x: float) -> float:
        return min(1.0, max(0.0, x))

    k0, k1 = box["kl"]
    d0, d1 = box["decode_tps"]
    v0, v1 = box["vram_gib"]
    return (clip((k1 - row.kl) / (k1 - k0)), clip((row.decode_tps - d0) / (d1 - d0)), clip((v1 - row.vram_gib) / (v1 - v0)))


def _hv2(points: list[tuple[float, float]]) -> float:
    pts = sorted(points, key=lambda p: -p[0])
    area, best_y = 0.0, 0.0
    for i, (x, y) in enumerate(pts):
        best_y = max(best_y, y)
        nxt = pts[i + 1][0] if i + 1 < len(pts) else 0.0
        area += (x - nxt) * best_y
    return area


def hypervolume(points: list[tuple[float, float, float]]) -> float:
    """Exact volume of the union of boxes [0, p] in [0,1]^3 (maximization)."""
    if not points:
        return 0.0
    pts = sorted(points, key=lambda p: -p[0])
    vol = 0.0
    for i, (x, _, _) in enumerate(pts):
        nxt = pts[i + 1][0] if i + 1 < len(pts) else 0.0
        if x > nxt:
            vol += (x - nxt) * _hv2([(p[1], p[2]) for p in pts[: i + 1]])
    return vol


def frontier_gain(row: Row, incumbents: list[Row], box: dict) -> float:
    """FG-1 of `row` against a set of already-accepted rows. Invalid rows gain nothing."""
    if not row.valid:
        return 0.0
    base = [normalize(r, box) for r in incumbents if r.valid]
    return max(0.0, hypervolume(base + [normalize(row, box)]) - hypervolume(base))


def rank(rows: list[Row], box: dict) -> list[Row]:
    """Mark frontier membership and each row's marginal gain over all *other* valid rows."""
    front = {id(r) for r in pareto(rows)}
    for r in rows:
        r.frontier = id(r) in front
        r.gain = frontier_gain(r, [o for o in rows if o is not r], box) if r.frontier else 0.0
    return rows
