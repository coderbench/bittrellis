"""Quality gates, noise-aware Pareto frontier and Frontier Gain.

Objectives (HPC-01): minimize KL to BF16, maximize decode tok/s, maximize prefill tok/s,
minimize peak VRAM.

Dominance is *epsilon*-dominance: a result only counts as better on an objective when it beats
the other by more than that objective's measurement noise (track `frontier.epsilon`). Without
it, a 0.3 tok/s decode wobble or a 0.0003 nats KL difference would let a result "dominate"
another by luck.

Frontier Gain (FG-2) is the increase in normalized dominated hypervolume a result adds to the
current frontier. Each objective is mapped to [0, 1] inside the track's fixed box (1 = best
edge), so FG-2 is a share of the 4-D box: 0 for a dominated or invalid result, and progress for
anything that opens new operating room on any axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FG_VERSION = "FG-2"
OBJECTIVES = (("kl", "min"), ("decode_tps", "max"), ("prefill_tps", "max"), ("vram_gib", "min"))


@dataclass
class Row:
    id: str
    name: str
    kind: str                      # "candidate" or "reference"
    kl: float
    decode_tps: float
    vram_gib: float
    prefill_tps: float
    top1: float | None = None
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


def _better(a: float, b: float, sense: str, eps: float) -> bool:
    return a < b - eps if sense == "min" else a > b + eps


def _worse(a: float, b: float, sense: str, eps: float) -> bool:
    return _better(b, a, sense, eps)


def dominates(a: Row, b: Row, epsilon: dict | None = None) -> bool:
    """a dominates b: not worse beyond noise on any objective, better beyond noise on one."""
    eps = epsilon or {}
    worse = any(_worse(getattr(a, k), getattr(b, k), s, eps.get(k, 0.0)) for k, s in OBJECTIVES)
    better = any(_better(getattr(a, k), getattr(b, k), s, eps.get(k, 0.0)) for k, s in OBJECTIVES)
    return better and not worse


def pareto(rows: list[Row], epsilon: dict | None = None) -> list[Row]:
    valid = [r for r in rows if r.valid]
    return [r for r in valid if not any(dominates(o, r, epsilon) for o in valid if o is not r)]


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
    """FG-2 of `row` against already-accepted rows. Invalid rows gain nothing."""
    if not row.valid:
        return 0.0
    base = [normalize(r, box) for r in incumbents if r.valid]
    return max(0.0, hypervolume(base + [normalize(row, box)]) - hypervolume(base))


def rank(rows: list[Row], box: dict, epsilon: dict | None = None) -> list[Row]:
    """Mark frontier membership and each frontier row's marginal gain over all other valid rows."""
    front = {id(r) for r in pareto(rows, epsilon)}
    for r in rows:
        r.frontier = id(r) in front
        r.gain = frontier_gain(r, [o for o in rows if o is not r], box) if r.frontier else 0.0
    return rows
