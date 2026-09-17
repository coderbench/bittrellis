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
    dominated_by: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.gate_failures


QualityCmp = Callable[[Row, Row], int]  # -1: a materially better, +1: a materially worse, 0: not distinguishable


def mcnemar_loss_p(lost: int, gained: int) -> float:
    """One-sided exact McNemar: probability of at least `lost` losses among the discordant items if
    the candidate were no worse than the incumbent."""
    from math import comb

    n = lost + gained
    return sum(comb(n, k) for k in range(lost, n + 1)) / 2**n if n else 1.0


def task_guard(cand: dict, inc: dict, cfg: dict) -> list[str]:
    """Paired, per-question comparison with the incumbent, overall and per suite.

    Fails when the candidate loses significantly more questions than it gains (exact McNemar), or keeps
    fewer than `min_suite_retention` of the questions the incumbent passes in any suite. A suite of five
    questions can no longer lose everything and pass.
    """
    if "items" not in inc:
        return ["task guard: the incumbent has no per-question results"]
    if "items" not in cand:
        return ["task guard: per-question results missing"]
    fails, lost_all, gained_all = [], 0, 0
    for suite, inc_items in sorted(inc["items"].items()):
        got = cand["items"].get(suite)
        if not got or set(got) != set(inc_items):
            fails.append(f"task guard {suite}: not the same questions as the incumbent")
            continue
        lost = sum(1 for q, v in inc_items.items() if v and not got[q])
        gained = sum(1 for q, v in inc_items.items() if not v and got[q])
        kept = sum(1 for q, v in inc_items.items() if v and got[q])
        lost_all, gained_all = lost_all + lost, gained_all + gained
        passed = sum(inc_items.values())
        p = mcnemar_loss_p(lost, gained)
        if p < cfg["suite_alpha"]:
            fails.append(f"task guard {suite}: lost {lost}, gained {gained} vs incumbent (p={p:.4f} < {cfg['suite_alpha']})")
        elif passed and kept / passed < cfg["min_suite_retention"]:
            fails.append(f"task guard {suite}: keeps {kept}/{passed} of the incumbent's passes (< {cfg['min_suite_retention']:.0%})")
    p = mcnemar_loss_p(lost_all, gained_all)
    if p < cfg["overall_alpha"]:
        fails.append(f"task guard overall: lost {lost_all}, gained {gained_all} vs incumbent (p={p:.4f} < {cfg['overall_alpha']})")
    return fails


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
        fails += task_guard(row.tasks, incumbent_tasks, gates["task_guard"])
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


def handicap(row: Row, floors: dict) -> Row:
    """`row` made worse by the noise margin on every objective: only a gain larger than noise survives."""
    from dataclasses import replace

    return replace(
        row,
        rp_kl=row.rp_kl + floors["rp_kl"],
        decode_tps=row.decode_tps * (1.0 - max(floors["decode_tps"], row.decode_spread)),
        prefill_tps=row.prefill_tps * (1.0 - max(floors["prefill_tps"], row.prefill_spread)),
        peak_gpu_gib=row.peak_gpu_gib + floors["peak_gpu_gib"],
    )


def distinct(row: Row, others: list[Row], floors: dict, quality_cmp: QualityCmp) -> bool:
    """Materially better than every other valid internal row on at least one objective (same test as dominance)."""
    for o in others:
        signs = [quality_cmp(row, o)] + [_cmp_perf(row, o, k, floors[k]) for k, _ in OBJECTIVES[1:]]
        if -1 not in signs:
            return False
    return True


def frontier_gain(row: Row, incumbents: list[Row], box: dict, floors: dict | None = None) -> float:
    """FG-2 of `row` against valid internal incumbents, with `row` handicapped by the noise margins.

    External or invalid rows gain nothing. Without the handicap, a result identical to V0 except for a
    0.01 tok/s decode difference would add a sliver of hypervolume and earn a tier.
    """
    if row.kind != "internal" or not row.valid:
        return 0.0
    base = [normalize(r, box) for r in incumbents if r.kind == "internal" and r.valid]
    point = normalize(handicap(row, floors) if floors else row, box)
    return max(0.0, hypervolume(base + [point]) - hypervolume(base))


def rank(rows: list[Row], box: dict, floors: dict, quality_cmp: QualityCmp) -> list[Row]:
    front = {id(r) for r in pareto(rows, floors, quality_cmp)}
    pool = [r for r in rows if r.kind == "internal" and r.valid]
    for r in rows:
        r.frontier = id(r) in front
        others = [o for o in pool if o is not r]
        credited = r.frontier and distinct(r, others, floors, quality_cmp)
        r.gain = frontier_gain(r, [o for o in rows if o is not r], box, floors) if credited else 0.0
        if r.kind == "internal" and r.valid and not r.frontier:
            r.dominated_by = [o.name for o in others if dominates(o, r, floors, quality_cmp)]
    return rows
