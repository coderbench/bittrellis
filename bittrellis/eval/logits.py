"""Distribution-level quality: how far a candidate's next-token distribution is from BF16.

For each scored position the reference stores the BF16 top-64 log-probabilities and the target
log-probability; the candidate dump stores its own top-128 and target log-probability. KL is the
coarse-grained KL(P_bf16 || Q_candidate) over the reference top-64 tokens plus one "everything
else" bucket. Coarse-graining can only lower KL, so it is a conservative (never inflated)
estimate; a candidate token outside its own top-128 is assigned the smallest listed probability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..runtime import ScoreDump

EPS = 1e-12


@dataclass
class StreamQuality:
    stream: str
    category: str
    n: int
    kl: np.ndarray          # per position
    top1: np.ndarray        # bool per position
    nll_delta: np.ndarray   # candidate NLL - reference NLL, per position
    needles: dict[str, dict]


def _candidate_lp(ref_ids: np.ndarray, cand_ids: np.ndarray, cand_lp: np.ndarray) -> np.ndarray:
    """Candidate log-prob for every reference top-k id (row-wise lookup, floor for unlisted ids)."""
    n, kr = ref_ids.shape
    order = np.argsort(cand_ids, axis=1)
    sid = np.take_along_axis(cand_ids, order, axis=1)
    slp = np.take_along_axis(cand_lp, order, axis=1)
    floor = cand_lp.min(axis=1)
    out = np.empty((n, kr), np.float64)
    kc = sid.shape[1]
    for r in range(n):
        j = np.searchsorted(sid[r], ref_ids[r])
        j = np.minimum(j, kc - 1)
        hit = sid[r, j] == ref_ids[r]
        out[r] = np.where(hit, slp[r, j], floor[r])
    return out


def compare(ref: ScoreDump, cand: ScoreDump, stream: dict) -> StreamQuality:
    """Align candidate positions to the reference and compute per-position metrics."""
    common, ri, ci = np.intersect1d(ref.pos, cand.pos, return_indices=True)
    if len(common) != len(ref.pos):
        raise ValueError(f"{stream['id']}: candidate scored {len(common)}/{len(ref.pos)} reference positions")
    r_ids, r_lp = ref.top_ids[ri], ref.top_lp[ri].astype(np.float64)
    c_ids, c_lp = cand.top_ids[ci], cand.top_lp[ci].astype(np.float64)
    q_lp = _candidate_lp(r_ids, c_ids, c_lp)
    p = np.exp(r_lp)
    q = np.exp(q_lp)
    p_tail = np.clip(1.0 - p.sum(axis=1), EPS, 1.0)
    q_tail = np.clip(1.0 - q.sum(axis=1), EPS, 1.0)
    kl = (p * (r_lp - q_lp)).sum(axis=1) + p_tail * (np.log(p_tail) - np.log(q_tail))
    kl = np.maximum(kl, 0.0)
    top1 = r_ids[:, 0] == cand.argmax[ci]
    nll_delta = (-cand.lp_target[ci].astype(np.float64)) - (-ref.lp_target[ri].astype(np.float64))

    needles = {}
    ids = stream["ids"]
    pos_to_c = {int(pp): k for k, pp in enumerate(cand.pos)}
    pos_to_r = {int(pp): k for k, pp in enumerate(ref.pos)}
    for nd in stream.get("needles", []):
        preds = [p_ - 1 for p_ in nd["target_positions"]]  # position i predicts token i+1
        want = [ids[p_] for p_ in nd["target_positions"]]
        cand_ok = all(cand.argmax[pos_to_c[i]] == w for i, w in zip(preds, want, strict=True))
        ref_ok = all(ref.top_ids[pos_to_r[i], 0] == w for i, w in zip(preds, want, strict=True))
        needles[nd["name"]] = {"depth": nd["depth"], "candidate": bool(cand_ok), "reference": bool(ref_ok)}
    return StreamQuality(stream["id"], stream["category"], len(common), kl, top1, nll_delta, needles)


def block_bootstrap_ci(values: list[np.ndarray], block: int = 128, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of the pooled mean, resampling contiguous blocks (positions are autocorrelated)."""
    blocks = [v[i : i + block] for v in values for i in range(0, len(v), block)]
    sums = np.array([b.sum() for b in blocks])
    counts = np.array([len(b) for b in blocks])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(blocks), size=(n_boot, len(blocks)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def save_positions(results: list[StreamQuality], path) -> None:
    """Per-position KL/top-1 per stream, so any two artifacts can be compared *paired*."""
    arrays = {}
    for r in results:
        arrays[f"{r.stream}.kl"] = r.kl.astype(np.float32)
        arrays[f"{r.stream}.top1"] = r.top1
    np.savez_compressed(path, **arrays)


def paired_delta(a: dict[str, np.ndarray], b: dict[str, np.ndarray], block: int = 128, n_boot: int = 2000,
                 seed: int = 0) -> dict:
    """Mean KL(b) - KL(a) over identical positions with a paired block-bootstrap 95% interval.

    Both artifacts score the same tokens, so position-level noise that is common to every
    precision map (chaotic or high-entropy positions) cancels in the difference.
    """
    streams = sorted(k[:-3] for k in a if k.endswith(".kl") and k in b)
    diffs = [b[f"{s}.kl"].astype(np.float64) - a[f"{s}.kl"].astype(np.float64) for s in streams]
    lo, hi = block_bootstrap_ci(diffs, block=block, n_boot=n_boot, seed=seed)
    d = np.concatenate(diffs)
    return {"delta_kl": float(d.mean()), "ci95": [lo, hi], "positions": int(len(d)),
            "significant": bool(lo > 0 or hi < 0)}


def summarize(results: list[StreamQuality]) -> dict:
    kl_all = np.concatenate([r.kl for r in results])
    top1_all = np.concatenate([r.top1 for r in results])
    nll_all = np.concatenate([r.nll_delta for r in results])
    lo, hi = block_bootstrap_ci([r.kl for r in results])
    by_cat: dict[str, dict] = {}
    for cat in sorted({r.category for r in results}):
        rs = [r for r in results if r.category == cat]
        by_cat[cat] = {
            "positions": int(sum(r.n for r in rs)),
            "kl": float(np.concatenate([r.kl for r in rs]).mean()),
            "top1": float(np.concatenate([r.top1 for r in rs]).mean()),
            "nll_delta": float(np.concatenate([r.nll_delta for r in rs]).mean()),
        }
    needles = {r.stream: r.needles for r in results if r.needles}
    counted = [n for s in needles.values() for n in s.values() if n["reference"]]
    return {
        "positions": int(len(kl_all)),
        "kl": float(kl_all.mean()),
        "kl_ci95": [lo, hi],
        "kl_p99": float(np.percentile(kl_all, 99)),
        "top1": float(top1_all.mean()),
        "nll_delta": float(nll_all.mean()),
        "by_category": by_cat,
        "by_stream": {r.stream: {"kl": float(r.kl.mean()), "top1": float(r.top1.mean())} for r in results},
        "needles": needles,
        "needle_recall": (sum(n["candidate"] for n in counted) / len(counted)) if counted else None,
    }
