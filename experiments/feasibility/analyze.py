"""Answer the feasibility questions A-G from seed artifacts; writes results/feasibility/analysis.json.

Usage: python experiments/feasibility/analyze.py [results/feasibility/artifacts] [results/feasibility]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from bittrellis.eval.logits import block_bootstrap_ci, paired_delta
from bittrellis.frontier.pareto import dominates
from bittrellis.frontier.report import load_rows, write_report
from bittrellis.track import REPO_ROOT, load_track

INCUMBENT = "V0-baseline-rebuild"


def _npz(art: Path) -> dict:
    return dict(np.load(art / "kl_positions.npz"))


def main(art_dir: Path, out_dir: Path) -> dict:
    track = load_track("HPC-01")
    arts = {d.name: d for d in sorted(art_dir.iterdir()) if (d / "quality.json").exists()}
    inc = arts[INCUMBENT]
    inc_kl = _npz(inc)
    inc_perf = json.loads((inc / "performance.json").read_text())
    res: dict = {"incumbent": INCUMBENT, "vs_incumbent": {}}
    for name, d in arts.items():
        if name == INCUMBENT:
            continue
        perf = json.loads((d / "performance.json").read_text())
        res["vs_incumbent"][name] = {
            "rp_kl": paired_delta(inc_kl, _npz(d)),
            "decode_rel": (perf["decode_tps"] - inc_perf["decode_tps"]) / inc_perf["decode_tps"],
            "prefill_rel": (perf["prefill_tps"] - inc_perf["prefill_tps"]) / inc_perf["prefill_tps"],
            "peak_gpu_delta_gib": perf["peak_gpu_gib"] - inc_perf["peak_gpu_gib"],
        }

    rep = inc / "reproducibility.json"
    ident = art_dir / "V0-tensor-identity.json"
    res["A_reproducible"] = bool(rep.exists() and all(v["identical"] for v in json.loads(rep.read_text()).values())
                                 and ident.exists() and json.loads(ident.read_text())["identical"])
    variants = {k: v for k, v in res["vs_incumbent"].items() if k.startswith("V")}
    res["B_quality_moves"] = [k for k, v in variants.items() if v["rp_kl"]["significant"]]
    res["C_speed_or_memory_moves"] = [k for k, v in variants.items()
                                      if abs(v["decode_rel"]) > 0.02 or abs(v["prefill_rel"]) > 0.03 or abs(v["peak_gpu_delta_gib"]) > 0.1]

    rows, cmp = load_rows([art_dir], track)
    floors = track["frontier"]["epsilon_floor"]
    inc_row = next(r for r in rows if r.name == INCUMBENT)
    internal = [r for r in rows if r.kind == "internal" and r.name != INCUMBENT]
    res["D_not_dominated_by_incumbent"] = [r.name for r in internal if r.valid and not dominates(inc_row, r, floors, cmp)]
    res["D_dominating_incumbent"] = [r.name for r in internal if r.valid and dominates(r, inc_row, floors, cmp)]

    names = ("V4-gdn-q4k", "V6-mlp-q4k", "V9-gdn-q4k-mlp-q4k")
    if all(n in arts for n in names):
        k4, k6, k9 = (_npz(arts[n]) for n in names)
        streams = sorted(k[:-3] for k in inc_kl if k.endswith(".kl"))
        terms = [k9[f"{s}.kl"].astype(np.float64) - k4[f"{s}.kl"] - k6[f"{s}.kl"] + inc_kl[f"{s}.kl"] for s in streams]
        lo, hi = block_bootstrap_ci(terms, n_boot=2000)
        res["F_interaction"] = {"interaction": float(np.concatenate(terms).mean()), "ci95": [lo, hi],
                                "significant": bool(lo > 0 or hi < 0),
                                "by_stream": {s: float(t.mean()) for s, t in zip(streams, terms, strict=True)}}
    res["frontier"] = [r.name for r in rows if r.frontier]
    res["rows"] = {r.name: {"kind": r.kind, "valid": r.valid, "frontier": r.frontier, "gain": r.gain,
                            "gates": r.gate_failures} for r in rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report([art_dir], track, out_dir)
    (out_dir / "analysis.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    a = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "results/feasibility/artifacts"
    o = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO_ROOT / "results/feasibility"
    print(json.dumps(main(a, o), indent=2))
