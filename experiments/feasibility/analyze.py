"""Answer feasibility questions A–G from artifacts and write results/feasibility/analysis.json.

Usage: python experiments/feasibility/analyze.py [artifacts/feasibility] [results/feasibility]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from bittrellis.eval.logits import paired_delta
from bittrellis.frontier.pareto import dominates
from bittrellis.frontier.report import feasibility_report, load_rows
from bittrellis.track import REPO_ROOT, load_track

GDN_PARAMS = 48 * (10240 * 5120 + 6144 * 5120 + 5120 * 6144)
ATTN_PARAMS = 16 * (12288 * 5120 + 1024 * 5120 * 2 + 5120 * 6144)
MLP_PARAMS = 64 * 3 * 17408 * 5120


def _npz(art: Path) -> dict:
    p = art / "kl_positions.npz"
    return dict(np.load(p)) if p.exists() else {}


def _perf(art: Path) -> dict | None:
    p = art / "performance.json"
    return json.loads(p.read_text()) if p.exists() else None


def _speed_delta(a: dict, b: dict, key: str, ctx: str = "4096") -> dict:
    """Relative change b vs a at one context; 'clear' only if > 2% (reps are lower medians of 2)."""
    va = a["contexts"][ctx][key] if "contexts" in a else a[key]
    vb = b["contexts"][ctx][key] if "contexts" in b else b[key]
    rel = (vb - va) / va
    return {"a": va, "b": vb, "rel": rel, "clear": abs(rel) > 0.02}


def main(art_dir: Path, out_dir: Path) -> dict:
    track = load_track("HPC-01")
    arts = {d.name: d for d in sorted(art_dir.iterdir()) if (d / "candidate.json").exists()}
    r0 = arts["R0"]
    r0_kl = _npz(r0)
    r0_perf = _perf(r0)
    res: dict = {"vs_R0": {}}
    for name, d in arts.items():
        if name == "R0" or not (d / "kl_positions.npz").exists():
            continue
        entry = {"kl": paired_delta(r0_kl, _npz(d))}
        perf = _perf(d)
        if perf and r0_perf:
            key_prefill = "prefill_pp" if "contexts" in perf else "prefill_tps"
            entry["decode"] = _speed_delta(r0_perf, perf, "decode_tps")
            entry["prefill"] = _speed_delta(r0_perf, perf, key_prefill) if "contexts" in perf else None
            entry["vram_gib"] = {"a": r0_perf["vram_gib"], "b": perf["vram_gib"], "delta": perf["vram_gib"] - r0_perf["vram_gib"]}
        res["vs_R0"][name] = entry

    rep = r0 / "reproducibility.json"
    ident = art_dir / "V0-tensor-identity.json"
    a_scores = json.loads(rep.read_text()) if rep.exists() else None
    a_tensors = json.loads(ident.read_text()) if ident.exists() else None
    res["A_reproducible"] = bool(a_scores and all(v["identical"] for v in a_scores.values())
                                 and a_tensors and a_tensors["identical"])
    variants = {k: v for k, v in res["vs_R0"].items() if k.startswith("V") and not k.startswith("V0")}
    res["B_quality_moves"] = any(v["kl"]["significant"] for v in variants.values())
    res["C_speed_moves"] = any((v.get("decode") or {}).get("clear") or (v.get("prefill") or {}).get("clear")
                               for v in variants.values())

    rows = load_rows([art_dir], track)
    by_id = {r.name: r for r in rows}
    r0_row = next(r for r in rows if r.id == "R0")
    mixed = [r for r in rows if r.kind == "candidate" and not r.name.startswith(("V0", "V1"))]
    res["D_mixed_not_dominated_by_R0"] = [r.name for r in mixed if r.valid and not dominates(r0_row, r)]

    def dkl(name: str) -> dict | None:
        return variants.get(name, {}).get("kl")

    v4, v5, v6, v9 = dkl("V4-gdn-q4k"), dkl("V5-attn-q4k"), dkl("V6-mlp-q4k"), dkl("V9-gdn-q4k-mlp-q4k")
    if v4 and v5 and v6:
        per_b = {"gdn": v4["delta_kl"] / (GDN_PARAMS / 1e9), "attn": v5["delta_kl"] / (ATTN_PARAMS / 1e9),
                 "mlp": v6["delta_kl"] / (MLP_PARAMS / 1e9)}
        res["E_sensitivity_per_billion_weights"] = per_b
    if v4 and v6 and v9:
        additive = v4["delta_kl"] + v6["delta_kl"]
        half = ((v4["ci95"][1] - v4["ci95"][0]) + (v6["ci95"][1] - v6["ci95"][0]) + (v9["ci95"][1] - v9["ci95"][0])) / 2
        res["F_interaction"] = {"sum_of_parts": additive, "together": v9["delta_kl"],
                                "difference": v9["delta_kl"] - additive, "tolerance": half,
                                "interaction": abs(v9["delta_kl"] - additive) > half}
    res["rows"] = {r.name: {"valid": r.valid, "frontier": r.frontier, "gain": r.gain, "gates": r.gate_failures} for r in rows}
    _ = by_id
    out_dir.mkdir(parents=True, exist_ok=True)
    feasibility_report([art_dir], track, out_dir)
    (out_dir / "analysis.json").write_text(json.dumps(res, indent=2) + "\n")
    return res


if __name__ == "__main__":
    a = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "artifacts/feasibility"
    o = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO_ROOT / "results/feasibility"
    print(json.dumps(main(a, o), indent=2))
