"""Feasibility question A: is the baseline reproducible?

  check_reproducible.py <R0 artifact dir> <R0 checkpoint>            re-score two streams, compare bytes
  check_reproducible.py --compare-tensors <V0 checkpoint> <R0 dir>   rebuilt tensors == shipped tensors
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from bittrellis.eval.corpus import load_corpus
from bittrellis.runtime import ScoreDump, SparkInfer
from bittrellis.safetensors_io import SafeTensorsDir
from bittrellis.track import REPO_ROOT, load_track


def compare_tensors(a: Path, b: Path) -> dict:
    with SafeTensorsDir(a) as x, SafeTensorsDir(b) as y:
        names = sorted(set(x.tensors) | set(y.tensors))
        diff = [n for n in names if n not in x.tensors or n not in y.tensors or x.sha256(n) != y.sha256(n)]
    return {"tensors": len(names), "different": diff[:50], "n_different": len(diff), "identical": not diff}


def rescore(art: Path, ckpt: Path) -> dict:
    track = load_track("HPC-01")
    si = SparkInfer(REPO_ROOT / "third_party/sparkinfer", track)
    corpus = load_corpus(REPO_ROOT / "data/corpus/hpc01-public-v2.json")
    out = {}
    for s in corpus["streams"]:
        if s["id"] not in ("short-code", "long-8k"):
            continue
        first = ScoreDump.load(art / "scores" / f"{s['id']}.npz")
        cfg = track["evaluation"]["score"]
        again = si.score(ckpt, s["ids"], cfg["topk"], prefix_len=s["score_from"], env=cfg.get("env"))
        same = all(np.array_equal(getattr(first, f), getattr(again, f)) for f in ("pos", "argmax", "lp_target", "top_ids", "top_lp"))
        out[s["id"]] = {"identical": bool(same), "max_abs_lp_diff": float(np.abs(first.top_lp - again.top_lp).max())}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--compare-tensors", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = compare_tensors(Path(args.a), Path(args.b)) if args.compare_tensors else rescore(Path(args.a), Path(args.b))
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    ok = res["identical"] if args.compare_tensors else all(v["identical"] for v in res.values())
    sys.exit(0 if ok else 1)
