"""Is the incumbent reproducible?

  check_reproducible.py <V0 artifact dir> <V0 checkpoint>           re-score two streams, compare bytes
  check_reproducible.py --compare-tensors <V0 checkpoint> <shipped>  rebuilt tensors == shipped tensors
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from bittrellis.eval.corpus import load_corpus
from bittrellis.eval.reference import load_reference
from bittrellis.runtime import RefScore, SparkInfer
from bittrellis.safetensors_io import SafeTensorsDir
from bittrellis.track import REPO_ROOT, load_track


def compare_tensors(a: Path, b: Path) -> dict:
    with SafeTensorsDir(a) as x, SafeTensorsDir(b) as y:
        names = sorted(set(x.tensors) | set(y.tensors))
        diff = [n for n in names if n not in x.tensors or n not in y.tensors or x.sha256(n) != y.sha256(n)]
    return {"tensors": len(names), "different": diff[:50], "n_different": len(diff), "identical": not diff}


def rescore(art: Path, ckpt: Path, reference: Path) -> dict:
    track = load_track("HPC-01")
    si = SparkInfer(REPO_ROOT / "third_party/sparkinfer", track)
    corpus = load_corpus(REPO_ROOT / track["evaluation"]["corpus"]["public"])
    refs = load_reference(reference, corpus)
    # Re-score two streams, each alone in its own process, and compare with the multi-stream run
    # stored in the artifact: this checks run-to-run determinism and that sharing one model load
    # across streams does not change a single byte.
    out = {}
    for s in corpus["streams"]:
        if s["id"] not in ("short-code", "long-8k"):
            continue
        first = RefScore.load(art / "scores" / f"{s['id']}.npz")
        again, _ = si.refscore_many(ckpt, [(s["id"], s["ids"], refs[s["id"]].top_ids, s["score_from"])],
                                    art / "scores" / "_repro", env=track["evaluation"]["score"].get("env"))
        again = again[s["id"]]
        same = all(np.array_equal(getattr(first, f), getattr(again, f)) for f in ("pos", "argmax", "lp_target", "lp_ref"))
        out[s["id"]] = {"identical": bool(same), "max_abs_lp_diff": float(np.abs(first.lp_ref - again.lp_ref).max())}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--compare-tensors", action="store_true")
    ap.add_argument("--reference", default=str(REPO_ROOT / "data/reference/hpc01-public-v2-k256"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = compare_tensors(Path(args.a), Path(args.b)) if args.compare_tensors else rescore(Path(args.a), Path(args.b), Path(args.reference))
    Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    ok = res["identical"] if args.compare_tensors else all(v["identical"] for v in res.values())
    sys.exit(0 if ok else 1)
