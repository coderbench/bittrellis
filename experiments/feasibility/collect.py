"""Copy the small, reviewable parts of every artifact into results/feasibility/artifacts/.

These are the seeds of the internal frontier. Raw per-stream dumps (scores/*.npz, ~40 MB per
checkpoint) stay on the evaluation host; per-position RP-KL (kl_positions.npz) is kept so every paired
comparison can be recomputed.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from bittrellis.track import REPO_ROOT

KEEP = ("candidate.json", "quality.json", "correctness.json", "performance.json", "tasks.json", "environment.json",
        "audit.json", "holdout.json", "reproducibility.json", "kl_positions.npz")


def main(src: Path, dst: Path) -> None:
    for art in sorted(p for p in src.iterdir() if (p / "candidate.json").exists()):
        out = dst / art.name
        shutil.rmtree(out, ignore_errors=True)
        out.mkdir(parents=True)
        for f in KEEP:
            if (art / f).exists():
                shutil.copy2(art / f, out / f)
        for name, drop in (("performance.json", ("log_tail",)), ("audit.json", ())):
            p = out / name
            if p.exists():
                d = json.loads(p.read_text())
                for run in d.get("runs", []):
                    run.pop("log_tail", None)
                for k in drop:
                    d.pop(k, None)
                p.write_text(json.dumps(d, indent=2) + "\n")
    for extra in src.glob("*.json"):
        shutil.copy2(extra, dst / extra.name)


if __name__ == "__main__":
    s = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "artifacts/seeds"
    d = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO_ROOT / "results/feasibility/artifacts"
    main(s, d)
