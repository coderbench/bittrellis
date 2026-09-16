"""Copy the small, reviewable parts of every artifact into results/feasibility/artifacts/.

Raw score dumps (scores/*.npz, ~25 MB per checkpoint) stay on the evaluation host; per-position
KL (kl_positions.npz) is kept so every paired comparison in the report can be recomputed.
Candidate ids are re-derived from each artifact's manifest with the current hashing rules.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from bittrellis.manifest import Manifest
from bittrellis.model.qwen38 import Qwen38Arch
from bittrellis.track import REPO_ROOT

KEEP = ("candidate.json", "quality.json", "performance.json", "tasks.json", "environment.json", "audit.json",
        "reproducibility.json", "kl_positions.npz")


def main(src: Path, dst: Path) -> None:
    units = Qwen38Arch().units()
    for art in sorted(p for p in src.iterdir() if (p / "candidate.json").exists()):
        cand = json.loads((art / "candidate.json").read_text())
        if cand.get("kind") == "candidate":
            m = Manifest.from_dict({k: v for k, v in cand["manifest"].items() if k != "expanded"})
            cand["id"] = m.candidate_id(units)
            cand["manifest"] = m.to_dict(units)
            (art / "candidate.json").write_text(json.dumps(cand, indent=2) + "\n")
        out = dst / art.name
        out.mkdir(parents=True, exist_ok=True)
        for f in KEEP:
            if (art / f).exists():
                shutil.copy2(art / f, out / f)
        perf = out / "performance.json"
        if perf.exists():
            d = json.loads(perf.read_text())
            d.pop("log_tail", None)
            perf.write_text(json.dumps(d, indent=2) + "\n")
        audit = out / "audit.json"
        if audit.exists():
            d = json.loads(audit.read_text())
            d.pop("fidelity", None)  # per-tensor numbers; summarized below
            audit.write_text(json.dumps(d, indent=2) + "\n")
    for extra in ("V0-tensor-identity.json",):
        if (src / extra).exists():
            shutil.copy2(src / extra, dst / extra)


if __name__ == "__main__":
    s = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "artifacts/feasibility"
    d = Path(sys.argv[2]) if len(sys.argv) > 2 else REPO_ROOT / "results/feasibility/artifacts"
    main(s, d)
