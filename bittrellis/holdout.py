"""Private rotating holdout: validator-side fidelity check on text miners never see.

Layout of a validator's private holdout directory (never committed, never published during its epoch):

    <private>/
      docs/general/*.txt   docs/math/*.txt   docs/code/*.txt   docs/tools/*.txt
      docs/multilingual/*.txt              docs/long/*.txt      (long filler documents)
      epoch.json            {"epoch": "hpc01-h2026w38", "seed": "<secret>"}
      corpus.json           built by `bittrellis holdout build`
      reference/            BF16 reference for corpus.json (`bittrellis reference --corpus ...`)
      results/<name>.json   full per-candidate numbers (private)

Each *.txt file is one document, already in the text form it should be scored in (chat-formatted
where that matters). The stream structure is identical to the public corpus.

Only PASS/FAIL leaves this directory: `holdout.json` in the candidate's artifact contains the epoch
and the verdict, nothing else, so repeated submissions cannot be used to fit the holdout.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .eval import logits
from .eval.candidate import evaluate_quality
from .eval.corpus import SHORT_CATEGORIES, assemble_streams, corpus_hash
from .runtime import SparkInfer
from .track import Track


def build_private_corpus(private_dir: Path, tokenizer_json: Path) -> dict:
    from tokenizers import Tokenizer

    private_dir = Path(private_dir)
    meta = json.loads((private_dir / "epoch.json").read_text())
    tok_bytes = Path(tokenizer_json).read_bytes()
    tok = Tokenizer.from_str(tok_bytes.decode())

    def docs(cat: str) -> list[str]:
        files = sorted((private_dir / "docs" / cat).glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"holdout needs documents in docs/{cat}/")
        ordered = sorted(files, key=lambda f: hashlib.sha256(f"{meta['seed']}:{f.name}".encode()).hexdigest())
        return [f.read_text() for f in ordered]

    short = {cat: docs(cat) for cat in SHORT_CATEGORIES}
    streams = assemble_streams(tok, short, docs("long"), meta["seed"])
    body = {"version": meta["epoch"], "split": "private-holdout",
            "tokenizer": {"sha256": hashlib.sha256(tok_bytes).hexdigest()}, "streams": streams}
    body["sha256"] = corpus_hash(body)
    (private_dir / "corpus.json").write_text(json.dumps(body, separators=(",", ":")) + "\n")
    return body


def check(si: SparkInfer, track: Track, checkpoint: Path, artifact: Path, private_dir: Path,
          incumbent_checkpoint: Path, incumbent_artifact: Path, log=print) -> str:
    """Score a candidate (and, once per epoch, the incumbent) on the private holdout; write PASS/FAIL."""
    private_dir = Path(private_dir)
    corpus = json.loads((private_dir / "corpus.json").read_text())
    meta = json.loads((private_dir / "epoch.json").read_text())
    results = private_dir / "results"
    results.mkdir(exist_ok=True)
    cand_name = json.loads((Path(artifact) / "candidate.json").read_text())["name"]

    def score(ckpt: Path, name: str) -> tuple[dict, dict]:
        work = private_dir / "work" / name
        work.mkdir(parents=True, exist_ok=True)
        q, corr = evaluate_quality(si, track, ckpt, corpus, private_dir / "reference", work, log=log)
        return q, dict(np.load(work / "kl_positions.npz"))

    inc_name = json.loads((Path(incumbent_artifact) / "candidate.json").read_text())["name"]
    inc_q, inc_pos = score(Path(incumbent_checkpoint), inc_name)
    cand_q, cand_pos = score(Path(checkpoint), cand_name)

    gates = track["gates"]
    reasons = []
    audit_ok = json.loads((Path(artifact) / "candidate.json").read_text()).get("audit_ok")
    if audit_ok is not True:
        reasons.append("audit")
    if cand_q["rp_kl"] > gates["rp_kl_max"]:
        reasons.append("rp_kl_max")
    for stream, got in cand_q["needles_by_length"].items():
        if got["required"] and got["retrieved"] < got["required"]:
            reasons.append(f"needles:{stream}")
    pub_inc = dict(np.load(Path(incumbent_artifact) / "kl_positions.npz"))
    pub_cand = dict(np.load(Path(artifact) / "kl_positions.npz"))
    pub = logits.paired_delta(pub_inc, pub_cand)          # candidate - incumbent, public
    hold = logits.paired_delta(inc_pos, cand_pos)         # candidate - incumbent, holdout
    public_gain, holdout_gain = -pub["delta"], -hold["delta"]
    ratio = track["evaluation"]["holdout"]["min_gain_ratio"]
    if pub["significant"] and public_gain > 0 and holdout_gain < ratio * public_gain:
        reasons.append("public gain does not carry over to the holdout")
    verdict = "FAIL" if reasons else "PASS"

    (results / f"{cand_name}.json").write_text(json.dumps({
        "epoch": meta["epoch"], "verdict": verdict, "reasons": reasons,
        "candidate_rp_kl": cand_q["rp_kl"], "incumbent_rp_kl": inc_q["rp_kl"],
        "public_gain": public_gain, "holdout_gain": holdout_gain, "public_ci95": pub["ci95"], "holdout_ci95": hold["ci95"],
    }, indent=2) + "\n")
    (Path(artifact) / "holdout.json").write_text(json.dumps({"epoch": meta["epoch"], "result": verdict}) + "\n")
    return verdict
