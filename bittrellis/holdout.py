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

MAX_REPEATED_LINES = 0.2   # above this share of duplicate lines a category is padding, not evidence


class NotEnoughText(ValueError):
    """The holdout directory does not hold enough text yet; the message lists what is missing."""


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
    inv = inventory(private_dir, tokenizer_json)
    inv.pop("epoch", None)
    missing = {cat: d["missing"] for cat, d in inv.items() if d["missing"]}
    if missing:
        raise NotEnoughText("not enough text yet: " + ", ".join(
            f"{cat} needs {n:,} more tokens (~{n * 3 // 4:,} words)" for cat, n in sorted(missing.items())))
    repeated = {cat: d["repeated_lines"] for cat, d in inv.items() if d["repeated_lines"] > MAX_REPEATED_LINES}
    if repeated:
        raise NotEnoughText("too much repeated text (a second copy of a passage is not new evidence): " + ", ".join(
            f"{cat} {share:.0%} repeated lines" for cat, share in sorted(repeated.items())))
    streams = assemble_streams(tok, short, docs("long"), meta["seed"])
    body = {"version": meta["epoch"], "split": "private-holdout",
            "tokenizer": {"sha256": hashlib.sha256(tok_bytes).hexdigest()}, "streams": streams}
    body["sha256"] = corpus_hash(body)
    (private_dir / "corpus.json").write_text(json.dumps(body, separators=(",", ":")) + "\n")
    return body


def inventory(private_dir: Path, tokenizer_json: Path) -> dict:
    """What the holdout directory has and what it still needs, per category."""
    from tokenizers import Tokenizer

    from .eval.corpus import LONG_LENGTHS, SHORT_TOKENS

    tok = Tokenizer.from_str(Path(tokenizer_json).read_text())
    need = {cat: SHORT_TOKENS for cat in SHORT_CATEGORIES}
    need["long"] = max(LONG_LENGTHS)  # every needle stream is filled from these documents
    out = {}
    for cat, want in need.items():
        files = sorted((Path(private_dir) / "docs" / cat).glob("*.txt"))
        have = sum(len(tok.encode(f.read_text()).ids) for f in files)
        # Repeated passages make a holdout look easy: predictions on a second copy are not independent.
        lines = [ln.strip() for f in files for ln in f.read_text().splitlines() if ln.strip()]
        repeated = 1 - len(set(lines)) / len(lines) if lines else 0.0
        # Two or more long documents keep the 8K/16K/32K streams from being prefixes of each other.
        want_more = max(LONG_LENGTHS) * 2 if cat == "long" else want
        out[cat] = {"files": len(files), "tokens": have, "needs": want, "missing": max(0, want - have),
                    "recommended": want_more, "short_of_recommended": max(0, want_more - have),
                    "repeated_lines": round(repeated, 3)}
    out["epoch"] = json.loads((Path(private_dir) / "epoch.json").read_text()) if (Path(private_dir) / "epoch.json").exists() else None
    return out


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
        work = private_dir / "work" / name  # "incumbent/<name>" or "candidate/<name>": a candidate never overwrites the cache
        work.mkdir(parents=True, exist_ok=True)
        q, corr = evaluate_quality(si, track, ckpt, corpus, private_dir / "reference", work, log=log)
        return q, dict(np.load(work / "kl_positions.npz"))

    inc_ident = json.loads((Path(incumbent_artifact) / "candidate.json").read_text())
    inc_name = inc_ident["name"]
    # The incumbent is scored once per epoch and corpus, then reused by every later check.
    inc_work = private_dir / "work" / "incumbent" / inc_name
    stamp = {"epoch": meta["epoch"], "corpus_sha256": corpus.get("sha256"), "candidate_id": inc_ident.get("id")}
    cached = inc_work / "incumbent.json"
    if cached.exists() and (inc_work / "kl_positions.npz").exists() and json.loads(cached.read_text()).get("stamp") == stamp:
        inc_q, inc_pos = json.loads(cached.read_text())["quality"], dict(np.load(inc_work / "kl_positions.npz"))
        log(f"[holdout] incumbent {inc_name}: cached for epoch {meta['epoch']}")
    else:
        inc_q, inc_pos = score(Path(incumbent_checkpoint), f"incumbent/{inc_name}")
        cached.write_text(json.dumps({"stamp": stamp, "quality": inc_q}) + "\n")
    cand_q, cand_pos = score(Path(checkpoint), f"candidate/{cand_name}")

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
