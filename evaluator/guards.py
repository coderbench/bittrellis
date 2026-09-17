"""Pre-GPU guards for the PR evaluator: who was first, copies, memory, queue share, encoders.

Everything here runs on the CPU in seconds, before a PR costs any GPU time. The evaluator's own
records live under its root directory, outside every submission's reach.

Copies are judged by what a submission IS, not how it is written:

* a manifest by its expanded per-unit assignment (so rewritten rules, reordered rules or other
  layer ranges that expand to the same recipe are the same recipe), weighted by parameter count;
* a quantizer by the bytes it produces (see bittrellis/fingerprint.py), so renamed or rewritten
  code that encodes identically is the same encoder.

The original is whoever the evaluator OBSERVED first at that head commit, not the lower PR number:
a PR opened early and force-pushed with copied content later gets the time of the copied head.

A near-copy is not rejected. It is measured, and its frontier gain is computed with the earlier
PR's result already in the frontier, so it earns exactly what it adds. Only exact duplicates are
closed without a measurement, and only a repeated pattern goes to a maintainer. No account is
blocked automatically: the search space is small and the repository ships a neighbour generator,
so independent miners will often find the same recipe.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

GIB = 1024**3


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")  # microseconds: one pass observes many PRs


# ── observations: the evaluator's append-only record ───────────────────────────────────────────


class Observations:
    """One write-once JSON file per (PR, head commit). `first_seen`, `author` and `head` never change."""

    IMMUTABLE = ("pr", "author", "head", "first_seen")

    def __init__(self, root: Path):
        self.dir = Path(root) / "observations"
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, pr: int, head: str) -> Path:
        return self.dir / f"pr-{pr:06d}-{head[:12]}.json"

    def observe(self, pr: int, author: str, head: str, now: str | None = None) -> dict:
        path = self._path(pr, head)
        if path.exists():
            return json.loads(path.read_text())
        rec = {"pr": pr, "author": author, "head": head, "first_seen": now or utcnow()}
        path.write_text(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def annotate(self, pr: int, head: str, **fields) -> dict:
        path = self._path(pr, head)
        rec = json.loads(path.read_text())
        rec.update({k: v for k, v in fields.items() if k not in self.IMMUTABLE})
        path.write_text(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def all(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.dir.glob("pr-*.json"))]


# ── verdicts ───────────────────────────────────────────────────────────────────────────────────


@dataclass
class Verdict:
    outcome: str                  # pass | duplicate | copy-review | memory | queued | same-encoder | nondeterministic
    reason: str = ""
    original: dict | None = None  # the earlier submission this one duplicates or derives from
    derivative_of: dict | None = None
    details: dict = field(default_factory=dict)

    @property
    def stop(self) -> bool:
        return self.outcome != "pass"

    def to_dict(self) -> dict:
        return {"outcome": self.outcome, "reason": self.reason, "original": self.original,
                "derivative_of": self.derivative_of, "details": self.details}


def keys_from_expanded(expanded: dict) -> dict[str, str]:
    """{unit: FORMAT@name@vN[+params]} from a manifest's expanded form (candidate.json)."""
    import hashlib

    out = {}
    for uid, e in expanded.items():
        key = f"{e['source_format']}@{e['quantizer']}"
        if e.get("params"):
            key += "+" + hashlib.sha256(json.dumps(sorted(e["params"].items()), sort_keys=True).encode()).hexdigest()[:12]
        out[uid] = key
    return out


def share_different(a: dict[str, str], b: dict[str, str], numel: dict[str, int]) -> float:
    """Share of searchable weights (by parameter count) assigned differently in `a` and `b`."""
    total = sum(numel.values())
    return sum(n for uid, n in numel.items() if a.get(uid) != b.get(uid)) / total if total else 0.0


def _ref(o: dict) -> dict:
    return {"pr": o["pr"], "author": o["author"], "first_seen": o["first_seen"]}


def judge_manifest(me: dict, candidate_id: str, keys: dict[str, str], numel: dict[str, int], earlier: list[dict],
                   known: dict[str, str], cfg: dict, epoch: str, cleared: bool = False) -> Verdict:
    """Duplicate, near-copy or pass for one manifest submission.

    `earlier`: observations of the current heads of PRs that are open or merged (a closed, unmerged PR
    earns nothing, so building on its public idea is not copying). `known`: {candidate id: name} of seeds
    and accepted results.
    """
    if candidate_id in known:
        return Verdict("duplicate", f"identical to `{known[candidate_id]}`, which is already measured",
                       original={"name": known[candidate_id]})
    before = [o for o in earlier if o["pr"] != me["pr"] and o["first_seen"] < me["first_seen"] and o.get("keys")]
    same = sorted((o for o in before if o.get("candidate_id") == candidate_id), key=lambda o: o["first_seen"])
    if same:
        return Verdict("duplicate", f"identical to #{same[0]['pr']}, observed earlier", original=_ref(same[0]))
    others = [o for o in before if o["author"].lower() != me["author"].lower()]
    near = sorted(((share_different(keys, o["keys"], numel), o) for o in others), key=lambda t: (t[0], t[1]["first_seen"]))
    if not near or near[0][0] > cfg["near_copy_max_share"]:
        return Verdict("pass", details={"closest_share": near[0][0] if near else None})
    share, orig = near[0]
    prior = len({o["pr"] for o in earlier if o["author"].lower() == me["author"].lower() and o["pr"] != me["pr"]
                 and o.get("derivative_of") and o.get("epoch") == epoch})
    details = {"share_different": share, "earlier_near_copies_by_author": prior}
    if prior + 1 >= cfg["copy_review_after"] and not cleared:
        return Verdict("copy-review", f"the author's {prior + 1}th recipe within {cfg['near_copy_max_share']:.0%} of "
                       "another author's earlier PR; waiting for a maintainer", original=_ref(orig), details=details)
    return Verdict("pass", f"{share:.2%} of weights differ from #{orig['pr']}; credited only for what it adds",
                   derivative_of=_ref(orig), details=details)


# ── memory ─────────────────────────────────────────────────────────────────────────────────────


def weights_gib(keys: dict[str, str], units) -> float:
    from bittrellis.precision import weight_bytes

    by_id = {u.id: u for u in units}
    return sum(weight_bytes(by_id[uid], key.split("@", 1)[0]) for uid, key in keys.items()) / GIB


def judge_memory(keys: dict[str, str], units, seeds: list[tuple[dict[str, str], float]], cfg: dict) -> Verdict:
    """Reject a recipe whose predicted peak (weights + the largest measured seed overhead + margin) cannot fit."""
    overhead = max(peak - weights_gib(k, units) for k, peak in seeds)
    predicted = weights_gib(keys, units) + overhead + cfg["memory_margin_gib"]
    details = {"predicted_peak_gib": round(predicted, 2), "limit_gib": cfg["peak_gpu_limit_gib"]}
    if predicted > cfg["peak_gpu_limit_gib"]:
        return Verdict("memory", f"predicted peak {predicted:.1f} GiB exceeds {cfg['peak_gpu_limit_gib']} GiB", details=details)
    return Verdict("pass", details=details)


# ── queue share ────────────────────────────────────────────────────────────────────────────────


def judge_queue(me: dict, pending: list[dict], cfg: dict) -> Verdict:
    """Hold a PR while the author already has `max_queued_per_author` earlier PRs waiting."""
    ahead = [o for o in pending if o["author"].lower() == me["author"].lower() and o["first_seen"] < me["first_seen"]
             and o["pr"] != me["pr"]]
    if len(ahead) >= cfg["max_queued_per_author"]:
        return Verdict("queued", f"{len(ahead)} earlier PRs by this author are still waiting", details={"ahead": [o["pr"] for o in ahead]})
    return Verdict("pass")


# ── quantizers ─────────────────────────────────────────────────────────────────────────────────


def judge_quantizers(new: dict[str, dict[str, bytes]], known: dict[str, dict[str, bytes]], cfg: dict,
                     repeat: dict[str, dict[str, bytes]] | None = None) -> Verdict:
    """Same-encoder check by output bytes.

    `new`: probe output of the quantizers this PR adds, by ref. `known`: probe output of main's
    quantizers and of earlier PRs' quantizers, by label. `repeat`: a second probe run of `new`,
    which must be byte-identical (the audit rejects nondeterministic encoders anyway).
    """
    from bittrellis.fingerprint import similarity

    if repeat is not None:
        for ref, fp in new.items():
            if repeat.get(ref) != fp:
                return Verdict("nondeterministic", f"`{ref}` produced different bytes on two identical runs")
    matches = []
    for ref, fp in new.items():
        for label, other in known.items():
            sim, compared = similarity(fp, other)
            if compared and sim >= cfg["quantizer_same_bytes"]:
                matches.append({"quantizer": ref, "same_as": label, "identical_bytes": round(sim, 4), "bytes_compared": compared})
    if matches:
        m = matches[0]
        return Verdict("same-encoder", f"`{m['quantizer']}` produces {m['identical_bytes']:.1%} of the same bytes as "
                       f"`{m['same_as']}`", original={"quantizer": m["same_as"]}, details={"matches": matches})
    return Verdict("pass")
