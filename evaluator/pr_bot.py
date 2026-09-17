"""BitTrellis PR evaluator: scores miner pull requests on the pinned RTX 5090 host.

Run it on the evaluation host (cron or a loop):

    GITHUB_TOKEN=... python evaluator/pr_bot.py --repo coderbench/bittrellis --root /workspace/bt-eval \
        --private /secure/holdout-epoch [--once]

Every pass:

0. OBSERVE every open PR head before evaluating anything, in an append-only record under <root>
   (evaluator/guards.py). The queue runs in first-seen order, so an original is always measured
   before a later near-copy of it.

For each PR, in that order:

1. CLASSIFY from its changed files.
   * manifest PR     exactly one added/changed `manifests/*.yaml`, nothing else
                     -> evaluated with the trusted `main` code; only the YAML is read from the PR
   * code PR         changes under bittrellis/quantizers/ or bittrellis/search.py (+ at most one manifest)
                     -> evaluated only after a maintainer adds `eval-approved`, from a worktree of the PR head
   * evaluator paths -> never evaluated automatically; labelled `bt:touches-evaluator`
2. SCREEN, no GPU (seconds):
   queue share per author · manifest valid · exact duplicate (seeds, accepted, earlier PRs) ·
   near-copy of an earlier PR by another author · predicted peak memory ·
   code PRs: new quantizers are deterministic and do not reproduce an existing encoder (probe bytes)
3. BUILD and AUDIT on the CPU. Code PRs: the new encoder's REAL stored bytes are compared with every
   existing encoder's bytes for the same tensors (sketches), still before any GPU time.
4. MEASURE in stages, stopping as soon as the outcome is known:
   quality -> quality gates -> performance (2 runs) -> dominated? stop (tasks and holdout can only fail
   a result, never lift a dominated one) -> tasks -> private holdout -> rank.
5. RANK against seeds + accepted results + earlier open PRs by other authors, so a later submission
   earns only what it adds over work already on the table. Comment once, label the status.

After the queue, RE-RANK evaluated open PRs whose reference set changed (an earlier PR merged or
closed): labels stay current, and a PR that becomes non-dominated is resumed for tasks and holdout.

State lives in <root>/state.json; artifacts in <root>/prs/<number>-<sha>/; accepted artifacts in
<root>/accepted/ (copied when a PR that the bot evaluated is merged).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import guards as G  # noqa: E402

MANIFEST_GLOB = "manifests/*.yaml"
CODE_GLOBS = ("bittrellis/quantizers/*", "bittrellis/search.py", "tests/*")
EVALUATOR_GLOBS = ("configs/*", "data/*", "bittrellis/eval/*", "bittrellis/frontier/*", "bittrellis/validate.py",
                   "bittrellis/lineage.py", "bittrellis/holdout.py", "bittrellis/runtime.py", "bittrellis/build.py",
                   "bittrellis/fingerprint.py", "bittrellis/synthetic.py", "bittrellis/manifest.py", "bittrellis/precision.py",
                   "bittrellis/cli.py", "evaluator/*", "tools/*", ".github/*", "scripts/*", "pyproject.toml")
LABELS = {
    "frontier": ("bt:frontier", "0e8a16", "moves the internal frontier"),
    "dominated": ("bt:dominated", "cccccc", "valid but dominated"),
    "gate": ("bt:gate-fail", "d93f0b", "failed a quality or runtime gate"),
    "audit": ("bt:audit-fail", "b60205", "failed the checkpoint audit"),
    "build": ("bt:build-fail", "b60205", "the checkpoint did not build"),
    "invalid": ("bt:invalid-manifest", "b60205", "manifest does not validate"),
    "duplicate": ("bt:duplicate", "cccccc", "same candidate as an accepted or earlier one; not measured"),
    "copy-review": ("bt:copy-review", "fbca04", "repeated near-copies of other authors' PRs; waiting for a maintainer"),
    "same-encoder": ("bt:same-encoder", "b60205", "the new quantizer reproduces an existing encoder"),
    "nondeterministic": ("bt:nondeterministic", "b60205", "the new quantizer is not deterministic"),
    "memory": ("bt:memory", "d93f0b", "predicted to exceed the GPU's memory; not measured"),
    "queued": ("bt:queued", "ededed", "waiting: the author's earlier PRs are ahead in the queue"),
    "evaluator": ("bt:touches-evaluator", "5319e7", "changes evaluator paths; maintainer review"),
    "needs_approval": ("bt:needs-approval", "fbca04", "executes contributed code; waiting for eval-approved"),
    "error": ("bt:eval-error", "b60205", "evaluation crashed; retried automatically"),
}
EXTRA_LABELS = {"derivative": ("bt:derivative", "c5def5", "close to an earlier PR by another author; credited for what it adds")}
APPROVED, COPY_CLEARED = "eval-approved", "copy-cleared"
RESCREEN = {"queued", "needs-approval", "copy-review"}   # re-checked every pass
RANKED = {"frontier", "dominated", "gate"}
MAX_ERRORS = 3


class GitHub:
    def __init__(self, repo: str, token: str):
        self.repo, self.token = repo, token

    def api(self, method: str, path: str, body: dict | None = None):
        url = path if path.startswith("https://") else f"https://api.github.com/repos/{self.repo}{path}"
        req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json",
                                              "X-GitHub-Api-Version": "2022-11-28"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
            return json.loads(data) if data else None

    def paged(self, path: str) -> list:
        out, page = [], 1
        while True:
            sep = "&" if "?" in path else "?"
            batch = self.api("GET", f"{path}{sep}per_page=100&page={page}")
            out += batch
            if len(batch) < 100:
                return out
            page += 1

    def ensure_labels(self) -> None:
        existing = {lab["name"] for lab in self.paged("/labels")}
        for name, color, desc in [*LABELS.values(), *EXTRA_LABELS.values()]:
            if name not in existing:
                self.api("POST", "/labels", {"name": name, "color": color, "description": desc})

    def set_status_label(self, number: int, key: str) -> None:
        current = {lab["name"] for lab in self.api("GET", f"/issues/{number}")["labels"]}
        for k, (name, _, _) in LABELS.items():
            if name in current and k != key:
                try:
                    self.api("DELETE", f"/issues/{number}/labels/{urllib.request.quote(name)}")
                except urllib.error.HTTPError:
                    pass
        if LABELS[key][0] not in current:
            self.api("POST", f"/issues/{number}/labels", {"labels": [LABELS[key][0]]})

    def add_label(self, number: int, name: str) -> None:
        self.api("POST", f"/issues/{number}/labels", {"labels": [name]})

    def comment(self, number: int, body: str) -> None:
        self.api("POST", f"/issues/{number}/comments", {"body": body})


def classify(files: list[str]) -> tuple[str, list[str]]:
    manifests = [f for f in files if fnmatch.fnmatch(f, MANIFEST_GLOB)]
    if any(any(fnmatch.fnmatch(f, g) for g in EVALUATOR_GLOBS) for f in files):
        return "evaluator", manifests
    others = [f for f in files if f not in manifests]
    if not others and len(manifests) == 1:
        return "manifest", manifests
    code = [f for f in others if any(fnmatch.fnmatch(f, g) for g in CODE_GLOBS)]
    if code and all(f in code or f.endswith(".md") for f in others) and len(manifests) <= 1:
        return "code", manifests
    return "other", manifests


def run(cmd: list[str], cwd: Path, log: Path, timeout: int = 6 * 3600) -> int:
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout).returncode


# ── pure decisions (unit-tested) ───────────────────────────────────────────────────────────────


def queue_order(prs: list[dict], first_seen: dict[tuple[int, str], str]) -> list[dict]:
    """Oldest observed head first; PR number only breaks exact ties."""
    return sorted(prs, key=lambda p: (first_seen[(p["number"], p["head"]["sha"])], p["number"]))


def quality_gate_failures(quality: dict, gates: dict) -> list[str]:
    """Gates decidable from the quality stage alone; failing one makes the speed runs pointless."""
    fails = []
    if quality.get("nonfinite_logprobs"):
        fails.append(f"{quality['nonfinite_logprobs']} non-finite log-probabilities")
    if quality["rp_kl"] > gates["rp_kl_max"]:
        fails.append(f"RP-KL {quality['rp_kl']:.4f} > {gates['rp_kl_max']}")
    if quality.get("top1") is not None and quality["top1"] < gates["top1_min"]:
        fails.append(f"top-1 {quality['top1']:.3f} < {gates['top1_min']}")
    for stream, need in gates["long_context_guard"]["required_success"].items():
        got = (quality.get("needles_by_length") or {}).get(stream)
        if got and got["required"] and got["retrieved"] / got["required"] < need:
            fails.append(f"long-context guard: {stream} {got['retrieved']}/{got['required']} needles")
    return fails


def reference_entries(me: dict, state: dict, live_heads: dict[int, str]) -> list[str]:
    """State keys of earlier, measured PRs by other authors that are still open or merged.

    `live_heads`: {PR number: head sha} for open PRs and merged PRs. A later PR is ranked with these
    already on the frontier, so it earns only what it adds over work already on the table.
    """
    out = []
    for key, e in state.items():
        if not isinstance(e, dict) or e.get("status") not in RANKED or not e.get("artifact") or "first_seen" not in e:
            continue
        if e["pr"] == me["pr"] or e["author"].lower() == me["author"].lower() or e["first_seen"] >= me["first_seen"]:
            continue
        if live_heads.get(e["pr"]) != e["head"]:
            continue
        out.append(key)
    return sorted(out)


def status_from_row(row: dict) -> str:
    if not row["valid"]:
        return "gate"
    return "frontier" if row["frontier"] and (row["frontier_gain"] or 0) > 0 else "dominated"


# ── comment ────────────────────────────────────────────────────────────────────────────────────


def render_comment(name: str, cid: str, frontier: dict | None, cmp: dict | None, label: str, notes: list[str],
                   screen: dict | None = None, timings: dict | None = None) -> str:
    epoch = frontier["evaluator_epoch"] if frontier else "—"
    lines = [f"### BitTrellis evaluation · `{name}` · `{cid}`", "",
             f"Epoch `{epoch}` · status **{LABELS[label][0]}**", ""]
    row = next((r for r in frontier["internal"] if r["name"] == name), None) if frontier else None
    if row:
        lines += ["| | RP-KL ↓ | decode tok/s ↑ | prefill 4K tok/s ↑ | peak GPU GiB ↓ | holdout | FG-2 |",
                  "|---|---:|---:|---:|---:|---|---:|",
                  f"| **this PR** | {row['rp_kl']:.4f} | {row['decode_tps']:.1f} | {row['prefill_tps']:,.0f} | "
                  f"{row['peak_gpu_gib']:.2f} | {row['holdout'] or 'not run'} | {100 * (row['frontier_gain'] or 0):.3f}% |"]
        inc = next((r for r in frontier["internal"] if r["name"] == frontier["incumbent"]), None)
        if inc:
            lines.append(f"| V0 incumbent | {inc['rp_kl']:.4f} | {inc['decode_tps']:.1f} | {inc['prefill_tps']:,.0f} | "
                         f"{inc['peak_gpu_gib']:.2f} | — | — |")
        if cmp:
            k = cmp["rp_kl"]
            lines += ["", f"Paired RP-KL vs V0: **{k['delta']:+.4f}** nats/token, 95% CI [{k['ci95'][0]:+.4f}, {k['ci95'][1]:+.4f}]"
                      f"{' (significant)' if k['significant'] else ' (not significant)'}; decode {100 * cmp['decode_tps']['rel']:+.1f}%, "
                      f"prefill {100 * cmp['prefill_tps']['rel']:+.1f}%, peak GPU {cmp['peak_gpu_gib']['delta']:+.2f} GiB."]
        if row["gate_failures"]:
            lines += ["", "Gate failures:", *[f"- {g}" for g in row["gate_failures"]]]
    if notes:
        lines += ["", *[f"- {n}" for n in notes]]
    if timings:
        lines += ["", "GPU time: " + " · ".join(f"{k.removesuffix('_seconds')} {v / 60:.1f} min" for k, v in timings.items())]
    lines += ["", f"<!-- bittrellis-result {json.dumps({'name': name, 'id': cid, 'status': LABELS[label][0], 'row': row, 'screen': screen})} -->"]
    return "\n".join(lines)


# ── evaluator ──────────────────────────────────────────────────────────────────────────────────


class Evaluator:
    def __init__(self, gh: GitHub, args):
        from bittrellis.track import load_track

        self.gh, self.args = gh, args
        self.root = Path(args.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.track = load_track("HPC-01")
        self.cfg = self.track["evaluation"]["screen"]
        self.epoch = self.track["evaluation"]["epoch"]
        self.obs = G.Observations(self.root)
        self.accepted = self.root / "accepted"
        self.accepted.mkdir(exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state: dict = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        self.state.setdefault("_merged", {})
        secret = self.root / "secret.txt"
        if not secret.exists():
            secret.write_text(secrets.token_hex(32) + "\n")
            secret.chmod(0o600)
        self.secret = secret.read_text().strip()
        self.probe_seed = int(hashlib.sha256(f"{self.secret}:probe".encode()).hexdigest()[:8], 16)
        self.py = [sys.executable, "-m", "bittrellis.cli"]
        self.env_args = ["--base", args.base, "--shipped", args.shipped, "--unsloth", args.unsloth]
        self.sources = {"base": Path(args.base), "gittensor_nvfp4": Path(args.shipped), "unsloth_nvfp4": Path(args.unsloth)}

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2) + "\n")

    # ---- known results --------------------------------------------------------------------

    def _artifacts(self, base: Path) -> list[Path]:
        return sorted(p for p in base.iterdir() if (p / "candidate.json").exists()) if base.exists() else []

    def known_ids(self) -> dict[str, str]:
        out = {}
        for p in self._artifacts(Path(self.args.seeds)) + self._artifacts(self.accepted):
            c = json.loads((p / "candidate.json").read_text())
            if c.get("kind") in ("internal", "candidate"):
                out[c["id"]] = c["name"]
        return out

    def seed_memory(self) -> list[tuple[dict[str, str], float]]:
        out = []
        for p in self._artifacts(Path(self.args.seeds)):
            c = json.loads((p / "candidate.json").read_text())
            perf = p / "performance.json"
            if c.get("kind") in ("internal", "candidate") and perf.exists():
                out.append((G.keys_from_expanded(c["manifest"]["expanded"]), json.loads(perf.read_text())["peak_gpu_gib"]))
        return out

    def live_heads(self, open_prs: list[dict]) -> dict[int, str]:
        heads = {p["number"]: p["head"]["sha"] for p in open_prs}
        heads.update({int(k): v for k, v in self.state["_merged"].items()})
        return heads

    # ---- one pass -------------------------------------------------------------------------

    def run_once(self) -> None:
        self.sync_merged()
        open_prs = self.gh.paged("/pulls?state=open&sort=created&direction=asc")
        first_seen = {}
        for pr in open_prs:  # observe everything before evaluating anything
            o = self.obs.observe(pr["number"], pr["user"]["login"], pr["head"]["sha"])
            first_seen[(pr["number"], pr["head"]["sha"])] = o["first_seen"]
        self.rerank(open_prs)  # first, so a PR lifted by a closed reference resumes in this pass
        for pr in queue_order(open_prs, first_seen):
            key = f"{pr['number']}-{pr['head']['sha'][:12]}"
            entry = self.state.get(key, {})
            status = entry.get("status")
            if status and status not in RESCREEN and status != "resume" and not (status == "error" and entry.get("errors", 0) < MAX_ERRORS):
                continue
            try:
                self.evaluate(pr, open_prs, resume=status == "resume")
            except Exception as e:  # noqa: BLE001 - one broken PR must not stop the queue
                self.state[key] = {**self.state.get(key, {}), "status": "error", "error": repr(e),
                                   "errors": entry.get("errors", 0) + 1}
                try:
                    self.gh.set_status_label(pr["number"], "error")
                except urllib.error.URLError:
                    pass
                traceback.print_exc()
            self.save()
        self.rerank(open_prs)
        self.save()

    def sync_merged(self) -> None:
        """Copy artifacts of merged, frontier-moving PRs into accepted/ so later PRs are ranked against them."""
        for pr in self.gh.paged("/pulls?state=closed&sort=updated&direction=desc")[:100]:
            if not pr.get("merged_at"):
                continue
            self.state["_merged"][str(pr["number"])] = pr["head"]["sha"]
            entry = self.state.get(f"{pr['number']}-{pr['head']['sha'][:12]}")
            if entry and entry.get("status") == "frontier" and not (self.accepted / entry["name"]).exists():
                shutil.copytree(entry["artifact"], self.accepted / entry["name"])

    def evaluate(self, pr: dict, open_prs: list[dict], resume: bool = False) -> None:
        number, sha, author = pr["number"], pr["head"]["sha"], pr["user"]["login"]
        key = f"{number}-{sha[:12]}"
        me = self.obs.observe(number, author, sha)
        work = self.root / "prs" / key
        work.mkdir(parents=True, exist_ok=True)
        log = work / "eval.log"
        labels = {lab["name"] for lab in pr["labels"]}
        files = [f["filename"] for f in self.gh.paged(f"/pulls/{number}/files")]
        kind, manifests = classify(files)
        base_entry = {"pr": number, "head": sha, "author": author, "first_seen": me["first_seen"], "kind": kind}

        def finish(status: str, label: str | None = None, body: str | None = None, **extra) -> None:
            self.state[key] = {**self.state.get(key, {}), **base_entry, "status": status, **extra}
            if label:
                self.gh.set_status_label(number, label)
            if body:
                self.gh.comment(number, body)

        if kind in ("evaluator", "other") or not manifests:
            return finish(kind, "evaluator", "BitTrellis evaluator: this PR changes evaluator or other protected paths "
                                             "(or adds no manifest), so it is not evaluated automatically. A maintainer will review it.")
        if kind == "code" and APPROVED not in labels:
            if self.state.get(key, {}).get("status") != "needs-approval":
                finish("needs-approval", "needs_approval")
            return None

        # queue share: an author's PRs wait their turn; nothing is rejected
        pending = [self.obs.observe(p["number"], p["user"]["login"], p["head"]["sha"]) for p in open_prs
                   if self.state.get(f"{p['number']}-{p['head']['sha'][:12]}", {}).get("status") in (None, "queued")]
        q = G.judge_queue(me, pending, self.cfg)
        if q.stop:
            if self.state.get(key, {}).get("status") != "queued":
                finish("queued", "queued")
            return None

        code = work / "code"
        if code.exists():
            run(["git", "worktree", "remove", "--force", str(code)], REPO_ROOT, log)
            shutil.rmtree(code, ignore_errors=True)
        run(["git", "fetch", "--quiet", "origin", f"pull/{number}/head", "main"], REPO_ROOT, log)
        run(["git", "worktree", "add", "--force", "--detach", str(code), sha if kind == "code" else "origin/main"], REPO_ROOT, log)
        try:
            return self._evaluate_in(pr, me, kind, manifests, labels, work, code, log, finish, open_prs, resume)
        finally:
            run(["git", "worktree", "remove", "--force", str(code)], REPO_ROOT, log)

    def _evaluate_in(self, pr, me, kind, manifests, labels, work, code, log, finish, open_prs, resume):
        from bittrellis import fingerprint as F
        from bittrellis.model.qwen38 import Qwen38Arch

        number, sha = pr["number"], pr["head"]["sha"]
        notes: list[str] = []
        manifest = work / "manifest.yaml"
        manifest.write_text(subprocess.run(["git", "show", f"{sha}:{manifests[0]}"], cwd=REPO_ROOT,
                                           capture_output=True, text=True).stdout)
        ids = work / "ids.json"
        if run(self.py + ["manifest", str(manifest), "--ids-out", str(ids), "--shipped", self.args.shipped], code, log) != 0 or not ids.exists():
            return finish("invalid", "invalid", "BitTrellis evaluator: the manifest does not validate:\n\n```\n"
                          + log.read_text()[-3000:] + "\n```")
        ident = json.loads(ids.read_text())
        cid, keys = ident["id"], ident["keys"]
        units = Qwen38Arch().units()
        numel = {u.id: u.numel for u in units}

        # ---- screen: duplicates and near-copies (by expanded recipe) ----
        live = self.live_heads(open_prs)
        earlier = [o for o in self.obs.all() if live.get(o["pr"]) == o["head"]]
        verdict = G.judge_manifest(me, cid, keys, numel, earlier, self.known_ids(), self.cfg, self.epoch,
                                   cleared=COPY_CLEARED in labels)
        self.obs.annotate(number, sha, candidate_id=cid, keys=keys, epoch=self.epoch, derivative_of=verdict.derivative_of)
        screen = {"manifest": verdict.to_dict()}
        if verdict.outcome == "duplicate":
            return finish("duplicate", "duplicate", f"BitTrellis evaluator: **not measured**. This recipe (`{cid}`) is "
                          f"{verdict.reason}. Identical recipes earn nothing, however their rules are written.", screen=screen)
        if verdict.outcome == "copy-review":
            if self.state.get(f"{number}-{sha[:12]}", {}).get("status") != "copy-review":
                finish("copy-review", "copy-review", f"BitTrellis evaluator: **waiting for a maintainer**. This is "
                       f"{verdict.reason}. A maintainer adds `{COPY_CLEARED}` to measure it.", screen=screen)
            return None
        if verdict.derivative_of:
            self.gh.add_label(number, EXTRA_LABELS["derivative"][0])
            notes.append(f"Close to #{verdict.derivative_of['pr']} by @{verdict.derivative_of['author']}, observed earlier "
                         f"({verdict.details['share_different']:.2%} of weights differ). Ranked with that result on the frontier, "
                         "so this PR is credited only for what it adds.")

        # ---- screen: memory ----
        mem = G.judge_memory(keys, units, self.seed_memory(), self.cfg)
        screen["memory"] = mem.to_dict()
        if mem.stop:
            return finish("memory", "memory", f"BitTrellis evaluator: **not measured**. {mem.reason}; it would run out of "
                          "memory on the 32 GB card.", screen=screen)

        # ---- screen: new quantizers, by probe bytes ----
        new_refs: list[str] = []
        if kind == "code":
            pr_probe, pr_repeat = work / "probe.npz", work / "probe-repeat.npz"
            if run(self.py + ["fingerprint", "--seed", str(self.probe_seed), "--out", str(pr_probe), "--repeat-out", str(pr_repeat)],
                   code, log, timeout=1800) != 0:
                return finish("build", "build", "BitTrellis evaluator: the quantizer probe failed to run:\n\n```\n"
                              + log.read_text()[-3000:] + "\n```", screen=screen)
            main_fp = F.by_quantizer(F.probe(None, self.probe_seed))
            mine, again = F.by_quantizer(F.load_probe(pr_probe)), F.by_quantizer(F.load_probe(pr_repeat))
            new = {ref: fp for ref, fp in mine.items() if ref not in main_fp}
            new_refs = sorted(new)
            known = {f"{ref} (main)": fp for ref, fp in main_fp.items()}
            probes = self.root / "probes"
            probes.mkdir(exist_ok=True)
            for o in earlier:
                path = probes / f"pr-{o['pr']:06d}-{o['head'][:12]}-{self.probe_seed}.npz"
                if o["pr"] != number and o["author"].lower() != me["author"].lower() and o["first_seen"] < me["first_seen"] and path.exists():
                    for ref, fp in F.by_quantizer(F.load_probe(path)).items():
                        known[f"{ref} (#{o['pr']})"] = fp
            qv = G.judge_quantizers(new, known, self.cfg, repeat={r: again.get(r) for r in new})
            screen["quantizers"] = {**qv.to_dict(), "new": new_refs}
            if qv.stop:
                return finish(qv.outcome, qv.outcome, f"BitTrellis evaluator: **not measured**. {qv.reason}.", screen=screen)
            if new:
                F.save_probe({f"{ref}|{k}": v for ref, fp in new.items() for k, v in fp.items()},
                             probes / f"pr-{number:06d}-{sha[:12]}-{self.probe_seed}.npz")
            self.obs.annotate(number, sha, quantizers=new_refs)

        # ---- build and audit (CPU) ----
        ckpt, art = work / "checkpoint", work / "artifact"
        if not (ckpt / "bittrellis_build.json").exists():
            shutil.rmtree(ckpt, ignore_errors=True)
            if run(self.py + ["build", str(manifest), "--out", str(ckpt)] + self.env_args, code, log) != 0:
                return finish("build", "build", "BitTrellis evaluator: the checkpoint did not build:\n\n```\n"
                              + log.read_text()[-3000:] + "\n```", screen=screen)

        # ---- screen: the new encoder's real stored bytes ----
        if new_refs:
            sv = self._sketch_guard(me, number, sha, keys, new_refs, units, ckpt, earlier)
            screen["stored_bytes"] = sv.to_dict()
            if sv.stop:
                return finish(sv.outcome, sv.outcome, f"BitTrellis evaluator: **not measured**. {sv.reason}.", screen=screen)

        ev = self.py + ["evaluate", str(ckpt), "--out", str(art), "--sparkinfer", self.args.sparkinfer,
                        "--reference", self.args.reference] + self.env_args
        reuse = ["--audit-json", str(art / "audit.json")]
        gates = self.track["gates"]
        skipped: list[str] = []
        if not resume:
            # ---- stage 1: quality ----
            rc = run(ev + ["--stages", "quality"], code, log)
            cand = json.loads((art / "candidate.json").read_text()) if (art / "candidate.json").exists() else {}
            if cand.get("audit_ok") is False:
                audit = json.loads((art / "audit.json").read_text())
                return finish("audit", "audit", "BitTrellis evaluator: **audit failed**.\n\n" + "\n".join(f"- {e}" for e in audit["errors"][:20]),
                              screen=screen)
            if rc != 0:
                raise RuntimeError("quality stage failed")
            fails = quality_gate_failures(json.loads((art / "quality.json").read_text()), gates)
            if fails:
                self._delete(ckpt)
                return finish("gate", "gate", render_comment(cand["name"], cand["id"], None, None, "gate",
                              notes + ["Quality gates failed; speed runs, tasks and holdout skipped:", *fails], screen,
                              self._timings(art)), screen=screen, artifact=str(art), name=cand["name"])
            # ---- stage 2: performance ----
            if run(ev + ["--stages", "performance"] + reuse, code, log) != 0:
                raise RuntimeError("performance stage failed")
            frontier, row, refs = self._rank(me, art, open_prs, work)
            if status_from_row(row) != "frontier":
                skipped = ["tasks", "holdout"]
                label = status_from_row(row)
                notes.append("Tasks and private holdout skipped: they can only fail a result, and this one is already "
                             f"{'dominated' if label == 'dominated' else 'invalid'} on the measured objectives.")
                return self._report(pr, cand, art, ckpt, frontier, label, notes, screen, refs, skipped, finish, work)
        # ---- stage 3: tasks and holdout ----
        cand = json.loads((art / "candidate.json").read_text())
        if run(ev + ["--stages", "tasks"] + reuse, code, log) != 0:
            raise RuntimeError("tasks stage failed")
        if self.args.private:
            incumbent = Path(self.args.seeds) / self.track["frontier"]["incumbent"]
            (art / "holdout.json").unlink(missing_ok=True)
            t0 = time.time()
            run(self.py + ["holdout", "check", str(ckpt), "--private", self.args.private, "--artifact", str(art),
                           "--incumbent-artifact", str(incumbent), "--shipped", self.args.shipped, "--sparkinfer", self.args.sparkinfer],
                REPO_ROOT, log)  # always trusted code: contributed code must never see the private holdout
            if not (art / "holdout.json").exists():  # a crash must never read as "no holdout failure"
                raise RuntimeError("holdout check did not produce a verdict")
            timings = self._timings(art)
            timings["holdout_seconds"] = round(time.time() - t0, 1)
            (art / "timings.json").write_text(json.dumps(timings) + "\n")
        else:
            notes.append("private holdout not configured on this evaluator; result is provisional")
        frontier, row, refs = self._rank(me, art, open_prs, work)
        return self._report(pr, cand, art, ckpt, frontier, status_from_row(row), notes, screen, refs, skipped, finish, work)

    # ---- helpers --------------------------------------------------------------------------

    def _sketch_guard(self, me, number, sha, keys, new_refs, units, ckpt, earlier) -> G.Verdict:
        from bittrellis import fingerprint as F

        mine_units = [u for u in units if keys[u.id].split("@", 1)[1].split("+")[0] in new_refs]
        mine_units.sort(key=lambda u: hashlib.sha256(f"{self.secret}:{u.id}".encode()).hexdigest())
        sample = mine_units[:4]
        prefixes = [lin.prefix for u in sample for lin in u.linears]
        pr_sketch = F.sketch(ckpt, prefixes, self.secret)
        known = {f"{ref} (main)": sk for ref, sk in
                 F.reference_sketches(self.sources, sample, {u.id: keys[u.id].split("@", 1)[0] for u in sample}, self.secret).items()}
        sketches = self.root / "sketches"
        sketches.mkdir(exist_ok=True)
        for o in earlier:
            path = sketches / f"pr-{o['pr']:06d}-{o['head'][:12]}.npz"
            if o["pr"] != number and o["author"].lower() != me["author"].lower() and o["first_seen"] < me["first_seen"] and path.exists():
                known[f"#{o['pr']}"] = F.load_sketches(path)
        verdict = G.judge_quantizers({"the stored bytes of " + ", ".join(new_refs): pr_sketch}, known, self.cfg)
        F.save_sketches(F.sketch(ckpt, [lin.prefix for u in mine_units for lin in u.linears][:64], self.secret),
                        sketches / f"pr-{number:06d}-{sha[:12]}.npz")
        return verdict

    def _rank(self, me: dict, art: Path, open_prs: list[dict], work: Path) -> tuple[dict, dict, list[str]]:
        refs = reference_entries(me, self.state, self.live_heads(open_prs))
        frontier_json = work / "frontier.json"
        paths = [self.args.seeds, str(self.accepted)] + [self.state[k]["artifact"] for k in refs] + [str(art)]
        if subprocess.run(self.py + ["frontier", *paths, "--out", str(frontier_json)], cwd=REPO_ROOT,
                          capture_output=True, text=True).returncode != 0:
            raise RuntimeError("frontier ranking failed")
        frontier = json.loads(frontier_json.read_text())
        cand = json.loads((art / "candidate.json").read_text())
        row = next(r for r in frontier["internal"] if r["id"] == cand["id"])
        return frontier, row, refs

    def _report(self, pr, cand, art, ckpt, frontier, label, notes, screen, refs, skipped, finish, work):
        cmp_out = subprocess.run(self.py + ["compare", str(Path(self.args.seeds) / self.track["frontier"]["incumbent"]), str(art)],
                                 cwd=REPO_ROOT, capture_output=True, text=True)
        cmp = json.loads(cmp_out.stdout) if cmp_out.returncode == 0 else None
        if refs:
            notes.append("Ranked with earlier open PRs on the frontier: " + ", ".join(f"#{self.state[k]['pr']}" for k in refs) + ".")
        body = render_comment(cand["name"], cand["id"], frontier, cmp, label, notes, screen, self._timings(art))
        self._delete(ckpt)  # a skipped result that a later re-rank lifts is rebuilt (deterministic, CPU)
        return finish(label, label, body, artifact=str(art), candidate=cand["id"], name=cand["name"], references=refs,
                      skipped=skipped, screen=screen)

    def _timings(self, art: Path) -> dict:
        p = art / "timings.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _delete(self, ckpt: Path) -> None:
        if not self.args.keep_checkpoints:
            shutil.rmtree(ckpt, ignore_errors=True)

    def rerank(self, open_prs: list[dict]) -> None:
        """Re-rank measured open PRs whose set of earlier references changed (CPU only)."""
        live = self.live_heads(open_prs)
        for pr in open_prs:
            key = f"{pr['number']}-{pr['head']['sha'][:12]}"
            e = self.state.get(key)
            if (not e or e.get("status") not in RANKED or not e.get("artifact") or "first_seen" not in e
                    or not (Path(e["artifact"]) / "performance.json").exists()):
                continue
            me = {"pr": e["pr"], "author": e["author"], "first_seen": e["first_seen"]}
            refs = reference_entries(me, self.state, live)
            if refs == e.get("references", []):
                continue
            work = self.root / "prs" / key
            try:
                frontier, row, refs = self._rank(me, Path(e["artifact"]), open_prs, work)
            except (RuntimeError, StopIteration, OSError):
                traceback.print_exc()
                continue
            label = status_from_row(row)
            e["references"] = refs
            if label == e["status"]:
                continue
            if label == "frontier" and e.get("skipped"):
                e["status"] = "resume"  # measured tasks and holdout never ran; the next pass rebuilds and finishes it
                self.gh.comment(pr["number"], "BitTrellis evaluator: an earlier PR this result was ranked against has "
                                "closed, so it is no longer dominated. Resuming: tasks and private holdout.")
                continue
            e["status"] = label
            self.gh.set_status_label(pr["number"], label)
            self.gh.comment(pr["number"], f"BitTrellis evaluator: re-ranked after the set of earlier PRs changed. Status is now "
                            f"**{LABELS[label][0]}**, FG-2 {100 * (row['frontier_gain'] or 0):.3f}%.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", required=True, help="evaluator working directory (state, records, artifacts)")
    ap.add_argument("--token-env", default="GITHUB_TOKEN")
    ap.add_argument("--base", default=str(REPO_ROOT / "models/Qwen3.8-27B"))
    ap.add_argument("--shipped", default=str(REPO_ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090"))
    ap.add_argument("--unsloth", default=str(REPO_ROOT / "models/Qwen3.8-27B-NVFP4-unsloth"))
    ap.add_argument("--sparkinfer", default=str(REPO_ROOT / "third_party/sparkinfer"))
    ap.add_argument("--reference", default=str(REPO_ROOT / "data/reference/hpc01-public-v2-k256"))
    ap.add_argument("--seeds", default=str(REPO_ROOT / "results/feasibility/artifacts"))
    ap.add_argument("--private", help="private holdout directory (see bittrellis/holdout.py)")
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--interval", type=int, default=600)
    args = ap.parse_args()

    gh = GitHub(args.repo, os.environ[args.token_env])
    gh.ensure_labels()
    ev = Evaluator(gh, args)
    while True:
        ev.run_once()
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
