"""BitTrellis PR evaluator: scores miner pull requests on the pinned RTX 5090 host.

Run it on the evaluation host (cron or a loop):

    GITHUB_TOKEN=... python evaluator/pr_bot.py --repo coderbench/bittrellis --root /workspace/bt-eval \
        --private /secure/holdout-epoch [--once]

What it does for every open PR head it has not evaluated yet:

1. Classifies the PR from its changed files.
   * manifest PR     exactly one added/changed `manifests/*.yaml`, nothing else
                     -> evaluated with the trusted `main` code; only the YAML is read from the PR
   * code PR         changes under bittrellis/quantizers/ or bittrellis/search.py (+ at most one manifest)
                     -> evaluated only after a maintainer adds the `eval-approved` label, because it
                        executes contributed code; runs from a clean worktree of the PR head
   * evaluator paths configs/, data/, bittrellis/eval/, bittrellis/frontier/, bittrellis/validate.py,
                     bittrellis/lineage.py, bittrellis/holdout.py, evaluator/, tools/, .github/
                     -> never evaluated automatically; labelled `bt:touches-evaluator`
2. Validates the manifest and rejects exact duplicates of accepted or open candidates.
3. Builds, audits (lineage + hashes), scores public fidelity, runs the task guard and the two
   performance runs, then the private holdout (PASS/FAIL only).
4. Ranks the result against the accepted internal frontier (seeds + merged candidates) and posts one
   comment with a human table and a machine-readable block, plus a status label.

State lives in <root>/state.json; artifacts in <root>/prs/<number>-<sha>/; accepted artifacts in
<root>/accepted/ (copied when a PR that the bot evaluated is merged).
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_GLOB = "manifests/*.yaml"
CODE_GLOBS = ("bittrellis/quantizers/*", "bittrellis/search.py", "tests/*")
EVALUATOR_GLOBS = ("configs/*", "data/*", "bittrellis/eval/*", "bittrellis/frontier/*", "bittrellis/validate.py",
                   "bittrellis/lineage.py", "bittrellis/holdout.py", "bittrellis/runtime.py", "bittrellis/build.py",
                   "evaluator/*", "tools/*", ".github/*", "scripts/*", "pyproject.toml")
LABELS = {
    "frontier": ("bt:frontier", "0e8a16", "moves the internal frontier"),
    "dominated": ("bt:dominated", "cccccc", "valid but dominated"),
    "gate": ("bt:gate-fail", "d93f0b", "failed a quality or runtime gate"),
    "audit": ("bt:audit-fail", "b60205", "failed the checkpoint audit"),
    "invalid": ("bt:invalid-manifest", "b60205", "manifest does not validate"),
    "duplicate": ("bt:duplicate", "cccccc", "same candidate as an accepted or earlier one"),
    "evaluator": ("bt:touches-evaluator", "5319e7", "changes evaluator paths; maintainer review"),
    "needs_approval": ("bt:needs-approval", "fbca04", "executes contributed code; waiting for eval-approved"),
    "error": ("bt:eval-error", "b60205", "evaluation crashed; maintainers notified"),
}


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
        for name, color, desc in LABELS.values():
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
        self.api("POST", f"/issues/{number}/labels", {"labels": [LABELS[key][0]]})

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


def render_comment(name: str, cid: str, frontier: dict, cmp: dict | None, label: str, notes: list[str]) -> str:
    row = next((r for r in frontier["internal"] if r["name"] == name), None)
    lines = [f"### BitTrellis evaluation · `{name}` · `{cid}`", "",
             f"Epoch `{frontier['evaluator_epoch']}` · {frontier['frontier_gain_version']} · status **{LABELS[label][0]}**", ""]
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
    lines += ["", *[f"- {n}" for n in notes]] if notes else []
    lines += ["", f"<!-- bittrellis-result {json.dumps({'name': name, 'id': cid, 'status': LABELS[label][0], 'row': row})} -->"]
    return "\n".join(lines)


def evaluate_pr(gh: GitHub, pr: dict, args, state: dict) -> None:
    number, sha = pr["number"], pr["head"]["sha"]
    key = f"{number}-{sha[:12]}"
    work = Path(args.root) / "prs" / key
    work.mkdir(parents=True, exist_ok=True)
    log = work / "eval.log"
    files = [f["filename"] for f in gh.paged(f"/pulls/{number}/files")]
    kind, manifests = classify(files)
    labels = {lab["name"] for lab in pr["labels"]}
    if kind in ("evaluator", "other") or not manifests:
        gh.set_status_label(number, "evaluator")
        gh.comment(number, "BitTrellis evaluator: this PR changes evaluator or other protected paths (or adds no manifest), "
                           "so it is not evaluated automatically. A maintainer will review it.")
        state[key] = {"status": kind}
        return
    if kind == "code" and "eval-approved" not in labels:
        gh.set_status_label(number, "needs_approval")
        state[key] = {"status": "needs-approval"}  # re-checked when the label appears (key changes on new commits only)
        return

    # Check out the code that will run: main for manifest PRs, the PR head for approved code PRs.
    code = work / "code"
    if code.exists():
        shutil.rmtree(code)
    ref = sha if kind == "code" else "origin/main"
    run(["git", "fetch", "--quiet", "origin", f"pull/{number}/head", "main"], REPO_ROOT, log)
    run(["git", "worktree", "add", "--force", "--detach", str(code), ref], REPO_ROOT, log)
    manifest_src = subprocess.run(["git", "show", f"{sha}:{manifests[0]}"], cwd=REPO_ROOT, capture_output=True, text=True)
    manifest = work / "manifest.yaml"
    manifest.write_text(manifest_src.stdout)
    py = [sys.executable, "-m", "bittrellis.cli"]
    env_args = ["--base", args.base, "--shipped", args.shipped, "--unsloth", args.unsloth]
    notes: list[str] = []
    try:
        accepted = Path(args.root) / "accepted"
        accepted.mkdir(exist_ok=True)
        main_manifests = work / "main-manifests"
        shutil.rmtree(main_manifests, ignore_errors=True)
        main_manifests.mkdir()
        listing = subprocess.run(["git", "ls-tree", "--name-only", "origin/main", "manifests/"], cwd=REPO_ROOT,
                                 capture_output=True, text=True).stdout.split()
        seeds = subprocess.run(["git", "ls-tree", "--name-only", "origin/main", "experiments/feasibility/variants/"],
                               cwd=REPO_ROOT, capture_output=True, text=True).stdout.split()
        for f in listing + seeds:  # seeds too, so copying a seed is rejected before any GPU time is spent
            if f.endswith(".yaml") and f != manifests[0]:
                (main_manifests / f.replace("/", "__")).write_text(
                    subprocess.run(["git", "show", f"origin/main:{f}"], cwd=REPO_ROOT, capture_output=True, text=True).stdout)
        if run(py + ["manifest", str(manifest), "--against", str(main_manifests)], code, log) != 0:
            gh.set_status_label(number, "invalid")
            gh.comment(number, "BitTrellis evaluator: the manifest does not validate or duplicates an existing one:\n\n```\n"
                               + log.read_text()[-3000:] + "\n```")
            state[key] = {"status": "invalid"}
            return
        ckpt = work / "checkpoint"
        if ckpt.exists():
            shutil.rmtree(ckpt)
        if run(py + ["build", str(manifest), "--out", str(ckpt)] + env_args, code, log) != 0:
            raise RuntimeError("build failed")
        art = work / "artifact"
        rc = run(py + ["evaluate", str(ckpt), "--out", str(art), "--sparkinfer", args.sparkinfer,
                       "--reference", args.reference] + env_args, code, log)
        cand = json.loads((art / "candidate.json").read_text()) if (art / "candidate.json").exists() else {}
        if cand.get("audit_ok") is False:
            gh.set_status_label(number, "audit")
            audit = json.loads((art / "audit.json").read_text())
            gh.comment(number, "BitTrellis evaluator: **audit failed**.\n\n" + "\n".join(f"- {e}" for e in audit["errors"][:20]))
            state[key] = {"status": "audit-fail"}
            return
        if rc != 0:
            raise RuntimeError("evaluate failed")
        known = [p for base in (Path(args.seeds), accepted) for p in base.iterdir() if (p / "candidate.json").exists()]
        dup = next((p.name for p in known if json.loads((p / "candidate.json").read_text())["id"] == cand["id"]), None)
        if dup:
            gh.set_status_label(number, "duplicate")
            gh.comment(number, f"BitTrellis evaluator: this candidate (`{cand['id']}`) is identical to `{dup}`.")
            state[key] = {"status": "duplicate"}
            return
        if args.private:
            incumbent = Path(args.seeds) / "V0-baseline-rebuild"
            (art / "holdout.json").unlink(missing_ok=True)
            run(py + ["holdout", "check", str(ckpt), "--private", args.private, "--artifact", str(art),
                      "--incumbent-artifact", str(incumbent), "--shipped", args.shipped, "--sparkinfer", args.sparkinfer], code, log)
            if not (art / "holdout.json").exists():  # a crash must never read as "no holdout failure"
                raise RuntimeError("holdout check did not produce a verdict")
        else:
            notes.append("private holdout not configured on this evaluator; result is provisional")
        frontier_json = work / "frontier.json"
        run(py + ["frontier", args.seeds, str(accepted), str(art), "--out", str(frontier_json)], code, log)
        frontier = json.loads(frontier_json.read_text())
        cmp_json = subprocess.run(py + ["compare", str(Path(args.seeds) / "V0-baseline-rebuild"), str(art)], cwd=code,
                                  capture_output=True, text=True)
        cmp = json.loads(cmp_json.stdout) if cmp_json.returncode == 0 else None
        row = next(r for r in frontier["internal"] if r["name"] == cand["name"])
        label = "frontier" if row["frontier"] and (row["frontier_gain"] or 0) > 0 else ("gate" if not row["valid"] else "dominated")
        gh.set_status_label(number, label)
        gh.comment(number, render_comment(cand["name"], cand["id"], frontier, cmp, label, notes))
        state[key] = {"status": label, "artifact": str(art), "candidate": cand["id"], "name": cand["name"]}
        if not args.keep_checkpoints:
            shutil.rmtree(ckpt, ignore_errors=True)
    finally:
        run(["git", "worktree", "remove", "--force", str(code)], REPO_ROOT, log)


def sync_merged(gh: GitHub, args, state: dict) -> None:
    """Copy artifacts of merged, frontier-moving PRs into accepted/ so later PRs are ranked against them."""
    accepted = Path(args.root) / "accepted"
    for pr in gh.paged("/pulls?state=closed&sort=updated&direction=desc")[:100]:
        if not pr.get("merged_at"):
            continue
        entry = state.get(f"{pr['number']}-{pr['head']['sha'][:12]}")
        if entry and entry.get("status") == "frontier" and not (accepted / entry["name"]).exists():
            shutil.copytree(entry["artifact"], accepted / entry["name"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", required=True, help="evaluator working directory (state, artifacts)")
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

    import os

    gh = GitHub(args.repo, os.environ[args.token_env])
    gh.ensure_labels()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "state.json"
    while True:
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        sync_merged(gh, args, state)
        for pr in gh.paged("/pulls?state=open&sort=created&direction=asc"):
            key = f"{pr['number']}-{pr['head']['sha'][:12]}"
            done = state.get(key, {}).get("status")
            if done and not (done == "needs-approval" and "eval-approved" in {lab["name"] for lab in pr["labels"]}):
                continue
            try:
                evaluate_pr(gh, pr, args, state)
            except Exception as e:  # noqa: BLE001 - one broken PR must not stop the queue
                state[key] = {"status": "error", "error": repr(e)}
                try:
                    gh.set_status_label(pr["number"], "error")
                except urllib.error.URLError:
                    pass
                traceback.print_exc()
            state_path.write_text(json.dumps(state, indent=2) + "\n")
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
