"""The PR bot end to end with a fake GitHub and fake GPU stages that replay measured seed artifacts."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bittrellis.cli import main as cli_main

ROOT = Path(__file__).resolve().parents[2]
SEEDS = ROOT / "results/feasibility/artifacts"
spec = importlib.util.spec_from_file_location("pr_bot", ROOT / "evaluator/pr_bot.py")
pr_bot = importlib.util.module_from_spec(spec)
sys.modules["pr_bot"] = pr_bot
spec.loader.exec_module(pr_bot)


def manifest_yaml(name, rules):
    return json.dumps({"schema": "bittrellis/manifest@2", "track": "HPC-01", "name": name, "default": "NVFP4", "rules": rules})


class FakeGitHub(pr_bot.GitHub):
    def __init__(self, prs):
        super().__init__("o/r", "t")
        self.prs, self.labels, self.comments = prs, {}, {}

    def paged(self, path):
        if path.startswith("/pulls?state=open"):
            return self.prs
        if path.startswith("/pulls?state=closed") or path == "/labels":
            return []
        number = int(path.split("/")[2])
        return [{"filename": f"manifests/{number}.yaml"}]

    def api(self, method, path, body=None):
        number = int(path.split("/")[2]) if path.startswith("/issues/") else None
        if method == "GET":
            return {"labels": [{"name": n} for n in self.labels.get(number, [])]}
        if method == "DELETE":
            self.labels[number].remove(urllib_unquote(path.rsplit("/", 1)[1]))
        elif path.endswith("/labels"):
            self.labels.setdefault(number, []).extend(body["labels"])
        elif path.endswith("/comments"):
            self.comments.setdefault(number, []).append(body["body"])
        return None


def urllib_unquote(s):
    from urllib.parse import unquote

    return unquote(s)


@pytest.fixture()
def bot(tmp_path, monkeypatch):
    stages_run: list[tuple[int, str]] = []
    recipes: dict[str, tuple[str, Path, dict]] = {}   # sha -> (manifest yaml, seed artifact to replay, perf overrides)

    def fake_run(cmd, cwd, log, timeout=0):
        if cmd[0] == "git":
            return 0
        sub = cmd[3]
        work = Path(cmd[4]).parent if sub in ("manifest", "build") else Path(cmd[cmd.index("--out") + 1]).parent
        sha = next(s for s in recipes if work.name.endswith(s[:12]))
        if sub == "manifest":
            return cli_main(cmd[3:])
        if sub == "build":
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "bittrellis_build.json").write_text("{}")
            return 0
        if sub == "evaluate":
            art = Path(cmd[cmd.index("--out") + 1])
            art.mkdir(parents=True, exist_ok=True)
            ids = json.loads((work / "ids.json").read_text())
            (art / "candidate.json").write_text(json.dumps({"id": ids["id"], "name": ids["name"], "kind": "internal", "audit_ok": True}))
            (art / "audit.json").write_text(json.dumps({"ok": True, "errors": []}))
            stage = cmd[cmd.index("--stages") + 1]
            stages_run.append((int(work.name.split("-")[0]), stage))
            _, src, perf = recipes[sha]
            files = {"quality": ["quality.json", "correctness.json", "kl_positions.npz"], "performance": ["performance.json"],
                     "tasks": ["tasks.json"]}[stage]
            for f in files:
                shutil.copy(src / f, art / f)
            if stage == "performance" and perf:
                p = json.loads((art / "performance.json").read_text())
                p.update(perf)
                (art / "performance.json").write_text(json.dumps(p))
            return 0
        raise AssertionError(cmd)

    real_run = subprocess.run

    def fake_subprocess_run(cmd, *a, **kw):
        if cmd[:2] == ["git", "show"]:
            sha = cmd[2].split(":")[0]
            return subprocess.CompletedProcess(cmd, 0, stdout=recipes[sha][0], stderr="")
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(pr_bot, "run", fake_run)
    monkeypatch.setattr(pr_bot.subprocess, "run", fake_subprocess_run)
    args = SimpleNamespace(root=str(tmp_path / "eval"), base="/nonexistent", shipped="/nonexistent", unsloth="/nonexistent",
                           sparkinfer="/nonexistent", reference="/nonexistent", seeds=str(SEEDS), private=None,
                           keep_checkpoints=False)
    return SimpleNamespace(args=args, recipes=recipes, stages=stages_run)


def pr(number, author, sha):
    return {"number": number, "user": {"login": author}, "head": {"sha": sha}, "labels": []}


RECIPE = [{"match": "L*.mlp", "layers": "0-55", "format": "NVFP4", "quantizer": "unsloth"},
          {"match": "L*.gdn.qkv", "layers": "48-63", "format": "FP8"}]


def test_frontier_duplicate_near_copy_and_staged_skip(bot):
    a, b, c, d = "a" * 40, "b" * 40, "c" * 40, "d" * 40
    faster = {"prefill_tps": 16000.0}
    bot.recipes[a] = (manifest_yaml("alice-calibrated", RECIPE), SEEDS / "V13-mlp-unsloth-bytes", faster)
    # same recipe, rules written differently
    split = [{**RECIPE[0], "layers": "0-27"}, {**RECIPE[0], "layers": "28-55"}, RECIPE[1]]
    bot.recipes[b] = (manifest_yaml("bob-same", split), SEEDS / "V13-mlp-unsloth-bytes", faster)
    # one attention projection different: a near-copy, measuring exactly like the original
    bot.recipes[c] = (json.dumps({**json.loads(manifest_yaml("carol-tweak", RECIPE)), "modules": {"L3.attn.o": "Q4_K"}}),
                      SEEDS / "V13-mlp-unsloth-bytes", faster)
    # a clearly dominated recipe
    bot.recipes[d] = (manifest_yaml("dave-q4k", [{"match": "L*.mlp", "layers": "0-62", "format": "Q4_K"}, {"match": "L*.gdn.*", "format": "Q4_K"}]),
                      SEEDS / "V9-gdn-q4k-mlp-q4k", {})
    gh = FakeGitHub([pr(4, "dave", d), pr(3, "carol", c), pr(2, "bob", b), pr(1, "alice", a)])
    ev = pr_bot.Evaluator(gh, bot.args)
    for i, p in enumerate(sorted(gh.prs, key=lambda p: p["number"])):  # observed in PR order, one second apart
        ev.obs.observe(p["number"], p["user"]["login"], p["head"]["sha"], now=f"2026-09-17T10:00:0{i}Z")
    ev.run_once()

    paid = {f"eval:{t}" for t in pr_bot.TIERS}
    assert {"bt:frontier", "bt:merge-first"} <= set(gh.labels[1]) and len(paid & set(gh.labels[1])) == 1
    assert set(gh.labels[2]) == {"bt:duplicate", "eval:none"}                  # not measured at all
    assert set(gh.labels[3]) == {"bt:derivative", "bt:dominated", "eval:none"}
    assert set(gh.labels[4]) == {"bt:dominated", "eval:none"}
    assert (2, "quality") not in bot.stages
    assert (4, "tasks") not in bot.stages and (3, "tasks") not in bot.stages  # dominated: tasks and holdout skipped
    assert (1, "tasks") in bot.stages
    assert "Ranked with earlier open PRs on the frontier: #1" in gh.comments[3][-1]
    assert "Claude" not in json.dumps(gh.comments)

    # alice closes #1 unmerged: carol's result is re-ranked, becomes non-dominated, and resumes for tasks
    gh.prs = [p for p in gh.prs if p["number"] != 1]
    ev.run_once()
    assert ev.state[f"3-{c[:12]}"]["status"] == "frontier"
    assert (3, "tasks") in bot.stages
    assert {"bt:frontier", "bt:merge-first"} <= set(gh.labels[3]) and len(paid & set(gh.labels[3])) == 1
    assert "eval:none" not in gh.labels[3]
