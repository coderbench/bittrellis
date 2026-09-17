import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"evaluator/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


L, P = _load("ledger"), _load("publish_ledger")
FRONTIER = {"internal": [{"name": "V0", "rp_kl": 0.1357, "decode_tps": 94.9, "prefill_tps": 14760.0,
                          "peak_gpu_gib": 22.02, "tasks_passed": 570, "tasks_n": 784, "frontier": True,
                          "frontier_gain": 0.00193}]}


def test_records_are_write_once_and_carry_no_private_data(tmp_path):
    led = L.Ledger(tmp_path, "hpc01-e3")
    entry = {"pr": 7, "head": "a" * 40, "author": "alice", "first_seen": "t0", "kind": "manifest",
             "status": "frontier", "tier": "L", "candidate": "cid", "name": "mine", "gain": 0.003,
             "artifact": "/workspace/bt-eval/prs/7/artifact"}
    path = led.record(entry, {"rp_kl": 0.12, "holdout": "PASS"})
    doc = json.loads(path.read_text())
    assert doc["tier"] == "L" and doc["row"]["holdout"] == "PASS"
    assert "artifact" not in doc and "/workspace" not in path.read_text()   # no evaluator paths
    before = path.read_text()
    led.record(entry, {"rp_kl": 0.12, "holdout": "PASS"})
    assert path.read_text() == before


def test_frontier_readme_lists_results_and_how_to_check(tmp_path):
    led = L.Ledger(tmp_path, "hpc01-e3")
    led.frontier(FRONTIER)
    readme = (tmp_path / "README.md").read_text()
    assert "570/784" in readme and "0.193%" in readme and "bittrellis frontier hpc01-e3/accepted" in readme
    assert json.loads((tmp_path / "hpc01-e3/frontier.json").read_text()) == FRONTIER


def test_publish_pushes_without_force_and_hides_the_token(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    led = L.Ledger(tmp_path / "ledger", "hpc01-e3")
    led.frontier(FRONTIER)
    seen = []
    real = subprocess.run

    def spy(args, **kw):
        seen.append(args)
        return real(args, **kw)

    monkeypatch.setattr(P.subprocess, "run", spy)
    commit = P.publish(tmp_path / "ledger", str(remote), "ghp_secret", "records: test")
    assert commit and P.publish(tmp_path / "ledger", str(remote), "ghp_secret", "records: test") is None
    assert not any("--force" in a for a in seen for a in a)
    assert not any("ghp_secret" in " ".join(a) for a in seen)                      # never on a command line
    assert "ghp_secret" not in (tmp_path / "ledger/.git/config").read_text()       # nor in a file
    out = real(["git", "--no-pager", "log", "--oneline", "-1", "--name-only", "main"], cwd=remote,
               capture_output=True, text=True).stdout
    assert "hpc01-e3/frontier.json" in out and "records: test" in out


def test_publish_refuses_a_remote_with_credentials(tmp_path):
    with pytest.raises(P.PublishError):
        P.publish(tmp_path, "https://user:token@github.com/o/r.git", None, "m")
