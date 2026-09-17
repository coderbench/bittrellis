import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("sandbox", Path(__file__).resolve().parents[2] / "evaluator/sandbox.py")
sandbox = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sandbox)


def test_sandbox_command_drops_the_environment(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    monkeypatch.setenv("LANG", "C.UTF-8")
    cmd = sandbox.command("bt-sandbox", ["python", "-m", "bittrellis.cli", "build"], "/home/bt-sandbox", "/venv/bin:/usr/bin")
    assert cmd[:6] == ["runuser", "-u", "bt-sandbox", "--", "env", "-i"]
    joined = " ".join(cmd)
    assert "ghp_secret" not in joined and "GITHUB_TOKEN" not in joined
    assert "CUDA_VISIBLE_DEVICES=" in cmd and "HOME=/home/bt-sandbox" in cmd and "LANG=C.UTF-8" in cmd
    assert cmd[-4:] == ["python", "-m", "bittrellis.cli", "build"]
