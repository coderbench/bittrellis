"""End-to-end CLI wiring on the tiny synthetic model (no GPU, no network)."""

import json

import yaml

from bittrellis.cli import main


def _manifest(path, **kw):
    d = {"schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "cli-test", "default": "NVFP4"}
    d.update(kw)
    path.write_text(yaml.safe_dump(d))
    return path


def test_cli_build_audit_describe_inventory_search(tiny_all, tmp_path, capsys):
    base, shipped, unsloth = tiny_all
    src = ["--base", str(base), "--shipped", str(shipped), "--unsloth", str(unsloth)]
    m = _manifest(tmp_path / "m.yaml", rules=[{"match": "L*.gdn.*", "format": "FP8"},
                                              {"match": "L*.mlp", "layers": "0-2", "format": "NVFP4", "quantizer": "unsloth"}])
    assert main(["manifest", str(m), "--shipped", str(shipped)]) == 0
    out = tmp_path / "ckpt"
    assert main(["build", str(m), "--out", str(out), "--no-verify", *src]) == 0
    assert main(["audit", str(out), "--fast", *src]) == 0
    assert "AUDIT PASS" in capsys.readouterr().out
    assert main(["describe", str(out)]) == 0
    assert main(["inventory", "--shipped", str(shipped), "--out", str(tmp_path / "inv.json")]) == 0
    inv = json.loads((tmp_path / "inv.json").read_text())
    head = next(u for u in inv["units"] if u["unit"] == "lm_head")
    assert any(a["execution"]["decode_b1"] == "Q4_K(nvfp4)" for a in head["legal_assignments"])
    assert main(["search", "neighbors", str(m), "--out", str(tmp_path / "prop"), "--shipped", str(shipped), "--limit", "3"]) == 0
    assert len(list((tmp_path / "prop").glob("*.yaml"))) == 3
    assert main(["quantizers"]) == 0


def test_cli_manifest_flags_duplicates(tiny_all, tmp_path, capsys):
    _, shipped, _ = tiny_all
    existing = tmp_path / "existing"
    existing.mkdir()
    _manifest(existing / "a.yaml", name="a", rules=[{"match": "L*.mlp", "format": "Q4_K"}])
    new = _manifest(tmp_path / "b.yaml", name="b", default="Q4_K",
                    rules=[{"match": "L*.gdn.*", "format": "NVFP4"}, {"match": "L*.attn.*", "format": "NVFP4"}],
                    modules={"lm_head": "NVFP4"})
    assert main(["manifest", str(new), "--against", str(existing), "--shipped", str(shipped)]) == 1
    assert "duplicate of" in capsys.readouterr().out


def test_cli_rejects_undeployable_manifest(tiny_all, tmp_path, capsys):
    _, shipped, _ = tiny_all
    bad = _manifest(tmp_path / "bad.yaml", rules=[{"match": "L*.attn.*", "format": "FP8"}])
    assert main(["manifest", str(bad), "--shipped", str(shipped)]) == 1
    assert "cannot run" in capsys.readouterr().out
