import json

import numpy as np
import pytest

from bittrellis.build import build
from bittrellis.manifest import Manifest
from bittrellis.model.qwen38 import Qwen38Arch
from bittrellis.precision import S_BF16, S_FP8, S_NVFP4, resolve_checkpoint, stored_format
from bittrellis.safetensors_io import SafeTensorsDir
from bittrellis.track import load_track
from bittrellis.validate import audit, describe


def manifest(**kw):
    d = {"schema": "bittrellis/precision-manifest@1", "track": "HPC-01", "name": "t", "default": "NVFP4"}
    d.update(kw)
    return Manifest.from_dict(d)


@pytest.fixture()
def track():
    return load_track("HPC-01")


def test_baseline_rebuild_is_byte_identical(tiny_models, tmp_path, track):
    base, bl = tiny_models
    out = tmp_path / "v0"
    build(manifest(), track, base, bl, out, log=lambda *_: None)
    with SafeTensorsDir(out) as a, SafeTensorsDir(bl) as b:
        assert set(a.tensors) == set(b.tensors)
        for name in b.tensors:
            assert bytes(a.raw(name)) == bytes(b.raw(name)), name
    assert audit(out, manifest(), base, bl).ok


def test_mixed_build_resolves_and_audits(tiny_models, tmp_path, track):
    base, bl = tiny_models
    m = manifest(rules=[{"match": "L*.gdn.*", "precision": "FP8"}, {"match": "L*.mlp", "layers": "1-2", "precision": "Q4_K"}],
                 modules={"lm_head": "Q4_K", "L3.attn.o": "Q4_K"})
    out = tmp_path / "mixed"
    rec = build(m, track, base, bl, out, log=lambda *_: None)
    assert rec["summary"]["gdn"] == {"FP8": 9}
    labels = describe(out)
    assert labels["L0.gdn.qkv"] == "FP8"
    assert labels["L1.mlp"] == "Q4_K" and labels["L0.mlp"] == "NVFP4"
    assert labels["L3.attn.o"] == "Q4_K" and labels["L3.attn.q"] == "NVFP4"
    assert labels["lm_head"] == "Q4_K"
    res = audit(out, m, base, bl)
    assert res.ok, res.errors
    cfg = json.loads((out / "config.json").read_text())
    assert "quantization_config" in cfg
    # determinism: same manifest, same shard hashes
    rec2 = build(m, track, base, bl, tmp_path / "mixed2", log=lambda *_: None)
    assert rec2["files"] == rec["files"]


def test_rtn_nvfp4_quantizer_audits(tiny_models, tmp_path, track):
    base, bl = tiny_models
    m = manifest(quantizers={"NVFP4": "rtn"})
    out = tmp_path / "rtn"
    build(m, track, base, bl, out, log=lambda *_: None)
    assert audit(out, m, base, bl).ok


def test_audit_rejects_modified_norm(tiny_models, tmp_path, track):
    base, bl = tiny_models
    out = tmp_path / "v0"
    build(manifest(), track, base, bl, out, log=lambda *_: None)
    with SafeTensorsDir(out) as ck:
        t = ck.get("model.language_model.norm.weight")
    with open(t.file, "r+b") as fh:
        fh.seek(t.offset)
        b = fh.read(2)
        fh.seek(t.offset)
        fh.write(bytes([b[0] ^ 0x01, b[1]]))
    res = audit(out, manifest(), base, bl)
    assert not res.ok and any("norm.weight" in e for e in res.errors)


def test_audit_rejects_substituted_weights(tiny_models, tmp_path, track):
    """A 'quantized' tensor that encodes different weights fails the fidelity check."""
    base, bl = tiny_models
    out = tmp_path / "v0"
    build(manifest(), track, base, bl, out, log=lambda *_: None)
    name = "model.language_model.layers.0.mlp.up_proj.weight"
    with SafeTensorsDir(out) as ck:
        t = ck.get(name)
    rng = np.random.default_rng(1)
    with open(t.file, "r+b") as fh:
        fh.seek(t.offset)
        fh.write(rng.integers(0, 256, t.nbytes, dtype=np.uint8).tobytes())
    res = audit(out, manifest(), base, bl, check_bytes=False)
    assert not res.ok and any("faithful" in e for e in res.errors)


def test_audit_detects_manifest_mismatch(tiny_models, tmp_path, track):
    base, bl = tiny_models
    out = tmp_path / "v0"
    build(manifest(), track, base, bl, out, log=lambda *_: None)
    res = audit(out, manifest(modules={"lm_head": "Q4_K"}), base, bl)
    assert not res.ok and any("lm_head" in e for e in res.errors)


def test_loader_rules_on_external_layouts(tiny_models):
    base, bl = tiny_models
    arch = Qwen38Arch.from_config(bl / "config.json")
    with SafeTensorsDir(bl) as ck, SafeTensorsDir(base) as b:
        units = arch.unit_map()
        lin = units["L0.gdn.qkv"].linears[0]
        assert stored_format(ck, lin)[0] == S_NVFP4
        assert stored_format(b, lin)[0] == S_BF16
        r = resolve_checkpoint(b, arch.units())
        assert r["L3.attn.q"].label == "Q4_K" and r["L0.gdn.z"].label == "Q4_K"
        assert all(stored_format(b, lin)[0] != S_FP8 for u in arch.units() for lin in u.linears)


def test_unsloth_quantizer_splices_ct_bytes(tiny_all, tmp_path, track):
    base, bl, ct = tiny_all
    m = manifest(rules=[{"match": "L*.mlp", "layers": "0-2", "precision": "NVFP4", "quantizer": "unsloth"}])
    out = tmp_path / "ct"
    rec = build(m, track, base, bl, out, log=lambda *_: None, sources={"unsloth": ct})
    assert rec["summary"]["mlp"] == {"NVFP4@unsloth": 3, "NVFP4": 1}
    with SafeTensorsDir(out) as ck:
        assert ck.get("model.language_model.layers.0.mlp.up_proj.weight_packed") is not None
        assert ck.get("model.language_model.layers.3.mlp.up_proj.weight_scale_2") is not None
    assert describe(out)["L0.mlp"] == "NVFP4"
    res = audit(out, m, base, bl)
    assert res.ok, res.errors
    assert m.candidate_id(Qwen38Arch.from_config(bl / "config.json").units()) != manifest().candidate_id(
        Qwen38Arch.from_config(bl / "config.json").units())


def test_unsloth_quantizer_refuses_units_it_lacks(tiny_all, tmp_path, track):
    base, bl, ct = tiny_all
    m = manifest(rules=[{"match": "L*.mlp", "precision": "NVFP4", "quantizer": "unsloth"}])
    with pytest.raises(ValueError, match="no NVFP4 bytes"):
        build(m, track, base, bl, tmp_path / "x", log=lambda *_: None, sources={"unsloth": ct})
