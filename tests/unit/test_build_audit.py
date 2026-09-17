import json

import numpy as np
import pytest

from bittrellis import quantizers as Q
from bittrellis.build import build
from bittrellis.manifest import Manifest
from bittrellis.model.qwen38 import Qwen38Arch
from bittrellis.precision import FP8, NVFP4, S_BF16, S_FP8, S_NVFP4, resolve_checkpoint, stored_format
from bittrellis.quantizers.base import f32_weight
from bittrellis.quantizers.builtin import RTN
from bittrellis.safetensors_io import SafeTensorsDir
from bittrellis.track import load_track
from bittrellis.validate import audit, describe

QUIET = {"log": lambda *_: None, "verify": False}


def manifest(**kw):
    d = {"schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "t", "default": "NVFP4"}
    d.update(kw)
    return Manifest.from_dict(d)


def sources(tiny_all):
    base, bl, ct = tiny_all
    return {"base": base, "gittensor_nvfp4": bl, "unsloth_nvfp4": ct}


def run_audit(out, m, tiny_all, **kw):
    return audit(out, m, sources(tiny_all), verify_sources=False, **kw)


def corrupt(path, name, rng_seed=1, flip_one=False):
    with SafeTensorsDir(path) as ck:
        t = ck.get(name)
    with open(t.file, "r+b") as fh:
        fh.seek(t.offset)
        if flip_one:
            b = fh.read(1)
            fh.seek(t.offset)
            fh.write(bytes([b[0] ^ 0x01]))
        else:
            fh.write(np.random.default_rng(rng_seed).integers(0, 256, t.nbytes, dtype=np.uint8).tobytes())


@pytest.fixture()
def track():
    return load_track("HPC-01")


def test_baseline_rebuild_is_byte_identical(tiny_all, tmp_path, track):
    out = tmp_path / "v0"
    build(manifest(), track, sources(tiny_all), out, **QUIET)
    with SafeTensorsDir(out) as a, SafeTensorsDir(tiny_all[1]) as b:
        assert set(a.tensors) == set(b.tensors)
        for name in b.tensors:
            assert bytes(a.raw(name)) == bytes(b.raw(name)), name
    res = run_audit(out, manifest(), tiny_all)
    assert res.ok, res.errors
    assert res.lineage["baseline@v1"]["lineage"] == "attested"


def test_mixed_build_resolves_and_audits(tiny_all, tmp_path, track):
    m = manifest(rules=[{"match": "L*.gdn.*", "format": "FP8"}, {"match": "L*.mlp", "layers": "1-2", "format": "Q4_K"}],
                 modules={"lm_head": "Q4_K", "L3.attn.o": "Q4_K"})
    out = tmp_path / "mixed"
    rec = build(m, track, sources(tiny_all), out, **QUIET)
    assert rec["summary"]["gdn"] == {"FP8": 9}
    labels = describe(out)
    assert labels["L0.gdn.qkv"] == "FP8" and labels["L1.mlp"] == "Q4_K" and labels["L0.mlp"] == "NVFP4"
    assert labels["L3.attn.o"] == "Q4_K" and labels["L3.attn.q"] == "NVFP4" and labels["lm_head"] == "Q4_K"
    res = run_audit(out, m, tiny_all)
    assert res.ok, res.errors
    assert res.lineage["rtn@v1"]["replay_mode"] == "independent"
    assert "quantization_config" in json.loads((out / "config.json").read_text())
    rec2 = build(m, track, sources(tiny_all), tmp_path / "mixed2", **QUIET)
    assert rec2["files"] == rec["files"]  # deterministic


def test_rtn_nvfp4_quantizer_audits(tiny_all, tmp_path, track):
    m = manifest(quantizers={"NVFP4": "rtn"})
    out = tmp_path / "rtn"
    build(m, track, sources(tiny_all), out, **QUIET)
    assert run_audit(out, m, tiny_all).ok


def test_audit_rejects_modified_frozen_tensor(tiny_all, tmp_path, track):
    out = tmp_path / "v0"
    build(manifest(), track, sources(tiny_all), out, **QUIET)
    corrupt(out, "model.language_model.norm.weight", flip_one=True)
    res = run_audit(out, manifest(), tiny_all)
    assert not res.ok and any("norm.weight" in e for e in res.errors)


def test_audit_rejects_attested_bytes_that_do_not_match_source(tiny_all, tmp_path, track):
    out = tmp_path / "v0"
    build(manifest(), track, sources(tiny_all), out, **QUIET)
    corrupt(out, "model.language_model.layers.0.mlp.up_proj.weight", flip_one=True)
    res = run_audit(out, manifest(), tiny_all)
    assert not res.ok and any("up_proj.weight: bytes differ from gittensor_nvfp4" in e for e in res.errors)


def test_audit_rejects_regenerable_bytes_the_quantizer_did_not_produce(tiny_all, tmp_path, track):
    m = manifest(rules=[{"match": "L*.gdn.*", "format": "FP8"}])
    out = tmp_path / "fp8"
    build(m, track, sources(tiny_all), out, **QUIET)
    corrupt(out, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", flip_one=True)
    res = run_audit(out, m, tiny_all)
    assert not res.ok and any("does not match a regeneration by rtn@v1" in e for e in res.errors)


def test_audit_rejects_substituted_bytes_as_anomaly(tiny_all, tmp_path, track):
    out = tmp_path / "v0"
    build(manifest(), track, sources(tiny_all), out, **QUIET)
    corrupt(out, "model.language_model.layers.0.mlp.up_proj.weight")
    res = run_audit(out, manifest(), tiny_all, check_bytes=False)
    assert not res.ok and any("substituted bytes" in e for e in res.errors)


def test_audit_detects_manifest_mismatch(tiny_all, tmp_path, track):
    out = tmp_path / "v0"
    build(manifest(), track, sources(tiny_all), out, **QUIET)
    res = run_audit(out, manifest(modules={"lm_head": "Q4_K"}), tiny_all)
    assert not res.ok and any("lm_head" in e for e in res.errors)


def test_unsloth_quantizer_splices_ct_bytes(tiny_all, tmp_path, track):
    m = manifest(rules=[{"match": "L*.mlp", "layers": "0-2", "format": "NVFP4", "quantizer": "unsloth"}])
    out = tmp_path / "ct"
    rec = build(m, track, sources(tiny_all), out, **QUIET)
    assert rec["summary"]["mlp"] == {"NVFP4@unsloth": 3, "NVFP4": 1}
    with SafeTensorsDir(out) as ck:
        assert ck.get("model.language_model.layers.0.mlp.up_proj.weight_packed") is not None
        assert ck.get("model.language_model.layers.3.mlp.up_proj.weight_scale_2") is not None
    assert describe(out)["L0.mlp"] == "NVFP4"
    res = run_audit(out, m, tiny_all)
    assert res.ok, res.errors
    units = Qwen38Arch.from_config(tiny_all[1] / "config.json").units()
    assert m.candidate_id(units) != manifest().candidate_id(units)


def test_unsloth_quantizer_refuses_units_it_lacks(tiny_all, tmp_path, track):
    m = manifest(rules=[{"match": "L*.mlp", "format": "NVFP4", "quantizer": "unsloth"}])
    with pytest.raises(ValueError, match="no NVFP4 bytes"):
        build(m, track, sources(tiny_all), tmp_path / "x", **QUIET)


class _ToySequential(RTN):
    """FP8 whose scale depends on how many units were encoded before it: only a replay reproduces it."""

    name = "toyseq"
    formats = (FP8,)
    replay_mode = "sequential"

    def begin(self, ctx):
        ctx.state["n"] = 0

    def encode(self, ctx, unit, lin, fmt):
        from bittrellis.quant.formats import quantize_fp8_per_channel

        ctx.state["n"] += 1
        w = f32_weight(ctx, lin) * (1.0 + 0.01 * ctx.state["n"])
        codes, scale = quantize_fp8_per_channel(w)
        return [(".weight", "F8_E4M3", codes.shape, codes), (".weight_scale", "BF16", scale.shape, scale)]


def test_sequential_quantizer_is_replayed_in_pipeline_order(tiny_all, tmp_path, track):
    Q.register(_ToySequential())
    try:
        m = manifest(rules=[{"match": "L*.gdn.*", "format": "FP8", "quantizer": "toyseq"}])
        out = tmp_path / "seq"
        build(m, track, sources(tiny_all), out, **QUIET)
        res = run_audit(out, m, tiny_all)
        assert res.ok, res.errors
        assert res.lineage["toyseq@v1"]["replay_mode"] == "sequential"
    finally:
        Q.REGISTRY.pop("toyseq", None)


def test_loader_rules_on_external_layouts(tiny_all):
    base, bl, _ = tiny_all
    arch = Qwen38Arch.from_config(bl / "config.json")
    with SafeTensorsDir(bl) as ck, SafeTensorsDir(base) as b:
        lin = arch.unit_map()["L0.gdn.qkv"].linears[0]
        assert stored_format(ck, lin)[0] == S_NVFP4
        assert stored_format(b, lin)[0] == S_BF16
        r = resolve_checkpoint(b, arch.units())
        assert r["L3.attn.q"].label == "Q4_K" and r["L0.gdn.z"].label == "Q4_K"
        assert all(stored_format(b, lin)[0] != S_FP8 for u in arch.units() for lin in u.linears)
    assert NVFP4 != FP8


def test_replay_cache_is_used_and_still_catches_tampering(tiny_all, tmp_path, track, monkeypatch):
    monkeypatch.setenv("BITTRELLIS_REPLAY_CACHE", str(tmp_path / "cache"))
    m = manifest(rules=[{"match": "L*.gdn.*", "format": "FP8"}])
    out = tmp_path / "fp8"
    build(m, track, sources(tiny_all), out, **QUIET)
    first = run_audit(out, m, tiny_all)
    assert first.ok and first.lineage["rtn@v1"]["replay_cache_hits"] == 0
    second = run_audit(out, m, tiny_all)
    assert second.ok and second.lineage["rtn@v1"]["replay_cache_hits"] > 0 and second.lineage["rtn@v1"]["replayed_linears"] == 0
    corrupt(out, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", flip_one=True)
    third = run_audit(out, m, tiny_all)
    assert not third.ok
