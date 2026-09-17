import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from bittrellis import fingerprint as F
from bittrellis import quantizers as Q
from bittrellis.manifest import Manifest
from bittrellis.model.qwen38 import Qwen38Arch
from bittrellis.quantizers.builtin import RTN
from bittrellis.track import load_track

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("guards", ROOT / "evaluator/guards.py")
G = importlib.util.module_from_spec(spec)
sys.modules["guards"] = G
spec.loader.exec_module(G)

CFG = load_track("HPC-01")["evaluation"]["screen"]
EPOCH = "hpc01-e2"
UNITS = Qwen38Arch().units()
NUMEL = {u.id: u.numel for u in UNITS}


def keys(**kw):
    d = {"schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "m", "default": "NVFP4"}
    d.update(kw)
    m = Manifest.from_dict(d)
    asg = m.expand_assignments(UNITS)
    return m.candidate_id(UNITS), {k: a.key() for k, a in asg.items()}


def obs(pr, author, t, cid=None, k=None, **extra):
    return {"pr": pr, "author": author, "head": f"{pr:040d}", "first_seen": t, "candidate_id": cid, "keys": k, **extra}


def test_observations_are_write_once(tmp_path):
    o = G.Observations(tmp_path)
    first = o.observe(7, "alice", "a" * 40, now="2026-09-17T10:00:00Z")
    again = o.observe(7, "alice", "a" * 40, now="2026-09-18T10:00:00Z")
    assert again["first_seen"] == first["first_seen"]
    o.annotate(7, "a" * 40, first_seen="1999-01-01T00:00:00Z", author="mallory", candidate_id="x")
    rec = o.all()[0]
    assert rec["first_seen"] == "2026-09-17T10:00:00Z" and rec["author"] == "alice" and rec["candidate_id"] == "x"
    # a force-pushed head is a new observation with its own time
    assert o.observe(7, "alice", "b" * 40, now="2026-09-19T10:00:00Z")["first_seen"] == "2026-09-19T10:00:00Z"


def test_rewritten_rules_are_the_same_recipe():
    a = keys(rules=[{"match": "L*.mlp", "layers": "0-55", "format": "NVFP4", "quantizer": "unsloth"}])
    b = keys(rules=[{"match": "L*.mlp", "layers": "0-27", "format": "NVFP4", "quantizer": "unsloth"},
                    {"match": "L*.mlp", "layers": "28-55", "format": "NVFP4", "quantizer": "unsloth"}])
    assert a == b


def test_duplicates_known_and_earlier():
    cid, k = keys(rules=[{"match": "L*.gdn.qkv", "layers": "48-63", "format": "FP8"}])
    me = obs(3, "bob", "2026-09-17T12:00:00Z")
    assert G.judge_manifest(me, cid, k, NUMEL, [], {cid: "V9"}, CFG, EPOCH).outcome == "duplicate"
    earlier = [obs(2, "alice", "2026-09-17T11:00:00Z", cid, k)]
    v = G.judge_manifest(me, cid, k, NUMEL, earlier, {}, CFG, EPOCH)
    assert v.outcome == "duplicate" and v.original["pr"] == 2
    later = [obs(4, "alice", "2026-09-17T13:00:00Z", cid, k)]
    assert G.judge_manifest(me, cid, k, NUMEL, later, {}, CFG, EPOCH).outcome == "pass"


def test_near_copy_is_measured_but_marked_derivative():
    cid_a, ka = keys(rules=[{"match": "L*.gdn.qkv", "layers": "48-63", "format": "FP8"}])
    cid_b, kb = keys(rules=[{"match": "L*.gdn.qkv", "layers": "48-63", "format": "FP8"}], modules={"L3.attn.o": "Q4_K"})
    assert cid_a != cid_b and G.share_different(ka, kb, NUMEL) < CFG["near_copy_max_share"]
    original = [obs(2, "alice", "2026-09-17T11:00:00Z", cid_a, ka)]
    v = G.judge_manifest(obs(3, "bob", "2026-09-17T12:00:00Z"), cid_b, kb, NUMEL, original, {}, CFG, EPOCH)
    assert v.outcome == "pass" and v.derivative_of["pr"] == 2
    # iterating on your own work is never a copy
    v = G.judge_manifest(obs(3, "alice", "2026-09-17T12:00:00Z"), cid_b, kb, NUMEL, original, {}, CFG, EPOCH)
    assert v.outcome == "pass" and v.derivative_of is None


def test_repeated_near_copies_wait_for_a_maintainer():
    cid_a, ka = keys(rules=[{"match": "L*.gdn.qkv", "layers": "48-63", "format": "FP8"}])
    cid_b, kb = keys(rules=[{"match": "L*.gdn.qkv", "layers": "48-63", "format": "FP8"}], modules={"L3.attn.o": "Q4_K"})
    history = [obs(2, "alice", "2026-09-17T11:00:00Z", cid_a, ka)]
    history += [obs(10 + i, "bob", f"2026-09-17T11:3{i}:00Z", "z", ka, derivative_of={"pr": 2}, epoch=EPOCH) for i in range(2)]
    me = obs(20, "bob", "2026-09-17T12:00:00Z")
    assert G.judge_manifest(me, cid_b, kb, NUMEL, history, {}, CFG, EPOCH).outcome == "copy-review"
    assert G.judge_manifest(me, cid_b, kb, NUMEL, history, {}, CFG, EPOCH, cleared=True).outcome == "pass"
    assert G.judge_manifest(me, cid_b, kb, NUMEL, history, {}, CFG, "hpc01-e3").outcome == "pass"  # counted per epoch


def test_memory_guard_uses_measured_seed_overhead():
    seeds = []
    for p in sorted((ROOT / "results/feasibility/artifacts").iterdir()):
        c, perf = p / "candidate.json", p / "performance.json"
        if c.exists() and perf.exists() and json.loads(c.read_text())["kind"] == "internal":
            seeds.append((G.keys_from_expanded(json.loads(c.read_text())["manifest"]["expanded"]),
                          json.loads(perf.read_text())["peak_gpu_gib"]))
    assert len(seeds) >= 5
    for k, _ in seeds:  # every measured seed fits by construction
        assert G.judge_memory(k, UNITS, seeds, CFG).outcome == "pass"
    all_fp8 = {u.id: "FP8@rtn@v1" for u in UNITS}
    assert G.judge_memory(all_fp8, UNITS, seeds, CFG).outcome == "memory"


def test_keys_from_expanded_match_manifest_keys():
    cid, k = keys(rules=[{"match": "L*.mlp", "layers": "0-55", "format": "NVFP4", "quantizer": "unsloth"}])
    m = Manifest.from_dict({"schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "m", "default": "NVFP4",
                            "rules": [{"match": "L*.mlp", "layers": "0-55", "format": "NVFP4", "quantizer": "unsloth"}]})
    assert G.keys_from_expanded(m.to_dict(UNITS)["expanded"]) == k


def test_queue_share_holds_later_prs():
    pending = [obs(i, "bob", f"2026-09-17T10:0{i}:00Z") for i in range(3)]
    assert G.judge_queue(obs(9, "bob", "2026-09-17T11:00:00Z"), pending, CFG).outcome == "queued"
    assert G.judge_queue(obs(9, "alice", "2026-09-17T11:00:00Z"), pending, CFG).outcome == "pass"
    assert G.judge_queue(obs(1, "bob", "2026-09-17T10:01:00Z"), pending, CFG).outcome == "pass"


class Renamed(RTN):
    name = "definitely_new"


class Clipped(RTN):
    name = "clipped"

    def encode(self, ctx, unit, lin, fmt):
        out = super().encode(ctx, unit, lin, fmt)
        return [(s, dt, sh, np.frombuffer(bytes(np.asarray(d, dtype=np.uint8).tobytes()), dtype=np.uint8) ^ 0x11
                 if s == ".weight" else d) for s, dt, sh, d in out]


class Flaky(RTN):
    name = "flaky"
    calls = 0

    def encode(self, ctx, unit, lin, fmt):
        Flaky.calls += 1
        out = super().encode(ctx, unit, lin, fmt)
        return [(s, dt, sh, np.frombuffer(np.asarray(d).tobytes(), dtype=np.uint8) ^ (Flaky.calls & 0xFF)
                 if s == ".weight" else d) for s, dt, sh, d in out]


@pytest.fixture()
def extra_quantizers():
    added = [Q.register(Renamed()), Q.register(Clipped()), Q.register(Flaky())]
    yield
    for q in added:
        Q.REGISTRY.pop(q.name)


def test_quantizer_guard_judges_bytes_not_names(extra_quantizers):
    fp = F.by_quantizer(F.probe(None, seed=11))
    main = {"rtn@v1 (main)": fp["rtn@v1"]}
    v = G.judge_quantizers({"definitely_new@v1": fp["definitely_new@v1"]}, main, CFG)
    assert v.outcome == "same-encoder" and v.details["matches"][0]["same_as"] == "rtn@v1 (main)"
    assert G.judge_quantizers({"clipped@v1": fp["clipped@v1"]}, main, CFG).outcome == "pass"
    again = F.by_quantizer(F.probe(["flaky"], seed=11))
    v = G.judge_quantizers({"flaky@v1": fp["flaky@v1"]}, main, CFG, repeat={"flaky@v1": again["flaky@v1"]})
    assert v.outcome == "nondeterministic"


def test_sketches_compare_stored_bytes_across_layouts(tiny_all):
    base, baseline, ct = tiny_all
    units = Qwen38Arch.from_config(json.loads((base / "config.json").read_text())).units()
    mlp = [u for u in units if u.kind == "mlp" and u.layer == 0]
    prefixes = [lin.prefix for u in mlp for lin in u.linears]
    stored = F.sketch(ct, prefixes, "s3cret")                   # unsloth layout: weight_packed
    refs = F.reference_sketches({"base": base, "gittensor_nvfp4": baseline, "unsloth_nvfp4": ct}, mlp,
                                {u.id: "NVFP4" for u in mlp}, "s3cret")
    assert F.similarity(stored, refs["unsloth@v1"])[0] == 1.0
    assert set(refs) >= {"baseline@v1", "unsloth@v1", "rtn@v1"}
    assert F.similarity(F.sketch(ct, prefixes, "s3cret", 16), F.sketch(ct, prefixes, "other", 16))[0] < 0.5  # offsets follow the secret


def test_observations_in_one_pass_are_strictly_ordered(tmp_path):
    o = G.Observations(tmp_path)
    times = [o.observe(i, "x", f"{i:040d}")["first_seen"] for i in range(5)]
    assert times == sorted(times) and len(set(times)) == 5
