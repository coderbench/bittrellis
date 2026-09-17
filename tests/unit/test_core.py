import numpy as np
import pytest

from bittrellis import quantizers as Q
from bittrellis.eval.logits import compare, paired_delta, summarize
from bittrellis.frontier.pareto import Row, apply_gates, dominates, frontier_gain, hypervolume, pareto, rank
from bittrellis.manifest import Manifest, ManifestError, distance
from bittrellis.model.qwen38 import Qwen38Arch
from bittrellis.precision import FP8, NVFP4, Q4_K, S_BF16, S_FP8, S_NVFP4, execution, resolve
from bittrellis.quant import formats as F
from bittrellis.runtime import RefScore, ScoreDump, parse_sweep, read_refscore_output, write_refscore_inputs
from bittrellis.search import neighbors

ARCH = Qwen38Arch()
UNITS = ARCH.units()
BY_ID = {u.id: u for u in UNITS}


# ---------------------------------------------------------------- architecture


def test_qwen38_units():
    kinds = {}
    for u in UNITS:
        kinds[u.kind] = kinds.get(u.kind, 0) + 1
    assert kinds == {"gdn": 144, "attn": 64, "mlp": 64, "lm_head": 1}
    assert [i for i in range(64) if not ARCH.is_linear(i)] == list(range(3, 64, 4))
    assert BY_ID["L0.gdn.qkv"].linears[0].rows == 10240
    assert BY_ID["L3.attn.q"].linears[0].rows == 12288
    assert sum(len(u.linears) for u in UNITS if u.kind != "lm_head") == 400  # ModelOpt's "400 Linears"


# ---------------------------------------------------------------- formats


def test_bf16_roundtrip_exhaustive():
    u = np.arange(65536, dtype=np.uint32).astype("<u2")
    f = F.bf16_to_f32(u)
    finite = np.isfinite(f)
    assert (F.f32_to_bf16(f[finite]) == u[finite]).all()


def test_e4m3_table_and_encode():
    assert F.E4M3_TABLE[0x7E] == 448.0 and np.isnan(F.E4M3_TABLE[0x7F])
    x = np.array([0.0, 1.0, -1.0, 448.0, 1e6, -1e6, 0.3], np.float32)
    back = F.e4m3_decode(F.e4m3_encode(x))
    assert back[3] == 448.0 and back[4] == 448.0 and back[5] == -448.0
    assert abs(back[6] - 0.3) < 0.02


def test_nvfp4_roundtrip_error_and_layout():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 256)).astype(np.float32) * 0.02
    packed, scale, ws2 = F.quantize_nvfp4(w)
    assert packed.shape == (64, 128) and scale.shape == (64, 16)
    assert np.isclose(ws2, np.abs(w).max() / (6 * 448))
    rel = np.linalg.norm(F.dequantize_nvfp4(packed, scale, ws2) - w) / np.linalg.norm(w)
    assert rel < 0.12
    codes = F.unpack_nibbles(packed)
    assert (F.pack_nibbles(codes) == packed).all()


def test_fp8_per_channel():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 64)).astype(np.float32)
    codes, scale = F.quantize_fp8_per_channel(w)
    assert scale.shape == (16, 1) and scale.dtype == np.dtype("<u2")
    rel = np.linalg.norm(F.dequantize_fp8_per_channel(codes, scale) - w) / np.linalg.norm(w)
    assert rel < 0.03


# ---------------------------------------------------------------- precision space + loader rules


@pytest.mark.parametrize("unit,stored,label", [
    ("L0.gdn.qkv", [S_NVFP4], NVFP4),
    ("L0.gdn.qkv", [S_FP8], FP8),
    ("L0.gdn.qkv", [S_BF16], Q4_K),
    ("L3.attn.q", [S_FP8], "Q4_K(fp8)"),   # unsloth's FP8 attention runs as a Q4_K refit
    ("L3.attn.q", [S_NVFP4], NVFP4),
    ("L0.mlp", [S_FP8] * 3, "Q4_K(fp8)"),
    ("L0.mlp", [S_BF16] * 3, Q4_K),
    ("lm_head", [S_NVFP4], NVFP4),
    ("lm_head", [S_FP8], "Q4_K(fp8)"),
])
def test_resolver(unit, stored, label):
    assert resolve(BY_ID[unit], stored).label == label


def test_mlp_gate_nvfp4_requires_up_down():
    r = resolve(BY_ID["L0.mlp"], [S_NVFP4, S_NVFP4, S_BF16])
    assert r.precision is None


# ---------------------------------------------------------------- manifests


def _m(**kw):
    d = {"schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "t", "default": "NVFP4"}
    d.update(kw)
    return Manifest.from_dict(d)


def test_manifest_rules_and_overrides():
    m = _m(rules=[{"match": "L*.gdn.*", "layers": "40-63", "format": "FP8"},
                  {"match": "L*.mlp", "format": "Q4_K"}], modules={"L0.mlp": "NVFP4"})
    exp = m.expand(UNITS)
    assert exp["L40.gdn.z"] == "FP8" and exp["L0.gdn.z"] == "NVFP4"
    assert exp["L1.mlp"] == "Q4_K" and exp["L0.mlp"] == "NVFP4"
    assert len(exp) == 273


@pytest.mark.parametrize("bad", [
    {"default": "BF16"},
    {"default": "FP8"},                                                    # FP8 is GDN-only
    {"rules": [{"match": "L*.attn.*", "format": "FP8"}]},                  # would silently run Q4_K
    {"rules": [{"match": "L*.nothing", "format": "Q4_K"}]},
    {"modules": {"L3.gdn.qkv": "FP8"}},                                    # layer 3 is attention
    {"quantizers": {"NVFP4": "gptq"}},                                     # not a registered quantizer
    {"rules": [{"match": "L*.mlp", "format": "Q4_K", "quantizer": "rtn"}]},  # Q4_K has no quantizer choice
    {"rules": [{"match": "L*.gdn.*", "format": "FP8", "quantizer": "unsloth"}]},  # unsloth only produces NVFP4
])
def test_manifest_rejects_undeployable(bad):
    with pytest.raises(ManifestError):
        _m(**bad).expand(UNITS)


def test_candidate_id_ignores_rule_spelling():
    a = _m(rules=[{"match": "L*.mlp", "format": "Q4_K"}])
    b = _m(default="Q4_K", rules=[{"match": "L*.gdn.*", "format": "NVFP4"}, {"match": "L*.attn.*", "format": "NVFP4"}],
           modules={"lm_head": "NVFP4"})
    assert a.candidate_id(UNITS) == b.candidate_id(UNITS)
    c = _m(rules=[{"match": "L*.mlp", "format": "Q4_K"}], modules={"L0.gdn.z": {"format": "NVFP4", "quantizer": "rtn"}})
    assert c.candidate_id(UNITS) != a.candidate_id(UNITS)
    assert distance(a.expand_assignments(UNITS), c.expand_assignments(UNITS)) == 1


def test_v1_manifests_still_read():
    m = Manifest.from_dict({"schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "old",
                            "default": "NVFP4", "rules": [{"match": "L*.mlp", "precision": "Q4_K"}]})
    assert m.expand(UNITS)["L0.mlp"] == "Q4_K"


def test_expanded_manifest_records_execution_semantics():
    d = _m().to_dict(UNITS)["expanded"]
    assert d["lm_head"]["execution"]["decode_b1"] == "Q4_K(nvfp4)"
    assert d["lm_head"]["quantizer"] == "baseline@v1" and d["L0.mlp"]["execution"]["decode_b1"] == "NVFP4"
    assert execution("lm_head", Q4_K)["decode_b1"] == "Q4_K"
    assert Q.get("runtime").formats == (Q4_K,)


def test_neighbors_are_legal_and_distinct():
    base = _m()
    ns = neighbors(base, UNITS)
    assert ns and len({m.candidate_id(UNITS) for m in ns}) == len(ns)
    assert all(m.candidate_id(UNITS) != base.candidate_id(UNITS) for m in ns)
    assert not any("unsloth" in m.name and "gdn" in m.name for m in ns)


# ---------------------------------------------------------------- runtime I/O


def test_parse_sweep_and_refscore_binary(tmp_path):
    sweep = ("VRAM used    : 29.7 GB\ndecode tg    : 93.60 tok/s\n"
             'SWEEP_JSON {"128":{"decode_tps":95.7,"prefill_pp":6942.0},"4096":{"decode_tps":93.6,"prefill_pp":14364.0}}\n')
    s = parse_sweep(sweep)
    assert s["sweep"][4096]["decode_tps"] == 93.6 and s["vram_gib_reported"] == 29.7
    ref_ids = np.array([[5, 7], [1, 2], [9, 3]], np.int32)
    write_refscore_inputs([1, 2, 3, 4, 5], ref_ids, tmp_path / "t.bin", tmp_path / "r.bin")
    assert np.fromfile(tmp_path / "r.bin", "<i4")[0] == 2
    out = tmp_path / "o.bin"
    with open(out, "wb") as fh:
        fh.write(b"BTRS" + np.array([3, 2], "<i4").tobytes() + np.array([5, 1, 9], "<i4").tobytes()
                 + np.array([-0.1, -0.2, -0.3], "<f4").tobytes() + np.arange(6, dtype="<f4").tobytes())
    rs = read_refscore_output(out, prefix_len=1)
    assert rs.pos.tolist() == [1, 2, 3] and rs.lp_ref.shape == (3, 2) and rs.lp_ref[2, 1] == 5.0


# ---------------------------------------------------------------- Reference-Partition KL


def _ref(pos, top_ids, top_lp, lp_t=None):
    top_ids = np.asarray(top_ids, np.int32)
    return ScoreDump(np.asarray(pos, np.int32), np.zeros(len(pos), np.int32), top_ids[:, 0].copy(),
                     np.asarray(lp_t if lp_t is not None else np.full(len(pos), -1.0), np.float32),
                     top_ids, np.asarray(top_lp, np.float32))


def _cand(pos, argmax, lp_ref, lp_t=None):
    return RefScore(np.asarray(pos, np.int32), np.asarray(argmax, np.int32),
                    np.asarray(lp_t if lp_t is not None else np.full(len(pos), -1.0), np.float32),
                    np.asarray(lp_ref, np.float32))


def test_rp_kl_zero_for_identical_positive_otherwise_and_bounded_by_full_kl():
    p = np.array([0.6, 0.3, 0.05, 0.05])            # full distribution over a 4-token vocab
    q = np.array([0.5, 0.2, 0.25, 0.05])
    full_kl = float((p * np.log(p / q)).sum())
    ref = _ref([0], [[0, 1]], np.log([p[:2]]))     # partition: top-2 + tail {2, 3}
    stream = {"id": "s", "category": "c", "ids": [0, 1]}
    same = compare(ref, _cand([0], [0], np.log([p[:2]])), stream)
    assert same.kl[0] < 1e-9 and same.top1[0]
    other = compare(ref, _cand([0], [0], np.log([q[:2]])), stream)
    projected = 0.6 * np.log(0.6 / 0.5) + 0.3 * np.log(0.3 / 0.2) + 0.1 * np.log(0.1 / 0.3)
    assert other.kl[0] == pytest.approx(projected, rel=1e-5)
    assert other.kl[0] <= full_kl + 1e-9
    assert other.outside_mass[0] == pytest.approx(0.3, rel=1e-5)
    s = summarize([same, other], k=2)
    assert s["metric"] == "reference-partition-kl" and s["positions"] == 2 and s["partition_k"] == 2


def test_rp_kl_rejects_misaligned_partitions():
    ref = _ref([0, 1], [[0, 1], [0, 1]], np.log([[0.5, 0.4], [0.5, 0.4]]))
    with pytest.raises(ValueError):
        compare(ref, _cand([0], [0], np.log([[0.5, 0.4]])), {"id": "s", "category": "c", "ids": [0, 1, 2]})


def test_needles_and_guard_counts():
    ids = [9, 9, 4, 5]
    stream = {"id": "long-8k", "category": "long", "ids": ids,
              "needles": [{"name": "red", "depth": 0.1, "target_positions": [2, 3]}]}
    ref = _ref([1, 2], [[4, 0], [5, 0]], np.log([[0.9, 0.05], [0.9, 0.05]]))
    good = compare(ref, _cand([1, 2], [4, 5], np.log([[0.9, 0.05], [0.9, 0.05]])), stream)
    bad = compare(ref, _cand([1, 2], [4, 6], np.log([[0.9, 0.05], [0.2, 0.05]])), stream)
    assert good.needles["red"]["candidate"] and not bad.needles["red"]["candidate"]
    assert summarize([bad], 2)["needles_by_length"]["long-8k"] == {"required": 1, "retrieved": 0}


def test_paired_delta_cancels_common_noise():
    rng = np.random.default_rng(0)
    common = rng.exponential(1.0, 4000) * (rng.random(4000) < 0.05) * 20
    d = paired_delta({"s.kl": common + 0.05}, {"s.kl": common + 0.06}, n_boot=300)
    assert abs(d["delta"] - 0.01) < 1e-9 and d["significant"]


# ---------------------------------------------------------------- frontier

BOX = {"rp_kl": [0.0, 0.3], "decode_tps": [60.0, 120.0], "prefill_tps": [2000.0, 20000.0], "peak_gpu_gib": [14.0, 32.0]}
FLOORS = {"rp_kl": 0.002, "decode_tps": 0.01, "prefill_tps": 0.03, "peak_gpu_gib": 0.1}
GUARD = {"required_success": {"long-8k": 1.0}}
GATES = {"rp_kl_max": 0.3, "top1_min": 0.8, "task_max_drop_items": 6, "long_context_guard": GUARD}
NEEDLES = {"long-8k": {"required": 3, "retrieved": 3}}


def _row(name, kl, dec, pre, mem, kind="internal", **kw):
    kw.setdefault("needles_by_length", NEEDLES)
    kw.setdefault("audit_ok", True if kind == "internal" else None)
    return Row(name, name, kind, rp_kl=kl, decode_tps=dec, prefill_tps=pre, peak_gpu_gib=mem, **kw)


def plain_quality(a, b):
    return -1 if a.rp_kl < b.rp_kl - 0.002 else (1 if a.rp_kl > b.rp_kl + 0.002 else 0)


def test_dominance_frontier_and_gain():
    a = _row("a", 0.05, 90, 15000, 22)
    b = _row("b", 0.07, 90, 15000, 22)     # dominated by a
    c = _row("c", 0.10, 100, 15000, 22)    # trade-off
    assert dominates(a, b, FLOORS, plain_quality) and not dominates(a, c, FLOORS, plain_quality)
    assert {r.id for r in pareto([a, b, c], FLOORS, plain_quality)} == {"a", "c"}
    rank([a, b, c], BOX, FLOORS, plain_quality)
    assert b.gain == 0 and a.gain > 0 and c.gain > 0


def test_noise_and_measured_spread_do_not_dominate():
    v0 = _row("v0", 0.1269, 94.0, 15238, 22.01)
    wobble = _row("w", 0.1266, 94.3, 15100, 22.01)
    assert not dominates(wobble, v0, FLOORS, plain_quality) and not dominates(v0, wobble, FLOORS, plain_quality)
    noisy = _row("n", 0.1269, 96.5, 15238, 22.01, decode_spread=0.04)   # +2.7% but its own runs differ by 4%
    assert not dominates(noisy, v0, FLOORS, plain_quality)
    prefill_trade = _row("p", 0.1266, 94.3, 8443, 20.83)
    assert not dominates(prefill_trade, v0, FLOORS, plain_quality) and not dominates(v0, prefill_trade, FLOORS, plain_quality)


def test_external_references_never_rank():
    v0 = _row("v0", 0.127, 94, 15238, 22)
    ext = _row("R2", 0.05, 99, 19000, 15, kind="external")
    for r in (v0, ext):
        apply_gates(r, GATES, None)
    rank([v0, ext], BOX, FLOORS, plain_quality)
    assert v0.frontier and not ext.frontier and ext.gain == 0.0
    assert frontier_gain(ext, [v0], BOX) == 0.0


def test_hypervolume_exact():
    assert hypervolume([(1, 1, 1)]) == pytest.approx(1.0)
    assert hypervolume([(0.5, 1, 1), (1, 0.5, 1)]) == pytest.approx(0.75)
    assert hypervolume([(0.5, 0.5, 0.5, 0.5), (1, 0.2, 1, 1)]) == pytest.approx(0.0625 + 0.2 - 0.5 * 0.2 * 0.5 * 0.5)


def test_gates():
    broken = _row("x", 0.5, 119, 19000, 19, top1=0.6, needles_by_length={"long-8k": {"required": 3, "retrieved": 2}},
                  correctness_ok=False, holdout="FAIL")
    fails = apply_gates(broken, GATES, None)
    assert len(fails) == 5 and frontier_gain(broken, [], BOX) == 0.0
    unaudited = _row("u", 0.1, 90, 15000, 22, audit_ok=None)
    assert any("audit" in f for f in apply_gates(unaudited, GATES, None))
    tasks = {"suites": {"gsm8k": {"passed": 20, "n": 33}}}
    weak = _row("t", 0.1, 90, 15000, 22, tasks={"suites": {"gsm8k": {"passed": 10, "n": 33}}})
    assert any("task guard" in f for f in apply_gates(weak, GATES, tasks))
