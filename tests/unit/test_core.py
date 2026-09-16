import numpy as np
import pytest

from bittrellis.eval.logits import compare, summarize
from bittrellis.frontier.pareto import Row, apply_gates, dominates, frontier_gain, hypervolume, pareto, rank
from bittrellis.manifest import Manifest, ManifestError
from bittrellis.model.qwen38 import Qwen38Arch
from bittrellis.precision import FP8, NVFP4, Q4_K, S_BF16, S_FP8, S_NVFP4, resolve
from bittrellis.quant import formats as F
from bittrellis.runtime import ScoreDump, parse_score, parse_sweep

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
    d = {"schema": "bittrellis/precision-manifest@1", "track": "HPC-01", "name": "t", "default": "NVFP4"}
    d.update(kw)
    return Manifest.from_dict(d)


def test_manifest_rules_and_overrides():
    m = _m(rules=[{"match": "L*.gdn.*", "layers": "40-63", "precision": "FP8"},
                  {"match": "L*.mlp", "precision": "Q4_K"}], modules={"L0.mlp": "NVFP4"})
    exp = m.expand(UNITS)
    assert exp["L40.gdn.z"] == "FP8" and exp["L0.gdn.z"] == "NVFP4"
    assert exp["L1.mlp"] == "Q4_K" and exp["L0.mlp"] == "NVFP4"
    assert len(exp) == 273


@pytest.mark.parametrize("bad", [
    {"default": "BF16"},
    {"default": "FP8"},                                                    # FP8 is GDN-only
    {"rules": [{"match": "L*.attn.*", "precision": "FP8"}]},               # would silently run Q4_K
    {"rules": [{"match": "L*.nothing", "precision": "Q4_K"}]},
    {"modules": {"L3.gdn.qkv": "FP8"}},                                    # layer 3 is attention
    {"quantizers": {"NVFP4": "gptq"}},
])
def test_manifest_rejects_undeployable(bad):
    with pytest.raises(ManifestError):
        _m(**bad).expand(UNITS)


def test_candidate_id_ignores_rule_spelling():
    a = _m(rules=[{"match": "L*.mlp", "precision": "Q4_K"}])
    b = _m(default="Q4_K", rules=[{"match": "L*.gdn.*", "precision": "NVFP4"}, {"match": "L*.attn.*", "precision": "NVFP4"}],
           modules={"lm_head": "NVFP4"})
    assert a.candidate_id(UNITS) == b.candidate_id(UNITS)


# ---------------------------------------------------------------- runtime output parsing


def test_parse_score_and_sweep():
    text = ("[load] ok\nS i=5 tgt=7 am=7 lp=-0.100000 top=7:-0.100000,3:-2.500000\n"
            "S i=6 tgt=1 am=4 lp=-3.000000 top=4:-0.200000,1:-3.000000\nPPL 1.2 over 2 positions\n")
    d = parse_score(text)
    assert d.pos.tolist() == [5, 6] and d.top_ids.shape == (2, 2) and d.argmax.tolist() == [7, 4]
    sweep = ("VRAM used    : 29.7 GB\ndecode tg    : 93.60 tok/s\n"
             'SWEEP_JSON {"128":{"decode_tps":95.7,"prefill_pp":6942.0},"4096":{"decode_tps":93.6,"prefill_pp":14364.0}}\n')
    s = parse_sweep(sweep)
    assert s["sweep"][4096]["decode_tps"] == 93.6 and s["vram_gib_reported"] == 29.7


# ---------------------------------------------------------------- quality metrics


def _dump(pos, top_ids, top_lp, argmax=None, lp_t=None):
    top_ids = np.asarray(top_ids, np.int32)
    return ScoreDump(np.asarray(pos, np.int32), np.zeros(len(pos), np.int32),
                     np.asarray(argmax if argmax is not None else top_ids[:, 0], np.int32),
                     np.asarray(lp_t if lp_t is not None else np.full(len(pos), -1.0), np.float32),
                     top_ids, np.asarray(top_lp, np.float32))


def test_kl_zero_for_identical_and_positive_otherwise():
    lp = np.log([[0.7, 0.2, 0.05]])
    ref = _dump([0], [[1, 2, 3]], lp)
    same = compare(ref, _dump([0], [[1, 2, 3]], lp), {"id": "s", "category": "c", "ids": [0, 1]})
    assert same.kl[0] < 1e-9 and same.top1[0]
    other = compare(ref, _dump([0], [[2, 1, 3]], np.log([[0.6, 0.3, 0.05]])), {"id": "s", "category": "c", "ids": [0, 1]})
    assert other.kl[0] > 0.1 and not other.top1[0]
    s = summarize([same, other])
    assert s["positions"] == 2 and 0 < s["kl"]


def test_needles():
    ids = [9, 9, 4, 5]
    stream = {"id": "long", "category": "long", "ids": ids, "needles": [{"name": "red", "depth": 0.1, "target_positions": [2, 3]}]}
    ref = _dump([1, 2], [[4, 0], [5, 0]], np.log([[0.9, 0.1], [0.9, 0.1]]))
    good = compare(ref, _dump([1, 2], [[4, 0], [5, 0]], np.log([[0.9, 0.1], [0.9, 0.1]])), stream)
    bad = compare(ref, _dump([1, 2], [[4, 0], [6, 0]], np.log([[0.9, 0.1], [0.9, 0.1]])), stream)
    assert good.needles["red"]["candidate"] and not bad.needles["red"]["candidate"]
    assert summarize([bad])["needle_recall"] == 0.0


# ---------------------------------------------------------------- frontier


BOX = {"kl": [0.0, 0.3], "decode_tps": [60.0, 120.0], "prefill_tps": [2000.0, 20000.0], "vram_gib": [14.0, 32.0]}
EPS = {"kl": 0.005, "decode_tps": 1.0, "prefill_tps": 450.0, "vram_gib": 0.1}
GATES = {"kl_max": 0.3, "top1_min": 0.8, "needle_min": 1.0, "task_max_drop_items": 6}


def _row(id_, kl, dec, pre, vram, **kw):
    return Row(id_, id_, "candidate", kl=kl, decode_tps=dec, prefill_tps=pre, vram_gib=vram, **kw)


def test_dominance_and_frontier():
    a = _row("a", 0.05, 90, 15000, 22)
    b = _row("b", 0.07, 90, 15000, 22)     # dominated by a
    c = _row("c", 0.10, 100, 15000, 22)    # trade-off
    assert dominates(a, b, EPS) and not dominates(a, c, EPS)
    assert {r.id for r in pareto([a, b, c], EPS)} == {"a", "c"}
    rank([a, b, c], BOX, EPS)
    assert b.gain == 0 and a.gain > 0 and c.gain > 0


def test_noise_does_not_dominate():
    r0 = _row("r0", 0.1269, 94.0, 15238, 22.01)
    wobble = _row("w", 0.1266, 94.3, 15100, 22.01)          # all differences inside epsilon
    assert not dominates(wobble, r0, EPS) and not dominates(r0, wobble, EPS)
    prefill_trade = _row("p", 0.1266, 94.3, 8443, 20.83)    # saves VRAM, pays prefill
    assert not dominates(prefill_trade, r0, EPS) and not dominates(r0, prefill_trade, EPS)


def test_hypervolume_exact():
    assert hypervolume([(1, 1, 1)]) == pytest.approx(1.0)
    assert hypervolume([(0.5, 1, 1), (1, 0.5, 1)]) == pytest.approx(0.75)
    assert hypervolume([(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)]) == pytest.approx(0.125)
    assert hypervolume([(0.5, 0.5, 0.5, 0.5), (1, 0.2, 1, 1)]) == pytest.approx(0.0625 + 0.2 - 0.5 * 0.2 * 0.5 * 0.5)


def test_gates_block_quality_trades():
    fast_but_broken = _row("x", 0.5, 119, 19000, 19, top1=0.6, needle_recall=0.5)
    fails = apply_gates(fast_but_broken, GATES, None)
    assert len(fails) == 3 and frontier_gain(fast_but_broken, [], BOX) == 0.0
