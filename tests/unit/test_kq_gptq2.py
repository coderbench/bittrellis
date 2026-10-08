import numpy as np
import pytest

from bittrellis import hpc02, kquant
from bittrellis.hpc02 import EncodeContext, Unit
from bittrellis.hpc02_encoders import kq_gptq2 as kg


class _Calib:
    """In-memory stand-in for the pinned HPC-02 statistics (a SafeTensorsDir with get/array)."""

    def __init__(self, tensors):
        self.t = tensors

    def get(self, name):
        return self.t.get(name)

    def array(self, name, dtype):
        return self.t[name].astype(dtype)


def _hessian(seed, n, tokens=2048):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((tokens, n)) @ (rng.standard_normal((n, n)) * 0.15 + np.eye(n))
    x *= np.exp(rng.standard_normal(n) * 0.7)
    return (x.T @ x / tokens).astype(np.float32)


def _rows(seed, shape):
    rng = np.random.default_rng(seed)
    return (rng.standard_t(5, size=shape) * 0.02).astype(np.float32)


def _out_err(w, data, fmt, h):
    e = w.astype(np.float64) - kquant.dequantize(fmt, data, w.shape[1])
    return float(np.einsum("ij,jk,ik->", e, h.astype(np.float64), e))


@pytest.mark.parametrize("fmt", ["Q4_K", "Q5_K"])
def test_dense_gptq_is_repeatable_and_beats_rtn(fmt):
    w, h = _rows(1, (64, 512)), _hessian(1, 512)
    a, b = kg.quantize(w, h, fmt), kg.quantize(w, h, fmt)
    assert a == b and len(a) == kquant.row_bytes(fmt, 512) * 64
    assert _out_err(w, a, fmt, h) < 0.7 * _out_err(w, kquant.RTN[fmt](w), fmt, h)


@pytest.mark.parametrize("fmt", ["Q4_K", "Q5_K"])
def test_fallback_fit_beats_rtn_on_its_own_weights(fmt):
    w = _rows(2, (32, 512))
    ours, rtn = kg._fallback(w, fmt), kquant.RTN[fmt](w)
    assert ours != rtn

    def err(data):
        e = w.astype(np.float64) - kquant.dequantize(fmt, data, 512)
        return float(((w.astype(np.float64) ** 2 + 1e-6) * e * e).sum())

    assert err(ours) < err(rtn)


def test_q8_0_keeps_the_rtn_bytes():
    w = _rows(3, (8, 256))
    ctx = EncodeContext(w, Unit("L0.gdn.qkv", "blk.0.attn_qkv.weight", 0, "gdn.qkv", 256, 8, 2048), {}, None)
    assert kg._encode(ctx, "Q8_0") == kquant.RTN["Q8_0"](w)


def test_q6k_fit_is_valid_repeatable_and_beats_rtn():
    w = _rows(8, (32, 512))
    ours = kg.fit_q6k(w)
    assert ours == kg.fit_q6k(w) and len(ours) == kquant.row_bytes("Q6_K", 512) * 32

    def err(data):
        e = w.astype(np.float64) - kquant.dequantize("Q6_K", data, 512)
        return float(((w.astype(np.float64) ** 2 + 1e-6) * e * e).sum())

    assert err(ours) < err(kquant.RTN["Q6_K"](w))


def test_refined_fit_is_at_least_as_good_as_the_first_fit():
    w = _rows(9, (32, 512))
    x = w.astype(np.float64)

    def err(data, fmt):
        e = x - kquant.dequantize(fmt, data, 512)
        return float(((x ** 2 + (x ** 2).mean()) * e * e).sum())

    for fmt in ("Q4_K", "Q5_K"):
        assert err(kg.fit_rows_v2(w, fmt), fmt) <= err(kg.fit_rows(w, fmt), fmt) * (1 + 1e-9)


def _layer_stats(cols=256, inner=256, seed=4):
    rng = np.random.default_rng(seed)
    count = rng.integers(0, 50, kg.N_EXPERTS).astype(np.float32)
    count[:3] = 0                                                   # experts no calibration token reached
    return {
        "blk.0.attn_input.xtx": _hessian(seed, cols),
        "blk.0.ffn_input.xtx": _hessian(seed + 1, cols),
        "blk.0.ffn_down_shexp.xtx": _hessian(seed + 2, inner),
        "blk.0.exps_input.sumsq": (rng.uniform(0.2, 3.0, (kg.N_EXPERTS, cols)) * count[:, None]).astype(np.float32),
        "blk.0.exps_down.sumsq": (rng.uniform(0.2, 3.0, (kg.N_EXPERTS, inner)) * count[:, None]).astype(np.float32),
        "blk.0.exps.count": count,
    }


def test_routed_experts_use_their_own_channel_energies():
    stats = _layer_stats()
    calib = _Calib(stats)
    per = 2
    w = _rows(5, (kg.N_EXPERTS * per, 256))
    unit = Unit("L0.exps.gate", "blk.0.ffn_gate_exps.weight", 0, "exps.gate", 256, w.shape[0], w.size)
    data = kg._encode(EncodeContext(w, unit, {}, calib), "Q4_K")
    assert data == kg._encode(EncodeContext(w.copy(), unit, {}, calib), "Q4_K")
    c, d = kg.expert_scales(stats["blk.0.ffn_input.xtx"].astype(np.float64), stats["blk.0.exps_input.sumsq"].astype(np.float64),
                            stats["blk.0.exps.count"].astype(np.float64))
    got = kquant.dequantize("Q4_K", data, 256)
    rtn = kquant.dequantize("Q4_K", kquant.RTN["Q4_K"](w), 256)
    ours = base = 0.0
    for e in range(kg.N_EXPERTS):
        he = c * np.outer(d[e], d[e])
        sl = slice(e * per, (e + 1) * per)
        for q, acc in ((got, "ours"), (rtn, "base")):
            err = w[sl].astype(np.float64) - q[sl]
            v = float(np.einsum("ij,jk,ik->", err, he, err))
            if acc == "ours":
                ours += v
            else:
                base += v
    assert ours < 0.8 * base


def test_every_unit_kind_encodes_to_a_valid_tensor():
    calib = _Calib(_layer_stats())
    shapes = {"gdn.qkv": (16, 256), "attn.q": (16, 256), "shexp.gate": (16, 256), "shexp.down": (16, 256),
              "exps.gate": (kg.N_EXPERTS, 256), "exps.down": (kg.N_EXPERTS, 256), "gdn.out": (16, 256)}
    for kind, shape in shapes.items():
        w = _rows(6, shape)
        unit = Unit(f"L0.{kind}", f"blk.0.{kind}.weight", 0, kind, shape[1], shape[0], w.size)
        for fmt in ("Q4_K", "Q5_K", "Q6_K"):
            data = kg._encode(EncodeContext(w, unit, {}, calib), fmt)
            assert data == kg._encode(EncodeContext(w.copy(), unit, {}, calib), fmt)
            back = kquant.dequantize(fmt, data, shape[1])
            assert back.shape == shape and np.isfinite(back).all()
            assert np.abs(back - w).mean() < np.abs(w).mean()


def test_without_statistics_every_kind_falls_back_to_the_fit():
    w = _rows(7, (16, 256))
    unit = Unit("L0.gdn.qkv", "blk.0.attn_qkv.weight", 0, "gdn.qkv", 256, 16, w.size)
    for fmt in ("Q4_K", "Q6_K"):
        assert kg._encode(EncodeContext(w, unit, {}, None), fmt) == kg._fallback(w, fmt)


def test_the_probe_is_repeatable_and_not_kq_rtn():
    from bittrellis.fingerprint import similarity

    hpc02.load_contributed()
    first = hpc02.probe(3, ["kq_gptq2"])
    assert first and first == hpc02.probe(3, ["kq_gptq2"])
    rtn = hpc02.probe(3, ["kq_rtn"])
    sim, compared = similarity({k.split("|", 1)[1]: v for k, v in first.items()},
                               {k.split("|", 1)[1]: v for k, v in rtn.items()})
    assert compared and sim < 0.99
