import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("gguf")

from bittrellis import hpc02, kquant  # noqa: E402
from bittrellis.hpc02 import EncodeContext, Unit  # noqa: E402
from bittrellis.hpc02_encoders import kq_imatrix_gptq as K  # noqa: E402
from bittrellis.safetensors_io import SafeTensorsDir, ShardWriter  # noqa: E402

from .test_hpc02 import QUIET, tiny  # noqa: E402,F401  (tiny: the HPC-02 template fixture)

FORMATS = ("Q4_K", "Q5_K", "Q6_K", "Q8_0")
RECIPE = Path(__file__).resolve().parents[2] / "manifests" / "hpc02" / "pr129-kq-imatrix-downs-gdn-qkv.yaml"


def _rows(seed, shape=(64, 512)):
    rng = np.random.default_rng(seed)
    x = rng.standard_t(5, size=shape) * 0.02
    x[rng.integers(0, shape[0], 16), rng.integers(0, shape[1], 16)] *= 8.0
    return x.astype(np.float32)


def _importance(seed, cols=512):
    """Per input channel mean x², skewed as the real statistics are (50x to 7,000x between channels)."""
    return np.exp(np.random.default_rng(seed).standard_normal(cols) * 1.5)


def _hessian(seed, n=512, tokens=2048):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((tokens, n)) @ (rng.standard_normal((n, n)) * 0.15 + np.eye(n))
    x *= np.exp(rng.standard_normal(n))
    return x.T @ x / tokens


def _decode(fmt, b, cols):
    return kquant.dequantize(fmt, b, cols).astype(np.float64)


def _werr(fmt, b, x, w):
    d = _decode(fmt, b, x.shape[1]) - x
    return float((d * d * w).sum())


def _herr(fmt, b, x, h):
    d = _decode(fmt, b, x.shape[1]) - x
    return float(np.einsum("ij,jk,ik->", d, h, d))


@pytest.mark.parametrize("fmt", FORMATS)
def test_bytes_repeat_and_decode(fmt):
    x, w = _rows(1), _importance(1)
    a, b = K.encode_weighted(x, w, fmt), K.encode_weighted(x.copy(), w.copy(), fmt)
    assert a == b and len(a) == x.shape[0] * kquant.row_bytes(fmt, x.shape[1])
    g1, g2 = K.encode_gptq(x, _hessian(1), fmt), K.encode_gptq(x.copy(), _hessian(1), fmt)
    assert g1 == g2 and len(g1) == len(a)
    for blob in (a, g1):                                      # gguf-py decodes it to the right values
        y = _decode(fmt, blob, x.shape[1])
        assert np.isfinite(y).all() and np.abs(y - x).max() < 0.25 * np.abs(x).max()


def test_layout_matches_kquant():
    """K.pack writes the same bytes as kquant for the same block fields (the llama.cpp layouts)."""
    x = _rows(2, (16, 256)).astype(np.float64)
    for fmt, nmax in (("Q4_K", 15), ("Q5_K", 31)):
        d, dmin, sc, m, q = kquant._asym_k(x.reshape(-1, 8, 32), nmax)
        f = {"d": d.astype(np.float64), "dmin": dmin.astype(np.float64), "sc": sc, "m": m}
        assert K.pack(fmt, f, q.reshape(len(q), -1)) == kquant.RTN[fmt](x)
    x6 = x.reshape(-1, 16, 16)
    scale = np.abs(x6).max(-1) / 31.0
    d = K._f16(scale.max(-1) / 127.0)
    sc = np.clip(np.rint(scale / d[:, None]), -128, 127)
    q = np.clip(np.rint(x6 / (d[:, None] * sc)[..., None]), -32, 31)
    assert K.pack("Q6_K", {"d": d, "sc": sc}, q.reshape(len(q), -1)) == kquant.RTN["Q6_K"](x)
    d8 = K._f16(np.abs(x.reshape(-1, 32)).max(-1) / 127.0)
    q8 = np.clip(np.rint(x.reshape(-1, 32) / d8[:, None]), -127, 127)
    assert K.pack("Q8_0", {"d": d8}, q8) == kquant.RTN["Q8_0"](x)


@pytest.mark.parametrize("fmt", FORMATS)
def test_lower_weighted_error_than_kq_rtn(fmt):
    x, w = _rows(3), _importance(3)
    ours, rtn = K.encode_weighted(x, w, fmt), kquant.RTN[fmt](x)
    assert _werr(fmt, ours, x, w) < 0.75 * _werr(fmt, rtn, x, w)
    # with uniform weights it is a better plain fit too
    flat = K.encode_weighted(x, np.ones(x.shape[1]), fmt)
    assert _werr(fmt, flat, x, 1.0) < 0.95 * _werr(fmt, rtn, x, 1.0)


@pytest.mark.parametrize("fmt", FORMATS)
def test_no_block_worse_than_kq_rtn(fmt):
    x, w = _rows(4, (32, 512)), _importance(4)
    nv = kquant.BLOCK[fmt][1]
    per_block = lambda b: ((_decode(fmt, b, 512) - x) ** 2 * w).reshape(-1, nv).sum(-1)  # noqa: E731
    ours, rtn = per_block(K.encode_weighted(x, w, fmt)), per_block(kquant.RTN[fmt](x))
    assert (ours <= rtn * (1 + 1e-5) + 1e-15).all()


@pytest.mark.parametrize("fmt", FORMATS)
def test_gptq_lowers_output_error(fmt):
    x, h = _rows(5, (48, 512)), _hessian(5)
    gptq = K.encode_gptq(x, h, fmt)
    diag = K.encode_weighted(x, np.diag(h), fmt)
    rtn = kquant.RTN[fmt](x)
    assert _herr(fmt, gptq, x, h) < 0.8 * _herr(fmt, diag, x, h) < 0.8 * _herr(fmt, rtn, x, h)


def test_gptq_rows_are_independent():
    x, h = _rows(6, (K.GPTQ_ROWS + 24, 256)), _hessian(6, 256, 1024)
    full = K.encode_gptq(x, h, "Q4_K")
    parts = b"".join(K.encode_gptq(x[r:r + 40], h, "Q4_K") for r in range(0, len(x), 40))
    assert full == parts


def test_factor_is_gptqs_and_needs_no_blas():
    h = _hessian(8, 128, 512)
    u = K.inverse_hessian_factor(h)
    hd = h + K.DAMP * np.mean(np.diag(h)) * np.eye(128)
    assert np.allclose(u, np.triu(u)) and np.allclose(u.T @ u, np.linalg.inv(hd), rtol=1e-8, atol=1e-10)


_THREADS_SCRIPT = """
import hashlib, numpy as np
from bittrellis.hpc02_encoders import kq_imatrix_gptq as K
rng = np.random.default_rng(3)
x = (rng.standard_normal((96, 512)) * 0.02).astype(np.float32)
t = rng.standard_normal((2048, 512)) @ (rng.standard_normal((512, 512)) * 0.15 + np.eye(512))
h = t.T @ t / 2048
print(hashlib.sha256(b"".join(K.encode_gptq(x, h, f) for f in ("Q4_K", "Q8_0"))).hexdigest())
"""


def test_gptq_bytes_do_not_depend_on_blas_threads():
    """OpenBLAS results change with its thread count; the GPTQ path must not (a build and its audit replay
    may run with different counts)."""
    out = set()
    for n in ("1", "4"):
        env = {**os.environ, **{v: n for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
        out.add(subprocess.run([sys.executable, "-c", _THREADS_SCRIPT], env=env, capture_output=True, text=True,
                               check=True).stdout.strip())
    assert len(out) == 1 and len(next(iter(out))) == len(hashlib.sha256().hexdigest())


def test_dead_inputs_and_zero_rows():
    x = _rows(7, (8, 256))
    x[2] = 0.0
    h = _hessian(7, 256, 512)
    h[:, 5] = h[5, :] = 0.0
    w = np.diag(h).copy()
    for fmt in FORMATS:
        for blob in (K.encode_weighted(x, w, fmt), K.encode_gptq(x, h, fmt)):
            y = _decode(fmt, blob, 256)
            assert np.isfinite(y).all() and not y[2].any()


# ------------------------------------------------------------------ statistics and fallbacks


def _calibration(root, layers=2, experts=4, cols=256, down_cols=256, seed=0, zero_expert=None):
    rng = np.random.default_rng(seed)
    w = ShardWriter(root, 1 << 30)
    for i in range(layers):
        count = rng.integers(20, 200, experts).astype(np.float64)
        if zero_expert is not None:
            count[zero_expert] = 0
        for key, c in (("exps_input", cols), ("exps_down", down_cols)):
            s = count[:, None] * np.exp(rng.standard_normal((experts, c)))
            w.add(f"blk.{i}.{key}.sumsq", "F32", s.shape, s.astype("<f4"))
        w.add(f"blk.{i}.exps.count", "F32", count.shape, count.astype("<f4"))
        for key, c in (("attn_input.xtx", cols), ("ffn_input.xtx", cols), ("ffn_down_shexp.xtx", down_cols)):
            w.add(f"blk.{i}.{key}", "F32", (c, c), _hessian(seed + i, c, 4 * c).astype("<f4"))
    w.close()
    return SafeTensorsDir(root)


def _ctx(kind, rows, cal, layer=0, params=None):
    return EncodeContext(rows, Unit(f"L{layer}.{kind}", "t", layer, kind, rows.shape[1], len(rows), rows.size),
                         params or {}, cal)


def test_importance_per_expert_and_fallbacks(tmp_path):
    cal = _calibration(tmp_path / "cal", zero_expert=2)
    x = _rows(8, (4 * 32, 256))
    w, h = K.importance(_ctx("exps.gate", x, cal))
    assert h is None and w.shape == (4, 256)
    s = cal.array("blk.0.exps_input.sumsq", "<f4").astype(np.float64)
    n = cal.array("blk.0.exps.count", "<f4").astype(np.float64)
    pooled = s.sum(0) / n.sum()
    assert np.allclose(w[2], pooled, rtol=1e-4)                 # never routed: the layer's pooled mean
    assert np.allclose(w[0], (s[0] + K.PRIOR_TOKENS * pooled) / (n[0] + K.PRIOR_TOKENS), rtol=1e-4)
    # each expert's rows are fit to its own importance
    got = K.ENCODER.fn(_ctx("exps.gate", x, cal), "Q4_K")
    want = b"".join(K.encode_weighted(x[e * 32:(e + 1) * 32], w[e], "Q4_K") for e in range(4))
    assert got == want
    # shared expert, attention, recurrent inputs: GPTQ on the full matrix; params gptq: false -> its diagonal
    assert K.importance(_ctx("exps.down", _rows(9, (4 * 256, 256)), cal))[0].shape == (4, 256)
    for kind, key in (("shexp.gate", "ffn_input.xtx"), ("shexp.down", "ffn_down_shexp.xtx"), ("gdn.qkv", "attn_input.xtx"),
                      ("attn.k", "attn_input.xtx")):
        hh = cal.array(f"blk.1.{key}", "<f4").astype(np.float64)
        assert K.ENCODER.fn(_ctx(kind, x, cal, 1), "Q6_K") == K.encode_gptq(x, hh, "Q6_K")
        assert K.ENCODER.fn(_ctx(kind, x, cal, 1, {"gptq": False}), "Q6_K") == \
            K.encode_weighted(x, K._floor(np.diag(hh)[None])[0], "Q6_K")
    # no statistics for the unit, or none at all: uniform weights
    flat = K.encode_weighted(x, np.ones(256), "Q5_K")
    assert K.ENCODER.fn(_ctx("attn.o", x, cal), "Q5_K") == flat
    assert K.ENCODER.fn(_ctx("exps.gate", x, None), "Q5_K") == flat
    top = EncodeContext(x, Unit("lm_head", "output.weight", None, "lm_head", 256, len(x), x.size), {}, cal)
    assert K.ENCODER.fn(top, "Q5_K") == flat
    with pytest.raises(ValueError, match="F32"):
        K.ENCODER.fn(_ctx("attn.o", x, cal), "F32")


def test_registered_and_distinct_from_kq_rtn():
    assert hpc02.ENCODERS["kq_imatrix"].ref == "kq_imatrix@v1"
    probe = hpc02.probe(11, ["kq_imatrix", "kq_rtn"])
    for fmt in FORMATS:
        a, b = probe[f"kq_imatrix@v1|{fmt}|probe"], probe[f"kq_rtn@v1|{fmt}|probe"]
        same = np.mean(np.frombuffer(a, np.uint8) == np.frombuffer(b, np.uint8))
        assert len(a) == len(b) and same < 0.9
    assert probe == hpc02.probe(11, ["kq_imatrix", "kq_rtn"])


def test_build_and_audit_with_calibration(tiny, tmp_path):  # noqa: F811
    tdir, ud = tiny
    cal = tmp_path / "cal"
    _calibration(cal, layers=2, experts=4, cols=256, down_cols=256)
    m = tmp_path / "m.yaml"
    m.write_text(yaml.safe_dump({"schema": "bittrellis/manifest@2", "track": "HPC-02", "name": "t", "default": "Q8_0",
                                 "encoders": {f: "kq_imatrix" for f in FORMATS},
                                 "rules": [{"match": "L*.exps.*", "format": "Q4_K"}, {"match": "L*.gdn.qkv", "format": "Q4_K"},
                                           {"match": "L1.exps.down", "format": "Q6_K"}],
                                 "modules": {"lm_head": "Q5_K"}}))
    out = tmp_path / "c.gguf"
    hpc02.build(m, tdir, out, ud=ud, calibration_dir=cal, jobs=2, **QUIET)
    res = hpc02.audit(out, m, tdir, ud=ud, calibration_dir=cal, secret="s", verify=False, **QUIET)
    assert res["ok"], res["errors"]
    assert res["lineage"]["kq_imatrix@v1"]["units"] == len(hpc02.units(hpc02.read_template(tdir)))
    # without the statistics the same recipe builds different (uniform-weight) bytes, which the audit catches
    res2 = hpc02.audit(out, m, tdir, ud=ud, calibration_dir=None, secret="s", verify=False, **QUIET)
    assert not res2["ok"]


def test_recipe_validates():
    """The recipe is #129's map with kq_imatrix on the covered Q4_K downs and recurrent qkv: #129's speed and memory."""
    us, udf = hpc02.read_units_file()
    a = hpc02.expand(hpc02.load_manifest(RECIPE), us, udf)
    base = hpc02.expand(hpc02.load_manifest(RECIPE.with_name("ud-downs-q4k-gdn-qkv-q4k-lmhead-embed-q4k.yaml")), us, udf)
    assert {k: x.format for k, x in a.items()} == {k: x.format for k, x in base.items()}
    assert sum(x.encoder == "kq_imatrix" for x in a.values()) == 67
