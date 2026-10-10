"""K-quants rounded for the inputs each tensor actually sees: importance-weighted scale search (llama.cpp's
imatrix idea, on the pinned HPC-02 statistics), a refit of the stored integers, and GPTQ where the full input
covariance is pinned.

What the model sees of a tensor's rounding error ΔW is the error of its output, x·ΔWᵀ, over its inputs x. For a
row that is Δ H Δᵀ with H = mean x xᵀ; on the diagonal alone it is Σ_c H_cc Δ_c². `kq_rtn` fits each block to
its own range and weighs every column the same, although the pinned statistics show input channels whose mean
x² differ by 50x to 7,000x within one layer. This encoder:

1. **weighs every column by its importance** (per input channel mean x²):

   | unit | importance | statistics |
   |---|---|---|
   | routed experts gate / up (expert e) | exps_input.sumsq[e] / count[e] | per-expert diagonal |
   | routed experts down (expert e) | exps_down.sumsq[e] / count[e] | per-expert diagonal |
   | shared expert gate / up | ffn_input.xtx | full H (GPTQ) |
   | shared expert down | ffn_down_shexp.xtx | full H (GPTQ) |
   | attention q / k / v, recurrent qkv / z | attn_input.xtx | full H (GPTQ) |
   | attention o, recurrent out, embed, lm_head | none pinned | uniform weights |

   An expert's mean is shrunk toward the layer's pooled mean by `PRIOR_TOKENS` pseudo-tokens
   ((sumsq[e] + k·pooled) / (count[e] + k)), so a rarely routed expert keeps a sane estimate and one never
   routed (count 0) gets the pooled mean. Without calibration statistics every unit gets uniform weights;

2. **searches each sub-block's scale (and minimum)** under those weights, as llama.cpp's make_qkx2_quants /
   make_qx_quants do: candidate inverse scales around the block's range, round, weighted least-squares scale
   (and minimum) for those integers, keep the lowest weighted error;

3. **refits what is stored**, which llama.cpp leaves at its first rounding: with the super-block's f16 scales
   fixed, each sub-block's 6-bit (Q4_K/Q5_K) or 8-bit (Q6_K) scale and minimum are re-chosen among their
   neighbours (jointly, scale and minimum each by -1, 0 or +1) with the values re-rounded, then the f16
   super-block scales are refit by weighted least squares to those integers, then the neighbours once more
   (Q8_0: its one f16 scale refit). Each block finally keeps whichever of this encoding and `kq_rtn`'s has the
   lower weighted error, so no block is worse than the baseline on its own measure;

4. **GPTQ** (default on, `params: {gptq: false}` turns it off) where the full H is pinned: columns are rounded
   left to right, each column's error pushed onto the columns not yet rounded through the damped inverse
   Hessian; every block's scales are chosen (steps 2-3, weights diag H) when GPTQ reaches it, and each later
   sub-block's 6/8-bit scale again when GPTQ reaches that sub-block, on the weights as the earlier feedback
   left them.

Output bytes are llama.cpp's block layouts (Q4_K, Q5_K, Q6_K, Q8_0), decoded by gguf-py in the tests.
Determinism: NumPy float64 on the CPU, fixed row chunks (one expert at a time for routed experts), and no BLAS
or LAPACK anywhere: GPTQ's Cholesky factor, triangular inverse and error propagation use NumPy's own einsum
loops, because OpenBLAS results change with its thread count (tested: the bytes are the same at 1 and 4
threads). About 5 CPU-hours for the whole model at Q4_K.
"""

from __future__ import annotations

import hashlib

import numpy as np

from bittrellis import kquant
from bittrellis.hpc02 import EncodeContext, Encoder, register

NAME, VERSION = "kq_imatrix", 1

PRIOR_TOKENS = 64.0          # pseudo-tokens of the layer's pooled mean added to each expert's own statistics
FLOOR = 1e-6                 # relative weight floor, so a dead input channel still has a defined fit
REFIT_ROUNDS = 1             # integer-neighbour / f16-refit rounds (2 measured within 0.5% of 1)
DAMP = 0.01                  # GPTQ: relative diagonal dampening of H
CHUNK_VALUES = 1 << 16       # values per pass of the weighted search (bounds the temporaries)
GPTQ_ROWS = 4096             # rows per GPTQ pass
GPTQ_SUBBLOCK_REFIT = True   # GPTQ: re-choose each later sub-block's scale when it is reached

MOVES = tuple((a, b) for a in (0, -1, 1) for b in (0, -1, 1))   # sub-block (scale, min) moves tried

# Sub-block search grids: Q4_K/Q5_K over llama.cpp's make_qkx2_quants range (-1.0 .. +1.0 around nmax) in
# steps of 0.2 (its 0.1 steps, and make_qkx3_quants' 0.05, measured no better on the real experts once the
# integers are refit, at two and four times the time); Q6_K as make_qx_quants (-9..9 tenths around nmax);
# Q8_0 searched the same way around 127.
ASYM_GRID = (-1.0, 0.2, 10)
SYM_STEPS = np.arange(-9, 10) * 0.1

# format -> (sub-block length, sub-blocks per block, integer range)
SPEC = {"Q4_K": (32, 8, 0, 15), "Q5_K": (32, 8, 0, 31), "Q6_K": (16, 16, -32, 31), "Q8_0": (32, 1, -127, 127)}

_EXPS_STATS = {"exps.gate": "exps_input", "exps.up": "exps_input", "exps.down": "exps_down"}
_XTX_STATS = {"shexp.gate": "ffn_input.xtx", "shexp.up": "ffn_input.xtx", "shexp.down": "ffn_down_shexp.xtx",
              "gdn.qkv": "attn_input.xtx", "gdn.z": "attn_input.xtx",
              "attn.q": "attn_input.xtx", "attn.k": "attn_input.xtx", "attn.v": "attn_input.xtx"}


def _f16(x) -> np.ndarray:
    return np.asarray(x, np.float64).astype(np.float16).astype(np.float64)


def _div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b != 0, a / np.where(b != 0, b, 1.0), 0.0)


# ------------------------------------------------------------------ importance


def _floor(w: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(w, np.float64), 0.0)
    m = w.mean(-1, keepdims=True)
    return w + FLOOR * np.where(m > 0, m, 1.0)


def importance(ctx: EncodeContext):
    """(weights, H): weights [groups, cols] float64 (one row of weights per expert, or one group), and the full
    input covariance H [cols, cols] when it is pinned (else None). (None, None) without statistics."""
    u, cal = ctx.unit, ctx.calibration
    if cal is None or u.layer is None:
        return None, None
    blk = f"blk.{u.layer}."
    if u.kind in _EXPS_STATS:
        sumsq, count = blk + _EXPS_STATS[u.kind] + ".sumsq", blk + "exps.count"
        if sumsq not in cal or count not in cal:
            return None, None
        n = np.asarray(cal.array(count, "<f4"), np.float64).reshape(-1)
        s = np.asarray(cal.array(sumsq, "<f4"), np.float64).reshape(len(n), -1)
        if (s.shape[1] != u.cols or len(ctx.rows) % len(n) or not np.isfinite(s).all() or not np.isfinite(n).all()
                or n.min() < 0 or n.sum() <= 0):
            return None, None
        pooled = s.sum(0) / n.sum()
        w = (s + PRIOR_TOKENS * pooled) / (n + PRIOR_TOKENS)[:, None]
        return _floor(w), None
    if u.kind in _XTX_STATS and blk + _XTX_STATS[u.kind] in cal:
        h = np.asarray(cal.array(blk + _XTX_STATS[u.kind], "<f4"), np.float64)
        if h.shape != (u.cols, u.cols) or not np.isfinite(h).all():
            return None, None
        return _floor(np.diag(h)[None, :]), h
    return None, None


# ------------------------------------------------------------------ sub-block search


def _dot(a, b):
    """Row sums of a·b over the last axis (no BLAS: NumPy's own loop, the same on every run)."""
    return np.einsum("...k,...k->...", a, b)


def _asym_search(x: np.ndarray, w: np.ndarray, nmax: int):
    """x, w [n, k] -> (scale, mn) [n]: x ≈ scale·L - mn, L in 0..nmax, mn >= 0 (make_qkx3_quants, weighted)."""
    lo = np.minimum(x.min(-1), 0.0)
    rng = x.max(-1) - lo
    inv_rng = _div(1.0, rng)
    wx = w * x
    sw, swx, swxx = w.sum(-1), wx.sum(-1), _dot(wx, x)
    xs = x - lo[:, None]
    best_s, best_m = rng / nmax, -lo                               # the plain range fit is the first candidate
    best_e = _err(x, w, best_s, best_m, _round(x, best_s, best_m, 0, nmax))
    rmin, rdelta, nstep = ASYM_GRID
    L = np.empty_like(x)
    for i in range(nstep + 1):
        np.multiply(xs, ((rmin + rdelta * i + nmax) * inv_rng)[:, None], out=L)
        np.rint(L, out=L)
        np.clip(L, 0, nmax, out=L)
        wl = w * L
        swl, swl2, swxl = wl.sum(-1), _dot(wl, L), _dot(wx, L)
        det = sw * swl2 - swl * swl
        sc = _div(sw * swxl - swx * swl, det)
        b = _div(swl2 * swx - swl * swxl, det)                   # x ≈ sc·L + b
        clamp = (b > 0) | (det <= 0)
        sc = np.where(clamp, _div(swxl, swl2), sc)
        b = np.where(clamp, 0.0, b)
        e = swxx - 2 * sc * swxl - 2 * b * swx + sc * sc * swl2 + 2 * sc * b * swl + b * b * sw
        better = (e < best_e) & (sc > 0)
        best_e = np.where(better, e, best_e)
        best_s = np.where(better, sc, best_s)
        best_m = np.where(better, -b, best_m)
    return best_s, best_m


def _sym_search(x: np.ndarray, w: np.ndarray, qmin: int, qmax: int):
    """x, w [n, k] -> signed scale [n]: x ≈ scale·L, L in qmin..qmax (make_qx_quants, weighted, both signs)."""
    imax = np.argmax(np.abs(x), -1)
    vmax = np.take_along_axis(x, imax[:, None], -1)[:, 0]
    inv_max = _div(1.0, vmax)
    wx = w * x
    swxx = _dot(wx, x)
    zero = np.zeros(len(x))
    best_s = np.abs(vmax) / qmax                                  # the plain range fit is the first candidate
    best_e = _err(x, w, best_s, zero, _round(x, best_s, zero, qmin, qmax))
    L = np.empty_like(x)
    for nmax, sign in ((-qmin, -1.0), (qmax, 1.0)):              # the extreme maps to qmin or to qmax
        for st in SYM_STEPS:
            np.multiply(x, (sign * (nmax + st) * inv_max)[:, None], out=L)
            np.rint(L, out=L)
            np.clip(L, qmin, qmax, out=L)
            swl2, swxl = _dot(w * L, L), _dot(wx, L)
            sc = _div(swxl, swl2)
            e = swxx - sc * swxl
            better = e < best_e
            best_e = np.where(better, e, best_e)
            best_s = np.where(better, sc, best_s)
    return best_s


# ------------------------------------------------------------------ block fits


def _round(x, step, off, qmin, qmax):
    """Nearest integers for x ≈ step·L - off (x [..., k], step/off [...]); a zero step gives 0."""
    L = x + off[..., None]
    L *= _div(1.0, step)[..., None]
    np.rint(L, out=L)
    return np.clip(L, qmin, qmax, out=L)


def _err(x, w, step, off, L):
    """Weighted squared error per sub-block of step·L - off against x."""
    r = L * step[..., None]
    r -= off[..., None]
    r -= x
    r *= r
    return _dot(w, r)


def _fit_asym(x, w, nmax):
    """x, w [n, 8, 32] -> d, dmin [n] (f16 values), sc, m [n, 8] (0..63)."""
    n = len(x)
    scale, mn = _asym_search(x.reshape(-1, 32), w.reshape(-1, 32), nmax)
    scale, mn = scale.reshape(n, 8), mn.reshape(n, 8)
    d, dmin = _f16(scale.max(-1) / 63.0), _f16(mn.max(-1) / 63.0)
    sc = np.clip(np.rint(_div(scale, d[:, None])), 0, 63)
    m = np.clip(np.rint(_div(mn, dmin[:, None])), 0, 63)
    for _ in range(REFIT_ROUNDS):
        sc, m = _neighbours_asym(x, w, d, dmin, sc, m, nmax)
        d, dmin = _refit_asym(x, w, d, dmin, sc, m, nmax)
    if REFIT_ROUNDS:
        sc, m = _neighbours_asym(x, w, d, dmin, sc, m, nmax)
    return d, dmin, sc, m


def _neighbours_asym(x, w, d, dmin, sc, m, nmax):
    """Each sub-block's 6-bit scale and minimum, jointly moved by up to one step where that lowers its error."""
    best, s0, m0 = None, sc, m
    for move in MOVES:
        s2, m2 = np.clip(s0 + move[0], 0, 63), np.clip(m0 + move[1], 0, 63)
        step, off = d[:, None] * s2, dmin[:, None] * m2
        e = _err(x, w, step, off, _round(x, step, off, 0, nmax))
        if best is None:
            best, sc, m = e, s2, m2
            continue
        better = e < best
        best, sc, m = np.where(better, e, best), np.where(better, s2, sc), np.where(better, m2, m)
    return sc, m


def _refit_asym(x, w, d, dmin, sc, m, nmax):
    """Weighted least squares of the two f16 super-block scales for the current integers; kept where better."""
    step, off = d[:, None] * sc, dmin[:, None] * m
    L = _round(x, step, off, 0, nmax)
    a = sc[..., None] * L                                         # x ≈ d·a - dmin·b
    b = np.broadcast_to(m[..., None], x.shape)
    saa, sbb, sab = (w * a * a).sum((1, 2)), (w * b * b).sum((1, 2)), (w * a * b).sum((1, 2))
    sxa, sxb = (w * x * a).sum((1, 2)), (w * x * b).sum((1, 2))
    det = saa * sbb - sab * sab
    nd = _div(sxa * sbb - sxb * sab, det)
    nm = -_div(saa * sxb - sab * sxa, det)
    only = (nm < 0) | (det <= 0)
    nd = np.where(only, _div(sxa, saa), nd)
    nm = np.where(only, 0.0, nm)
    nd, nm = _f16(np.maximum(nd, 0.0)), _f16(np.maximum(nm, 0.0))
    old = _err(x, w, step, off, L).sum(-1)
    new = _err(x, w, nd[:, None] * sc, nm[:, None] * m, L).sum(-1)
    keep = (new < old) & np.isfinite(new)
    return np.where(keep, nd, d), np.where(keep, nm, dmin)


def _fit_q6k(x, w):
    """x, w [n, 16, 16] -> d [n] (f16 value), sc [n, 16] (-128..127)."""
    n = len(x)
    scale = _sym_search(x.reshape(-1, 16), w.reshape(-1, 16), -32, 31).reshape(n, 16)
    imax = np.argmax(np.abs(scale), -1)
    smax = np.take_along_axis(scale, imax[:, None], -1)[:, 0]
    d = _f16(-smax / 128.0)                                       # llama.cpp: the largest scale maps to -128
    sc = np.clip(np.rint(_div(scale, d[:, None])), -128, 127)
    for _ in range(REFIT_ROUNDS):
        sc = _neighbours_sym(x, w, d, sc, -128, 127, -32, 31)
        d = _refit_sym(x, w, d, sc, -32, 31)
    return d, (_neighbours_sym(x, w, d, sc, -128, 127, -32, 31) if REFIT_ROUNDS else sc)


def _neighbours_sym(x, w, d, sc, smin, smax, qmin, qmax):
    zero = np.zeros(sc.shape)
    best = None
    for ds in (0, -1, 1):
        s2 = np.clip(sc + ds, smin, smax)
        step = d[:, None] * s2
        e = _err(x, w, step, zero, _round(x, step, zero, qmin, qmax))
        if best is None:
            best, bs = e, s2
        else:
            better = e < best
            best, bs = np.where(better, e, best), np.where(better, s2, bs)
    return bs


def _refit_sym(x, w, d, sc, qmin, qmax):
    zero = np.zeros(sc.shape)
    step = d[:, None] * sc
    L = _round(x, step, zero, qmin, qmax)
    a = sc[..., None] * L
    nd = _f16(_div((w * x * a).sum((1, 2)), (w * a * a).sum((1, 2))))
    old = _err(x, w, step, zero, L).sum(-1)
    new = _err(x, w, nd[:, None] * sc, zero, L).sum(-1)
    keep = (new < old) & np.isfinite(new) & (np.sign(nd) == np.sign(d))
    return np.where(keep, nd, d)


def _fit_q8(x, w):
    """x, w [n, 1, 32] -> d [n] (f16 value): weighted search, then the f16 scale re-rounded and refit."""
    s = np.abs(_sym_search(x[:, 0], w[:, 0], -127, 127))
    d = _f16(s)
    zero = np.zeros((len(x), 1))
    for _ in range(REFIT_ROUNDS):
        d = _refit_sym(x, w, d, np.ones((len(x), 1)), -127, 127)
    rtn = _f16(np.abs(x).max((1, 2)) / 127.0)                    # never worse than the plain range fit
    e_d = _err(x, w, d[:, None], zero, _round(x, d[:, None], zero, -127, 127)).sum(-1)
    e_r = _err(x, w, rtn[:, None], zero, _round(x, rtn[:, None], zero, -127, 127)).sum(-1)
    return np.where(e_r < e_d, rtn, d)


def block_params(fmt: str, x: np.ndarray, w: np.ndarray):
    """x, w [n, blocks of `fmt`] -> (fields for packing, step [n, sub], off [n, sub]) after search and refit,
    with each block replaced by kq_rtn's where that has the lower weighted error."""
    S, nsub, qmin, qmax = SPEC[fmt]
    x = x.reshape(-1, nsub, S)
    w = w.reshape(-1, nsub, S)
    if fmt in ("Q4_K", "Q5_K"):
        d, dmin, sc, m = _fit_asym(x, w, qmax)
        rd, rdm, rsc, rm, _ = kquant._asym_k(x, qmax)
        rd, rdm = rd.astype(np.float64), rdm.astype(np.float64)
        f = {"d": d, "dmin": dmin, "sc": sc, "m": m}
        r = {"d": rd, "dmin": rdm, "sc": rsc.astype(np.float64), "m": rm.astype(np.float64)}
        steps = lambda g: (g["d"][:, None] * g["sc"], g["dmin"][:, None] * g["m"])  # noqa: E731
    elif fmt == "Q6_K":
        d, sc = _fit_q6k(x, w)
        f = {"d": d, "sc": sc}
        amax = np.abs(x).max(-1) / 31.0
        rd = _f16(amax.max(-1) / 127.0)
        r = {"d": rd, "sc": np.clip(np.rint(_div(amax, rd[:, None])), -128, 127)}
        steps = lambda g: (g["d"][:, None] * g["sc"], np.zeros(g["sc"].shape))  # noqa: E731
    else:
        f = {"d": _fit_q8(x, w)}
        steps = lambda g: (g["d"][:, None], np.zeros((len(g["d"]), 1)))  # noqa: E731
        r = f
    if r is not f:
        st, of = steps(f)
        rst, rof = steps(r)
        e = _err(x, w, st, of, _round(x, st, of, qmin, qmax)).sum(-1)
        er = _err(x, w, rst, rof, _round(x, rst, rof, qmin, qmax)).sum(-1)
        use_r = er < e
        f = {k: np.where(use_r.reshape((-1,) + (1,) * (v.ndim - 1)), r[k], v) for k, v in f.items()}
    step, off = steps(f)
    return f, step, off


# ------------------------------------------------------------------ packing (llama.cpp layouts)


def pack(fmt: str, f: dict, L: np.ndarray) -> bytes:
    """Block fields and integers [n, values per block] -> GGUF block bytes."""
    n = len(L)
    if fmt in ("Q4_K", "Q5_K"):
        q = L.reshape(n, 8, 32).astype(np.uint8)
        head = [_f16(f["d"]).astype(np.float16).view(np.uint8).reshape(-1, 2),
                _f16(f["dmin"]).astype(np.float16).view(np.uint8).reshape(-1, 2),
                kquant._pack_scales_q4k(f["sc"], f["m"])]
        if fmt == "Q4_K":
            qs = (q[:, 0::2, :] | (q[:, 1::2, :] << 4)).reshape(n, 128)
            return np.concatenate(head + [qs], axis=1).tobytes()
        lo, hi = q & 0x0F, q >> 4
        qs = (lo[:, 0::2, :] | (lo[:, 1::2, :] << 4)).reshape(n, 128)
        qh = np.zeros((n, 32), np.uint8)
        for j in range(8):
            qh |= (hi[:, j, :] & 1) << j
        return np.concatenate(head + [qh, qs], axis=1).tobytes()
    if fmt == "Q6_K":
        q = (L.reshape(n, 256) + 32).astype(np.uint8)
        # element e: row r = e // 32, column c = e % 32; half h = r // 4, rr = r % 4 (as kquant.quantize_q6k)
        q = q.reshape(n, 2, 4, 32)                                # [n, h, rr, c]
        ql = np.empty((n, 2, 2, 32), np.uint8)                    # [n, h, rr % 2, c]
        ql[:, :, 0] = (q[:, :, 0] & 0x0F) | ((q[:, :, 2] & 0x0F) << 4)
        ql[:, :, 1] = (q[:, :, 1] & 0x0F) | ((q[:, :, 3] & 0x0F) << 4)
        qh = ((q[:, :, 0] >> 4) | ((q[:, :, 1] >> 4) << 2) | ((q[:, :, 2] >> 4) << 4) | ((q[:, :, 3] >> 4) << 6))
        sc = f["sc"].astype(np.int8).view(np.uint8)
        d = _f16(f["d"]).astype(np.float16).view(np.uint8).reshape(-1, 2)
        return np.concatenate([ql.reshape(n, 128), qh.reshape(n, 64).astype(np.uint8), sc, d], axis=1).tobytes()
    d = _f16(f["d"]).astype(np.float16).view(np.uint8).reshape(-1, 2)
    return np.concatenate([d, L.reshape(n, 32).astype(np.int8).view(np.uint8)], axis=1).tobytes()


# ------------------------------------------------------------------ encoders


def encode_weighted(rows: np.ndarray, weights: np.ndarray, fmt: str) -> bytes:
    """Rows [r, cols] with per-column weights [cols] (or per-row [r, cols]) -> GGUF bytes, no error feedback."""
    S, nsub, qmin, qmax = SPEC[fmt]
    r, cols = rows.shape
    per = max(1, CHUNK_VALUES // cols)
    out = []
    for a in range(0, r, per):
        x = np.asarray(rows[a:a + per], np.float64)             # float64 one chunk at a time (lm_head: 0.5 B values)
        w = np.broadcast_to(weights if weights.ndim == 1 else weights[a:a + per], x.shape)
        f, step, off = block_params(fmt, x, w)
        xb = x.reshape(len(step), nsub, S)
        out.append(pack(fmt, f, _round(xb, step, off, qmin, qmax).reshape(len(step), -1)))
    return b"".join(out)


def _cholesky_lower(a: np.ndarray) -> np.ndarray:
    """a = L Lᵀ with L lower triangular: left-looking, one column at a time, in NumPy's own loops."""
    n = len(a)
    lo = np.zeros_like(a)
    for j in range(n):
        s = a[j:, j] - np.einsum("ik,k->i", lo[j:, :j], lo[j, :j])
        if not s[0] > 0:
            raise np.linalg.LinAlgError("the damped input covariance is not positive definite")
        lo[j, j] = np.sqrt(s[0])
        lo[j + 1:, j] = s[1:] / lo[j, j]
    return lo


def _inv_upper(r: np.ndarray) -> np.ndarray:
    """Inverse of an upper triangular matrix by back substitution, in NumPy's own loops."""
    n = len(r)
    x = np.zeros_like(r)
    for i in range(n - 1, -1, -1):
        x[i, i] = 1.0 / r[i, i]
        x[i, i + 1:] = -np.einsum("k,kj->j", r[i, i + 1:], x[i + 1:, i + 1:]) / r[i, i]
    return x


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ik,kj->ij", a, b)


_FACTORS: dict[str, np.ndarray] = {}


def inverse_hessian_factor(h: np.ndarray, damp: float = DAMP) -> np.ndarray:
    """Upper triangular U with (H + λI)⁻¹ = UᵀU (GPTQ's factor), float64.

    No BLAS or LAPACK: their results change with the number of threads (measured: OpenBLAS Cholesky and matrix
    products on one machine at 1 and 8 threads differ in the last bits, enough to flip Q8_0 roundings), and a
    build and its audit replay need not run with the same number. H + λI = R Rᵀ with R upper triangular (the
    Cholesky factor of the index-reversed matrix, reversed back), and U = R⁻¹."""
    h = np.array(h, dtype=np.float64)
    key = f"{damp}:" + hashlib.sha256(h.tobytes()).hexdigest()
    if key in _FACTORS:
        return _FACTORS[key]
    dead = np.diag(h) <= 0
    h[dead, dead] = 1.0
    h[np.diag_indices_from(h)] += damp * float(np.mean(np.diag(h)))
    r = _cholesky_lower(np.ascontiguousarray(h[::-1, ::-1]))[::-1, ::-1]
    u = _inv_upper(np.ascontiguousarray(r))
    _FACTORS.clear()                                              # one factor kept: units of a layer share H
    _FACTORS[key] = u
    return u


def _refit_subblock(fmt, f, step, off, s, x, w):
    """Sub-block `s` of a K-quant block: its 6/8-bit scale (and minimum) re-searched with the block's f16
    scales fixed; updates `f`, `step` and `off` in place."""
    if fmt in ("Q4_K", "Q5_K"):
        nmax = SPEC[fmt][3]
        scale, mn = _asym_search(x, w, nmax)
        d, dmin = f["d"], f["dmin"]
        sc = np.clip(np.rint(_div(scale, d)), 0, 63)[:, None]
        m = np.clip(np.rint(_div(mn, dmin)), 0, 63)[:, None]
        sc, m = _neighbours_asym(x[:, None], w[:, None], d, dmin, sc, m, nmax)
        f["sc"][:, s], f["m"][:, s] = sc[:, 0], m[:, 0]
        step[:, s], off[:, s] = d * sc[:, 0], dmin * m[:, 0]
    elif fmt == "Q6_K":
        d = f["d"]
        sc = np.clip(np.rint(_div(_sym_search(x, w, -32, 31), d)), -128, 127)[:, None]
        sc = _neighbours_sym(x[:, None], w[:, None], d, sc, -128, 127, -32, 31)
        f["sc"][:, s] = sc[:, 0]
        step[:, s] = d * sc[:, 0]


def encode_gptq(rows: np.ndarray, h: np.ndarray, fmt: str) -> bytes:
    """GPTQ over K-quant blocks: each block's scales are fit (weights diag H) when GPTQ reaches it."""
    S, nsub, qmin, qmax = SPEC[fmt]
    B = S * nsub                                                  # values per block (256, or 32 for Q8_0)
    rows = np.asarray(rows, np.float64)
    r, cols = rows.shape
    u = inverse_hessian_factor(h)
    hd = _floor(np.diag(h)[None, :])[0]
    span = max(B, 256)                                            # lazy-update width (a multiple of B)
    out = []
    for a in range(0, r, GPTQ_ROWS):
        w = rows[a:a + GPTQ_ROWS].copy()
        n = len(w)
        L = np.empty((n, cols))
        fields = []
        for b0 in range(0, cols, span):
            b1 = min(b0 + span, cols)
            err = np.empty((n, b1 - b0))
            for k0 in range(b0, b1, B):
                f, step, off = block_params(fmt, w[:, k0:k0 + B], np.broadcast_to(hd[k0:k0 + B], (n, B)))
                fields.append(f)
                for s0 in range(k0, k0 + B, S):                   # one sub-block: its own scale and minimum
                    s, s1 = (s0 - k0) // S, s0 + S
                    if s and GPTQ_SUBBLOCK_REFIT:                 # re-chosen on the weights as feedback left them
                        _refit_subblock(fmt, f, step, off, s, w[:, s0:s1], np.broadcast_to(hd[s0:s1], (n, S)))
                    for j in range(s0, s1):
                        col = w[:, j]
                        q = _round(col[:, None], step[:, s], off[:, s], qmin, qmax)[:, 0]
                        L[:, j] = q
                        e = (col - (step[:, s] * q - off[:, s])) / u[j, j]
                        err[:, j - b0] = e
                        if j + 1 < s1:
                            w[:, j + 1:s1] -= np.outer(e, u[j, j + 1:s1])
                    if s1 < b1:
                        w[:, s1:b1] -= _matmul(err[:, s0 - b0:s1 - b0], u[s0:s1, s1:b1])
            if b1 < cols:
                w[:, b1:] -= _matmul(err, u[b0:b1, b1:])
        # blocks of one row are consecutive in GGUF: interleave the per-block fields back to row-major order
        nb = cols // B
        f = {k: np.stack([g[k] for g in fields], 1).reshape((n * nb,) + fields[0][k].shape[1:]) for k in fields[0]}
        out.append(pack(fmt, f, L.reshape(n * nb, B)))
    return b"".join(out)


def _encode(ctx: EncodeContext, fmt: str) -> bytes:
    if fmt not in SPEC:
        raise ValueError(f"{NAME} encodes {', '.join(SPEC)}, not {fmt}")
    rows = np.asarray(ctx.rows, np.float32)
    rows = rows.reshape(-1, rows.shape[-1])
    if not all(np.isfinite(rows[r:r + 4096]).all() for r in range(0, len(rows), 4096)):
        raise ValueError(f"{ctx.unit.id}: weights are not finite")
    weights, h = importance(ctx)
    if h is not None and ctx.params.get("gptq", True):
        return encode_gptq(rows, h, fmt)
    if weights is None:
        return encode_weighted(rows, np.ones(rows.shape[1]), fmt)
    if len(weights) == 1:
        return encode_weighted(rows, weights[0], fmt)
    per = len(rows) // len(weights)                               # expert-major rows: one expert per pass
    return b"".join(encode_weighted(rows[e * per:(e + 1) * per], weights[e], fmt) for e in range(len(weights)))


ENCODER = register(Encoder(NAME, VERSION, "regenerable", _encode))
