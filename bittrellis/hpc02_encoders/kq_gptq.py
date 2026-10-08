"""K-quant GPTQ: Q4_K / Q5_K / Q6_K bytes rounded to minimise each tensor's output error on the pinned statistics.

`kq_rtn` rounds every weight to its nearest level. What the model sees is the error of the tensor's output,
(w - q) H (w - q)ᵀ per row with H = mean x xᵀ of its input. Where the pinned HPC-02 statistics give H, this
encoder rounds by GPTQ: columns left to right, each column's rounding error pushed onto the columns not yet
rounded through the Cholesky factor of the damped inverse Hessian, so later roundings cancel earlier errors.

* Dense inputs with a full second moment: attention q/k/v and the recurrent qkv/z (`attn_input.xtx`), the
  shared expert gate/up (`ffn_input.xtx`) and down (`ffn_down_shexp.xtx`).
* Routed experts: only a per-expert diagonal is pinned (`exps_input.sumsq`, `exps_down.sumsq`). Gate/up use the
  layer's full `ffn_input.xtx` as the correlation structure at each expert's own channel energies
  (H_e = D_e C D_e, C the correlation of the layer's H, D_e the expert's channel RMS), which lets one Cholesky
  factor serve all 256 experts; down gets an importance-weighted grid fit with its expert's channel energies.
* No statistics (recurrent/attention output, lm_head, embeddings): the weighted grid fit with llama.cpp's default
  importance (x² plus the block's mean x²).
* Q6_K and Q8_0: `kq_rtn`'s bytes.

GGUF layouts are kept exactly (bittrellis/kquant.py). Each 256-value super-block's grid (6-bit sub-block scales
and minimums under f16 super-scales) is fitted when GPTQ reaches it, on the weights as earlier error feedback has
left them, by a weighted least-squares search over llama.cpp's make_qkx2 candidate steps, weighted by the
Hessian's diagonal; values then round to the nearest level of that grid.

Determinism: float64 NumPy on the CPU, fixed row chunks and column blocks.
"""

from __future__ import annotations

import numpy as np

from bittrellis import kquant
from bittrellis.hpc02 import Encoder, register

DAMP = 0.01
CHUNK_ROWS = 2048
NMAX = {"Q4_K": 15, "Q5_K": 31}


def inverse_hessian_factor(h: np.ndarray, damp: float = DAMP) -> np.ndarray:
    """Upper Cholesky factor U of the damped inverse Hessian (H + λI)⁻¹ = UᵀU, float64."""
    h = np.array(h, dtype=np.float64)
    dead = np.diag(h) <= 0
    h[dead, dead] = 1.0
    h[np.diag_indices_from(h)] += damp * float(np.mean(np.diag(h)))
    lo = np.linalg.cholesky(h)
    inv = np.linalg.inv(lo)
    return np.linalg.cholesky(inv.T @ inv).T


# Candidate steps for a sub-block, as fractions of its range over nmax: llama.cpp's make_qkx2 search (rmin -1,
# rdelta 0.1, 20 steps) tries iscale = (nmax + rmin + rdelta·k) / range; these are those candidates.
_STEPS = tuple(-1.0 + 0.1 * k for k in range(21))


def fit_asym(x: np.ndarray, nmax: int, wt: np.ndarray):
    """Weighted Q4_K/Q5_K grid. x [n, 8, 32], weights wt (broadcastable, > 0) -> (d, dmin f16 [n], sc, m [n, 8]).

    Per sub-block, every candidate step rounds the values, then the scale and minimum are refitted by weighted least
    squares for those levels; the candidate with the lowest weighted error wins (llama.cpp's make_qkx2 search, with
    the pinned input energies as weights where they exist). The sub-block scales and minimums are then stored as 6-bit
    multiples of the super-block's f16 d and dmin, as `kq_rtn` stores them."""
    x = np.asarray(x, np.float64)
    wt = np.broadcast_to(np.asarray(wt, np.float64), x.shape)
    lo = np.minimum(x.min(-1), 0.0)
    rng = x.max(-1) - lo
    best_s = rng / nmax
    best_m = -lo
    with np.errstate(divide="ignore", invalid="ignore"):
        q0 = np.where(best_s[..., None] > 0, np.clip(np.rint((x - lo[..., None]) / best_s[..., None]), 0, nmax), 0)
    best_e = (wt * (x - (q0 * best_s[..., None] - best_m[..., None])) ** 2).sum(-1)
    sw, swx = wt.sum(-1), (wt * x).sum(-1)
    for k in _STEPS:
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(rng > 0, (nmax + k) / rng, 0)
            q = np.clip(np.rint((x - lo[..., None]) * inv[..., None]), 0, nmax)
            swq, swqq, swqx = (wt * q).sum(-1), (wt * q * q).sum(-1), (wt * q * x).sum(-1)
            det = sw * swqq - swq * swq
            s = np.where(det > 0, (sw * swqx - swq * swx) / det, 0)
            mn = np.where(det > 0, (swq * swqx - swqq * swx) / det, 0)       # x ≈ s·q - mn (mn: the stored min)
        ok = (s > 0) & (mn >= 0)
        e = (wt * (x - (q * s[..., None] - mn[..., None])) ** 2).sum(-1)
        take = ok & (e < best_e)
        best_e = np.where(take, e, best_e)
        best_s = np.where(take, s, best_s)
        best_m = np.where(take, mn, best_m)
    d = (best_s.max(-1) / 63.0).astype(np.float16)
    dmin = (best_m.max(-1) / 63.0).astype(np.float16)
    df, dmf = d.astype(np.float64), dmin.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = np.where(df[:, None] > 0, np.clip(np.rint(best_s / df[:, None]), 0, 63), 0).astype(np.uint8)
        m = np.where(dmf[:, None] > 0, np.clip(np.rint(best_m / dmf[:, None]), 0, 63), 0).astype(np.uint8)
    return d, dmin, sc, m


def _round_asym(x: np.ndarray, d, dmin, sc, m, nmax: int) -> np.ndarray:
    step = d.astype(np.float64)[:, None] * sc
    off = dmin.astype(np.float64)[:, None] * m
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(step[..., None] > 0, np.rint((x + off[..., None]) / step[..., None]), 0)
    return np.clip(q, 0, nmax).astype(np.uint8)


def fit_rows(w: np.ndarray, fmt: str, wt: np.ndarray | None = None) -> bytes:
    """No statistics: the weighted grid fit alone, weights x² + mean x² (llama.cpp's default importance)."""
    x = np.asarray(w, np.float64).reshape(-1, 8, 32)
    if wt is None:
        x2 = x * x
        wt = x2 + x2.mean(axis=(1, 2), keepdims=True)
    d, dmin, sc, m = fit_asym(x, NMAX[fmt], wt)
    q = _round_asym(x, d, dmin, sc, m, NMAX[fmt])
    n = len(x)
    return PACK[fmt](d.reshape(n, 1), dmin.reshape(n, 1), sc.reshape(n, 1, 8), m.reshape(n, 1, 8), q.reshape(n, 256))


def _gptq_asym(w: np.ndarray, u: np.ndarray, nmax: int, scale: np.ndarray | None = None,
               hdiag: np.ndarray | None = None):
    """GPTQ for Q4_K/Q5_K. w [rows, cols] -> (d, dmin [rows, nb] f16, sc, m [rows, nb, 8], q [rows, cols]).

    `scale` [rows, cols] (optional, positive): run GPTQ on w·scale with the factor `u` of the scaled problem, rounding
    each value on its grid in the original space. Row r then minimises (w - q) D C D (w - q)ᵀ with D = diag(scale[r])
    and C the matrix `u` factors, so one factor serves rows whose Hessians differ only by channel energies.
    """
    w = np.array(w, dtype=np.float64)
    rows, cols = w.shape
    if scale is None:
        scale = np.ones_like(w)
    w *= scale
    hdiag = np.ones(cols) if hdiag is None else np.asarray(hdiag, np.float64)
    nb = cols // 256
    d = np.empty((rows, nb), np.float16)
    dmin = np.empty((rows, nb), np.float16)
    sc = np.empty((rows, nb, 8), np.uint8)
    mm = np.empty((rows, nb, 8), np.uint8)
    q = np.empty((rows, cols), np.uint8)
    for b in range(nb):
        c0, c1 = b * 256, b * 256 + 256
        w1 = w[:, c0:c1]
        s1 = scale[:, c0:c1]
        imp = (s1 * s1) * hdiag[None, c0:c1]                               # diag of D C D per row
        bd, bdm, bsc, bm = fit_asym((w1 / s1).reshape(rows, 8, 32), nmax, imp.reshape(rows, 8, 32))
        d[:, b], dmin[:, b], sc[:, b], mm[:, b] = bd, bdm, bsc, bm
        step = bd.astype(np.float64)[:, None] * bsc                          # [rows, 8]
        off = bdm.astype(np.float64)[:, None] * bm
        err = np.empty((rows, 256))
        u1 = u[c0:c1, c0:c1]
        for g in range(0, 256, 32):
            s, o = step[:, g // 32], off[:, g // 32]
            wg = w1[:, g:g + 32]
            for j in range(32):
                col = wg[:, j]
                sj = s1[:, g + j]
                with np.errstate(divide="ignore", invalid="ignore"):
                    qq = np.where(s > 0, np.clip(np.rint((col / sj + o) / s), 0, nmax), 0)
                q[:, c0 + g + j] = qq
                e = (col - (qq * s - o) * sj) / u1[g + j, g + j]
                err[:, g + j] = e
                if j < 31:
                    wg[:, j + 1:] -= e[:, None] * u1[g + j, g + j + 1:g + 32]
            if g + 32 < 256:
                w1[:, g + 32:] -= err[:, g:g + 32] @ u1[g:g + 32, g + 32:]
        if c1 < cols:
            w[:, c1:] -= err @ u[c0:c1, c1:]
    return d, dmin, sc, mm, q


def _pack_q4k(d, dmin, sc, m, q) -> bytes:
    rows, nb = d.shape
    n = rows * nb
    qb = q.reshape(n, 8, 32)
    qs = (qb[:, 0::2, :] | (qb[:, 1::2, :] << 4)).reshape(n, 128)
    blocks = np.concatenate([d.reshape(n).view(np.uint8).reshape(-1, 2), dmin.reshape(n).view(np.uint8).reshape(-1, 2),
                             kquant._pack_scales_q4k(sc.reshape(n, 8), m.reshape(n, 8)), qs], axis=1)
    return blocks.tobytes()


def _pack_q5k(d, dmin, sc, m, q) -> bytes:
    rows, nb = d.shape
    n = rows * nb
    qb = q.reshape(n, 8, 32)
    lo, hi = qb & 0x0F, qb >> 4
    qs = (lo[:, 0::2, :] | (lo[:, 1::2, :] << 4)).reshape(n, 128)
    qh = np.zeros((n, 32), np.uint8)
    for j in range(8):
        qh |= (hi[:, j, :] & 1) << j
    blocks = np.concatenate([d.reshape(n).view(np.uint8).reshape(-1, 2), dmin.reshape(n).view(np.uint8).reshape(-1, 2),
                             kquant._pack_scales_q4k(sc.reshape(n, 8), m.reshape(n, 8)), qh, qs], axis=1)
    return blocks.tobytes()


PACK = {"Q4_K": _pack_q4k, "Q5_K": _pack_q5k}


def quantize(w: np.ndarray, h: np.ndarray, fmt: str = "Q4_K", scale: np.ndarray | None = None,
             u: np.ndarray | None = None) -> bytes:
    """float rows [rows, cols] and H [cols, cols] (or its factor `u`) -> GGUF `fmt` bytes; `scale` as in _gptq_asym."""
    w = np.asarray(w, np.float64)
    hdiag = np.ones(w.shape[1]) if h is None else np.maximum(np.diag(np.asarray(h, np.float64)), 0)
    if u is None:
        u = inverse_hessian_factor(h)
    return b"".join(PACK[fmt](*_gptq_asym(w[r:r + CHUNK_ROWS], u, NMAX[fmt],
                                          None if scale is None else scale[r:r + CHUNK_ROWS], hdiag))
                    for r in range(0, len(w), CHUNK_ROWS))


# ------------------------------------------------------------------ the encoder

DENSE = {"gdn.qkv": "attn_input", "gdn.z": "attn_input", "attn.q": "attn_input", "attn.k": "attn_input",
         "attn.v": "attn_input", "shexp.gate": "ffn_input", "shexp.up": "ffn_input", "shexp.down": "ffn_down_shexp"}
EXPERT_IN = ("exps.gate", "exps.up")
N_EXPERTS = 256


def _stat(ctx, name: str) -> np.ndarray | None:
    cal = ctx.calibration
    if cal is None or cal.get(name) is None:
        return None
    return np.asarray(cal.array(name, "<f4"), np.float64)


def expert_scales(h: np.ndarray, sumsq: np.ndarray, count: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(correlation C of the layer's MoE input H, per-expert channel RMS D [experts, cols]).

    Expert e's Hessian is taken as D_e C D_e: the layer's correlation structure at the expert's own channel energies
    (exps_input.sumsq / count). An expert no calibration token reached gets the layer's energies."""
    dg = np.sqrt(np.maximum(np.diag(h), 0))
    floor = 1e-6 * float(dg.mean()) if dg.mean() > 0 else 1e-12
    dg = np.maximum(dg, floor)
    c = h / np.outer(dg, dg)
    d = np.where(count[:, None] > 0, np.sqrt(np.maximum(sumsq, 0) / np.maximum(count, 1)[:, None]), dg[None, :])
    return c, np.maximum(d, floor)


def _fallback(rows: np.ndarray, fmt: str) -> bytes:
    if fmt not in NMAX:
        return kquant.RTN[fmt](rows)
    w = np.asarray(rows).reshape(-1, np.asarray(rows).shape[-1])
    return b"".join(fit_rows(w[r:r + CHUNK_ROWS], fmt) for r in range(0, len(w), CHUNK_ROWS))


def _encode(ctx, fmt: str) -> bytes:
    unit, rows = ctx.unit, ctx.rows
    if fmt not in NMAX or unit.layer is None:
        return _fallback(rows, fmt)
    if unit.kind in DENSE:
        h = _stat(ctx, f"blk.{unit.layer}.{DENSE[unit.kind]}.xtx")
        if h is None or h.shape != (unit.cols, unit.cols):
            return _fallback(rows, fmt)
        return quantize(rows, h, fmt)
    if unit.kind in EXPERT_IN:
        h = _stat(ctx, f"blk.{unit.layer}.ffn_input.xtx")
        ss = _stat(ctx, f"blk.{unit.layer}.exps_input.sumsq")
        cnt = _stat(ctx, f"blk.{unit.layer}.exps.count")
        if h is None or ss is None or cnt is None or h.shape != (unit.cols, unit.cols) or rows.shape[0] % N_EXPERTS:
            return _fallback(rows, fmt)
        c, d = expert_scales(h, ss, cnt)
        per = rows.shape[0] // N_EXPERTS
        u = inverse_hessian_factor(c)
        out = []
        for r in range(0, rows.shape[0], CHUNK_ROWS):
            r1 = min(r + CHUNK_ROWS, rows.shape[0])
            scale = d[np.arange(r, r1) // per]
            out.append(PACK[fmt](*_gptq_asym(rows[r:r1], u, NMAX[fmt], scale, np.diag(c))))
        return b"".join(out)
    if unit.kind == "exps.down" and fmt in NMAX:
        sd = _stat(ctx, f"blk.{unit.layer}.exps_down.sumsq")
        cnt = _stat(ctx, f"blk.{unit.layer}.exps.count")
        if sd is not None and cnt is not None and sd.shape == (N_EXPERTS, unit.cols) and rows.shape[0] % N_EXPERTS == 0:
            per = rows.shape[0] // N_EXPERTS
            energy = sd / np.maximum(cnt, 1)[:, None]
            energy = np.where(cnt[:, None] > 0, energy, energy[cnt > 0].mean(0) if (cnt > 0).any() else 1.0)
            energy = np.maximum(energy, 1e-12 * max(float(energy.mean()), 1e-30))
            w = np.asarray(rows)
            out = []
            for r in range(0, w.shape[0], CHUNK_ROWS):
                r1 = min(r + CHUNK_ROWS, w.shape[0])
                wt = energy[np.arange(r, r1) // per].reshape(r1 - r, -1, 8, 32).reshape(-1, 8, 32)
                out.append(fit_rows(w[r:r1], fmt, wt))
            return b"".join(out)
    return _fallback(rows, fmt)


register(Encoder("kq_gptq", 1, "regenerable", _encode))
