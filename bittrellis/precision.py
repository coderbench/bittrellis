"""The precision space HPC-01 can actually deploy, and a resolver that mirrors the pinned
SparkInfer loader: given the bytes a checkpoint *stores*, what does the runtime *execute*?

This is the single most important fact in the project. SparkInfer does not execute arbitrary
precisions: it keeps NVFP4 native everywhere, keeps FP8 native only on the Gated DeltaNet
projections, and fits Q4_K at load time to anything else. A manifest therefore names the
precision that *runs*, and the builder chooses the stored bytes that make it run.

Rules below cite `runtime/src/models/qwen35.cpp` at the track's pinned commit.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model.qwen38 import Linear, Unit
from .safetensors_io import SafeTensorsDir

NVFP4 = "NVFP4"
FP8 = "FP8"
Q4_K = "Q4_K"
PRECISIONS = (NVFP4, FP8, Q4_K)

# What each unit kind may be assigned. A precision is listed only if the pinned runtime executes
# it natively for that kind (docs/precision_space.md has the evidence).
SPACE: dict[str, tuple[str, ...]] = {
    "gdn": (NVFP4, FP8, Q4_K),   # keep_native(): NVFP4 -> NVFP4, FP8 -> FP8, else Q4_K
    "attn": (NVFP4, Q4_K),       # attn_w(): NVFP4 -> NVFP4, anything else -> Q4_K
    "mlp": (NVFP4, Q4_K),        # ffn_is_nvfp4(gate) ? all three NVFP4 : all three Q4_K
    "lm_head": (NVFP4, Q4_K),    # always a Q4_K fit for AR decode; NVFP4 adds a packed-batch copy
}

# Stored formats, as the loader detects them from tensor names and dtypes.
S_NVFP4 = "nvfp4"   # ModelOpt (.weight U8 + .weight_scale + .weight_scale_2) or CT (.weight_packed ...)
S_FP8 = "fp8"       # .weight F8_E4M3 + per-row .weight_scale
S_BF16 = "bf16"
S_MISSING = "missing"
S_INVALID = "invalid"

# Bits per weight a precision costs in resident decode weights (group/row scales included).
BITS_PER_WEIGHT = {NVFP4: 4.5, FP8: 8.0, Q4_K: 4.5}


@dataclass(frozen=True)
class Resolved:
    """What the runtime does with one unit."""

    precision: str | None       # NVFP4 / FP8 / Q4_K, or None if the unit fails to load
    source: str                 # stored format the runtime reads (for Q4_K: what it fits from)
    note: str = ""

    @property
    def label(self) -> str:
        """The unit's selected format as a manifest names it (lm_head: its stored format)."""
        if self.precision is None:
            return "LOAD-ERROR"
        if self.precision == Q4_K and self.source != S_BF16:
            return f"Q4_K({self.source})"   # a refit of already-lossy bytes
        return self.precision


def stored_format(ck: SafeTensorsDir, lin: Linear) -> tuple[str, str]:
    """Detect a Linear's stored format exactly as nvfp4_src()/keep_fp8()/dequant_any() do."""
    p = lin.prefix
    gs = ck.get(p + ".weight_scale")
    wp = ck.get(p + ".weight_packed")
    glob = ck.get(p + ".weight_global_scale") if wp else None
    if wp is None:
        wp = ck.get(p + ".weight")
        glob = ck.get(p + ".weight_scale_2")
    if wp is not None and gs is not None and glob is not None and wp.dtype == "U8":
        if wp.numel != lin.numel // 2 or gs.numel != lin.numel // 16 or glob.numel != 1:
            return S_INVALID, "malformed NVFP4 shapes"
        return S_NVFP4, ""
    w = ck.get(p + ".weight")
    if w is None:
        return S_MISSING, f"missing {p}.weight"
    if w.dtype == "F8_E4M3":
        if gs is None or gs.numel != lin.rows or w.numel != lin.numel:
            return S_INVALID, "FP8 needs one scale per output row (per-tensor/block FP8 is rejected)"
        if gs.dtype != "BF16":
            return S_INVALID, f"FP8 scale must be BF16 (the loader copies it as BF16), got {gs.dtype}"
        return S_FP8, ""
    if w.dtype == "BF16" and w.numel == lin.numel:
        return S_BF16, ""
    return S_INVALID, f"unsupported dtype {w.dtype} or shape {w.shape}"


def resolve(unit: Unit, stored: list[str]) -> Resolved:
    """Runtime precision for a unit given the stored format of each of its Linears."""
    if any(s in (S_MISSING, S_INVALID) for s in stored):
        return Resolved(None, ",".join(stored), "unloadable tensor")
    kind = unit.kind
    if kind == "gdn":
        s = stored[0]
        if s == S_NVFP4:
            return Resolved(NVFP4, s)
        if s == S_FP8:
            return Resolved(FP8, s)
        return Resolved(Q4_K, s)
    if kind == "attn":
        s = stored[0]
        return Resolved(NVFP4, s) if s == S_NVFP4 else Resolved(Q4_K, s)
    if kind == "mlp":
        gate, up, down = stored
        if gate == S_NVFP4:
            if up != S_NVFP4 or down != S_NVFP4:
                return Resolved(None, ",".join(stored), "gate_proj NVFP4 requires up/down NVFP4")
            return Resolved(NVFP4, S_NVFP4)
        srcs = set(stored)
        src = srcs.pop() if len(srcs) == 1 else "mixed"
        return Resolved(Q4_K, src)
    if kind == "lm_head":
        s = stored[0]
        if s == S_NVFP4:
            return Resolved(NVFP4, s, "batch-1 decode executes a Q4_K fit of these bytes; NVFP4 copy serves packed batches")
        return Resolved(Q4_K, s)
    raise ValueError(kind)


def execution(kind: str, fmt: str) -> dict[str, str]:
    """What SparkInfer executes for a unit kind whose selected (stored) format is `fmt`.

    Batch-1 decode is the official path. The lm_head is the one unit with conditional semantics:
    AR decode always reads a Q4_K fit of the stored head; a stored NVFP4 head is additionally kept
    as an NVFP4 operand for wide packed decode, and only when >= 3 GiB of VRAM is free at load.
    """
    if kind == "lm_head":
        if fmt == NVFP4:
            return {"decode_b1": "Q4_K(nvfp4)", "packed_wide": "NVFP4 if free VRAM at load > payload + 3 GiB, else Q4_K(nvfp4)"}
        return {"decode_b1": "Q4_K", "packed_wide": "Q4_K"}
    return {"decode_b1": fmt, "packed_wide": fmt}


def stored_for(precision: str) -> str:
    """The stored format BitTrellis writes for a precision (Q4_K is always fitted from BF16)."""
    return {NVFP4: S_NVFP4, FP8: S_FP8, Q4_K: S_BF16}[precision]


def resolve_checkpoint(ck: SafeTensorsDir, units: list[Unit]) -> dict[str, Resolved]:
    out: dict[str, Resolved] = {}
    for u in units:
        fmts = [stored_format(ck, lin) for lin in u.linears]
        r = resolve(u, [f for f, _ in fmts])
        notes = "; ".join(n for _, n in fmts if n)
        out[u.id] = Resolved(r.precision, r.source, "; ".join(x for x in (r.note, notes) if x))
    return out


def weight_bytes(unit: Unit, precision: str) -> float:
    """Resident decode-weight bytes for a unit (an estimate; official footprint is measured)."""
    return unit.numel * BITS_PER_WEIGHT[precision] / 8.0


def stored_bytes(lin: Linear, fmt: str) -> int:
    if fmt == S_NVFP4:
        return lin.numel // 2 + lin.numel // 16 + 8
    if fmt == S_FP8:
        return lin.numel + 2 * lin.rows
    if fmt == S_BF16:
        return 2 * lin.numel
    raise ValueError(fmt)
