"""Render README visuals from real data: a banner and per-manifest layer maps (SVG).

    python scripts/render_readme_assets.py            # writes docs/assets/*.svg

Every cell in a layer map is one searchable unit of Qwen3.8-27B, colored by what the manifest
assigns, so the pictures stay truthful when manifests change.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from bittrellis.manifest import Manifest  # noqa: E402
from bittrellis.model.qwen38 import Qwen38Arch  # noqa: E402

OUT = REPO / "docs/assets"
BG, FG, MUTED, GRID = "#0b1020", "#e5e7eb", "#94a3b8", "#1e293b"
COLORS = {
    "NVFP4@baseline": ("#3b82f6", "NVFP4 · shipped encoder"),
    "NVFP4@unsloth": ("#a855f7", "NVFP4 · calibrated encoder"),
    "NVFP4@rtn": ("#60a5fa", "NVFP4 · round-to-nearest"),
    "FP8@rtn": ("#f59e0b", "FP8"),
    "Q4_K@runtime": ("#14b8a6", "Q4_K (fitted at load)"),
}
ROWS = (("GDN q·k·v", "gdn.qkv"), ("GDN gate z", "gdn.z"), ("GDN out", "gdn.out"), ("attention", "attn"), ("MLP", "mlp"))


def _cells(manifest: Manifest) -> tuple[dict[tuple[str, int], str], str]:
    units = Qwen38Arch().units()
    asg = manifest.expand_assignments(units)
    cells: dict[tuple[str, int], str] = {}
    for u in units:
        if u.layer is None:
            continue
        row = "attn" if u.kind == "attn" else u.role
        key = f"{asg[u.id].format}@{asg[u.id].quantizer}"
        # attention has four units per layer: a cell shows a non-default assignment if any unit has one
        prev = cells.get((row, u.layer))
        if prev is None or (prev == "NVFP4@baseline" and key != prev):
            cells[(row, u.layer)] = key
    return cells, f"{asg['lm_head'].format}@{asg['lm_head'].quantizer}"


def layer_map(manifest: Manifest, title: str, subtitle: str, path: Path, legend: bool = True) -> None:
    cells, head = _cells(manifest)
    cw, ch, gap = 13, 18, 2
    left, top = 118, 58
    width = left + 64 * (cw + gap) + 70
    height = top + len(ROWS) * (ch + gap) + (70 if legend else 34)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             f'font-family="Inter,Segoe UI,Helvetica,Arial,sans-serif">',
             f'<rect width="100%" height="100%" rx="14" fill="{BG}"/>',
             f'<text x="20" y="26" fill="{FG}" font-size="15" font-weight="700">{title}</text>',
             f'<text x="20" y="45" fill="{MUTED}" font-size="11">{subtitle}</text>']
    for r, (label, key) in enumerate(ROWS):
        y = top + r * (ch + gap)
        parts.append(f'<text x="{left - 10}" y="{y + ch - 5}" fill="{MUTED}" font-size="11" text-anchor="end">{label}</text>')
        for layer in range(64):
            x = left + layer * (cw + gap)
            k = cells.get((key, layer))
            if k is None:
                parts.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="2" fill="{GRID}" opacity="0.45"/>')
            else:
                parts.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="2" fill="{COLORS[k][0]}"/>')
    hx = left + 64 * (cw + gap) + 14
    parts.append(f'<rect x="{hx}" y="{top}" width="{cw + 8}" height="{len(ROWS) * (ch + gap) - gap}" rx="3" fill="{COLORS[head][0]}"/>')
    parts.append(f'<text x="{hx + 10}" y="{top - 8}" fill="{MUTED}" font-size="10" text-anchor="middle">head</text>')
    ybase = top + len(ROWS) * (ch + gap) + 14
    for layer in (0, 16, 32, 48, 63):
        parts.append(f'<text x="{left + layer * (cw + gap) + cw / 2}" y="{ybase}" fill="{MUTED}" font-size="10" text-anchor="middle">L{layer}</text>')
    if legend:
        lx = 20
        for color, label in COLORS.values():
            parts.append(f'<rect x="{lx}" y="{ybase + 18}" width="12" height="12" rx="2" fill="{color}"/>')
            parts.append(f'<text x="{lx + 18}" y="{ybase + 28}" fill="{FG}" font-size="11">{label}</text>')
            lx += 36 + 6.2 * len(label)
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


EXAMPLE = {
    "schema": "bittrellis/manifest@2", "track": "HPC-01", "name": "readme-example", "default": "NVFP4",
    "rules": [{"match": "L*.mlp", "layers": "0-55", "format": "NVFP4", "quantizer": "unsloth"},
              {"match": "L*.gdn.qkv", "layers": "48-63", "format": "FP8"},
              {"match": "L*.mlp", "layers": "60-63", "format": "Q4_K"}],
}


def banner(path: Path) -> None:
    """Title banner whose trellis is a real, legal mixed recipe rendered unit by unit."""
    cells, head = _cells(Manifest.from_dict(EXAMPLE))
    w, h = 1200, 320
    cw, ch, gap = 6, 22, 2
    x0, y0 = 640, 70
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
             f'font-family="Inter,Segoe UI,Helvetica,Arial,sans-serif">',
             '<defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#60a5fa"/>'
             '<stop offset="1" stop-color="#c084fc"/></linearGradient></defs>',
             f'<rect width="100%" height="100%" rx="18" fill="{BG}"/>']
    for r, (_, key) in enumerate(ROWS):
        y = y0 + r * (ch + 8)
        for layer in range(64):
            x = x0 + layer * (cw + gap)
            k = cells.get((key, layer))
            fill, op = (COLORS[k][0], 1.0) if k else (GRID, 0.6)
            parts.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="1.5" fill="{fill}" opacity="{op}"/>')
    hx = x0 + 64 * (cw + gap) + 10
    parts.append(f'<rect x="{hx}" y="{y0}" width="14" height="{len(ROWS) * (ch + 8) - 8}" rx="3" fill="{COLORS[head][0]}"/>')
    parts.append(f'<text x="{hx + 7}" y="{y0 - 10}" fill="{MUTED}" font-size="10" text-anchor="middle">head</text>')
    parts.append(f'<text x="{x0}" y="{y0 - 10}" fill="{MUTED}" font-size="10">layer 0</text>')
    parts.append(f'<text x="{x0 + 63 * (cw + gap) + cw}" y="{y0 - 10}" fill="{MUTED}" font-size="10" text-anchor="end">layer 63</text>')
    ly = y0 + len(ROWS) * (ch + 8) + 16
    lx = x0
    for key, label in (("NVFP4@baseline", "NVFP4"), ("NVFP4@unsloth", "NVFP4 calibrated"), ("FP8@rtn", "FP8"), ("Q4_K@runtime", "Q4_K")):
        parts.append(f'<rect x="{lx}" y="{ly}" width="11" height="11" rx="2" fill="{COLORS[key][0]}"/>')
        parts.append(f'<text x="{lx + 16}" y="{ly + 10}" fill="{FG}" font-size="11">{label}</text>')
        lx += 34 + 6.2 * len(label)
    parts.append(f'<text x="{x0}" y="{ly + 34}" fill="{MUTED}" font-size="11">one square = one part of one layer · a real, legal recipe</text>')
    parts += [
        '<text x="52" y="128" fill="url(#g)" font-size="62" font-weight="800" letter-spacing="-1">BitTrellis</text>',
        f'<text x="54" y="172" fill="{FG}" font-size="22" font-weight="600">Every layer doesn\'t deserve the same bits.</text>',
        f'<text x="54" y="206" fill="{MUTED}" font-size="16">Find the best compressed LLM for the GPU you actually run.</text>',
        f'<text x="54" y="268" fill="{MUTED}" font-size="13">Qwen3.8-27B · 1× RTX 5090 · SparkInfer · measured on real hardware</text>',
        "</svg>",
    ]
    path.write_text("\n".join(parts) + "\n")


LABELS = {
    "V0-baseline-rebuild": "V0 · today's shipped checkpoint",
    "V1-all-q4k": "V1 · everything Q4_K",
    "V3-gdn-fp8": "V3 · recurrent path FP8",
    "V4-gdn-q4k": "V4 · recurrent path Q4_K",
    "V5-attn-q4k": "V5 · attention Q4_K",
    "V6-mlp-q4k": "V6 · MLP Q4_K",
    "V7-mlp-q4k-early": "V7 · early MLP Q4_K",
    "V9-gdn-q4k-mlp-q4k": "V9 · recurrent + MLP Q4_K",
    "V13-mlp-unsloth-bytes": "V13 · calibrated MLP encoder",
}


def _label(row: dict) -> str:
    if row["kind"] == "external":
        return {"R1": "unsloth checkpoint (external)", "R2": "llama.cpp best GGUF (external)"}.get(row["id"], row["name"])
    return LABELS.get(row["name"], row["name"])


def quality_chart(frontier: dict, path: Path) -> None:
    """Horizontal bars: how far each checkpoint's answers drift from the original model (lower is better)."""
    rows = sorted(frontier["internal"] + frontier["external"], key=lambda r: r["rp_kl"])
    inc = next(r for r in frontier["internal"] if r["name"] == frontier["incumbent"])
    w, left, bar_h, gap, top = 1100, 300, 22, 10, 86
    h = top + len(rows) * (bar_h + gap) + 60
    scale = (w - left - 170) / max(r["rp_kl"] for r in rows)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
             f'font-family="Inter,Segoe UI,Helvetica,Arial,sans-serif">',
             f'<rect width="100%" height="100%" rx="16" fill="{BG}"/>',
             f'<text x="24" y="36" fill="{FG}" font-size="19" font-weight="700">How far does each checkpoint drift from the original model?</text>',
             f'<text x="24" y="60" fill="{MUTED}" font-size="12">Reference-Partition KL vs BF16 on 21,624 scored tokens · shorter bar = answers closer to the original · measured on one RTX 5090</text>']
    for i, r in enumerate(rows):
        y = top + i * (bar_h + gap)
        length = r["rp_kl"] * scale
        if r["kind"] == "external":
            fill, stroke, op = "none", "#94a3b8", 1
        elif r["name"] == frontier["incumbent"]:
            fill, stroke, op = "#64748b", "none", 1
        else:
            fill, stroke, op = ("#a855f7" if r["frontier"] else "#3b82f6"), "none", (1 if r["valid"] else 0.4)
        parts.append(f'<text x="{left - 12}" y="{y + bar_h - 6}" fill="{FG if r["kind"] == "internal" else MUTED}" '
                     f'font-size="13" text-anchor="end">{_label(r)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{length:.1f}" height="{bar_h}" rx="4" fill="{fill}" stroke="{stroke}" '
                     f'stroke-width="1.5" stroke-dasharray="{"4 3" if r["kind"] == "external" else "0"}" opacity="{op}"/>')
        delta = (r["rp_kl"] - inc["rp_kl"]) / inc["rp_kl"]
        tag = "today" if r["name"] == frontier["incumbent"] else f"{abs(delta) * 100:.0f}% {'closer' if delta < 0 else 'further'}"
        color = "#22c55e" if delta < -0.02 else ("#f87171" if delta > 0.02 else MUTED)
        parts.append(f'<text x="{left + length + 10}" y="{y + bar_h - 6}" fill="{color if r["name"] != frontier["incumbent"] else MUTED}" '
                     f'font-size="12" font-weight="600">{r["rp_kl"]:.3f} · {tag}</text>')
    ly = h - 22
    for x, fill, stroke, label in ((24, "#a855f7", "none", "on the frontier"), (170, "#3b82f6", "none", "measured seed"),
                                   (310, "#64748b", "none", "today's checkpoint"), (470, "none", "#94a3b8", "external reference, not ranked")):
        parts.append(f'<rect x="{x}" y="{ly - 10}" width="14" height="12" rx="2" fill="{fill}" stroke="{stroke}" stroke-dasharray="3 2"/>')
        parts.append(f'<text x="{x + 20}" y="{ly}" fill="{MUTED}" font-size="12">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def tradeoff_chart(frontier: dict, path: Path) -> None:
    """Two scatter panels: closeness to the original vs speed, and vs GPU memory."""
    rows = frontier["internal"] + frontier["external"]
    w, h = 1100, 430
    panels = (("decode_tps", "generation speed (tokens/s) →  faster", False), ("peak_gpu_gib", "← less GPU memory (GiB)", True))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
             f'font-family="Inter,Segoe UI,Helvetica,Arial,sans-serif">',
             f'<rect width="100%" height="100%" rx="16" fill="{BG}"/>',
             f'<text x="24" y="36" fill="{FG}" font-size="19" font-weight="700">Quality vs speed, quality vs memory</text>',
             f'<text x="24" y="58" fill="{MUTED}" font-size="12">up = closer to the original model · purple = frontier (nothing measured beats it on every axis) · hollow = external, not ranked</text>']
    kl_max = max(r["rp_kl"] for r in rows) * 1.08
    kl_min = min(r["rp_kl"] for r in rows) * 0.9
    for p_i, (key, xlabel, invert) in enumerate(panels):
        px, py, pw, ph = 70 + p_i * 520, 84, 440, 290
        vals = [r[key] for r in rows]
        lo, hi = min(vals), max(vals)
        pad = (hi - lo) * 0.12 or 1
        lo, hi = lo - pad, hi + pad
        parts.append(f'<rect x="{px}" y="{py}" width="{pw}" height="{ph}" fill="none" stroke="{GRID}"/>')
        for t in range(5):
            gy = py + ph * t / 4
            parts.append(f'<line x1="{px}" y1="{gy}" x2="{px + pw}" y2="{gy}" stroke="{GRID}" stroke-width="0.6"/>')
        parts.append(f'<text x="{px + pw / 2}" y="{py + ph + 32}" fill="{MUTED}" font-size="12" text-anchor="middle">{xlabel}</text>')
        parts.append(f'<text x="{px - 44}" y="{py + ph / 2}" fill="{MUTED}" font-size="12" text-anchor="middle" '
                     f'transform="rotate(-90 {px - 44} {py + ph / 2})">↑ closer to original</text>')
        for v in (lo + pad, hi - pad):
            vx = px + (v - lo) / (hi - lo) * pw
            vx = px + pw - (vx - px) if invert else vx
            parts.append(f'<text x="{vx}" y="{py + ph + 16}" fill="{MUTED}" font-size="10" text-anchor="middle">{v:.1f}</text>')
        for r in rows:
            x = px + (r[key] - lo) / (hi - lo) * pw
            x = px + pw - (x - px) if invert else x
            y = py + (r["rp_kl"] - kl_min) / (kl_max - kl_min) * ph  # lowest RP-KL (closest to BF16) at the top
            if r["kind"] == "external":
                parts.append(f'<rect x="{x - 6}" y="{y - 6}" width="12" height="12" fill="none" stroke="#94a3b8" stroke-width="1.6"/>')
            else:
                color = "#64748b" if r["name"] == frontier["incumbent"] else ("#a855f7" if r["frontier"] else "#3b82f6")
                parts.append(f'<circle cx="{x}" cy="{y}" r="{7 if r["frontier"] else 5}" fill="{color}" opacity="{1 if r["valid"] else 0.4}"/>')
            tag = r["id"] if r["kind"] == "external" else r["name"].split("-")[0]
            parts.append(f'<text x="{x + 9}" y="{y + 4}" fill="{MUTED}" font-size="10">{tag}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    banner(OUT / "banner.svg")
    variants = REPO / "experiments/feasibility/variants"
    maps = {
        "V0-baseline-rebuild": ("Today: one recipe everywhere", "the shipped checkpoint (V0): every part NVFP4 with the same encoder"),
        "V13-mlp-unsloth-bytes": ("Same format, better encoder", "V13: MLP layers 0-55 use calibrated NVFP4 bytes; speed and memory unchanged"),
        "V3-gdn-fp8": ("Protect the recurrent path", "V3: every Gated DeltaNet projection at FP8; better long-context fidelity, slower decode"),
        "V7-mlp-q4k-early": ("Trade prefill for memory", "V7: MLP layers 0-31 as Q4_K; less GPU memory, slower prompt reading"),
    }
    for name, (title, sub) in maps.items():
        layer_map(Manifest.load(variants / f"{name}.yaml"), title, sub, OUT / f"layers-{name}.svg")
    frontier_json = REPO / "results/feasibility/frontier.json"
    if frontier_json.exists():
        import json

        frontier = json.loads(frontier_json.read_text())
        quality_chart(frontier, OUT / "results-quality.svg")
        tradeoff_chart(frontier, OUT / "results-tradeoffs.svg")
        print("wrote results charts")
    print(f"wrote {len(maps) + 1} SVGs to {OUT}")


if __name__ == "__main__":
    main()
