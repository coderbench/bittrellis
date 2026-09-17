"""Render the README figures from real data (SVG, light and dark theme).

    python scripts/render_readme_assets.py            # writes docs/assets/*.svg

- hero.svg          what the project does, in one picture
- how-it-works.svg  the five steps every candidate goes through
- recipes.svg       three real manifests, drawn layer by layer, with their measured result
- scorecard.svg     every measured checkpoint against today's checkpoint (V0)

Layer colors come from the manifests and every number from results/feasibility, so the pictures stay
truthful when either changes. CI re-renders them and fails if the committed files differ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from bittrellis.manifest import Manifest  # noqa: E402
from bittrellis.model.qwen38 import Qwen38Arch  # noqa: E402

OUT = REPO / "docs/assets"
RESULTS = REPO / "results/feasibility"
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"

# One palette, readable on GitHub's light and dark backgrounds.
STYLE = """<style>
.bg{fill:#ffffff;stroke:#d0d7de}.card{fill:#f6f8fa;stroke:#d0d7de}.accent{fill:#f3efff;stroke:#8b5cf6}
.t{fill:#1f2328}.m{fill:#59636e}.rule{stroke:#d0d7de}.track{fill:#e6e9ed}.gpu{stroke:#1f2328}
@media (prefers-color-scheme: dark){
.bg{fill:#0d1117;stroke:#30363d}.card{fill:#161b22;stroke:#30363d}.accent{fill:#1d1633;stroke:#8b5cf6}
.t{fill:#e6edf3}.m{fill:#9198a1}.rule{stroke:#30363d}.track{fill:#21262d}.gpu{stroke:#e6edf3}}
</style>"""

FORMATS = {  # manifest assignment → (color, legend label)
    "NVFP4@baseline": ("#5b8def", "NVFP4, standard encoder"),
    "NVFP4@rtn": ("#5b8def", "NVFP4, standard encoder"),
    "NVFP4@unsloth": ("#8b5cf6", "NVFP4, calibrated encoder"),
    "FP8@rtn": ("#e3a008", "FP8"),
    "Q4_K@runtime": ("#14b8a6", "Q4_K"),
}
BETTER, WORSE, SAME = "#1a7f37", "#cf222e", "#8c959f"


def svg(w: int, h: int, body: list[str]) -> str:
    return "\n".join([f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" font-family="{FONT}">',
                      STYLE, f'<rect class="bg" x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="12"/>', *body, "</svg>"]) + "\n"


def text(x: float, y: float, s: str, cls: str = "t", size: int = 13, weight: int = 400, anchor: str = "start", fill: str | None = None) -> str:
    paint = f'fill="{fill}"' if fill else f'class="{cls}"'
    return f'<text x="{x:.1f}" y="{y:.1f}" {paint} font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{s}</text>'


# ── hero ────────────────────────────────────────────────────────────────────────────────────────


def hero(path: Path, shipped_gib: float) -> None:
    w, h = 1200, 330
    b = [text(40, 64, "BitTrellis", size=40, weight=700),
         text(40, 96, "Finds the best way to compress a large language model for one specific GPU,", "m", 17),
         text(40, 120, "keeping as much of the original model's quality as possible.", "m", 17)]
    cards = [
        ("Original model", "Qwen3.8-27B in full precision (BF16)", 52.0, "too big for a 32 GB GPU", WORSE, "card"),
        ("BitTrellis", "chooses the format and encoder", None, "for each part of the model", None, "accent"),
        ("Compressed checkpoint", "runs on one RTX 5090 with SparkInfer", shipped_gib, "fits, close to the original", BETTER, "card"),
    ]
    cw, gap, top, ch = 336, 56, 152, 150
    for i, (title, line1, gib, line2, color, cls) in enumerate(cards):
        x = 40 + i * (cw + gap)
        b.append(f'<rect class="{cls}" x="{x}" y="{top}" width="{cw}" height="{ch}" rx="10"/>')
        b.append(text(x + 20, top + 34, title, size=17, weight=600))
        b.append(text(x + 20, top + 58, line1, "m", 13))
        if gib is None:
            b.append(text(x + 20, top + 78, line2, "m", 13))
            for j, (fmt, label) in enumerate((("NVFP4@baseline", "NVFP4"), ("NVFP4@unsloth", "NVFP4 calibrated"), ("FP8@rtn", "FP8"), ("Q4_K@runtime", "Q4_K"))):
                bx, by = x + 20 + (j % 2) * 150, top + 104 + (j // 2) * 24
                b.append(f'<rect x="{bx}" y="{by - 11}" width="12" height="12" rx="2" fill="{FORMATS[fmt][0]}"/>')
                b.append(text(bx + 18, by, label, "m", 13))
            continue
        # memory bar: the dashed line is the GPU's 32 GB
        bx, by, scale = x + 20, top + 86, (cw - 40) / 56
        b.append(f'<rect class="track" x="{bx}" y="{by}" width="{32 * scale:.1f}" height="16" rx="3"/>')
        fit = min(gib, 32) * scale
        b.append(f'<rect x="{bx}" y="{by}" width="{fit:.1f}" height="16" rx="3" fill="{color}" opacity="0.85"/>')
        if gib > 32:
            b.append(f'<rect x="{bx + 32 * scale:.1f}" y="{by}" width="{(gib - 32) * scale:.1f}" height="16" rx="3" fill="{color}" opacity="0.35"/>')
        gx = bx + 32 * scale
        b.append(f'<line class="gpu" x1="{gx:.1f}" y1="{by - 8}" x2="{gx:.1f}" y2="{by + 24}" stroke-width="1.5" stroke-dasharray="3 3"/>')
        b.append(text(gx, by - 12, "32 GB GPU", "m", 11, anchor="middle"))
        b.append(text(bx, by + 44, f"{gib:.0f} GB · {line2}", size=13, weight=600, fill=color))
    for i in range(2):
        ax = 40 + cw + i * (cw + gap) + 12
        ay = top + ch / 2
        b.append(f'<path d="M{ax} {ay} h{gap - 24}" class="gpu" stroke-width="2" fill="none"/>')
        b.append(f'<path d="M{ax + gap - 24} {ay - 6} l8 6 l-8 6 z" class="t"/>')
    path.write_text(svg(w, h, b))


# ── how it works ────────────────────────────────────────────────────────────────────────────────

STEPS = (
    ("Recipe", ["a short YAML file: which", "format and encoder each", "part of the model gets"]),
    ("Build", ["turn the original weights", "into a real checkpoint"]),
    ("Audit", ["prove the checkpoint holds", "only legal encodings of", "the original weights"]),
    ("Measure", ["quality, speed and memory", "on the real RTX 5090"]),
    ("Frontier", ["kept only if nothing else", "beats it on every measure"]),
)


def how_it_works(path: Path) -> None:
    w, h = 1200, 200
    step = (w - 80) / len(STEPS)
    cy = 52
    b = [f'<line class="rule" x1="{40 + step / 2}" y1="{cy}" x2="{40 + step * (len(STEPS) - 0.5)}" y2="{cy}" stroke-width="2"/>']
    for i, (title, lines) in enumerate(STEPS):
        cx = 40 + step * (i + 0.5)
        last = i == len(STEPS) - 1
        b.append(f'<circle cx="{cx}" cy="{cy}" r="18" fill="{"#8b5cf6" if last else "#5b8def"}"/>')
        b.append(text(cx, cy + 5, str(i + 1), size=15, weight=700, anchor="middle", fill="#ffffff"))
        b.append(text(cx, cy + 50, title, size=16, weight=600, anchor="middle"))
        for j, line in enumerate(lines):
            b.append(text(cx, cy + 74 + j * 19, line, "m", 13, anchor="middle"))
    path.write_text(svg(w, h, b))


# ── recipes ─────────────────────────────────────────────────────────────────────────────────────


def _layers(manifest: Manifest) -> dict[tuple[str, int], str]:
    """(row, layer) → assignment key. Row "mixer" is the GDN or attention block, "mlp" the MLP."""
    units = Qwen38Arch().units()
    asg = manifest.expand_assignments(units)
    cells: dict[tuple[str, int], str] = {}
    for u in units:
        if u.layer is None:
            continue
        row = "mlp" if u.kind == "mlp" else "mixer"
        key = f"{asg[u.id].format}@{asg[u.id].quantizer}"
        prev = cells.get((row, u.layer))
        if prev is None or (prev.startswith("NVFP4@") and not key.startswith("NVFP4@")) or (prev == "NVFP4@baseline" and key != prev):
            cells[(row, u.layer)] = key
    return cells


def recipes(path: Path, entries: list[tuple[str, str, str, list[tuple[str, str]]]]) -> None:
    """entries: (manifest name, title, description, [(result text, color)])."""
    arch = Qwen38Arch()
    attention = {u.layer for u in arch.units() if u.kind == "attn"}
    w, left, cw, cg, rh, block = 1200, 330, 11, 2, 18, 124
    h = 104 + block * len(entries) + 30
    b = [text(40, 42, "Same model, different recipes", size=20, weight=700),
         text(40, 66, "One column is one of the 64 layers. Top row: the layer's attention or recurrent block. Bottom row: its MLP.", "m", 13)]
    for layer in (0, 16, 32, 48, 63):
        b.append(text(left + layer * (cw + cg) + cw / 2, 92, f"layer {layer}", "m", 11, anchor="middle"))
    variants = REPO / "experiments/feasibility/variants"
    used: set[str] = set()
    for i, (name, title, desc, results) in enumerate(entries):
        y = 104 + i * block
        if i:
            b.append(f'<line class="rule" x1="40" y1="{y - 12}" x2="{w - 40}" y2="{y - 12}"/>')
        b.append(text(40, y + 16, title, size=15, weight=600))
        b.append(text(40, y + 37, desc, "m", 13))
        for j, (s, color) in enumerate(results):
            b.append(text(40, y + 62 + j * 19, s, size=13, weight=600, fill=color))
        cells = _layers(Manifest.load(variants / f"{name}.yaml"))
        used.update(cells.values())
        for r, row in enumerate(("mixer", "mlp")):
            ry = y + 4 + r * (rh + 6)
            for layer in range(64):
                x = left + layer * (cw + cg)
                color = FORMATS[cells[(row, layer)]][0]
                if row == "mixer" and layer in attention:
                    b.append(f'<rect x="{x + 1}" y="{ry + 1}" width="{cw - 2}" height="{rh - 2}" rx="2" fill="none" stroke="{color}" stroke-width="2"/>')
                else:
                    b.append(f'<rect x="{x}" y="{ry}" width="{cw}" height="{rh}" rx="2" fill="{color}"/>')
    ly = h - 24
    lx = 40
    seen = []
    for key, (color, label) in FORMATS.items():
        if label in seen or key not in used:
            continue
        seen.append(label)
        b.append(f'<rect x="{lx}" y="{ly - 11}" width="12" height="12" rx="2" fill="{color}"/>')
        b.append(text(lx + 18, ly, label, "m", 13))
        lx += 18 + 7.4 * len(label) + 24
    b.append(f'<rect x="{lx}" y="{ly - 11}" width="12" height="12" rx="2" fill="none" class="gpu" stroke-width="1.5"/>')
    b.append(text(lx + 18, ly, "outlined = attention layer, filled = recurrent layer", "m", 13))
    path.write_text(svg(w, h, b))


# ── scorecard ───────────────────────────────────────────────────────────────────────────────────

LABELS = {
    "V1-all-q4k": "everything Q4_K",
    "V3-gdn-fp8": "recurrent layers FP8",
    "V4-gdn-q4k": "recurrent layers Q4_K",
    "V5-attn-q4k": "attention Q4_K",
    "V6-mlp-q4k": "MLP Q4_K",
    "V7-mlp-q4k-early": "MLP Q4_K, first half",
    "V9-gdn-q4k-mlp-q4k": "recurrent + MLP Q4_K",
    "V13-mlp-unsloth-bytes": "calibrated MLP encoder",
    "R1": "unsloth NVFP4 checkpoint",
    "R2": "llama.cpp, UD-Q4_K_M GGUF",
}


def _pct(x: float) -> str:
    if abs(x) < 0.0005:
        return "0%"
    digits = 1 if abs(x) < 0.01 else 0
    return f"{'+' if x > 0 else '−'}{abs(x) * 100:.{digits}f}%"


def scorecard(path: Path, frontier: dict, analysis: dict, floors: dict) -> None:
    inc = next(r for r in frontier["internal"] if r["name"] == frontier["incumbent"])
    vs = analysis["vs_incumbent"]

    def cells(r: dict) -> list[tuple[float, str, str]]:
        """(bar value as a fraction, label, color); positive = better than V0."""
        key = r["id"] if r["kind"] == "external" else r["name"]
        q = -(r["rp_kl"] - inc["rp_kl"]) / inc["rp_kl"]
        q_same = not vs[key]["rp_kl"]["significant"]
        d = r["decode_tps"] / inc["decode_tps"] - 1
        p = r["prefill_tps"] / inc["prefill_tps"] - 1
        m = r["peak_gpu_gib"] - inc["peak_gpu_gib"]
        mem_same = abs(m) < floors["peak_gpu_gib"]
        return [
            (q, _pct(q), SAME if q_same else (BETTER if q > 0 else WORSE)),
            (d, _pct(d), SAME if abs(d) < floors["decode_tps"] else (BETTER if d > 0 else WORSE)),
            (p, _pct(p), SAME if abs(p) < floors["prefill_tps"] else (BETTER if p > 0 else WORSE)),
            (-m / 10, "same" if mem_same else f"{'+' if m > 0 else '−'}{abs(m):.1f} GB", SAME if mem_same else (BETTER if m < 0 else WORSE)),
        ]

    internal = sorted((r for r in frontier["internal"] if r["name"] != inc["name"]), key=lambda r: r["rp_kl"])
    external = sorted(frontier["external"], key=lambda r: r["rp_kl"])
    w, left, colw, rowh = 1200, 330, 210, 36
    top = 150
    h = top + rowh * (len(internal) + len(external)) + 56 + 64
    cols = ("Closeness to original", "Generation speed", "Prompt reading, 4K", "GPU memory")
    b = [text(40, 42, "Every measured recipe, compared with today's checkpoint", size=20, weight=700),
         text(40, 66, f"Today's checkpoint (V0): {inc['decode_tps']:.0f} tokens/s generation · {inc['prefill_tps']:,.0f} tokens/s prompt reading · "
                      f"{inc['peak_gpu_gib']:.0f} GB peak memory. One RTX 5090, 2 runs each.", "m", 13)]
    for c, name in enumerate(cols):
        cx = left + c * colw + colw / 2
        b.append(text(cx, 118, name, size=13, weight=600, anchor="middle"))
    b.append(f'<line class="rule" x1="40" y1="130" x2="{w - 40}" y2="130"/>')

    def row(y: float, label: str, sub: str, r: dict, star: bool = False) -> None:
        if star:
            b.append(text(40, y + 5, "★", size=14, weight=600))
        b.append(text(60, y + 5, label, size=14, weight=600))
        b.append(text(106, y + 5, sub, "m", 13))
        for c, (v, s, color) in enumerate(cells(r)):
            cx = left + c * colw + colw / 2
            half = 62
            length = max(min(abs(v) / 0.6, 1.0) * half, 2)
            x0 = cx if v >= 0 else cx - length
            b.append(f'<rect class="track" x="{cx - half}" y="{y - 6}" width="{2 * half}" height="12" rx="2"/>')
            b.append(f'<rect x="{x0:.1f}" y="{y - 6}" width="{length:.1f}" height="12" rx="2" fill="{color}"/>')
            b.append(f'<line class="gpu" x1="{cx}" y1="{y - 10}" x2="{cx}" y2="{y + 10}" stroke-width="1"/>')
            b.append(text(cx + half + 8, y + 5, s, size=12, weight=600, fill=color))

    y = top
    for r in internal:
        row(y, r["name"].split("-")[0], LABELS.get(r["name"], ""), r, star=r["frontier"])
        y += rowh
    y += 18
    b.append(f'<line class="rule" x1="40" y1="{y - 16}" x2="{w - 40}" y2="{y - 16}"/>')
    b.append(text(40, y + 4, "Outside references, for context only (not ranked)", "m", 13, weight=600))
    y += 34
    for r in external:
        row(y, r["id"], LABELS.get(r["id"], r["name"]), r)
        y += rowh
    ly = h - 26
    lx = 40
    for color, label in ((BETTER, "better than today"), (WORSE, "worse than today"), (SAME, "within measurement noise"), (None, "★ on the frontier: nothing measured beats it on every measure")):
        if color:
            b.append(f'<rect x="{lx}" y="{ly - 10}" width="12" height="12" rx="2" fill="{color}"/>')
            b.append(text(lx + 18, ly, label, "m", 13))
            lx += 18 + 7.2 * len(label) + 26
        else:
            b.append(text(lx, ly, label, "m", 13))
    path.write_text(svg(w, h, b))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("*.svg"):
        stale.unlink()
    frontier = json.loads((RESULTS / "frontier.json").read_text())
    analysis = json.loads((RESULTS / "analysis.json").read_text())
    floors = yaml.safe_load((REPO / "configs/hpc01.yaml").read_text())["frontier"]["epsilon_floor"]
    rows = {r["name"]: r for r in frontier["internal"]}
    inc = rows[frontier["incumbent"]]

    hero(OUT / "hero.svg", inc["peak_gpu_gib"])
    how_it_works(OUT / "how-it-works.svg")

    def result(name: str) -> list[tuple[str, str]]:
        r = rows[name]
        out = [(f"{-(r['rp_kl'] / inc['rp_kl'] - 1) * 100:.0f}% closer to the original", BETTER)]
        for key, what, floor in (("decode_tps", "generation", floors["decode_tps"]), ("prefill_tps", "prompt reading", floors["prefill_tps"])):
            rel = r[key] / inc[key] - 1
            if rel < -floor:
                out.append((f"{-rel * 100:.0f}% slower {what}", WORSE))
        return out

    recipes(OUT / "recipes.svg", [
        ("V0-baseline-rebuild", "V0 · today's checkpoint", "the standard NVFP4 encoder everywhere", [("the baseline for every comparison", SAME)]),
        ("V13-mlp-unsloth-bytes", "V13 · better encoder for the MLP", "calibrated bytes, MLP layers 0–55", result("V13-mlp-unsloth-bytes")),
        ("V3-gdn-fp8", "V3 · more bits for recurrent layers", "FP8 on every recurrent block", result("V3-gdn-fp8")),
    ])
    scorecard(OUT / "scorecard.svg", frontier, analysis, floors)
    print(f"wrote {len(list(OUT.glob('*.svg')))} SVGs to {OUT}")


if __name__ == "__main__":
    main()
