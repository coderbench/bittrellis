"""Load artifact directories into rows, rank them, and write frontier + feasibility outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..track import Track
from .pareto import FG_VERSION, Row, apply_gates, rank

GIB = 1024**3


def _read(p: Path) -> dict | None:
    return json.loads(p.read_text()) if p.exists() else None


def load_row(art: Path) -> Row | None:
    cand = _read(art / "candidate.json")
    q = _read(art / "quality.json")
    perf = _read(art / "performance.json")
    if not (cand and q and perf):
        return None
    return Row(
        id=cand["id"], name=cand["name"], kind=cand["kind"],
        kl=q["kl"], top1=q["top1"], needle_recall=q.get("needle_recall"),
        decode_tps=perf["decode_tps"], prefill_tps=perf.get("prefill_tps"), vram_gib=perf["vram_gib"],
        checkpoint_gib=(cand.get("checkpoint_bytes") or 0) / GIB or None,
        tasks=_read(art / "tasks.json"), audit_ok=cand.get("audit_ok"),
    )


def load_rows(paths: list[Path], track: Track) -> list[Row]:
    arts: list[Path] = []
    for p in paths:
        if (p / "candidate.json").exists():
            arts.append(p)
        else:
            arts.extend(sorted(d for d in p.iterdir() if (d / "candidate.json").exists()))
    rows = [r for r in (load_row(a) for a in arts) if r is not None]
    ref0 = next((r for r in rows if r.id == "R0"), None)
    for r in rows:
        apply_gates(r, track["gates"], ref0.tasks if ref0 and r is not ref0 else None)
    return rank(rows, track["frontier"]["box"], track["frontier"].get("epsilon"))


def render_table(rows: list[Row]) -> str:
    head = f"{'':2s}{'id':18s} {'name':34s} {'KL':>7s} {'top1':>6s} {'decode':>7s} {'prefill':>8s} {'VRAM':>6s} {'FG-2':>6s}  gates"
    lines = [head, "-" * len(head)]
    for r in sorted(rows, key=lambda r: (r.kind != "reference", r.kl)):
        mark = "★" if r.frontier else " "
        gates = "PASS" if r.valid else "; ".join(r.gate_failures)
        lines.append(
            f"{mark} {r.id:18s} {r.name[:34]:34s} {r.kl:7.4f} {r.top1 or 0:6.3f} {r.decode_tps:7.1f} "
            f"{r.prefill_tps:8.0f} {r.vram_gib:6.2f} {100 * r.gain:5.2f}%  {gates}"
        )
    return "\n".join(lines)


def row_dict(r: Row) -> dict:
    return {
        "id": r.id, "name": r.name, "kind": r.kind, "kl": r.kl, "top1": r.top1, "needle_recall": r.needle_recall,
        "decode_tps": r.decode_tps, "prefill_tps": r.prefill_tps, "vram_gib": r.vram_gib,
        "checkpoint_gib": r.checkpoint_gib, "tasks_rate": (r.tasks or {}).get("rate"),
        "valid": r.valid, "gate_failures": r.gate_failures, "frontier": r.frontier, "frontier_gain": r.gain,
    }


def write_frontier(rows: list[Row], track: Track, out: Path | None) -> dict:
    doc = {
        "track": track.id, "track_version": track["version"], "frontier_gain_version": FG_VERSION,
        "objectives": track["frontier"]["objectives"], "epsilon": track["frontier"].get("epsilon"),
        "box": track["frontier"]["box"], "gates": track["gates"],
        "rows": [row_dict(r) for r in rows], "frontier": [r.id for r in rows if r.frontier],
    }
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2) + "\n")
    return doc


def _plots(rows: list[Row], out: Path) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    out.mkdir(parents=True, exist_ok=True)
    made = []
    for key, label, fname in (("decode_tps", "decode tok/s @ 4K, batch 1  →  better", "quality_vs_decode.png"),
                              ("prefill_tps", "prefill tok/s @ 4K  →  better", "quality_vs_prefill.png"),
                              ("vram_gib", "peak VRAM GiB  ←  better", "quality_vs_vram.png")):
        fig, ax = plt.subplots(figsize=(8, 5.2), dpi=150)
        for r in rows:
            x = getattr(r, key)
            if x is None:
                continue
            tag = r.id if r.kind == "reference" else r.name.split("-")[0]
            if r.kind == "reference":
                style = {"marker": "s", "s": 70, "color": "#9ca3af" if r.id != "R0" else "#f59e0b", "edgecolor": "#111827"}
            elif not r.valid:
                style = {"marker": "x", "s": 50, "color": "#dc2626"}
            else:
                style = {"marker": "o", "s": 70 if r.frontier else 38, "color": "#2563eb" if r.frontier else "#93c5fd",
                         "edgecolor": "#111827" if r.frontier else "none"}
            ax.scatter(x, r.kl, zorder=3, **style)
            ax.annotate(tag, (x, r.kl), textcoords="offset points", xytext=(6, -3), fontsize=8,
                        fontweight="bold" if r.frontier or r.id == "R0" else "normal")
        ax.set_xlabel(label)
        ax.set_ylabel("KL(BF16 ‖ checkpoint), nats/token  ↓ better")
        ax.set_title("HPC-01 · Qwen3.8-27B · RTX 5090   (■ reference · ● BitTrellis · bold = frontier)", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out / fname)
        plt.close(fig)
        made.append(fname)
    return made


def feasibility_report(paths: list[Path], track: Track, out: Path) -> Path:
    """Write frontier.json, comparison CSVs and plots; the narrative is written by hand from these."""
    rows = load_rows(paths, track)
    out.mkdir(parents=True, exist_ok=True)
    write_frontier(rows, track, out / "frontier.json")
    with open(out / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row_dict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            d = row_dict(r)
            d["gate_failures"] = "; ".join(d["gate_failures"])
            w.writerow(d)
    _plots(rows, out / "plots")
    (out / "frontier_table.txt").write_text(render_table(rows) + "\n")
    return out / "frontier.json"
