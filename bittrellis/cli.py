"""bittrellis — command-line entry point."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

from . import __version__
from .manifest import Manifest, ManifestError, summarize
from .model.qwen38 import Qwen38Arch
from .precision import PRECISIONS, SPACE, stored_bytes, stored_for, weight_bytes
from .track import REPO_ROOT, load_track

DEFAULTS = {
    "base": os.environ.get("BITTRELLIS_BASE", str(REPO_ROOT / "models/Qwen3.8-27B")),
    "baseline": os.environ.get("BITTRELLIS_BASELINE", str(REPO_ROOT / "models/Qwen3.8-27B-NVFP4-RTX5090")),
    "sparkinfer": os.environ.get("BITTRELLIS_SPARKINFER", str(REPO_ROOT / "third_party/sparkinfer")),
    "corpus": str(REPO_ROOT / "data/corpus/hpc01-public-v2.json"),
    "reference": os.environ.get("BITTRELLIS_REFERENCE", str(REPO_ROOT / "data/reference/hpc01-public-v2")),
}
GIB = 1024**3


def _arch(args) -> Qwen38Arch:
    cfg = Path(args.baseline) / "config.json"
    return Qwen38Arch.from_config(cfg) if cfg.exists() else Qwen38Arch()


def cmd_track(args) -> int:
    t = load_track(args.track)
    print(yaml.safe_dump(t.data, sort_keys=False), end="")
    return 0


def cmd_inventory(args) -> int:
    units = _arch(args).units()
    rows = []
    for u in units:
        rows.append({
            "unit": u.id, "kind": u.kind, "role": u.role, "layer": u.layer,
            "tensors": [{"name": lin.prefix, "rows": lin.rows, "cols": lin.cols} for lin in u.linears],
            "params": u.numel, "allowed": list(SPACE[u.kind]),
            "weight_bytes": {p: weight_bytes(u, p) for p in SPACE[u.kind]},
            "stored_bytes": {p: sum(stored_bytes(lin, stored_for(p)) for lin in u.linears) for p in SPACE[u.kind]},
        })
    doc = {"track": args.track, "model": "Qwen3.8-27B", "units": rows,
           "totals": {"units": len(rows), "params": sum(r["params"] for r in rows),
                      "by_kind": {k: sum(1 for r in rows if r["kind"] == k) for k in SPACE}}}
    text = json.dumps(doc, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}: {doc['totals']}")
    else:
        print(json.dumps(doc["totals"], indent=2))
    return 0


def cmd_manifest(args) -> int:
    track = load_track(args.track)
    units = _arch(args).units()
    by_id = {u.id: u for u in units}
    status = 0
    for path in args.manifests:
        try:
            m = Manifest.load(path)
            if m.track != track.id:
                raise ManifestError(f"manifest track {m.track} != {track.id}")
            exp = m.expand(units)
        except (ManifestError, OSError, yaml.YAMLError) as e:
            print(f"✗ {path}: {e}")
            status = 1
            continue
        est = sum(weight_bytes(by_id[k], p) for k, p in exp.items()) / GIB
        disk = sum(stored_bytes(lin, stored_for(p)) for k, p in exp.items() for lin in by_id[k].linears) / GIB
        print(f"✓ {path}: {m.name}  id={m.candidate_id(units)}")
        for kind, counts in summarize(exp, units).items():
            print(f"    {kind:8s} " + "  ".join(f"{p}×{counts[p]}" for p in PRECISIONS if p in counts))
        print(f"    searchable decode weights ≈ {est:.2f} GiB, stored Linears ≈ {disk:.2f} GiB (estimate)")
        if args.expand:
            print(json.dumps(exp, indent=1))
    return status


def cmd_build(args) -> int:
    from .build import build

    track = load_track(args.track)
    m = Manifest.load(args.manifest)
    out = Path(args.out) if args.out else REPO_ROOT / "models/candidates" / f"{m.name}-{m.candidate_id(_arch(args).units())}"
    build(m, track, Path(args.base), Path(args.baseline), out)
    print(out)
    return 0


def cmd_audit(args) -> int:
    from .validate import audit

    m = Manifest.load(args.manifest) if args.manifest else Manifest.load(Path(args.checkpoint) / "precision_manifest.yaml")
    res = audit(args.checkpoint, m, args.base, args.baseline, check_bytes=not args.fast, check_fidelity=not args.fast)
    d = res.to_dict()
    if args.out:
        Path(args.out).write_text(json.dumps(d, indent=2) + "\n")
    for e in res.errors[:30]:
        print(f"  ✗ {e}")
    print("AUDIT PASS" if res.ok else f"AUDIT FAIL ({len(res.errors)} errors)")
    return 0 if res.ok else 1


def cmd_describe(args) -> int:
    from collections import Counter

    from .validate import describe

    labels = describe(args.checkpoint)
    counts = Counter((uid.split(".")[1] if uid.startswith("L") else uid, lab) for uid, lab in labels.items())
    for (kind, lab), n in sorted(counts.items()):
        print(f"  {kind:8s} {lab:12s} ×{n}")
    if args.out:
        Path(args.out).write_text(json.dumps(labels, indent=1) + "\n")
    return 0


def cmd_corpus(args) -> int:
    from .eval.corpus import build_corpus, load_corpus, save_corpus

    if args.action == "build":
        c = build_corpus(Path(args.cache), split=args.split)
        save_corpus(c, Path(args.out))
        print(f"{args.out}: {len(c['streams'])} streams, sha256 {c['sha256']}")
    else:
        c = load_corpus(Path(args.out))
        print(f"{args.out}: OK sha256 {c['sha256']}")
    return 0


def cmd_reference(args) -> int:
    from .eval.corpus import load_corpus
    from .eval.reference import build_reference

    build_reference(Path(args.base), load_corpus(Path(args.corpus)), Path(args.out), gpu_gib=args.gpu_gib)
    return 0


def cmd_evaluate(args) -> int:
    from .eval.candidate import checkpoint_bytes, evaluate
    from .eval.corpus import load_corpus
    from .runtime import SparkInfer
    from .validate import audit, describe

    track = load_track(args.track)
    ckpt = Path(args.checkpoint)
    corpus = load_corpus(Path(args.corpus))
    si = SparkInfer(args.sparkinfer, track)
    if args.reference_id:
        ref = track["references"][args.reference_id]
        identity = {"id": args.reference_id, "name": ref["name"], "kind": "reference",
                    "repo": ref.get("repo"), "revision": ref.get("revision")}
        audit_ok = None
    else:
        m = Manifest.load(ckpt / "precision_manifest.yaml")
        units = Qwen38Arch.from_config(ckpt / "config.json").units()
        identity = {"id": m.candidate_id(units), "name": m.name, "kind": "candidate",
                    "manifest": m.to_dict(units), "build": json.loads((ckpt / "bittrellis_build.json").read_text())}
        res = audit(ckpt, m, args.base, args.baseline, check_bytes=not args.fast_audit, check_fidelity=True)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit.json").write_text(json.dumps(res.to_dict(), indent=2) + "\n")
        audit_ok = res.ok
        if not res.ok:
            print("AUDIT FAIL — not evaluating:", *res.errors[:10], sep="\n  ")
            return 1
    identity["runtime_precision"] = describe(ckpt)
    identity["checkpoint_bytes"] = checkpoint_bytes(ckpt)
    identity["audit_ok"] = audit_ok
    stages = tuple(s for s in args.stages.split(",") if s)
    evaluate(track, si, ckpt, Path(args.out), corpus, Path(args.reference), identity, stages)
    print(f"wrote {args.out}")
    return 0


def cmd_evaluate_llamacpp(args) -> int:
    from .eval.corpus import load_corpus
    from .eval.llamacpp import evaluate_llamacpp

    track = load_track(args.track)
    evaluate_llamacpp(track, args.reference_id, Path(args.llamacpp), Path(args.gguf), load_corpus(Path(args.corpus)),
                      Path(args.reference), Path(args.out))
    print(f"wrote {args.out}")
    return 0


def cmd_frontier(args) -> int:
    from .frontier.report import load_rows, render_table, write_frontier

    track = load_track(args.track)
    rows = load_rows([Path(p) for p in args.artifacts], track)
    doc = write_frontier(rows, track, Path(args.out) if args.out else None)
    print(render_table(rows))
    print(f"\n{len(doc['frontier'])} on the frontier (FG version {doc['frontier_gain_version']})")
    return 0


def cmd_report(args) -> int:
    from .frontier.report import feasibility_report

    track = load_track(args.track)
    path = feasibility_report([Path(p) for p in args.artifacts], track, Path(args.out))
    print(f"wrote {path}")
    return 0


def cmd_doctor(args) -> int:
    track = load_track(args.track)
    ok = True

    def check(name: str, good: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= good
        print(f"  {'✓' if good else '✗'} {name}{(' — ' + detail) if detail else ''}")

    print(f"bittrellis {__version__} · track {track.id}")
    check("nvidia-smi", shutil.which("nvidia-smi") is not None)
    for key in ("base", "baseline"):
        p = Path(getattr(args, key))
        check(f"{key} checkpoint", (p / "config.json").exists(), str(p))
    si = Path(args.sparkinfer)
    if (si / ".git").exists():
        from .runtime import RuntimeError_, SparkInfer

        try:
            SparkInfer(si, track).check_pinned()
            check("SparkInfer pinned + built", True, track["runtime"]["commit"][:12])
        except RuntimeError_ as e:
            check("SparkInfer pinned + built", False, str(e))
    else:
        check("SparkInfer checkout", False, f"{si} (run scripts/setup_sparkinfer.sh)")
    check("corpus", Path(args.corpus).exists(), args.corpus)
    check("BF16 reference", (Path(args.reference) / "reference.json").exists(), args.reference)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="bittrellis", description="Find the precision map the hardware actually wants.")
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("--track", default="HPC-01")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def paths(p, *keys):
        for k in keys:
            p.add_argument(f"--{k}", default=DEFAULTS[k])

    p = sub.add_parser("doctor", help="check the environment against the track pins")
    paths(p, "base", "baseline", "sparkinfer", "corpus", "reference")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("track", help="print the pinned track definition")
    p.set_defaults(fn=cmd_track)

    p = sub.add_parser("inventory", help="list every searchable unit, its shapes and legal precisions")
    paths(p, "baseline")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_inventory)

    p = sub.add_parser("manifest", help="validate manifests and show what they expand to")
    paths(p, "baseline")
    p.add_argument("manifests", nargs="+")
    p.add_argument("--expand", action="store_true", help="print the full unit → precision map")
    p.set_defaults(fn=cmd_manifest)

    p = sub.add_parser("build", help="build a deployable checkpoint from a manifest (CPU only)")
    paths(p, "base", "baseline")
    p.add_argument("manifest")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("audit", help="check a built checkpoint against the HPC-01 rules")
    paths(p, "base", "baseline")
    p.add_argument("checkpoint")
    p.add_argument("--manifest")
    p.add_argument("--fast", action="store_true", help="skip byte hashing and fidelity sampling")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_audit)

    p = sub.add_parser("describe", help="show what precision SparkInfer executes for any checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_describe)

    p = sub.add_parser("corpus", help="build or verify the evaluation corpus")
    p.add_argument("action", choices=["build", "verify"])
    p.add_argument("--out", default=DEFAULTS["corpus"])
    p.add_argument("--cache", default=str(REPO_ROOT / "data/cache"))
    p.add_argument("--split", default="public", choices=["public", "holdout"])
    p.set_defaults(fn=cmd_corpus)

    p = sub.add_parser("reference", help="compute BF16 reference distributions (transformers, GPU+CPU)")
    paths(p, "base", "corpus")
    p.add_argument("--out", default=DEFAULTS["reference"])
    p.add_argument("--gpu-gib", type=int, default=22)
    p.set_defaults(fn=cmd_reference)

    p = sub.add_parser("evaluate", help="audit, score, benchmark and task-check one checkpoint")
    paths(p, "base", "baseline", "sparkinfer", "corpus", "reference")
    p.add_argument("checkpoint")
    p.add_argument("--out", required=True)
    p.add_argument("--reference-id", help="evaluate an external reference (R0, R1, ...) instead of a candidate")
    p.add_argument("--stages", default="quality,performance,tasks")
    p.add_argument("--fast-audit", action="store_true")
    p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("evaluate-llamacpp", help="measure a GGUF reference point through pinned llama.cpp")
    paths(p, "corpus", "reference")
    p.add_argument("--reference-id", default="R2")
    p.add_argument("--gguf", default=str(REPO_ROOT / "models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf"))
    p.add_argument("--llamacpp", default=os.environ.get("BITTRELLIS_LLAMACPP", str(REPO_ROOT / "third_party/llama.cpp")))
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_evaluate_llamacpp)

    p = sub.add_parser("frontier", help="gate, rank and compute Frontier Gain over artifact directories")
    p.add_argument("artifacts", nargs="+")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_frontier)

    p = sub.add_parser("report", help="write the feasibility report")
    p.add_argument("artifacts", nargs="+")
    p.add_argument("--out", default=str(REPO_ROOT / "results/feasibility"))
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
