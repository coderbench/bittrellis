"""Evaluate one checkpoint end to end and write its artifact directory.

    artifacts/<candidate>/
      candidate.json      what was evaluated (manifest hash, runtime precision map, checkpoint hashes)
      audit.json          HPC-01 rule checks (candidates only)
      quality.json        KL / top-1 / NLL vs BF16, per category, needles
      performance.json    decode + prefill tok/s per context, VRAM
      tasks.json          SparkInfer quality-suite guard
      environment.json    GPU, driver, CUDA, SparkInfer commit, BitTrellis commit
      scores/*.npz        raw teacher-forced dumps (re-scorable without a GPU)
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

from .. import __version__
from ..runtime import ScoreDump, SparkInfer, gpu_memory_used_mib, wait_gpu_idle
from ..track import REPO_ROOT, Track
from . import logits
from .reference import load_reference
from .tasks import run_tasks


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def environment(si: SparkInfer) -> dict:
    return {
        "gpu": _sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "cuda": _sh(["nvcc", "--version"]).splitlines()[-1] if _sh(["nvcc", "--version"]) else "",
        "cpu": _sh(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"]).strip(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "sparkinfer_commit": si.commit(),
        "bittrellis_version": __version__,
        "bittrellis_commit": _sh(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def evaluate_quality(si: SparkInfer, track: Track, model_dir: Path, corpus: dict, ref_dir: Path, out: Path,
                     log=print) -> dict:
    refs = load_reference(ref_dir, corpus)
    cfg = track["evaluation"]["score"]
    (out / "scores").mkdir(parents=True, exist_ok=True)
    results = []
    for s in corpus["streams"]:
        dump_path = out / "scores" / f"{s['id']}.npz"
        if dump_path.exists():
            dump = ScoreDump.load(dump_path)
        else:
            t0 = time.time()
            dump = si.score(model_dir, s["ids"], cfg["topk"], prefix_len=s["score_from"], env=cfg.get("env"))
            dump.save(dump_path)
            log(f"[quality] {s['id']}: {len(dump.pos)} positions in {time.time() - t0:.0f}s")
        results.append(logits.compare(refs[s["id"]], dump, s))
    logits.save_positions(results, out / "kl_positions.npz")
    q = logits.summarize(results)
    q["corpus_sha256"] = corpus["sha256"]
    return q


def evaluate_performance(si: SparkInfer, track: Track, model_dir: Path, corpus: dict, out: Path, log=print) -> dict:
    cfg = track["evaluation"]["performance"]
    longest = max(corpus["streams"], key=lambda s: len(s["ids"]))
    prompt = out / "bench_prompt_ids.txt"
    prompt.write_text(" ".join(str(i) for i in longest["ids"]))
    wait_gpu_idle()
    idle = gpu_memory_used_mib()
    t0 = time.time()
    r = si.bench(model_dir, cfg["contexts"], cfg["reps"], cfg["decode_tokens"], prompt)
    log(f"[perf] sweep {cfg['contexts']} x{cfg['reps']} in {time.time() - t0:.0f}s")
    primary = int(cfg["primary_context"])
    sweep = r["sweep"]
    return {
        "contexts": {str(k): v for k, v in sorted(sweep.items())},
        "decode_tps": sweep[primary]["decode_tps"],
        "prefill_tps": sweep[primary]["prefill_pp"],
        "primary_context": primary,
        # Official footprint: peak device memory over the whole sweep (weights, KV at 16K, prefill
        # arena, CUDA context), minus what was in use before the run. The bench's own "VRAM used"
        # is sampled right after load and misses the arena, so it is kept only for reference.
        "vram_gib": round((r["peak_mib"] - (idle or 0)) / 1024.0, 3),
        "vram_gib_after_load": r["vram_gib_reported"],
        "gpu_idle_mib_before": idle,
        "reps": cfg["reps"],
        "log_tail": r["log_tail"],
    }


def evaluate(track: Track, si: SparkInfer, model_dir: Path, out: Path, corpus: dict, ref_dir: Path,
             identity: dict, stages: tuple[str, ...] = ("quality", "performance", "tasks"), log=print) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    si.check_pinned()
    _write(out / "candidate.json", identity)
    _write(out / "environment.json", environment(si))
    result: dict = {"candidate": identity}
    if "quality" in stages:
        result["quality"] = evaluate_quality(si, track, model_dir, corpus, ref_dir, out, log)
        _write(out / "quality.json", result["quality"])
    if "performance" in stages:
        result["performance"] = evaluate_performance(si, track, model_dir, corpus, out, log)
        _write(out / "performance.json", result["performance"])
    if "tasks" in stages:
        tcfg = track["evaluation"]["tasks"]
        result["tasks"] = run_tasks(si, model_dir, out / "tasks", tcfg["tier"], tcfg["server_ctx"], log=log)
        _write(out / "tasks.json", result["tasks"])
    return result


def checkpoint_bytes(model_dir: Path) -> int:
    return sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors"))
