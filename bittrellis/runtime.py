"""Thin, strict wrappers around the pinned SparkInfer binaries and their text output."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .track import Track

_SWEEP = re.compile(r"^SWEEP_JSON\s+(\{.*\})\s*$", re.M)
_VRAM = re.compile(r"^VRAM used\s*:\s*([\d.]+)\s*GB", re.M)
_SCORE_LINE = re.compile(r"^S i=(\d+) tgt=(\d+) am=(\d+) lp=(\S+) top=(.*)$")


class RuntimeError_(RuntimeError):
    pass


@dataclass
class ScoreDump:
    """Teacher-forced output of qwen3_gguf_score for one token stream."""

    pos: np.ndarray        # int32 [N] position i (predicting token i+1)
    target: np.ndarray     # int32 [N]
    argmax: np.ndarray     # int32 [N]
    lp_target: np.ndarray  # float32 [N]
    top_ids: np.ndarray    # int32 [N, K]
    top_lp: np.ndarray     # float32 [N, K]

    def save(self, path: Path) -> None:
        np.savez_compressed(path, pos=self.pos, target=self.target, argmax=self.argmax,
                            lp_target=self.lp_target, top_ids=self.top_ids, top_lp=self.top_lp)

    @classmethod
    def load(cls, path: Path) -> ScoreDump:
        z = np.load(path)
        return cls(z["pos"], z["target"], z["argmax"], z["lp_target"], z["top_ids"], z["top_lp"])


def parse_score(text: str) -> ScoreDump:
    rows = []
    for line in text.splitlines():
        m = _SCORE_LINE.match(line)
        if not m:
            continue
        top = [kv.split(":") for kv in m.group(5).split(",")]
        rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)),
                     [int(a) for a, _ in top], [float(b) for _, b in top]))
    if not rows:
        raise RuntimeError_("no score lines in output")
    k = min(len(r[4]) for r in rows)
    return ScoreDump(
        pos=np.array([r[0] for r in rows], np.int32),
        target=np.array([r[1] for r in rows], np.int32),
        argmax=np.array([r[2] for r in rows], np.int32),
        lp_target=np.array([r[3] for r in rows], np.float32),
        top_ids=np.array([r[4][:k] for r in rows], np.int32),
        top_lp=np.array([r[5][:k] for r in rows], np.float32),
    )


def parse_sweep(text: str) -> dict:
    m = _SWEEP.findall(text)
    if not m:
        raise RuntimeError_("benchmark printed no SWEEP_JSON line")
    sweep = json.loads(m[-1])
    vram = [float(v) for v in _VRAM.findall(text)]
    return {"sweep": {int(k): v for k, v in sweep.items()}, "vram_gib_reported": max(vram) if vram else None}


class SparkInfer:
    def __init__(self, root: str | Path, track: Track):
        self.root = Path(root)
        self.track = track
        self.bench_bin = self.root / "build/runtime/qwen3_gguf_bench"
        self.score_bin = self.root / "build/runtime/qwen3_gguf_score"
        self.server_bin = self.root / "build/server/sparkinfer_server"

    def commit(self) -> str:
        return subprocess.run(["git", "-C", str(self.root), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()

    def check_pinned(self) -> None:
        want = self.track["runtime"]["commit"]
        got = self.commit()
        if got != want:
            raise RuntimeError_(f"SparkInfer at {got}, track pins {want}")
        dirty = subprocess.run(["git", "-C", str(self.root), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        if dirty:
            raise RuntimeError_("SparkInfer checkout has local modifications")
        for b in (self.bench_bin, self.score_bin):
            if not b.exists():
                raise RuntimeError_(f"missing binary {b}; run scripts/setup_sparkinfer.sh")

    def score(self, model_dir: Path, ids: list[int], topk: int, prefix_len: int = 0,
              env: dict[str, str] | None = None, timeout: int = 7200) -> ScoreDump:
        extra = dict(env or {})
        extra["SPARKINFER_SCORE_MAX_SEQ"] = str(len(ids) + 16)
        if prefix_len:
            extra["SPARKINFER_PREFIX_CACHE_LEN"] = str(prefix_len)
        cmd = [str(self.score_bin), str(model_dir), str(topk)] + [str(i) for i in ids]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=self.track.clean_env(extra))
        if r.returncode != 0 or "[FAIL]" in r.stdout:
            raise RuntimeError_(f"score failed ({r.returncode}): {r.stdout[-800:]} {r.stderr[-1500:]}")
        return parse_score(r.stdout)

    def bench(self, model_dir: Path, contexts: list[int], reps: int, n_decode: int, prompt_file: Path,
              env: dict[str, str] | None = None, timeout: int = 3600) -> dict:
        extra = dict(env or {})
        extra.update({
            "SPARKINFER_BENCH_SWEEP_CTXS": ",".join(str(c) for c in contexts),
            "SPARKINFER_BENCH_SWEEP_REPS": str(reps),
            "SPARKINFER_BENCH_PROMPT_FILE": str(prompt_file),
        })
        cmd = [str(self.bench_bin), str(model_dir), str(n_decode), "sweep"]
        t0 = time.time()
        with PeakMemory() as peak:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=self.track.clean_env(extra))
        if r.returncode != 0:
            raise RuntimeError_(f"bench failed ({r.returncode}): {r.stdout[-800:]} {r.stderr[-1500:]}")
        out = parse_sweep(r.stdout)
        out["peak_mib"] = peak.peak_mib
        out["wall_seconds"] = round(time.time() - t0, 1)
        out["log_tail"] = r.stdout[-3000:]
        return out

    def start_server(self, model_dir: Path, port: int, ctx: int, log: Path,
                     env: dict[str, str] | None = None) -> subprocess.Popen:
        cmd = [str(self.server_bin), "-m", str(model_dir), "--tokenizer", str(model_dir / "tokenizer.json"),
               "--ctx", str(ctx), "--host", "127.0.0.1", "--port", str(port)]
        extra = {"SPARKINFER_SAMPLING_DEFAULTS": "greedy", **(env or {})}
        fh = open(log, "w")
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=self.track.clean_env(extra))
        url = f"http://127.0.0.1:{port}/health"
        for _ in range(900):
            if proc.poll() is not None:
                raise RuntimeError_(f"server exited with {proc.returncode}; see {log}")
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return proc
            except OSError:
                pass
            time.sleep(1)
        proc.kill()
        raise RuntimeError_("server did not become healthy in 900s")


def gpu_memory_used_mib() -> int | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return int(r.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class PeakMemory:
    """Poll device memory in a background thread; `peak_mib` is the maximum seen."""

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self.peak_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            used = gpu_memory_used_mib()
            if used is not None:
                self.peak_mib = max(self.peak_mib, used)
            self._stop.wait(self.interval)

    def __enter__(self) -> PeakMemory:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)


def wait_gpu_idle(max_mib: int = 1024, timeout: int = 300) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        used = gpu_memory_used_mib()
        if used is None or used < max_mib:
            return
        time.sleep(2)
    raise RuntimeError_(f"GPU still has {gpu_memory_used_mib()} MiB in use; refusing to benchmark on a busy card")
