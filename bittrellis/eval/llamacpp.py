"""Reference point R2: the same model through llama.cpp on its best-known precision map.

llama.cpp cannot load NVFP4 safetensors, so its comparable artifact is a GGUF: unsloth's
UD-Q4_K_M, an imatrix-guided *mixed* precision map (Q4_K/Q5_K/Q6_K/Q8_0 per tensor). It is the
llama.cpp ecosystem's hand-tuned answer to the question BitTrellis automates.

Quality is scored by `tools/llamacpp_score.cpp` against the same BF16 reference and corpus as
every candidate; speed comes from `llama-bench`; VRAM from `llama-server` loaded at the same
maximum context the SparkInfer sweep uses.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from ..runtime import PeakMemory, ScoreDump, gpu_memory_used_mib, parse_score, wait_gpu_idle
from ..track import Track
from . import logits
from .reference import load_reference


class LlamaCpp:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.bin = self.root / "build/bin"
        self.score_bin = self.bin / "llamacpp_score"

    def score(self, gguf: Path, ids: list[int], topk: int, prefix_len: int) -> ScoreDump:
        with tempfile.NamedTemporaryFile("w", suffix=".ids", delete=False) as fh:
            fh.write(" ".join(map(str, ids)))
        r = subprocess.run([str(self.score_bin), str(gguf), str(topk), str(prefix_len), fh.name],
                           capture_output=True, text=True, timeout=7200,
                           env={"LD_LIBRARY_PATH": str(self.bin), "PATH": "/usr/bin:/bin"})
        Path(fh.name).unlink(missing_ok=True)
        if r.returncode != 0 or "[FAIL]" in r.stdout:
            raise RuntimeError(f"llamacpp_score failed: {r.stdout[-500:]} {r.stderr[-1500:]}")
        return parse_score(r.stdout)

    def bench(self, gguf: Path, depth: int, reps: int, n_decode: int) -> dict:
        def run(args: list[str]) -> list[dict]:
            cmd = [str(self.bin / "llama-bench"), "-m", str(gguf), "-ngl", "99", "-r", str(reps), "-o", "json", *args]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                               env={"LD_LIBRARY_PATH": str(self.bin), "PATH": "/usr/bin:/bin"})
            if r.returncode != 0:
                raise RuntimeError(f"llama-bench failed: {r.stderr[-1500:]}")
            return json.loads(r.stdout)

        dec = run(["-p", "0", "-n", str(n_decode), "-d", str(depth)])
        pre = run(["-p", str(depth), "-n", "0"])
        return {"decode_tps": dec[0]["avg_ts"], "decode_samples": dec[0].get("samples_ts"),
                "prefill_tps": pre[0]["avg_ts"], "prefill_samples": pre[0].get("samples_ts")}

    def vram_at_ctx(self, gguf: Path, ctx: int, port: int = 18181) -> float:
        wait_gpu_idle()
        idle = gpu_memory_used_mib() or 0
        cmd = [str(self.bin / "llama-server"), "-m", str(gguf), "-ngl", "99", "-c", str(ctx), "--port", str(port),
               "--host", "127.0.0.1", "-np", "1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                env={"LD_LIBRARY_PATH": str(self.bin), "PATH": "/usr/bin:/bin"})
        try:
            with PeakMemory() as peak:
                for _ in range(600):
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                            if resp.status == 200:
                                break
                    except OSError:
                        time.sleep(1)
                # Fill the context once so the KV cache and compute buffers are really touched.
                body = json.dumps({"prompt": [int(x) for x in range(100, 100 + ctx - 256)], "n_predict": 16}).encode()
                req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=600) as resp:
                    resp.read()
            used = peak.peak_mib
        finally:
            proc.terminate()
            proc.wait(timeout=60)
        return (used - idle) / 1024.0


def evaluate_llamacpp(track: Track, ref_id: str, llama_root: Path, gguf: Path, corpus: dict, ref_dir: Path,
                      out: Path, log=print) -> None:
    from .candidate import _write

    out.mkdir(parents=True, exist_ok=True)
    ref = track["references"][ref_id]
    lc = LlamaCpp(llama_root)
    _write(out / "candidate.json", {"id": ref_id, "name": ref["name"], "kind": "reference", "repo": ref["repo"],
                                    "revision": ref["revision"], "file": ref["file"],
                                    "llama_cpp_commit": ref["llama_cpp_commit"], "checkpoint_bytes": gguf.stat().st_size})
    refs = load_reference(ref_dir, corpus)
    (out / "scores").mkdir(exist_ok=True)
    results = []
    for s in corpus["streams"]:
        p = out / "scores" / f"{s['id']}.npz"
        if p.exists():
            dump = ScoreDump.load(p)
        else:
            t0 = time.time()
            dump = lc.score(gguf, s["ids"], track["evaluation"]["score"]["topk"], s["score_from"])
            dump.save(p)
            log(f"[llama.cpp quality] {s['id']}: {time.time() - t0:.0f}s")
        results.append(logits.compare(refs[s["id"]], dump, s))
    logits.save_positions(results, out / "kl_positions.npz")
    q = logits.summarize(results)
    q["corpus_sha256"] = corpus["sha256"]
    _write(out / "quality.json", q)
    perf_cfg = track["evaluation"]["performance"]
    wait_gpu_idle()
    b = lc.bench(gguf, perf_cfg["primary_context"], perf_cfg["reps"], perf_cfg["decode_tokens"])
    b["vram_gib"] = lc.vram_at_ctx(gguf, max(perf_cfg["contexts"]))
    b["primary_context"] = perf_cfg["primary_context"]
    b["reps"] = perf_cfg["reps"]
    _write(out / "performance.json", b)
