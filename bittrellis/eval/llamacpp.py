"""External reference R2: the same model through llama.cpp on its best-known precision map.

llama.cpp cannot load NVFP4 safetensors, so its comparable artifact is unsloth's UD-Q4_K_M GGUF, an
imatrix-guided mixed Q4_K/Q5_K/Q6_K/Q8_0 map. It is scored on the same BF16 reference partition by
tools/llamacpp_score.cpp, benchmarked with llama-bench (2 runs), and its peak GPU memory is taken
from llama-server with the 16K context filled. It is context only: it never enters the internal
frontier.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from ..runtime import (
    PeakMemory,
    RefScore,
    gpu_memory_used_mib,
    read_refscore_output,
    wait_gpu_idle,
    write_refscore_inputs,
)
from ..track import Track
from . import logits
from .reference import load_reference


class LlamaCpp:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.bin = self.root / "build/bin"
        self.env = {"LD_LIBRARY_PATH": str(self.bin), "PATH": "/usr/bin:/bin"}

    def refscore(self, gguf: Path, tokens: list[int], ref_ids, prefix_len: int) -> RefScore:
        with tempfile.TemporaryDirectory() as td:
            tok, ref, out = Path(td) / "t.bin", Path(td) / "r.bin", Path(td) / "o.bin"
            write_refscore_inputs(tokens, ref_ids, tok, ref)
            r = subprocess.run([str(self.bin / "llamacpp_score"), str(gguf), str(tok), str(ref), str(out), str(prefix_len)],
                               capture_output=True, text=True, timeout=7200, env=self.env)
            if r.returncode != 0 or not r.stdout.startswith("OK"):
                raise RuntimeError(f"llamacpp_score failed: {r.stdout[-500:]} {r.stderr[-1500:]}")
            return read_refscore_output(out, prefix_len)

    def bench_once(self, gguf: Path, depth: int, n_decode: int) -> dict:
        def run(args: list[str]) -> float:
            cmd = [str(self.bin / "llama-bench"), "-m", str(gguf), "-ngl", "99", "-r", "1", "-o", "json", *args]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=self.env)
            if r.returncode != 0:
                raise RuntimeError(f"llama-bench failed: {r.stderr[-1500:]}")
            return float(json.loads(r.stdout)[0]["avg_ts"])

        return {"decode_tps": run(["-p", "0", "-n", str(n_decode), "-d", str(depth)]),
                "prefill_tps": run(["-p", str(depth), "-n", "0"])}

    def peak_gpu_at_ctx(self, gguf: Path, ctx: int, port: int = 18181) -> float:
        wait_gpu_idle()
        idle = gpu_memory_used_mib() or 0
        cmd = [str(self.bin / "llama-server"), "-m", str(gguf), "-ngl", "99", "-c", str(ctx), "--port", str(port),
               "--host", "127.0.0.1", "-np", "1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=self.env)
        try:
            with PeakMemory() as peak:
                for _ in range(600):
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
                            if resp.status == 200:
                                break
                    except OSError:
                        time.sleep(1)
                body = json.dumps({"prompt": list(range(100, 100 + ctx - 1024)), "n_predict": 512}).encode()
                req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=900) as resp:
                    resp.read()
            return (peak.peak_mib - idle) / 1024.0
        finally:
            proc.terminate()
            proc.wait(timeout=60)


def evaluate_llamacpp(track: Track, ref_id: str, llama_root: Path, gguf: Path, corpus: dict, ref_dir: Path,
                      out: Path, log=print) -> None:
    from .candidate import _write

    out.mkdir(parents=True, exist_ok=True)
    ref = track["external_references"][ref_id]
    lc = LlamaCpp(llama_root)
    _write(out / "candidate.json", {"id": ref_id, "name": ref["name"], "kind": "external", "source": ref["source"],
                                    "file": ref["file"], "llama_cpp_commit": ref["llama_cpp_commit"],
                                    "checkpoint_bytes": gguf.stat().st_size})
    k = track["evaluation"]["score"]["reference_topk"]
    refs = load_reference(ref_dir, corpus, expected_k=k)
    (out / "scores").mkdir(exist_ok=True)
    results = []
    for s in corpus["streams"]:
        p = out / "scores" / f"{s['id']}.npz"
        if p.exists():
            rs = RefScore.load(p)
        else:
            t0 = time.time()
            rs = lc.refscore(gguf, s["ids"], refs[s["id"]].top_ids, s["score_from"])
            rs.save(p)
            log(f"[llama.cpp quality] {s['id']}: {time.time() - t0:.0f}s")
        results.append(logits.compare(refs[s["id"]], rs, s))
    logits.save_positions(results, out / "kl_positions.npz")
    q = logits.summarize(results, k)
    q["corpus"] = {"version": corpus["version"], "split": corpus["split"], "sha256": corpus["sha256"]}
    _write(out / "quality.json", q)

    perf = track["evaluation"]["performance"]
    runs = []
    for _ in range(int(perf["runs"])):
        wait_gpu_idle()
        runs.append(lc.bench_once(gguf, perf["primary_context"], perf["decode_tokens"]))

    def stat(key: str) -> tuple[float, float]:
        vals = [r[key] for r in runs]
        mean = sum(vals) / len(vals)
        return mean, (max(vals) - min(vals)) / mean

    decode, dspread = stat("decode_tps")
    prefill, pspread = stat("prefill_tps")
    _write(out / "performance.json", {
        "decode_tps": decode, "decode_spread": dspread, "prefill_tps": prefill, "prefill_spread": pspread,
        "peak_gpu_gib": lc.peak_gpu_at_ctx(gguf, max(perf["contexts"])), "runs": runs,
        "note": "llama-bench -r 1 per run (tg at depth 4096, pp 4096); peak GPU from llama-server with 16K filled",
    })
