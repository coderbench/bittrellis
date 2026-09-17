"""Source lineage: every byte BitTrellis reads comes from a pinned source whose files are hashed.

`configs/sources.lock.json` pins each source (repo, revision, and every file by sha256 for LFS
files or by git blob id for small files). `verify_source()` checks a local copy against the lock.
Hashing ~50 GB takes minutes, so a successful verification is cached next to the files, keyed by
path, size and mtime; any change to a file invalidates its entry.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .track import CONFIGS

CACHE_NAME = ".bittrellis-verified.json"
CHUNK = 64 * 1024 * 1024


class LineageError(RuntimeError):
    pass


@dataclass
class VerifyResult:
    source: str
    ok: bool = True
    checked: int = 0
    cached: int = 0
    errors: list[str] = field(default_factory=list)


def load_lock(path: Path | None = None) -> dict:
    return json.loads(Path(path or CONFIGS / "sources.lock.json").read_text())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                return h.hexdigest()
            h.update(b)


def _git_blob(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify_source(source_id: str, local_dir: str | Path, files: list[str] | None = None,
                  lock: dict | None = None, log=None) -> VerifyResult:
    """Verify `files` (default: every pinned file present in the lock) of a local checkout."""
    lock = lock or load_lock()
    entry = lock["sources"].get(source_id)
    if entry is None:
        raise LineageError(f"unknown source {source_id!r}")
    local_dir = Path(local_dir)
    res = VerifyResult(source_id)
    cache_path = local_dir / CACHE_NAME
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        cache = {}
    wanted = files if files is not None else sorted(entry["files"])
    changed = False
    for name in wanted:
        pin = entry["files"].get(name)
        if pin is None:
            res.ok = False
            res.errors.append(f"{name}: not pinned in the lock for {source_id}")
            continue
        p = local_dir / name
        if not p.exists():
            res.ok = False
            res.errors.append(f"{name}: missing")
            continue
        st = p.stat()
        if st.st_size != pin["size"]:
            res.ok = False
            res.errors.append(f"{name}: size {st.st_size} != pinned {pin['size']}")
            continue
        key = f"{name}|{st.st_size}|{st.st_mtime_ns}"
        want = pin.get("sha256") or pin.get("git_blob")
        if cache.get(key) == want:
            res.cached += 1
            continue
        if log:
            log(f"[lineage] hashing {source_id}/{name} ({st.st_size / 1e9:.1f} GB)")
        got = _sha256(p) if "sha256" in pin else _git_blob(p)
        res.checked += 1
        if got != want:
            res.ok = False
            res.errors.append(f"{name}: hash {got[:16]}… != pinned {want[:16]}…")
            continue
        cache[key] = want
        changed = True
    if changed:
        try:
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache, indent=1))
            os.replace(tmp, cache_path)
        except OSError:
            pass  # read-only checkout: verification still happened, just not cached
    return res


def weight_files(source_id: str, lock: dict | None = None) -> list[str]:
    lock = lock or load_lock()
    return sorted(n for n in lock["sources"][source_id]["files"] if n.endswith((".safetensors", ".gguf")))


def require_verified(source_id: str, local_dir: str | Path, log=None) -> None:
    """Raise unless every pinned weight file of the source matches the lock."""
    names = [n for n in weight_files(source_id) if (Path(local_dir) / n).exists() or not n.startswith("model_mtp")]
    res = verify_source(source_id, local_dir, names, log=log)
    if not res.ok:
        raise LineageError(f"source {source_id} at {local_dir} does not match the lock: " + "; ".join(res.errors[:5]))
