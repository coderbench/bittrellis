import hashlib
import os

from bittrellis.lineage import verify_source


def _lock(tmp_path):
    big = b"x" * 1000
    small = b'{"a": 1}'
    (tmp_path / "model.safetensors").write_bytes(big)
    (tmp_path / "config.json").write_bytes(small)
    return {"sources": {"s": {"repo": "r", "revision": "v", "files": {
        "model.safetensors": {"size": len(big), "sha256": hashlib.sha256(big).hexdigest()},
        "config.json": {"size": len(small), "git_blob": hashlib.sha1(b"blob %d\0" % len(small) + small).hexdigest()},
    }}}}


def test_verify_source_hashes_caches_and_detects_tampering(tmp_path):
    lock = _lock(tmp_path)
    first = verify_source("s", tmp_path, lock=lock)
    assert first.ok and first.checked == 2
    second = verify_source("s", tmp_path, lock=lock)
    assert second.ok and second.cached == 2 and second.checked == 0
    (tmp_path / "model.safetensors").write_bytes(b"y" * 1000)          # same size, new mtime
    os.utime(tmp_path / "model.safetensors", ns=(1, 1))
    third = verify_source("s", tmp_path, lock=lock)
    assert not third.ok and any("hash" in e for e in third.errors)


def test_verify_source_reports_missing_and_size(tmp_path):
    lock = _lock(tmp_path)
    (tmp_path / "config.json").write_bytes(b"{}")
    res = verify_source("s", tmp_path, lock=lock)
    assert not res.ok and any("size" in e for e in res.errors)
    (tmp_path / "model.safetensors").unlink()
    assert any("missing" in e for e in verify_source("s", tmp_path, lock=lock).errors)
