"""Dependency-free safetensors reading (mmap, sharded) and streaming writing.

The builder copies most tensors byte-for-byte, so it never needs a framework dtype: a tensor is
its header entry plus a memoryview of the mapped file.
"""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


@dataclass(frozen=True)
class TensorRef:
    name: str
    dtype: str
    shape: tuple[int, ...]
    file: Path
    offset: int  # absolute byte offset in file
    nbytes: int

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n


class SafeTensorsDir:
    """A checkpoint directory: one `model.safetensors`, or shards listed by an index file.

    Like SparkInfer's reader, shards named in the index but absent on disk are skipped.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._maps: dict[Path, mmap.mmap] = {}
        self._fhs: list = []
        self.tensors: dict[str, TensorRef] = {}
        index = self.path / "model.safetensors.index.json"
        if index.exists():
            files = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        else:
            files = sorted(p.name for p in self.path.glob("*.safetensors"))
        for fname in files:
            f = self.path / fname
            if f.exists():
                self._read_header(f)
        if not self.tensors:
            raise FileNotFoundError(f"no safetensors found in {self.path}")

    def _read_header(self, f: Path) -> None:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            self.tensors[name] = TensorRef(name, meta["dtype"], tuple(meta["shape"]), f, base + a, b - a)

    def _map(self, f: Path) -> mmap.mmap:
        m = self._maps.get(f)
        if m is None:
            fh = open(f, "rb")
            self._fhs.append(fh)
            m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            self._maps[f] = m
        return m

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def get(self, name: str) -> TensorRef | None:
        return self.tensors.get(name)

    def raw(self, name: str) -> memoryview:
        t = self.tensors[name]
        return memoryview(self._map(t.file))[t.offset : t.offset + t.nbytes]

    def array(self, name: str, np_dtype: str) -> np.ndarray:
        """Zero-copy view of a tensor's bytes as `np_dtype` (e.g. '<u2' for BF16, 'u1' for U8)."""
        t = self.tensors[name]
        return np.frombuffer(self.raw(name), dtype=np_dtype).reshape(t.shape)

    def sha256(self, name: str) -> str:
        return hashlib.sha256(self.raw(name)).hexdigest()

    def close(self) -> None:
        for m in self._maps.values():
            m.close()
        for fh in self._fhs:
            fh.close()
        self._maps.clear()
        self._fhs.clear()

    def __enter__(self) -> SafeTensorsDir:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


@dataclass
class PendingTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes | memoryview | np.ndarray

    @property
    def nbytes(self) -> int:
        if isinstance(self.data, np.ndarray):
            return self.data.nbytes
        return len(self.data)


class ShardWriter:
    """Writes tensors into size-capped shards plus `model.safetensors.index.json`.

    Tensors are buffered only until a shard is full; a shard's header needs every entry, so the
    shard's tensors are held (as views into mmaps or small arrays) until it is flushed.
    """

    def __init__(self, out_dir: str | os.PathLike, max_shard_bytes: int = 5 * 1024**3):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.max = max_shard_bytes
        self._pending: list[PendingTensor] = []
        self._pending_bytes = 0
        self._shards: list[tuple[str, list[str]]] = []
        self.total_bytes = 0

    def add(self, name: str, dtype: str, shape, data) -> None:
        t = PendingTensor(name, dtype, tuple(int(s) for s in shape), data)
        expect = DTYPE_BYTES[dtype]
        for s in t.shape:
            expect *= s
        if t.nbytes != expect:
            raise ValueError(f"{name}: {t.nbytes} bytes, expected {expect} for {dtype}{list(t.shape)}")
        if self._pending and self._pending_bytes + t.nbytes > self.max:
            self._flush()
        self._pending.append(t)
        self._pending_bytes += t.nbytes

    def _flush(self) -> None:
        if not self._pending:
            return
        tmp_name = f"shard-{len(self._shards):05d}.safetensors"
        header: dict = {"__metadata__": {"format": "pt"}}
        off = 0
        for t in self._pending:
            header[t.name] = {"dtype": t.dtype, "shape": list(t.shape), "data_offsets": [off, off + t.nbytes]}
            off += t.nbytes
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * ((8 - len(hb) % 8) % 8)
        with open(self.out / tmp_name, "wb") as fh:
            fh.write(struct.pack("<Q", len(hb)))
            fh.write(hb)
            for t in self._pending:
                data = t.data
                if isinstance(data, np.ndarray):
                    data = np.ascontiguousarray(data).tobytes()
                fh.write(data)
        self._shards.append((tmp_name, [t.name for t in self._pending]))
        self.total_bytes += off
        self._pending = []
        self._pending_bytes = 0

    def close(self, metadata: dict | None = None) -> dict[str, str]:
        self._flush()
        n = len(self._shards)
        weight_map: dict[str, str] = {}
        for i, (tmp, names) in enumerate(self._shards):
            final = f"model-{i + 1:05d}-of-{n:05d}.safetensors"
            os.replace(self.out / tmp, self.out / final)
            for name in names:
                weight_map[name] = final
        index = {"metadata": {"total_size": self.total_bytes, **(metadata or {})},
                 "weight_map": dict(sorted(weight_map.items()))}
        (self.out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
        return weight_map


def iter_file_sha256(path: Path, chunk: int = 64 * 1024 * 1024) -> Iterator[bytes]:
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                return
            yield b


def file_sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    for b in iter_file_sha256(Path(path)):
        h.update(b)
    return h.hexdigest()
