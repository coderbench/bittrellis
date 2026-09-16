"""Qwen3.8-27B as the pinned SparkInfer loader sees it: layer pattern, Linear shapes, and the
searchable *units* a precision manifest assigns.

A unit is the smallest thing the runtime lets you choose a precision for independently:

* `L{i}.gdn.qkv | .z | .out`   Gated DeltaNet projections, one tensor each (48 layers)
* `L{i}.attn.q | .k | .v | .o` full-attention projections, one tensor each (16 layers)
* `L{i}.mlp`                    gate/up/down together: the loader keys the whole FFN on gate_proj
* `lm_head`

Everything else (embeddings, norms, conv1d, in_proj_a/b, A_log, dt_bias, the vision tower) is
BF16-only in the loader and therefore not searchable.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path

LM = "model.language_model."


@dataclass(frozen=True)
class Linear:
    prefix: str  # tensor name without the ".weight" suffix
    rows: int
    cols: int

    @property
    def numel(self) -> int:
        return self.rows * self.cols


@dataclass(frozen=True)
class Unit:
    id: str
    role: str  # gdn.qkv, gdn.z, gdn.out, attn.q, attn.k, attn.v, attn.o, mlp, lm_head
    layer: int | None
    linears: tuple[Linear, ...]

    @property
    def numel(self) -> int:
        return sum(lin.numel for lin in self.linears)

    @property
    def kind(self) -> str:
        """Precision-space key: gdn, attn, mlp or lm_head."""
        return self.role.split(".")[0]


@dataclass(frozen=True)
class Qwen38Arch:
    hidden: int = 5120
    intermediate: int = 17408
    n_layers: int = 64
    full_attention_interval: int = 4
    n_heads: int = 24
    n_kv_heads: int = 4
    head_dim: int = 256
    attn_output_gate: bool = True
    linear_k_heads: int = 16
    linear_v_heads: int = 48
    linear_k_dim: int = 128
    linear_v_dim: int = 128
    vocab: int = 248320

    @classmethod
    def from_config(cls, config: dict | str | Path) -> Qwen38Arch:
        if not isinstance(config, dict):
            config = json.loads(Path(config).read_text())
        t = config.get("text_config", config)
        arch = cls(
            hidden=t["hidden_size"],
            intermediate=t["intermediate_size"],
            n_layers=t["num_hidden_layers"],
            full_attention_interval=t.get("full_attention_interval", 4),
            n_heads=t["num_attention_heads"],
            n_kv_heads=t["num_key_value_heads"],
            head_dim=t["head_dim"],
            attn_output_gate=t.get("attn_output_gate", True),
            linear_k_heads=t["linear_num_key_heads"],
            linear_v_heads=t["linear_num_value_heads"],
            linear_k_dim=t["linear_key_head_dim"],
            linear_v_dim=t["linear_value_head_dim"],
            vocab=t["vocab_size"],
        )
        # SparkInfer derives the pattern from the interval and ignores layer_types; refuse a
        # config where the two disagree rather than silently mis-describing the model.
        lt = t.get("layer_types")
        if lt is not None:
            derived = ["linear_attention" if arch.is_linear(i) else "full_attention" for i in range(arch.n_layers)]
            if list(lt) != derived:
                raise ValueError("config layer_types disagree with full_attention_interval")
        return arch

    def is_linear(self, layer: int) -> bool:
        return (layer + 1) % self.full_attention_interval != 0

    def units(self) -> list[Unit]:
        H, inter = self.hidden, self.intermediate
        qkv = 2 * self.linear_k_heads * self.linear_k_dim + self.linear_v_heads * self.linear_v_dim
        vdim = self.linear_v_heads * self.linear_v_dim
        qdim = self.n_heads * self.head_dim
        kvdim = self.n_kv_heads * self.head_dim
        out: list[Unit] = []
        for i in range(self.n_layers):
            b = f"{LM}layers.{i}."
            if self.is_linear(i):
                la = b + "linear_attn."
                out.append(Unit(f"L{i}.gdn.qkv", "gdn.qkv", i, (Linear(la + "in_proj_qkv", qkv, H),)))
                out.append(Unit(f"L{i}.gdn.z", "gdn.z", i, (Linear(la + "in_proj_z", vdim, H),)))
                out.append(Unit(f"L{i}.gdn.out", "gdn.out", i, (Linear(la + "out_proj", H, vdim),)))
            else:
                sa = b + "self_attn."
                q_rows = 2 * qdim if self.attn_output_gate else qdim
                out.append(Unit(f"L{i}.attn.q", "attn.q", i, (Linear(sa + "q_proj", q_rows, H),)))
                out.append(Unit(f"L{i}.attn.k", "attn.k", i, (Linear(sa + "k_proj", kvdim, H),)))
                out.append(Unit(f"L{i}.attn.v", "attn.v", i, (Linear(sa + "v_proj", kvdim, H),)))
                out.append(Unit(f"L{i}.attn.o", "attn.o", i, (Linear(sa + "o_proj", H, qdim),)))
            m = b + "mlp."
            out.append(Unit(f"L{i}.mlp", "mlp", i, (
                Linear(m + "gate_proj", inter, H), Linear(m + "up_proj", inter, H), Linear(m + "down_proj", H, inter),
            )))
        out.append(Unit("lm_head", "lm_head", None, (Linear("lm_head", self.vocab, H),)))
        return out

    def unit_map(self) -> dict[str, Unit]:
        return {u.id: u for u in self.units()}


def searchable_prefixes(units: list[Unit]) -> set[str]:
    return {lin.prefix for u in units for lin in u.linears}


def select_units(units: list[Unit], pattern: str, layers: str | None = None) -> list[Unit]:
    """Glob on unit ids (`L*.gdn.*`, `lm_head`), optionally restricted to a layer range `a-b,c`."""
    allowed = parse_layers(layers) if layers else None
    return [
        u for u in units
        if fnmatch.fnmatchcase(u.id, pattern) and (allowed is None or (u.layer is not None and u.layer in allowed))
    ]


def parse_layers(spec: str) -> set[int]:
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out
