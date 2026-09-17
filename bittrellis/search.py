"""Baseline search helpers.

`neighbors()` proposes every legal one-group change around a manifest: for each unit kind (and GDN
projection) in each depth half, switch the whole group to another legal format/quantizer. It is the
starting move of the simple greedy search in docs/search.md -- deliberately naive, so better search
is a place miners can win.
"""

from __future__ import annotations

import copy

from . import quantizers as Q
from .manifest import Manifest, ManifestError
from .model.qwen38 import Unit
from .precision import SPACE

GROUPS = (
    ("gdn-qkv", "L*.gdn.qkv"), ("gdn-z", "L*.gdn.z"), ("gdn-out", "L*.gdn.out"),
    ("attn", "L*.attn.*"), ("mlp", "L*.mlp"),
)
DEPTHS = (("shallow", "0-31"), ("deep", "32-63"))


def neighbors(base: Manifest, units: list[Unit], limit: int = 0) -> list[Manifest]:
    current = base.expand_assignments(units)
    base_id = base.candidate_id(units)
    out: list[Manifest] = []
    seen = {base_id}
    for gname, pattern in GROUPS:
        kind = gname.split("-")[0]
        for dname, layers in DEPTHS:
            for fmt in SPACE[kind]:
                for q in Q.REGISTRY.values():
                    if fmt not in q.formats:
                        continue
                    m = copy.deepcopy(base)
                    m.name = f"{base.name}+{gname}-{dname}-{fmt.lower()}-{q.name}"
                    m.rules = list(m.rules) + [{"match": pattern, "layers": layers, "format": fmt, "quantizer": q.name}]
                    try:
                        asg = m.expand_assignments(units)
                        cid = m.candidate_id(units)
                    except ManifestError:
                        continue
                    if cid in seen or all(asg[k] == current[k] for k in asg):
                        continue
                    # Attested sources only cover some units (unsloth: MLP layers 0-55); proposals that
                    # need bytes a source does not have are dropped here rather than failing at build.
                    if q.name == "unsloth" and (kind != "mlp" or dname == "deep"):
                        continue
                    seen.add(cid)
                    out.append(m)
                    if limit and len(out) >= limit:
                        return out
    return out
