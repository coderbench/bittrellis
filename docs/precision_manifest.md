# Manifests

> Per unit: the format SparkInfer runs and the quantizer writing its bytes ([terms](../README.md#key-terms)).

```yaml
schema: bittrellis/manifest@2
track: HPC-01
name: gdn-fp8-deep-calibrated-mlp       # lowercase, hyphenated; becomes the artifact name
description: >
  FP8 for deep recurrent layers; calibrated NVFP4 bytes for MLP 0-55.
authors: [your-github-handle]
default: NVFP4                          # every unit starts here (NVFP4 or Q4_K)
quantizers: {NVFP4: baseline}           # default quantizer per format (optional)
rules:                                  # applied top to bottom; later rules win
  - match: "L*.gdn.*"                   # glob over unit ids
    layers: "32-63"                     # optional layer filter: "a-b,c,d-e"
    format: FP8
  - match: "L*.mlp"
    layers: "0-55"
    format: NVFP4
    quantizer: unsloth                  # optional per-rule quantizer
modules:                                # exact unit overrides, applied last
  lm_head: Q4_K
  L3.attn.o: {format: NVFP4, quantizer: rtn, params: {}}
```

The older `bittrellis/precision-manifest@1` schema (key `precision`) is still read.

## Legal assignments

```text
L0.gdn.qkv   L0.gdn.z   L0.gdn.out   L0.mlp            layers 0,1,2 · 4,5,6 · …  (Gated DeltaNet)
L3.attn.q    L3.attn.k  L3.attn.v    L3.attn.o   L3.mlp   layers 3,7,…,63       (full attention)
lm_head
```

Formats:

- `gdn.*`: NVFP4 · FP8 · Q4_K
- `attn.*`, `mlp`: NVFP4 · Q4_K
- `lm_head`: NVFP4 (executes Q4_K(nvfp4) at batch 1) · Q4_K

Quantizers:

- `baseline` (NVFP4, attested): shipped bytes; NVFP4 default
- `unsloth` (NVFP4, attested): calibrated bytes, MLP layers 0–55 only
- `rtn` (NVFP4/FP8, regenerable): round-to-nearest from BF16; FP8 default
- `runtime` (Q4_K, runtime): the only Q4_K quantizer; SparkInfer fits it at load

Rejected: FP8 outside GDN, a quantizer on a format it can't produce, a Q4_K "quantizer", an
attested source with no bytes for a unit, unknown units or quantizers
([add one](quantizer_contract.md)).

## Expansion and id

Expanded per unit into `precision_manifest.yaml` and `candidate.json`:

```yaml
L40.gdn.z: {source_format: FP8, quantizer: rtn@v1, params: {}, execution: {decode_b1: FP8, packed_wide: FP8}}
lm_head:   {source_format: NVFP4, quantizer: baseline@v1, params: {},
            execution: {decode_b1: Q4_K(nvfp4), packed_wide: "NVFP4 if free VRAM at load > payload + 3 GiB, else Q4_K(nvfp4)"}}
```

**Candidate id** = first 16 hex chars of SHA-256(track + every unit's `FORMAT@quantizer@vN[+params]`).
Identical expansions share an id; a quantizer version bump changes every id using it.

## Commands

```bash
bittrellis inventory --out module_inventory.json          # 273 units: shapes, legal/executed formats, lineage, loader constraints
bittrellis manifest my.yaml                               # validate, per-kind summary, id
bittrellis manifest my.yaml --expand                      # full per-unit assignment
bittrellis manifest my.yaml --against manifests/          # flag duplicates, near-duplicates
bittrellis build my.yaml --out models/candidates/mine     # CPU only, sources hash-verified
bittrellis audit models/candidates/mine                   # what validators check before scoring
bittrellis search neighbors my.yaml --out proposals/      # every legal one-group change
```
