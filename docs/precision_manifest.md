# Manifests

A manifest is the whole submission: a few lines of YAML that say, for each part of the model, which
format SparkInfer should execute and which quantizer produces its bytes. BitTrellis turns it into a
checkpoint, audits it and measures it.

```yaml
schema: bittrellis/manifest@2
track: HPC-01
name: gdn-fp8-deep-calibrated-mlp       # lowercase, hyphenated; becomes the artifact name
description: >
  Protect the deep recurrent layers with FP8 and use calibrated NVFP4 bytes for MLP 0-55.
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

## Unit ids

```text
L0.gdn.qkv   L0.gdn.z   L0.gdn.out   L0.mlp            layers 0,1,2 · 4,5,6 · …  (Gated DeltaNet)
L3.attn.q    L3.attn.k  L3.attn.v    L3.attn.o   L3.mlp   layers 3,7,…,63       (full attention)
lm_head
```

`bittrellis inventory --out module_inventory.json` lists all 273 units with their shapes, legal
assignments, executed formats, lineage and loader constraints.

## Legal assignments

| Unit kind | NVFP4 | FP8 | Q4_K |
|---|:-:|:-:|:-:|
| `gdn.*` | ✓ | ✓ | ✓ |
| `attn.*` | ✓ | | ✓ |
| `mlp` | ✓ | | ✓ |
| `lm_head` | ✓ (executes Q4_K(nvfp4) at batch 1) | | ✓ |

| Quantizer | Formats | Lineage | Notes |
|---|---|---|---|
| `baseline` | NVFP4 | attested | the shipped bytes (default for NVFP4) |
| `unsloth` | NVFP4 | attested | calibrated bytes, MLP layers 0–55 only |
| `rtn` | NVFP4, FP8 | regenerable | round-to-nearest from BF16 (default for FP8) |
| `runtime` | Q4_K | runtime | the only Q4_K quantizer: SparkInfer fits Q4_K at load |

Anything else is rejected with a reason: FP8 outside GDN, a quantizer on a format it can't produce,
a Q4_K "quantizer", an attested source that lacks bytes for a unit, or an unknown unit or quantizer.
Adding quantizers: [quantizer_contract.md](quantizer_contract.md).

## Expansion and identity

A manifest *expands* to one assignment per unit. The expanded form is stored in the built checkpoint
(`precision_manifest.yaml`) and in `candidate.json`:

```yaml
L40.gdn.z: {source_format: FP8, quantizer: rtn@v1, params: {}, execution: {decode_b1: FP8, packed_wide: FP8}}
lm_head:   {source_format: NVFP4, quantizer: baseline@v1, params: {},
            execution: {decode_b1: Q4_K(nvfp4), packed_wide: "NVFP4 if free VRAM at load > payload + 3 GiB, else Q4_K(nvfp4)"}}
```

The **candidate id** is the first 16 hex characters of a SHA-256 over the track and every unit's
`FORMAT@quantizer@vN[+params]`. Different rule spellings that expand identically are the same
candidate. A quantizer version bump changes the id of every candidate that uses it.

## Commands

```bash
bittrellis manifest my.yaml                               # validate, per-kind summary, id
bittrellis manifest my.yaml --expand                      # full per-unit assignment
bittrellis manifest my.yaml --against manifests/          # flag duplicates and near-duplicates
bittrellis build my.yaml --out models/candidates/mine     # CPU only, sources hash-verified
bittrellis audit models/candidates/mine                   # what validators check before scoring
bittrellis search neighbors my.yaml --out proposals/      # every legal one-group change
```
