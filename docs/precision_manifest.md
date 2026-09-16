# Precision manifests

A manifest is the whole submission: a few lines of YAML that say which precision each part of
the model runs at. BitTrellis turns it into a checkpoint, audits it, and measures it.

```yaml
schema: bittrellis/precision-manifest@1
track: HPC-01
name: gdn-fp8-late-head-q4k           # lowercase, hyphenated; becomes the artifact name
description: >
  Protect the recurrent path where depth is largest; fit the head once from BF16.
authors: [your-github-handle]
default: NVFP4                        # every unit starts here (NVFP4 or Q4_K)
rules:                                # applied top to bottom; later rules win
  - match: "L*.gdn.*"                 # glob over unit ids
    layers: "40-63"                   # optional layer filter: "a-b,c,d-e"
    precision: FP8
  - match: "L*.mlp"
    layers: "0-7"
    precision: Q4_K
modules:                              # exact unit overrides, applied last
  lm_head: Q4_K
quantizers:                           # how stored bytes are produced
  NVFP4: baseline                     # baseline | rtn
  FP8: rtn
```

## Unit ids

```text
L0.gdn.qkv   L0.gdn.z   L0.gdn.out   L0.mlp        ← layers 0,1,2 · 4,5,6 · …  (Gated DeltaNet)
L3.attn.q    L3.attn.k  L3.attn.v    L3.attn.o   L3.mlp   ← layers 3,7,…,63  (full attention)
lm_head
```

`bittrellis inventory --out module_inventory.json` lists all 273 units, with shapes, parameter
counts, legal precisions and bytes per precision.

## Legal precisions

| kind | NVFP4 | FP8 | Q4_K |
|---|:-:|:-:|:-:|
| `gdn.*` | ✓ | ✓ | ✓ |
| `attn.*` | ✓ | | ✓ |
| `mlp` | ✓ | | ✓ |
| `lm_head` | ✓ | | ✓ |

Anything else is rejected with the reason; see [precision_space.md](precision_space.md) for why.

## Quantizers

| Precision | Stored bytes | Quantizer options |
|---|---|---|
| NVFP4 | ModelOpt `.weight` U8 + UE4M3 block scales + F32 tensor scale | `baseline`: the shipped checkpoint's bytes (per unit, identical to R0) · `rtn`: round-to-nearest from BF16 with max calibration |
| FP8 | E4M3 weight + one BF16 scale per output row | `rtn` |
| Q4_K | BF16, byte-identical to the base model | none: SparkInfer fits Q4_K at load |

New quantizers (for example error-feedback rounding or scale search) are welcome as PRs. Whatever
they produce must still pass the audit's fidelity bound: on sampled rows, reconstruction error at
most 1.35× round-to-nearest plus 0.01.

## Identity

A candidate's id is the first 16 hex characters of a SHA-256 over:

- the track;
- the fully **expanded** map (every unit and its precision);
- the quantizer of each precision it uses.

Different rule spellings that expand to the same map are the same candidate.

## Commands

```bash
bittrellis manifest my.yaml               # validate, print the per-kind summary and the id
bittrellis manifest my.yaml --expand      # print the full unit → precision map
bittrellis build my.yaml                  # CPU-only: writes models/candidates/<name>-<id>/
bittrellis audit models/candidates/...    # the checks validators run before scoring
bittrellis describe <any checkpoint dir>  # what SparkInfer executes for any Qwen3.8 checkpoint
```
