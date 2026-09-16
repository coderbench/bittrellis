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
| NVFP4 | ModelOpt `.weight` U8 + UE4M3 block scales + F32 tensor scale (or the compressed-tensors layout for `unsloth`) | `baseline`: the shipped R0 bytes · `rtn`: round-to-nearest from BF16 · `unsloth`: the pinned R1 checkpoint's calibrated bytes (MLP layers 0–55 only) |
| FP8 | E4M3 weight + one BF16 scale per output row | `rtn` |
| Q4_K | BF16, byte-identical to the base model | none: SparkInfer fits Q4_K at load |

A rule or module override can pick its own quantizer:

```yaml
rules:
  - match: "L*.mlp"
    layers: "0-55"
    precision: NVFP4
    quantizer: unsloth
modules:
  L3.attn.o: {precision: NVFP4, quantizer: rtn}
```

A manifest never carries bytes. It can only name quantizers implemented in `bittrellis/build.py`
or pinned checkpoints listed under `model.quantizer_sources` in the track, so the evaluator
regenerates every byte itself.

New quantizers (for example GPTQ-style error feedback or scale search) are welcome as PRs. Whatever
they produce must still pass the audit's fidelity bound: on sampled rows, reconstruction error at
most 2× round-to-nearest plus 0.01 (calibrated encoders measure 1.2–1.6×; substituted bytes > 5×).

## Identity

A candidate's id is the first 16 hex characters of a SHA-256 over the track and the fully
**expanded** map: every unit's `precision@quantizer`.

Different rule spellings that expand to the same map are the same candidate.

## Commands

```bash
bittrellis manifest my.yaml               # validate, print the per-kind summary and the id
bittrellis manifest my.yaml --expand      # print the full unit → precision map
bittrellis build my.yaml                  # CPU-only: writes models/candidates/<name>-<id>/
bittrellis audit models/candidates/...    # the checks validators run before scoring
bittrellis describe <any checkpoint dir>  # what SparkInfer executes for any Qwen3.8 checkpoint
```
