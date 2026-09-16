# Miner guide

You don't touch the inference engine. You submit a better **precision map** for it, and the
evaluator proves on real hardware whether it moved the frontier.

```text
  you                                   evaluator (pinned RTX 5090)
  ───                                   ───────────────────────────
  write manifests/<name>.yaml ──PR──▶   build  → audit → score → bench → tasks
                                                                   │
                         ◀── comment: gates · frontier · FG-1 ◀────┘
```

## 1. Set up (no GPU needed)

```bash
git clone https://github.com/coderbench/bittrellis && cd bittrellis
pip install -e ".[dev]"
pytest -q
bittrellis inventory --out module_inventory.json   # 273 units, shapes, legal precisions
```

## 2. Form a hypothesis

Useful starting points, all of which the evaluator can test:

- **Where does 8-bit pay?** FP8 exists only for GDN projections. Which layers, and which of `qkv`/`z`/`out`?
- **Which 4-bit format, where?** NVFP4 and Q4_K cost the same bits but differ in error and in kernels (Q4_K decodes through tuned int8 MMVQ; NVFP4 prefill uses CUTLASS FP4 GEMMs).
- **Don't quantize twice.** Any unit whose runtime path refits already-quantized bytes, like the head.
- **Interactions.** Two individually harmless changes can compound; see feasibility question F.
- **Better NVFP4 bytes.** A quantizer that lowers error at the same format (a PR to `bittrellis/quant/`).

Read [`results/feasibility/feasibility_report.md`](../results/feasibility/feasibility_report.md)
first. It holds the measured sensitivity of each unit kind, so you don't have to rediscover it.

## 3. Write a manifest

```yaml
schema: bittrellis/precision-manifest@1
track: HPC-01
name: head-once-gdn-fp8-deep
description: Fit lm_head from BF16 and protect the recurrent path in the deepest 16 GDN layers.
authors: [your-handle]
default: NVFP4
rules:
  - match: "L*.gdn.*"
    layers: "40-63"
    precision: FP8
modules:
  lm_head: Q4_K
```

```bash
bittrellis manifest manifests/head-once-gdn-fp8-deep.yaml
```

Full syntax: [precision_manifest.md](precision_manifest.md).

## 4. (Optional) measure it yourself

On a machine with an RTX 5090, ~80 GB of disk for models, and 60 GB of RAM for the one-time reference:

```bash
scripts/setup_sparkinfer.sh
scripts/setup_models.sh base baseline
bittrellis reference                       # once, ~2 min on a 5090 with CPU offload
bittrellis build manifests/head-once-gdn-fp8-deep.yaml --out models/candidates/mine
bittrellis evaluate models/candidates/mine --out artifacts/mine
bittrellis frontier artifacts/mine results/frontier/R0
```

Self-reported numbers are never scored. They help you iterate.

## 5. Open a PR

One manifest per PR, placed in `manifests/`, with the hypothesis stated in `description`.
Don't modify evaluator paths: `configs/`, `data/corpus/`, `bittrellis/eval/`,
`bittrellis/frontier/`, `bittrellis/validate.py`.

## How a result is scored

1. **Audit.** A pure precision transformation of the pinned weights, or it is rejected.
2. **Gates.** KL ≤ 0.30 · top-1 ≥ 0.80 · every needle retrieved · no task suite more than 6 items below R0.
3. **Frontier Gain (FG-1).** The normalized hypervolume your result adds to the current frontier (KL × decode × VRAM). Dominated results earn 0; see [frontier.md](frontier.md).
4. **Holdout.** Frontier-moving results are re-scored on a sealed holdout corpus drawn from the same sources. A map tuned to the public tokens does not survive it.

## What is not allowed

- **Weights:** changing weights beyond quantization (the fidelity audit bounds reconstruction error), or pruning, fine-tuning, distillation or architecture edits.
- **Model and runtime:** changing the tokenizer, config, or any non-searchable tensor; setting `SPARKINFER_*` runtime variables (they are cleared anyway); patching SparkInfer.
- **Evaluation:** special-casing corpus tokens, or modifying the evaluator.
