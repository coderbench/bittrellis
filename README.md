# BitTrellis

### Every layer doesn't deserve the same bits.

**One model. One GPU. Find the precision map the hardware actually wants.**

A 27-billion-parameter model is not numerically uniform. Some projections shrug off 4-bit
weights; some carry recurrent state that quietly compounds every rounding error across 32,000
tokens. The kernel that runs a format fast on one GPU can be the slow path on the next.

Yet almost every checkpoint ships with **one precision stamped on every layer.**

**BitTrellis searches for the map instead:** which parts of the model run at which precision,
built into a real checkpoint, audited, and measured on the real card.

```text
                           B I T T R E L L I S

   Qwen3.8-27B (BF16)   ──┐
                          │   ┌───────────────────────────────┐
   precision manifest   ──┼──▶│ build   ▸ bytes the runtime   │
   (273 units)            │   │           actually executes   │
                          │   │ audit   ▸ nothing but          │
   pinned SparkInfer    ──┘   │           precision changed    │
                              │ measure ▸ KL vs BF16 · tok/s · │
                              │           VRAM · tasks         │
                              └───────────────┬───────────────┘
                                              ▼
                                  deployable checkpoint
                                              │
                              SparkInfer ──▶ 1× RTX 5090 ──▶ frontier
```

<!-- STATUS -->

## Status: Phase 1–2 feasibility

Before building a search engine, BitTrellis had to prove there is something to search for. The
feasibility experiment and its verdict are in
[`results/feasibility/feasibility_report.md`](results/feasibility/feasibility_report.md).

<!-- /STATUS -->

---

## What we race against

A benchmark that only beats itself can report any number it likes. BitTrellis measures three
**external reference points** with exactly the same harness, corpus and BF16 reference as every
candidate:

| | Reference point | Precision map | Why it's here |
|---|---|---|---|
| **R0** | [gittensor NVFP4](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090) on SparkInfer | NVFP4 on all 400 Linears + head | what SparkInfer ships and scores today |
| **R1** | [unsloth NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) on SparkInfer | hand-tuned NVFP4 MLP + FP8 attention/GDN/head | an expert's mixed map for the same model |
| **R2** | [unsloth UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF) on **llama.cpp** | imatrix-guided Q4_K/Q5_K/Q6_K mix | the llama.cpp ecosystem's best answer |

NVIDIA's official NVFP4 build is recorded too. It **does not load** on the pinned runtime: its
FP8 scales are per-tensor, and SparkInfer reads only per-row FP8.

<!-- FRONTIER -->
<!-- /FRONTIER -->

---

## 🎯 Current target — HPC-01

| | |
|---|---|
| Model | Qwen3.8-27B: 64 layers, 48 Gated DeltaNet + 16 full attention, pinned revision |
| GPU | 1× NVIDIA GeForce RTX 5090 (32 GB) |
| Runtime | SparkInfer v0.5.8 @ `b1ed168`, every runtime knob pinned |
| Search space | 273 units × {NVFP4, FP8, Q4_K} where the runtime executes them |
| Quality | KL divergence from BF16 on a fixed 49K-token corpus, plus 8K/16K/32K needles |
| Objectives | KL ↓ · decode tok/s ↑ · VRAM ↓ |
| Frozen | architecture, tokenizer, every non-Linear tensor, the weights themselves |

Everything is pinned in [`configs/hpc01.yaml`](configs/hpc01.yaml).

> **Which parts of the model should spend precision, and which can give it up?**

---

## The precision space is what the runtime executes

A checkpoint can *store* any format; what matters is what the loader *runs*. At the pinned commit:

```text
                        stored ▸   NVFP4          FP8 (per row)      BF16
   ─────────────────────────────────────────────────────────────────────────
   GDN  qkv · z · out   runs  ▸   NVFP4          FP8                Q4_K fit
   attention q·k·v·o    runs  ▸   NVFP4          Q4_K refit (!)     Q4_K fit
   MLP  gate·up·down    runs  ▸   NVFP4          Q4_K refit (!)     Q4_K fit
   lm_head (batch 1)    runs  ▸   Q4_K refit     Q4_K refit         Q4_K fit
```

So a manifest names the precision that **runs**, and anything the runtime would silently
convert is rejected. BF16 never executes for a Linear; FP8 only executes on GDN. Details and
line-level evidence: [docs/precision_space.md](docs/precision_space.md).

---

## Five words that do all the work

| Term | Meaning |
|---|---|
| **Unit** | The smallest piece whose precision the runtime lets you choose: `L7.attn.q`, `L40.gdn.z`, `L12.mlp`, `lm_head`. There are 273. |
| **Manifest** | A few lines of YAML that assign a precision to every unit. It *is* the submission. |
| **Audit** | Proof that a checkpoint is nothing but a precision transformation of the pinned weights: bytes, tensor set, reconstruction fidelity, loader resolution. |
| **KL** | How far the checkpoint's next-token distribution is from BF16, teacher-forced through SparkInfer itself. Deterministic, compared *paired* position by position. |
| **Frontier Gain** | The normalized hypervolume a result adds to the KL × decode × VRAM frontier. It replaces XS/S/M/L/XL labels, which measure effort, not effect. |

---

## Quickstart (no GPU)

```bash
git clone https://github.com/coderbench/bittrellis && cd bittrellis
pip install -e ".[dev]"
pytest -q                                          # build → audit → tamper detection on tiny synthetic models

bittrellis inventory                               # 273 units · 144 GDN · 64 attention · 64 MLP · lm_head
bittrellis manifest experiments/feasibility/variants/V2-lmhead-bf16-source.yaml
```

A manifest:

```yaml
schema: bittrellis/precision-manifest@1
track: HPC-01
name: head-once-gdn-fp8-deep
default: NVFP4
rules:
  - match: "L*.gdn.*"
    layers: "40-63"
    precision: FP8
modules:
  lm_head: Q4_K
```

## On the GPU host

```bash
scripts/setup_sparkinfer.sh                 # pinned SparkInfer build
scripts/setup_models.sh base baseline       # BF16 base + NVFP4 baseline
bittrellis doctor                           # everything pinned and present?
bittrellis reference                        # BF16 reference distributions, once

bittrellis build my.yaml --out models/candidates/mine     # CPU only, deterministic
bittrellis evaluate models/candidates/mine --out artifacts/mine
bittrellis frontier artifacts/*
```

Reproduce the whole feasibility experiment: `experiments/feasibility/run.sh`.

---

## For miners

You don't rewrite the inference engine. You submit a better map for it.

1. Read the measured sensitivities in the [feasibility report](results/feasibility/feasibility_report.md).
2. Write `manifests/<name>.yaml` with a hypothesis in its description.
3. Open a PR. The evaluator builds, audits, scores and benchmarks it on the pinned RTX 5090.
4. Your result earns **Frontier Gain** only if it passes every gate and opens new space.

[Miner guide](docs/miner_guide.md) · [Manifest format](docs/precision_manifest.md) ·
[Evaluation](docs/evaluation.md) · [Frontier](docs/frontier.md)

---

## Repository

```text
bittrellis/
├── configs/hpc01.yaml          every pin: model, runtime, references, gates, frontier box
├── bittrellis/
│   ├── model/qwen38.py         architecture → 273 searchable units
│   ├── precision.py            the runtime-executed precision space + loader resolver
│   ├── manifest.py             rules → expanded map → content-addressed candidate id
│   ├── quant/formats.py        bit-exact BF16 · FP8 E4M3 · NVFP4 codecs (numpy)
│   ├── build.py                manifest → deployable checkpoint (CPU, deterministic)
│   ├── validate.py             audit + describe any checkpoint
│   ├── eval/                   corpus · BF16 reference · KL · bench · tasks · llama.cpp
│   └── frontier/               gates · Pareto · Frontier Gain · reports
├── data/corpus/                pinned, hash-verified evaluation tokens
├── experiments/feasibility/    V0–V9 manifests, runner, analysis
├── results/feasibility/        frontier, comparison, plots, report
├── tools/llamacpp_score.cpp    teacher-forced scoring for the llama.cpp reference
└── docs/
```

## What this repo is not

- not another inference runtime ([SparkInfer](https://github.com/gittensor-ai-lab/sparkinfer) runs the artifact)
- not a generic quantization library
- not a model zoo, a model server, or a training framework
- not a wrapper around `quantize(model, bits=4)`

Existing tools provide number formats. **BitTrellis finds where to put them.**

## Final goal

A sequence of deep, reproducible, hardware-specific frontiers, one model × GPU pair at a time:

```text
today   Qwen3.8-27B × RTX 5090 × SparkInfer
later   next model × RTX 5090   ·   Qwen3.8 × next GPU   ·   …
```

Not every model badly. One pair, optimized properly.

---

Why the design differs from the original blueprint: [docs/blueprint_review.md](docs/blueprint_review.md).
License: MIT.
