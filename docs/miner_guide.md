# Miner guide

You don't touch the inference engine. You submit a better **quantization topology** for it: which
format each part of Qwen3.8-27B executes, and which quantizer produces its bytes. The evaluator
proves on a real RTX 5090 whether it moved the frontier.

```text
  you                                         evaluator (pinned RTX 5090)
  ───                                         ───────────────────────────
  manifests/<name>.yaml          ──PR──▶     validate → build → audit → RP-KL → tasks
  (optionally a new quantizer)                → 2 perf runs → private holdout → ε-frontier
                                                                               │
                    ◀── PR comment: table, paired Δ vs V0, gates, FG-2, label ◀─┘
```

## 1. Set up (no GPU needed)

```bash
git clone https://github.com/coderbench/bittrellis && cd bittrellis
pip install -e ".[dev]"
pytest -q                          # build → audit → lineage → tamper detection on tiny synthetic models
bittrellis quantizers              # runtime · baseline · unsloth · rtn
bittrellis inventory --out module_inventory.json
```

## 2. Know what you are competing against

The internal frontier already contains V0 (the shipped checkpoint) and every seed candidate in
[`results/feasibility/`](../results/feasibility/feasibility_report.md). Read it first. It tells you:

- **Quantizer quality is a big lever.** Calibrated NVFP4 bytes for the MLP (V13) cut RP-KL by 11.5%
  at the same decode speed and memory, but cost 4.4% prefill. Whether that cost is the byte layout
  or the values is open.
- **FP8 on GDN buys fidelity for decode.** It cuts long-context drift by 62% at 13% decode cost.
- **MLP bytes guard math, the recurrent path guards long context.** Tool calling is the worst
  category for every checkpoint.
- **The first layer of each kind picks the batched-prefill path.** Changing layer 0 can cost 20–50%
  prefill.
- **Q4_K saves memory and costs prefill.**
- **Effects interact.** Test combinations, not just single changes.

## 3. Write a manifest

```yaml
schema: bittrellis/manifest@2
track: HPC-01
name: calibrated-mlp-gdn-fp8-deep-qkv
description: >
  Calibrated MLP bytes where available, plus FP8 on the state-writing qkv projection of the
  deepest GDN layers; keeps layer 0 NVFP4 so batched prefill stays on the fast path.
authors: [your-handle]
default: NVFP4
rules:
  - match: "L*.mlp"
    layers: "0-55"
    format: NVFP4
    quantizer: unsloth
  - match: "L*.gdn.qkv"
    layers: "48-63"
    format: FP8
```

```bash
bittrellis manifest manifests/calibrated-mlp-gdn-fp8-deep-qkv.yaml --against manifests/
```

Syntax: [precision_manifest.md](precision_manifest.md). Exploring neighbors:
`bittrellis search neighbors <manifest> --out proposals/`.

## 4. Or write a quantizer

A better encoder for a format the runtime already executes is often the biggest win. Implement the
[quantizer contract](quantizer_contract.md) and submit it together with one manifest that uses it.
Code PRs are evaluated after a maintainer reviews them and adds `eval-approved`.

## 5. (Optional) measure it yourself

On an RTX 5090 host with about 120 GB of disk and 60 GB of RAM:

```bash
scripts/setup_sparkinfer.sh                 # pinned runtime + fixed-partition scorer
scripts/setup_models.sh                     # base, shipped, unsloth; hash-verified
bittrellis reference                        # BF16 reference, once
bittrellis build manifests/mine.yaml --out models/candidates/mine
bittrellis evaluate models/candidates/mine --out artifacts/mine
bittrellis compare results/feasibility/artifacts/V0-baseline-rebuild artifacts/mine
bittrellis frontier --with-seeds artifacts/mine
```

Self-reported numbers are never scored; they help you iterate.

## 6. Open a PR

One manifest per PR in `manifests/`, with the hypothesis in `description`. Don't modify evaluator
paths: `configs/`, `data/`, `bittrellis/eval/`, `bittrellis/frontier/`, `bittrellis/validate.py`,
`bittrellis/lineage.py`, `bittrellis/holdout.py`, `bittrellis/build.py`, `bittrellis/runtime.py`,
`evaluator/`, `tools/`, `scripts/`, `.github/`.

## How a result is scored

1. **Audit.** Legal formats, legal quantizers, lineage proven, frozen tensors untouched.
2. **Gates.**
   - runtime correctness;
   - RP-KL ≤ 0.30 and top-1 ≥ 0.80;
   - every BF16-retrievable needle at 8K/16K/32K;
   - no task suite more than 6 items below V0;
   - private holdout PASS.
3. **ε-frontier.** RP-KL ↓ · decode ↑ · 4K prefill ↑ · peak GPU memory ↓, with noise-aware dominance
   against V0, the seeds, accepted results and earlier open PRs by other authors.
4. **FG-2.** The normalized hypervolume your result adds. Dominated, invalid or duplicate results
   earn 0; see [frontier.md](frontier.md).
5. **Tier.** FG-2 is bucketed into `eval:XL` (≥ 0.60%) … `eval:XS`; Gittensor pays the tier when a
   maintainer merges the PR. The highest open result is marked `bt:merge-first`. See [rewards.md](rewards.md).

Before any GPU time, the PR is screened ([guards.md](guards.md)):

- **Identical recipe** to a seed, an accepted result or an earlier PR → not measured.
- **Within 2% of the weights** of an earlier PR by another author → measured, but ranked with that PR
  on the frontier, so it earns only what it adds. Three in one epoch wait for a maintainer.
- **Predicted to exceed GPU memory** → not measured.
- **A new quantizer that reproduces an existing encoder's bytes** → not measured, however the code is
  written.
- **More than 3 of your PRs waiting** → later ones wait their turn.

The GPU stages stop early: a quality-gate failure skips the speed runs, and a result that is already
dominated after the speed runs skips tasks and the holdout.

## What earns nothing

- relabeling metadata, or changing stored formats when execution is unchanged;
- Q4_K "quantizers";
- changing frozen weights;
- `SPARKINFER_*` tricks (they are cleared);
- fitting the public corpus (the holdout catches it);
- copying another PR, including with rewritten rules or a renamed encoder;
- disk-size savings with no runtime effect;
- resubmitting a seed or an accepted candidate.
