# Feasibility protocol (Phase 1–2)

The question this experiment answers before any search engine is built:

> **Can a non-uniform precision map produce a Qwen3.8-27B checkpoint that SparkInfer runs on one
> RTX 5090 and that is not dominated by the checkpoint SparkInfer ships today?**

Nothing about the answer is assumed. If it is no, the project stops.

## Fixed inputs

Everything is pinned in [`configs/hpc01.yaml`](../configs/hpc01.yaml):

- base BF16 model;
- NVFP4 baseline;
- SparkInfer `b1ed168` (v0.5.8) and the build flags;
- the corpus `hpc01-public-v2`, with its source datasets and tokenizer pinned by revision.

## Steps

```text
0. lock      pins · SparkInfer build · models · corpus hash · BF16 reference hash
1. baseline  R0 through the full harness, plus a determinism check (score twice, compare bytes)
2. variants  V0–V9 built from manifests, audited, scored, benchmarked
3. refs      R1 unsloth mixed NVFP4/FP8 · R2 llama.cpp UD-Q4_K_M
4. decide    frontier + answers A–G below → STRONG PASS / PASS / BORDERLINE / FAIL
```

## Measurements per checkpoint

| What | How | Repeats |
|---|---|---|
| KL(BF16 ‖ candidate), top-1, ΔNLL | `qwen3_gguf_score`, teacher-forced, `SPARKINFER_DETERMINISTIC=1`, 5 × 4K category streams + 8K/16K/32K tails | 1 (bit-deterministic) |
| Needle recall at 8K/16K/32K | argmax on the digits of three codes planted at 10/50/90% depth | same run |
| Decode / prefill tok/s at 128, 4K, 16K | `qwen3_gguf_bench` sweep, real-text prompt ids | **2** (lower median) |
| VRAM | bench `VRAM used` after load at 16K max context, on an idle card | same run |
| Task guard | SparkInfer `bench/quality` benchmark tier (196 items) via `sparkinfer_server` | 1 (greedy) |
| Checkpoint audit | `bittrellis audit` (bytes, tensor set, fidelity, loader resolution) | 1 |

**Uncertainty.**
- **Quality:** a 95% block-bootstrap interval over positions (block = 128 tokens). Scoring is deterministic, so run-to-run noise is zero by construction; step 1 verifies that.
- **Speed:** the spread of the two repetitions. A speed difference counts only if it exceeds 2% *and* both of the candidate's reps beat both of the reference's.

## Variants

| ID | Map | Hypothesis under test |
|---|---|---|
| V0 | NVFP4 everywhere, baseline bytes | builder is exact: reproduces R0 tensor for tensor |
| V1 | Q4_K everywhere | the other uniform 4.5-bit corner; Q4_K decode kernels vs NVFP4 |
| V2 | lm_head stored BF16 | the baseline quantizes the head twice (→NVFP4→Q4_K); one fit is better, and costs no speed |
| V3 | GDN FP8 | spending 8 bits on the recurrent path buys long-context quality |
| V4 | GDN Q4_K | GDN Q4_K is faster (SparkInfer measured +19–23% for a refit); a fit from BF16 limits the damage |
| V5 | attention Q4_K | attention is a small share of weights; how sensitive is it? |
| V6 | MLP Q4_K | the bulk of weights; decode kernel effect |
| V7 / V8 | MLP Q4_K in layers 0–31 / 32–63 | depth sensitivity is measured, not assumed |
| V9 | V4 + V6 | interaction: is ΔKL(V9) ≈ ΔKL(V4) + ΔKL(V6)? |

### Follow-ups (added after the first pass, before the verdict)

The first pass raised three questions: GDN FP8 (V3) bought quality at a speed cost, and unsloth's
checkpoint beat it on KL with the same GDN precision. These variants were added to answer them.
They are labelled as post-hoc, and the verdict rule below was not changed.

| ID | Map | Question |
|---|---|---|
| V10 / V11 | GDN FP8 in layers 32–63 / 0–31 | Which depth carries V3's gain, at what share of its cost? |
| V12 | GDN FP8 on `qkv` only | Is the state-writing projection enough? |
| V13 | R0's map with unsloth's calibrated NVFP4 bytes for MLP 0–55 | How much of R1's quality comes from the quantizer rather than the topology? |

## Decision questions

| | Question | YES if |
|---|---|---|
| A | Is the baseline reproducible? | two score runs are byte-identical and V0 reproduces R0's tensors |
| B | Do maps change quality meaningfully? | some variants' KL 95% intervals do not overlap R0's |
| C | Do maps change real speed? | some variant's decode or prefill differs from R0 by > 2%, both reps agreeing |
| D | Does a mixed map beat the baseline frontier? | some non-uniform variant is valid and not dominated by R0 |
| E | Are GDN / attention / MLP sensitivities different? | ΔKL per billion weights differs across V4/V5/V6 by more than their intervals |
| F | Do interactions matter? | ΔKL(V9) is outside ΔKL(V4) + ΔKL(V6) ± the combined intervals |
| G | Is automated search justified? | see verdict rule |

## Verdict rule (fixed before running)

- **STRONG PASS:** D is yes, and one valid mixed map either improves one objective by ≥ 5% at statistically equivalent KL, or improves KL by ≥ 10% at ≤ 2% speed and VRAM cost. B and C are also yes.
- **PASS:** D is yes, with smaller margins, and B and C are yes.
- **BORDERLINE:** B or C is yes, but no mixed map escapes R0's dominance.
- **FAIL:** neither B nor C, or the toolchain cannot express module-level maps.

Raw artifacts land in `artifacts/feasibility/<id>/`. `bittrellis report` writes
`results/feasibility/` (frontier.json, comparison.csv, plots). The narrative answers are in
[`results/feasibility/feasibility_report.md`](../results/feasibility/feasibility_report.md).
