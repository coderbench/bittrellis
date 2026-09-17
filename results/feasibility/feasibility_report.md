# HPC-01 feasibility report (epoch hpc01-e2)

> **Verdict: PASS.** Automated precision and quantizer search is justified.

Measured on one RTX 5090 with SparkInfer `b1ed168`, 2026-09-17, evaluator epoch `hpc01-e2`. Every
number is reproducible from [`artifacts/`](artifacts/) and
[`experiments/feasibility/`](../../experiments/feasibility/). Replaces the e1 pass in
[`../feasibility-e1/`](../feasibility-e1/), whose STRONG PASS did not survive the stricter e2
measurement ([what changed](#what-changed-from-e1)). Terms: [README](../../README.md#key-terms).

## In short

- **Nothing beats today's checkpoint (V0) on every axis at once, and V0 beats none of them.** All
  eight mixed seeds are valid and trade something. The search space is open.
- **Best trade-off: V13.** V0's precision map with calibrated NVFP4 bytes for 56 MLP layers. It is
  **11.5% closer to the original model** (paired 95% CI excludes 0) at the same decode speed (−0.5%)
  and peak memory, but **reads 4K prompts 4.4% slower**. Under the pre-registered rule, that prefill
  cost makes it PASS, not STRONG PASS.
- **Maps matter on real hardware.** Same model, different maps: prefill up to −49%, peak memory
  −1.8 to +1.9 GiB, decode up to −13%.
- **Different layers protect different abilities.**
  - Calibrated MLP bytes (V13) halve math drift (0.175 → 0.084) and leave long context alone.
  - GDN FP8 (V3) cuts long-context drift by 62% (0.095 → 0.036) but costs 13% decode.
- **Effects do not add up.** GDN Q4_K × MLP Q4_K interaction: −0.0135 nats, 95% CI [−0.0264, −0.0023].
- **Lots of room left.** External references drift far less than any internal candidate: unsloth's
  checkpoint is 35% closer than V0, llama.cpp with a dynamic GGUF 53% closer, but both are slower.
  Closing that gap on SparkInfer is the miner's job.

## Setup

| | |
|---|---|
| Model | Qwen/Qwen3.8-27B (BF16 reference); baseline gittensor NVFP4; every source sha256-pinned in [`configs/sources.lock.json`](../../configs/sources.lock.json) |
| Runtime | SparkInfer v0.5.8 @ `b1ed168e`, all `SPARKINFER_*` cleared; scoring with `SPARKINFER_DETERMINISTIC=1` |
| Hardware | 1× RTX 5090 32 GB |
| Quality | Reference-Partition KL (BF16 top-256 + tail bucket) over 21,624 teacher-forced positions: 5 × 4K category streams + 8K/16K/32K tails with 9 needles. Candidate log-probs come from the unmodified SparkInfer runtime |
| Speed | `qwen3_gguf_bench` at 128/4K/16K context, 512 decode tokens, 2 independent processes; objectives at 4K |
| Memory | peak device memory polled over the whole benchmark process |
| Tasks | SparkInfer `bench/quality` benchmark tier (196 items) through `sparkinfer_server` |

Protocol, variants and verdict rule were fixed before running:
[feasibility_protocol.md](../../docs/feasibility_protocol.md). Seeds e1 had shown dominated or
gate-failing (V2, V8, V10, V11, V12) were not re-run.

## Results

Δ is against V0 over identical positions (paired 95% block-bootstrap CI). ★ = internal ε-frontier
over RP-KL, decode, prefill and peak memory (differences count only beyond noise). External
references are context only, never ranked.

| | Checkpoint | RP-KL | ΔRP-KL vs V0 [95% CI] | decode tok/s | prefill tok/s | peak GiB | tasks /196 |
|---|---|---:|---|---:|---:|---:|---:|
| ★ | **V0** today's shipped checkpoint | 0.1357 | — | 94.9 | 14,760 | 22.02 | 147 |
| | V3 GDN FP8 | 0.1128 | **−16.9%** [−0.0329, −0.0137] | 83.0 (−12.6%) | 11,848 (−19.7%) | 23.93 (+1.91) | 148 |
| ★ | **V13** calibrated MLP bytes | 0.1201 | **−11.5%** [−0.0285, −0.0041] | 94.4 (−0.5%) | 14,110 (−4.4%) | 22.02 (±0) | 144 |
| ★ | V1 everything Q4_K | 0.1242 | −8.5% [−0.0266, +0.0039] | 94.8 (−0.2%) | 7,537 (−48.9%) | 20.25 (−1.77) | 145 |
| | V9 GDN + MLP Q4_K | 0.1277 | −5.9% [−0.0224, +0.0063] | 94.0 (−0.9%) | 7,560 (−48.8%) | 20.41 (−1.61) | 148 |
| | V7 early MLP Q4_K | 0.1284 | −5.4% [−0.0197, +0.0042] | 94.3 (−0.7%) | 8,199 (−44.4%) | 21.40 (−0.61) | 147 |
| ★ | V6 MLP Q4_K | 0.1346 | −0.8% [−0.0114, +0.0097] | 94.4 (−0.6%) | 8,358 (−43.4%) | 20.84 (−1.18) | 145 |
| ★ | V5 attention Q4_K | 0.1356 | −0.1% [−0.0119, +0.0107] | 94.6 (−0.4%) | 13,571 (−8.1%) | 21.85 (−0.16) | 144 |
| ★ | V4 GDN Q4_K | 0.1424 | +4.9% [−0.0071, +0.0215] | 94.9 (±0.0%) | 12,011 (−18.6%) | 21.58 (−0.44) | 145 |
| ext | R1 unsloth NVFP4 checkpoint | 0.0879 | −35.3% [−0.0653, −0.0321] | 82.4 (−13.2%) | 10,305 (−30.2%) | 23.63 (+1.61) | 149 |
| ext | R2 llama.cpp + UD-Q4_K_M GGUF | 0.0638 | −53.0% [−0.0924, −0.0549] | 79.9 (−15.8%) | 3,616 (−75.5%) | 16.20 (−5.82) | — |

All internal rows pass every gate: audit, runtime correctness (no non-finite log-probs), RP-KL
≤ 0.30, top-1 ≥ 0.80, every 8K/16K/32K needle retrieved, at most 6 task items lost vs V0. Run-to-run
spread ≤ 0.2% on every speed figure. Plots: [`plots/`](plots/).

### Drift by category

RP-KL per category (lower = closer to the original). Bold = best internal value.

| | code | general | math | multilingual | tools | long |
|---|---:|---:|---:|---:|---:|---:|
| V0 | 0.081 | 0.047 | 0.175 | 0.082 | **0.305** | 0.095 |
| V3 GDN FP8 | 0.074 | **0.039** | 0.102 | **0.068** | 0.303 | **0.036** |
| V13 calibrated MLP | 0.093 | 0.045 | **0.084** | 0.078 | 0.307 | 0.097 |
| V1 all Q4_K | 0.101 | 0.040 | 0.087 | 0.074 | 0.343 | 0.042 |
| V4 GDN Q4_K | 0.086 | 0.050 | 0.129 | 0.078 | 0.396 | 0.046 |
| V5 attention Q4_K | 0.088 | 0.049 | 0.154 | 0.078 | 0.331 | 0.061 |
| V6 MLP Q4_K | **0.065** | 0.042 | 0.210 | **0.068** | **0.299** | 0.097 |
| V7 early MLP Q4_K | 0.079 | 0.043 | 0.112 | 0.071 | 0.347 | 0.092 |
| V9 GDN + MLP Q4_K | 0.089 | 0.048 | 0.125 | 0.072 | 0.330 | 0.042 |
| R1 (external) | 0.081 | 0.029 | 0.041 | 0.050 | 0.255 | 0.031 |
| R2 (external) | 0.043 | 0.014 | 0.032 | 0.036 | 0.208 | 0.013 |

- **Math is fixed by the MLP, long context by the recurrent path.** Every map touching GDN (V1, V3,
  V4, V9) roughly halves long-context drift; MLP-only maps (V6, V13) do not.
- **Tool calling is hardest for everyone**, both external references included. No seed moves it
  much: an open target.
- **No map wins every column.** A search aware of per-category sensitivity can combine them.

## Epoch hpc01-e3 update (scoring fixes)

A Gittensor review found three scoring flaws. The fixes changed the rules, so the epoch moved to
`hpc01-e3`; drift, speed and memory numbers above are unchanged and were not re-measured.

| Fix | Effect on these seeds |
|---|---|
| **FG-2 counts only gains beyond noise** (each result handicapped by the ε floors, and it must beat every other result beyond noise somewhere) | gains fall to V13 0.30%, V0 0.19%, V4 0.10%, V1 0.04%, V6 0.02%; a copy of V0 that differs by 0.01 tok/s now scores 0 instead of earning a tier |
| **Task guard: all 784 questions, compared with V0 question by question** (exact McNemar; no suite may keep under half of V0's passes) | **V5 now fails the guard** (lost 33, gained 19, p = 0.035) and leaves the frontier; every other seed passes |
| **A paid tier needs a private holdout PASS** | results measured without one are `bt:provisional` and earn nothing |

Task scores on the full set (V0 570/784): V1 575, V3 573, V13 566, V4 566, V6 563, V9 562, V7 561,
V5 556. The old guard used 196 questions, with suites as small as five, and allowed six lost answers
per suite.

**Fidelity is not accuracy.** V13's headline is 11.5% less drift from the original model. On tasks it
scores 566 against V0's 570, a 4-question difference that the paired guard does not call a
regression. Neither number shows V13 is *better* at tasks.

## Decision questions

| | Question | Answer | Evidence |
|---|---|---|---|
| A | Is the baseline reproducible? | **YES** | two scoring runs are byte-identical (max log-prob difference 0.0 on `short-code` and `long-8k`); V0 rebuilds the shipped checkpoint's 2,387 tensors byte for byte |
| B | Do maps change quality meaningfully? | **YES** | V13 and V3 differ from V0 with paired 95% intervals that exclude 0 |
| C | Do maps change real speed or memory? | **YES** | all eight seeds move prefill, decode or peak memory beyond ε |
| D | Does a mixed map beat the baseline frontier? | **YES** | all eight seeds are valid and not dominated by V0; V1, V4, V5, V6 and V13 join V0 on the frontier |
| E | Are GDN / attention / MLP sensitivities different? | **NOT RESOLVED** | the whole-family Q4_K moves (V4 +0.0066, V5 −0.0001, V6 −0.0011) all have intervals that include 0; per-category they clearly differ (see above), but the pre-registered aggregate test does not separate them |
| F | Do interactions matter? | **YES** | ΔRP-KL(V9) − ΔRP-KL(V4) − ΔRP-KL(V6) = −0.0135 [−0.0264, −0.0023] |
| G | Is automated search justified? | **YES (PASS)** | D, B and C are yes; see below |

### Why PASS and not STRONG PASS

The pre-registered rule needs a valid mixed map that improves one objective by ≥ 5% at
statistically equivalent KL, or KL by ≥ 10% at ≤ 2% speed and memory cost.

- V13 improves RP-KL by 11.5% at 0.5% decode cost and no memory cost, but prefill is 4.4% slower.
  Prefill is a speed objective, so the 2% bound is not met.
- V1, V6 and V7 save 0.6–1.8 GiB (3–8%) at statistically equivalent KL, but that is under 5% for V6
  and V7, and V1 pays 49% prefill.

Result: PASS. The margins are real, but smaller than e1 suggested.

## What changed from e1

| | e1 | e2 |
|---|---|---|
| Quality metric | top-64 KL estimate from SparkInfer's score tool | Reference-Partition KL, top-256 + tail, exact candidate log-probs from the pinned runtime |
| Decode tokens | 128 | 512 |
| Speed runs | 2 repetitions in one process | 2 independent processes |
| Shipped checkpoint | R0, ranked | V0 rebuilt by the pipeline, the incumbent |
| Headline | V13 −12.1% KL at −1.9% prefill → STRONG PASS | V13 −11.5% RP-KL at −4.4% prefill → PASS |

Every e1 finding kept its direction. V13's prefill cost grew once decode runs were longer and
processes independent, which moved the verdict one step.

## Open questions for miners

- **Where does V13's prefill cost come from?** The unsloth bytes use the compressed-tensors layout.
  An in-repo calibrated quantizer writing the ModelOpt layout would show whether the cost is layout
  or values. If layout, V13's quality comes for free.
- **Can GDN FP8 be cheaper?** V3 buys the biggest long-context gain at 13% decode. Partial placements
  (depth halves, `qkv` only) measured in e1 were cheaper but weaker; a finer search may do better.
- **Combine the category winners.** Calibrated MLP bytes fix math, GDN changes fix long context, on
  different tensors. Interactions are real, so measure the combination.
- **Tool calling.** The worst category for every checkpoint, internal and external.
- **Close the gap to the references.** R1 and R2 prove the model can stay much closer to BF16 at
  this size; how much survives on SparkInfer's fast kernels?
