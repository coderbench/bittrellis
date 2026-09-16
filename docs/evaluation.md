# Evaluation

Every checkpoint goes through the same four steps, in the same order, on the same pinned RTX 5090.

```text
  manifest ──build──▶ checkpoint ──audit──▶ ✓ ──score──▶ quality.json
                                              ├─bench──▶ performance.json
                                              └─tasks──▶ tasks.json
                                                             │
                                          gates ◀────────────┘
                                            │
                                            ▼
                                  frontier · Frontier Gain
```

## 1. Audit (before any GPU time)

`bittrellis audit` fails a checkpoint that is anything other than a precision transformation of the
pinned weights. See [blueprint_review.md §9](blueprint_review.md#9-anti-gaming-needed-teeth)
for the list of checks. A failed audit is never scored.

## 2. Quality: divergence from BF16

**Metric.** Mean KL(P<sub>BF16</sub> ‖ Q<sub>candidate</sub>) in nats per token, over every scored position.

**How it is measured**
- **Candidate:** SparkInfer's `qwen3_gguf_score` teacher-forces the fixed corpus through the decode path, with `SPARKINFER_DETERMINISTIC=1`. Long streams are prefilled up to their tail with batched prefill, then scored token by token.
- **Reference:** transformers BF16, computed once and hash-pinned (`bittrellis reference`).
- **Estimator:** coarse-grained KL over the reference top-64 tokens plus one tail bucket. It can only under-estimate the true KL, never inflate it.

| Field in `quality.json` | Meaning |
|---|---|
| `kl`, `kl_ci95` | mean KL, 95% block-bootstrap interval (blocks of 128 positions) |
| `kl_p99` | 99th percentile, the tail |
| `top1` | argmax agreement with BF16 |
| `nll_delta` | candidate NLL − BF16 NLL on the true next token |
| `by_category` | general · math · code · tools · multilingual · long |
| `needles`, `needle_recall` | vault codes at 10/50/90% depth of 8K/16K/32K contexts (a needle counts only if BF16 retrieves it) |

**Comparing two checkpoints.** A few positions in any text are chaotic: BF16 itself is
uncertain there, and every 4-bit map disagrees with it differently. Those positions are shared
by every candidate, so differences are computed **paired**: per-position ΔKL against the same
positions of the other artifact (`kl_positions.npz`), with a paired block bootstrap
(`bittrellis.eval.logits.paired_delta`). A change is significant when its interval excludes 0.

## 3. Performance

`qwen3_gguf_bench … sweep` in one process: contexts 128 / 4,096 / 16,384, **2 repetitions**,
128 decode tokens, real prompt text (the 32K corpus stream), batch 1.

| Field | Meaning |
|---|---|
| `decode_tps` | decode tok/s at 4,096 context: **official objective** |
| `prefill_tps` | prefill tok/s at 4,096 context: reported |
| `contexts` | full sweep |
| `vram_gib` | **peak** device memory during the sweep (weights, 16K KV, prefill arena, CUDA context), minus idle usage, polled every 250 ms: **official objective** |
| `vram_gib_after_load` | the bench's own reading right after load (reference only) |

Benchmarks refuse to start on a card with more than 1 GiB already in use.

## 4. Task guard

This stage reuses SparkInfer's `bench/quality` benchmark tier: its data, prompt builders and
scorers for IFEval, GSM8K, MMLU-Pro, HumanEval and function calling (196 items, greedy, thinking
off). The checkpoint is loaded once in `sparkinfer_server`.

Two deliberate differences from `run_quality.py`, which predates Qwen3.8:

- **Endpoint.** Requests go to `/v1/chat/completions`. `/v1/completions` returns the
  end-of-turn token inside the text (`F<|im_end|>`), and the MMLU-Pro scorer then reads the
  "D" of "END".
- **Token caps.** Raised (GSM8K 1024, IFEval/HumanEval 768), because Qwen3.8 reasons step by
  step and the original caps cut the final answer off.

A candidate fails the guard if any suite drops more than 6 passed items below R0.

## Gates → frontier

A result enters the frontier only if it passes **all** gates in `configs/hpc01.yaml`:

- audit;
- KL ≤ 0.30;
- top-1 ≥ 0.80;
- needle recall = 1.0;
- task guard.

Then see [frontier.md](frontier.md).

## Reproducing a published number

```bash
bittrellis evaluate <checkpoint> --out artifacts/<name>       # candidate
bittrellis evaluate <checkpoint> --reference-id R0 --out ...  # reference point
bittrellis evaluate-llamacpp --out artifacts/R2                # llama.cpp reference
bittrellis frontier artifacts/*                                # gates, frontier, FG-2
```

Raw score dumps are kept in `scores/*.npz`, so quality can be recomputed without a GPU.
