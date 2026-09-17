# Evaluation

> What happens to a candidate between PR and score? Rules: [specification](specification.md); terms: [Key terms](../README.md#key-terms).

```text
screen (no GPU) ─▶ build ─▶ audit ─▶ public RP-KL ──gate fail──▶ stop
                                     + correctness
                                     + needles
                                          │
                      2 perf runs ──dominated──▶ stop: tasks and holdout cannot lift it
                                          │
                      task guard ─▶ private holdout ─▶ gates ─▶ ε-frontier ─▶ FG-2 ─▶ PR comment
```

The CPU screen decides duplicates, near-copies, memory, queue share and new encoders in seconds
([guards.md](guards.md)). On the pinned RTX 5090, quality takes ~4.5 min, two speed runs ~1 min, the
task guard ~4 min: a dominated result costs ~6 GPU minutes, not 15.

## 1. Build and audit (CPU)

`bittrellis build` verifies sources against `configs/sources.lock.json`, then writes the checkpoint
deterministically. `bittrellis audit` rejects all but legal precision/quantizer transformations of the
pinned weights ([audit.md](audit.md)).

## 2. Fidelity: Reference-Partition KL (RP-KL), next-token drift from BF16

- **Reference:** BF16 via transformers, once per corpus and epoch; top-256 ids and log-probs per position, hash-pinned.
- **Candidate:** [`tools/sparkinfer_refscore.cpp`](../tools/sparkinfer_refscore.cpp) links the unmodified pinned runtime, teacher-forces the corpus through the decode path (`SPARKINFER_DETERMINISTIC=1`), returns log-probs of exactly the reference's 256 ids per position. Long streams are batch-prefilled to their tail first.
- **Metric:** KL of both distributions projected onto {BF16 top-256, tail}. Same partition for every candidate; never above full-vocabulary KL.

| `quality.json` | Meaning |
|---|---|
| `rp_kl` · `rp_kl_ci95` · `rp_kl_p99` | mean drift, block-bootstrap 95% interval, tail |
| `top1` | argmax agreement with BF16 |
| `nll_delta` | candidate − BF16 NLL on the true next token |
| `outside_mass_mean` | candidate probability outside BF16's top-256 (diagnostic) |
| `by_category` | general, math, code, tools, multilingual, long |
| `needles_by_length` | needles retrieved / required at 8K, 16K, 32K |

`correctness.json` holds the runtime-correctness checks and the effective scoring env.

Paired per-position comparison (`kl_positions.npz`) cancels shared noise:
`bittrellis compare artifacts/V0-baseline-rebuild artifacts/mine`.

## 3. Task guard

SparkInfer's `bench/quality` benchmark tier (data, prompts, scorers): IFEval, GSM8K, MMLU-Pro,
HumanEval, function calling. 196 items, greedy, thinking off, via `sparkinfer_server`'s chat endpoint
with Qwen3.8-sized token caps. No suite may drop more than 6 passed items below V0.

## 4. Performance and memory: exactly two runs

Each run: fresh process, one load, a 128 / 4,096 / 16,384 context sweep, 512 decode tokens, no
repeats. Benchmarks refuse a card with more than 1 GiB already in use.

| `performance.json` | Meaning |
|---|---|
| `decode_tps` · `decode_spread` | 4K decode, mean of both runs, relative spread |
| `prefill_tps` · `prefill_spread` | the same for 4K prefill |
| `peak_gpu_gib` | **official memory objective**: peak device memory during the run minus idle |
| `runs` | both full sweeps |
| `resident_after_load_gib` · `peak_host_gib` | reported only: memory right after load, peak host RSS |

## 5. Gates and commands

Entering the [frontier](frontier.md) needs: audit, runtime correctness, RP-KL ≤ 0.30,
top-1 ≥ 0.80, long-context guard (every BF16-retrievable needle at 8K/16K/32K), task guard, and a
validator-only holdout PASS ([holdout.md](holdout.md)).

```bash
bittrellis evaluate <checkpoint> --out artifacts/<name>          # everything
bittrellis evaluate-public <checkpoint> --out artifacts/<name>   # audit + fidelity only
bittrellis benchmark <checkpoint> --out artifacts/<name>         # audit + performance only
bittrellis evaluate <dir> --external R1 --out artifacts/R1       # external reference
bittrellis evaluate-llamacpp --out artifacts/R2                  # llama.cpp reference
bittrellis frontier --with-seeds artifacts/<name>                # rank against the seeds
```
