# Evaluation

Every candidate goes through the same steps on the same pinned RTX 5090. The rules are in the
[specification](specification.md); this page is the practical view.

```text
manifest ─▶ validate + duplicates ─▶ build ─▶ audit ─▶ public RP-KL ─▶ task guard ─▶ 2 perf runs
                                     (hash-verified)   + correctness                     │
                                                      + needles                          ▼
                                    PR comment ◀─ FG-2 ◀─ ε-frontier ◀─ gates ◀─ private holdout
```

## 1. Build and audit (CPU)

`bittrellis build` verifies every source against `configs/sources.lock.json`, then writes the
checkpoint deterministically. `bittrellis audit` rejects anything that is not a legal
precision/quantizer transformation of the pinned weights; see [audit.md](audit.md).

## 2. Public fidelity: Reference-Partition KL

- **Reference.** BF16 via transformers, computed once per corpus and epoch; top-256 ids and
  log-probs per position, hash-pinned.
- **Candidate.** [`tools/sparkinfer_refscore.cpp`](../tools/sparkinfer_refscore.cpp) links the
  unmodified pinned runtime and teacher-forces the corpus through the decode path with
  `SPARKINFER_DETERMINISTIC=1`. It returns the candidate's log-probs of exactly the reference's
  256 ids per position. Long streams are prefilled to their tail with batched prefill first.
- **Metric.** RP-KL, the KL between both distributions projected onto {BF16 top-256, tail}. It is
  identical in partition for every candidate and never larger than full-vocabulary KL.

| `quality.json` field | Meaning |
|---|---|
| `rp_kl`, `rp_kl_ci95`, `rp_kl_p99` | mean, block-bootstrap 95% interval, tail |
| `top1` | argmax agreement with BF16 |
| `nll_delta` | candidate NLL − BF16 NLL on the true next token |
| `outside_mass_mean` | candidate probability outside the BF16 top-256 (diagnostic) |
| `by_category` | general · math · code · tools · multilingual · long |
| `needles_by_length` | long-context guard: needles retrieved / required at 8K, 16K, 32K |

**Comparing two artifacts** always uses per-position paired differences (`kl_positions.npz`), so
position-level noise shared by every checkpoint cancels:

```bash
bittrellis compare artifacts/V0-baseline-rebuild artifacts/mine
```

`correctness.json` records the runtime-correctness checks and the effective scoring environment.

## 3. Task guard

SparkInfer's `bench/quality` benchmark tier: its data, prompts and scorers for IFEval, GSM8K,
MMLU-Pro, HumanEval and function calling, 196 items, greedy, thinking off. It runs through
`sparkinfer_server`'s chat endpoint with Qwen3.8-sized token caps. A suite may not drop more than 6
passed items below V0.

## 4. Performance and memory: exactly two runs

Each run is a fresh process: one model load, then a 128 / 4,096 / 16,384 context sweep with 512
decode tokens. Nothing repeats inside a run.

| `performance.json` field | Meaning |
|---|---|
| `decode_tps`, `decode_spread` | mean of the two runs at 4K, relative spread between them |
| `prefill_tps`, `prefill_spread` | same, for 4K prefill |
| `peak_gpu_gib` | **official memory objective**: peak device memory during the run, minus idle usage |
| `resident_after_load_gib` | the bench's reading right after load (reported) |
| `peak_host_gib` | peak host RSS of the benchmark process (reported) |
| `runs` | both runs, full sweeps |

Benchmarks refuse to start on a card that already has more than 1 GiB in use.

## 5. Private holdout

Validators only; PASS or FAIL. See [holdout.md](holdout.md).

## Gates → frontier

All must hold:

- audit;
- runtime correctness;
- RP-KL ≤ 0.30 and top-1 ≥ 0.80;
- long-context guard (every BF16-retrievable needle at 8K/16K/32K);
- task guard;
- holdout PASS.

Then see [frontier.md](frontier.md).

## Commands

```bash
bittrellis evaluate <checkpoint> --out artifacts/<name>                   # everything
bittrellis evaluate-public <checkpoint> --out artifacts/<name>            # audit + fidelity only
bittrellis benchmark <checkpoint> --out artifacts/<name>                  # audit + performance only
bittrellis evaluate <dir> --external R1 --out artifacts/R1                # external reference
bittrellis evaluate-llamacpp --out artifacts/R2                           # llama.cpp reference
bittrellis frontier --with-seeds artifacts/<name>                         # rank against the seeds
```
