# BitTrellis specification v2.1 (authoritative)

> **Every layer doesn't deserve the same bits. And the same bits don't deserve the same quantizer.**

BitTrellis is a hardware-aware LLM artifact optimizer. It searches legal execution formats,
quantizers and module mappings for a fixed model, runtime and GPU, and validates every resulting
artifact against canonical BF16 behavior and the real deployment runtime.

This version supersedes the original blueprint and specification v2. v2.1 keeps v2's rules and
fixes six points that were not implementable as written. [§14](#14-changes-from-v2) lists them.

---

## 1. Track HPC-01

| | |
|---|---|
| Model | Qwen3.8-27B, frozen architecture, tokenizer and BF16 weights |
| Runtime | SparkInfer `b1ed168` (v0.5.8), kernels unmodified, every `SPARKINFER_*` knob pinned |
| GPU | 1× RTX 5090 32 GB |
| Search dimensions | execution format · quantizer/encoder · module mapping |
| Objectives | RP-KL ↓ · decode tok/s ↑ · 4K prefill tok/s ↑ · peak GPU memory ↓ |
| Hard guards | audit · runtime correctness · long-context needles · task guard · private holdout |

Out of scope for this track: pruning, distillation, fine-tuning or QAT, adapters, architecture or
tokenizer changes, kernel changes, other models, other GPUs and other runtimes.

Every pin lives in [`configs/hpc01.yaml`](../configs/hpc01.yaml) and
[`configs/sources.lock.json`](../configs/sources.lock.json). A change to evaluation, frontier or
gates requires a new **evaluator epoch** (currently `hpc01-e2`).

## 2. Runtime truth

A manifest names what SparkInfer **executes**. The loader rules and their code references are in
[precision_space.md](precision_space.md).

| Unit kind | Units | Legal formats | Notes |
|---|---:|---|---|
| `L{i}.gdn.qkv/z/out` | 144 | NVFP4 · FP8 · Q4_K | FP8 needs one BF16 scale per output row |
| `L{i}.attn.q/k/v/o` | 64 | NVFP4 · Q4_K | only layers 3, 7, …, 63 have attention |
| `L{i}.mlp` | 64 | NVFP4 · Q4_K | gate/up/down move together |
| `lm_head` | 1 | NVFP4 · Q4_K | **conditional execution**, see below |

`bittrellis inventory` derives the 273 units from the config and the loader contract. The count is
asserted in tests, not hard-coded.

**Q4_K is a runtime fit.** Its quantizer is `runtime`, and the stored tensor is the frozen BF16
weight. BitTrellis offers no Q4_K encoder choice and no source-weight preparation.

**lm_head has dual semantics.** Whatever the head stores, batch-1 decode executes a Q4_K fit of
those bytes. A stored NVFP4 head is also kept as an NVFP4 operand for wide packed decode, but only
if at least 3 GiB of VRAM is free at load. The expanded manifest records both paths:

```yaml
lm_head: {source_format: NVFP4, quantizer: baseline@v1,
          execution: {decode_b1: Q4_K(nvfp4), packed_wide: "NVFP4 if free VRAM ... else Q4_K(nvfp4)"}}
```

A runtime Q4_K fit may come from the unit's legal stored representation. For every kind except the
head that is BF16; for the head it is BF16 or NVFP4. This is what makes V0, the shipped checkpoint,
a legal manifest (v2 §6 contradicted this).

## 3. Quantizers and lineage

Each unit is assigned `format @ quantizer@version [+params]`. Quantizers implement the contract in
[`bittrellis/quantizers/base.py`](../bittrellis/quantizers/base.py); see
[quantizer_contract.md](quantizer_contract.md).

| Lineage | How legitimacy is established | Built-in |
|---|---|---|
| **runtime** | stored BF16 is byte-identical to the base | `runtime` (Q4_K) |
| **regenerable** | the evaluator rebuilds sampled units and compares bytes; `sequential` quantizers are replayed in pipeline order from the first unit | `rtn` (NVFP4, FP8) |
| **attested** | bytes are identical to a maintainer-approved source whose every file matches the hash lock | `baseline` (shipped NVFP4), `unsloth` (calibrated NVFP4, MLP 0–55) |

- Miners cannot add attested sources; only a maintainer change to `sources.lock.json` can.
- A quantizer may produce only its own Linear's tensors. Moving scales into norms or neighbouring
  layers (SmoothQuant or AWQ-style migration) is illegal. Scale, clipping, rounding and act-order
  search inside the tensor is legal.
- Reconstruction error relative to round-to-nearest is a diagnostic. It warns above 2× and rejects
  above 5× as substituted bytes, because legitimate calibrated encoders reach 1.6×.
- Sequential replay is expensive, so regenerated units are cached by replay key and shared across
  candidates. The key covers the quantizer version, the base revision and the unit's pipeline
  prefix: every assignment to that quantizer up to and including the unit, because a GPTQ-style
  tensor depends on everything quantized before it. Independent quantizers key on the unit's own
  assignment alone.

## 4. Fidelity: Reference-Partition KL

- **Reference.** The canonical BF16 model runs once through transformers with CPU offload. Per
  scored position it stores the top-256 token ids and log-probabilities. The files are hash-pinned
  per corpus and evaluator epoch.
- **Fixed partition.** The BF16 top-256 plus one tail bucket, chosen before any candidate exists.
  The partition is never candidate-dependent.
- **Candidate scoring.** [`tools/sparkinfer_refscore.cpp`](../tools/sparkinfer_refscore.cpp) links the
  *unmodified* pinned SparkInfer runtime library and drives the same load, `cache_prefix` and
  `forward_token` path as SparkInfer's own score tool. For each position it returns exact log-probs
  of the 256 reference ids. SparkInfer's shipped tools cannot do this: `qwen3_gguf_score` prints
  only its own top-k, and `/v1/score` caps at 20. This addition is allowed because the runtime
  library, kernels and loader are untouched.
- **Metric.** `RP-KL = Σ p_i log(p_i/q_i) + p_tail log(p_tail/q_tail)`. This is the KL between the
  two projections onto the same partition, so it never exceeds the full-vocabulary KL. It is not
  called "KL". A candidate's probability mass outside the reference ids is reported as a
  diagnostic.
- **Uncertainty.** Scoring is deterministic (`SPARKINFER_DETERMINISTIC=1`), so uncertainty is corpus
  uncertainty: a paired block bootstrap over contiguous 128-position blocks.

## 5. Corpora

| Split | Who sees it | Use |
|---|---|---|
| `public` ([`data/corpus/hpc01-public-v2.json`](../data/corpus/)) | everyone | development fidelity set; official RP-KL |
| `public-validation` | everyone (rebuildable) | local overfitting check; **not** a holdout |
| private holdout | validators only, rotated per epoch | PASS/FAIL gate, [holdout.md](holdout.md) |

The public corpus is 77,824 tokens with 21,624 scored positions:

- 5 × 4K streams: general, math, code, tools, multilingual;
- 8K, 16K and 32K streams, each with three needles and a 384-position scored tail.

Documents begin with `<|endoftext|>`. Adding a category (for example instruction following) changes
the corpus and the reference, so it requires a new epoch.

## 6. Performance and memory

- **Repetitions.** Exactly **two** independent processes, each with one model load and a
  128/4K/16K sweep with **512 decode tokens**. Nothing repeats inside a run
  (`SPARKINFER_BENCH_SWEEP_REPS=1`).
- **Objectives.** Decode and prefill at 4K are the means of the two runs. Their relative spreads
  are recorded.
- **Memory objective.** Peak device memory over the whole run (weights, KV at `max_seq` 16,384,
  prefill arena, CUDA context), minus idle usage. It is sampled every 250 ms, so the reported value
  is a lower bound on the true peak.
- **Reported, not ranked.** Resident memory after load, peak host RSS and checkpoint bytes.
- **Workload choice.** The memory workload is the 16K sweep, not SparkInfer's 262K deployment
  default. At 262K, context allocation dominates and throughput changes with it. A long-context
  memory track would be a new epoch.

## 7. Frontier

- **Internal rows only.** Legal manifests on the pinned runtime. The incumbent is **V0** (the
  shipped checkpoint, rebuilt byte for byte). The internal frontier is **seeded** with every
  maintainer-measured feasibility candidate, so miners compete against V1–V13 as well, and
  resubmitting one earns nothing.
- **External references.** R1 (unsloth, whose FP8 attention bytes run as a Q4_K refit and so is not
  a legal manifest), R2 (llama.cpp UD-Q4_K_M) and R3 (NVIDIA, a compatibility boundary) are shown
  beside the frontier and never enter dominance or FG-2.
- **ε-dominance.** A is *materially better* on:
  - RP-KL if it is lower by more than 0.002 and the paired 95% interval excludes 0;
  - decode or prefill if it is higher by more than max(floor, either row's two-run spread), with
    floors of 1% and 3%;
  - peak memory if it is lower by more than 0.1 GiB.

  A dominates B if it is materially better somewhere and materially worse nowhere.
- **FG-2.** The increase in normalized 4-D dominated hypervolume a valid internal row adds. It is
  versioned. There are no manual size tiers.

## 8. Gates

A candidate enters the frontier only if all of these hold:

1. the audit passes (sources, config, tensor set, frozen bytes, execution map, lineage, anomaly);
2. runtime correctness holds:
   - every scoring and benchmark process loads the checkpoint and exits 0;
   - no NaN or Inf log-probs;
   - argmax ids are inside the vocabulary;
   - the executed format map matches the manifest;
   - no unpinned `SPARKINFER_*` variable reaches the runtime;
3. RP-KL ≤ 0.30 and top-1 ≥ 0.80;
4. the long-context guard holds: every needle BF16 retrieves at 8K, 16K and 32K is retrieved
   (required success 1.0 at each length). Long-context RP-KL is reported but not gated;
5. the task guard holds: no SparkInfer quality suite drops more than 6 passed items below V0;
6. the private holdout returns PASS, under the rule in [holdout.md](holdout.md).

## 9. Holdout

- **Text.** Private text, the same stream structure, and a validator-held seed and epoch.
- **Output.** Only `holdout.json` = `{epoch, result}` is published.
- **PASS rule.** PASS requires all of:
  - the audit is valid;
  - holdout RP-KL ≤ 0.30;
  - every retrievable holdout needle is kept;
  - if the public RP-KL gain over V0 is significant, the holdout gain over V0 is at least 50% of it.
- **Probing defense.** PASS/FAIL only, rotation per epoch, no item-level output, and evaluator rate
  limits.
- **Attested sources.** Their calibration data is unknown, so non-overlap with the holdout cannot be
  proven. This provenance risk is accepted, and the holdout is the protection against it.

## 10. Evaluation flow

```text
manifest ─▶ validate + duplicate check ─▶ build (sources hash-verified) ─▶ audit ──fail──▶ reject
                                                                           │
   public RP-KL + correctness + needles ◀────────────────────────────────┘
        │
   task guard ─▶ 2 performance runs (decode, 4K prefill, peak GPU, host RAM)
        │
   private holdout (PASS/FAIL) ─▶ gates ─▶ ε-frontier vs seeds + accepted ─▶ FG-2 ─▶ PR comment + label
```

[`evaluator/pr_bot.py`](../evaluator/pr_bot.py) runs this flow for pull requests:

- **Manifest-only PRs** are evaluated with trusted `main` code.
- **PRs that execute contributed code** (quantizers, search) wait for a maintainer's `eval-approved`
  label.
- **PRs that touch evaluator paths** are never evaluated automatically.

## 11. Artifacts

```text
artifacts/<name>/
  candidate.json     manifest, expanded assignments (source format, quantizer@version, execution), build record
  audit.json         audit result, lineage per quantizer, anomaly ratios
  quality.json       RP-KL, CI, per category, outside mass, needles by length
  correctness.json   runtime-correctness checks and the effective scoring environment
  tasks.json         task guard
  performance.json   both runs, spreads, peak GPU, resident, host RAM, protocol
  holdout.json       {epoch, result}
  environment.json   GPU, driver, CUDA, kernel, SparkInfer commit, evaluator epoch, effective env
  kl_positions.npz   per-position RP-KL for paired comparisons
```

Candidates are cached by candidate id plus evaluator epoch. The id already includes every unit's
quantizer name and version, so a quantizer code change must bump its version.

## 12. Realistic results

Measured on this runtime:

- the shipped map already uses the fastest kernels;
- Q4_K and FP8 cost prefill;
- no map beats V0's decode beyond noise.

Credible headlines are therefore:

- fidelity at a stated, small cost ("RP-KL −11.5% at unchanged decode and memory, −4.4% prefill");
- memory savings with the prefill cost stated alongside;
- long-context fidelity at the same execution format.

Claims of +x% decode or prefill need a measured, beyond-ε result.

## 13. What does not count

- metadata relabeling, or stored-format changes with the same execution;
- Q4_K "quantizers";
- edits to frozen source weights;
- environment-variable tricks;
- overfitting the public corpus;
- architecture changes;
- fine-tuning disguised as quantization;
- disk-size savings with no runtime effect.

## 14. Changes from v2

1. **Fixed-partition scoring needs a new tool.** The pinned binaries cannot emit log-probs for
   chosen ids, so a scorer linked to the unmodified runtime library is part of the evaluator (§4).
2. **The V0 legality contradiction is resolved.** A runtime Q4_K fit may come from the unit's legal
   stored representation (§2).
3. **ε from two runs has floors** (§7). A two-sample spread can be near zero by chance.
4. **The holdout PASS rule is defined** (§9). The trade-off is explicit: ranking uses public RP-KL,
   and the holdout guards transfer.
5. **Sequential replay is cached by pipeline prefix, and the candidate cache key includes quantizer
   versions** (§3, §11). Sharing by quantizer configuration alone would be wrong for sequential
   encoders.
6. **Benchmarks are split in reporting, not in processes.** Decode, prefill and memory come from the
   same two runs (§6). Separate commands would multiply model loads.

Decisions taken for this epoch: peak memory uses the 16K sweep workload (§6), and the internal
frontier is seeded with the measured feasibility candidates (§7).
