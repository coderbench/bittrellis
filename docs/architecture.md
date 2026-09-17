# Architecture

```text
                          configs/hpc01.yaml  ·  configs/sources.lock.json
                                          │  (every pin)
     manifest.yaml                        ▼
          │               ┌───────────────────────────────┐
          ├──────────────▶│ manifest.py   expand → FORMAT@quantizer@vN per unit, candidate id
          │               │ model/qwen38  273 units from config + loader contract
          │               │ precision.py  legal formats, loader resolver, execution semantics
          │               └──────────────┬────────────────┘
          │                              ▼
          │               ┌───────────────────────────────┐
          │               │ lineage.py    hash-verify sources
          │               │ quantizers/   runtime · attested (baseline, unsloth) · regenerable (rtn, yours)
          │               │ build.py      deterministic checkpoint writer (CPU)
          │               └──────────────┬────────────────┘
          │                              ▼
          │               ┌───────────────────────────────┐
          │               │ validate.py   audit: config, tensor set, frozen bytes, execution map,
          │               │               lineage (replay + cache), anomaly
          │               └──────────────┬────────────────┘
          │                              ▼
          │  pinned SparkInfer ┌─────────────────────────────────────────────────────┐
          │  (unmodified lib)  │ tools/sparkinfer_refscore.cpp  fixed-partition log-probs │
          │                    │ runtime.py   refscore · bench runs · server · peak GPU/RAM │
          │                    └──────────────┬──────────────────────────────────────┘
          │                                   ▼
          │               ┌───────────────────────────────┐
          │               │ eval/corpus      public streams (+ assemble_streams for holdouts)
          │               │ eval/reference   BF16 top-256 per position (transformers, once)
          │               │ eval/logits      RP-KL, needles, paired bootstrap
          │               │ eval/tasks       SparkInfer quality suite via server
          │               │ eval/candidate   correctness, 2 perf runs, artifact directory
          │               │ holdout.py       private PASS/FAIL
          │               │ eval/llamacpp    external reference R2
          │               └──────────────┬────────────────┘
          │                              ▼
          │               ┌───────────────────────────────┐
          │               │ frontier/pareto  gates, ε-dominance, FG-2 (internal only)
          │               │ frontier/report  tables, frontier.json, plots, compare
          │               └──────────────┬────────────────┘
          ▼                              ▼
   evaluator/pr_bot.py ── PR comment + label ── accepted/ (frontier grows)
```

## Trust boundaries

| Component | Trusted? | Why |
|---|---|---|
| Manifest YAML from a PR | no | parsed only; it can name registered quantizers but never carry bytes |
| Quantizer code from a PR | only after review | executed on the evaluator host once a maintainer labels `eval-approved` |
| Source checkpoints on disk | no, verified | every file hashed against the lock before use |
| SparkInfer | pinned | commit checked, checkout must be clean, env sanitized |
| Evaluator paths | maintainers | changes bump the evaluator epoch |

## Data that persists

| Path | Committed | Content |
|---|---|---|
| `data/corpus/hpc01-public-v2.json` | yes | public token streams, hash-verified |
| `data/reference/hpc01-public-v2-k256/` | no (regenerable, hash-recorded) | BF16 top-256 per position |
| `results/feasibility/artifacts/` | yes | seed candidates of the internal frontier |
| `<eval root>/accepted/` | evaluator host | merged, frontier-moving miner artifacts |
| private holdout directory | never | validator-only text, reference, results |
| `~/.cache/bittrellis/replay/` | host cache | regenerated tensor hashes by replay key |
