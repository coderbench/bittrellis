# Architecture

> Manifest → PR label: code path, trust boundaries, persistent data.

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
   evaluator/pr_bot.py ── screen (guards.py) · sandbox (sandbox.py) ── PR comment + eval:* tier ── accepted/ (frontier grows)
```

## Trust boundaries

| Component | Trusted? | Why |
|---|---|---|
| Manifest YAML from a PR | no | parsed only; names registered quantizers, never bytes |
| Quantizer code from a PR | no | after maintainer `eval-approved`, runs only in the `bt-sandbox` account (build, probe, regenerate); trusted `main` code judges its output ([guards](guards.md#isolation-contributed-code-produces-trusted-code-judges)) |
| Source checkpoints on disk | no, verified | each file hash-checked against the lock first |
| SparkInfer | pinned | commit checked, clean checkout required, env sanitized |
| Evaluator paths | maintainers | never auto-evaluated; a change that alters scores starts a new evaluator epoch |

## Persistent data

| Path | Committed | Content |
|---|---|---|
| `data/corpus/hpc01-public-v2.json` | yes | public token streams, hash-verified |
| `data/reference/hpc01-public-v2-k256/` | no (regenerable, hash-recorded) | BF16 top-256 per position |
| `results/feasibility/artifacts/` | yes | internal-frontier seed candidates |
| `<eval root>/accepted/` | evaluator host | merged, frontier-moving miner artifacts |
| `<eval root>/ledger/` | pushed to its own repository each pass | public score records: one write-once record per evaluated PR head, the first-seen records, the frontier and the accepted artifacts ([ledger.py](../evaluator/ledger.py)) |
| private holdout directory | never | validator-only text, reference, results |
| `~/.cache/bittrellis/replay/` | host cache | regenerated tensor hashes by replay key |
