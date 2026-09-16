<!-- One PR = one thing: a new precision map, a new quantizer, a search method, or a fix. -->

## What this PR adds

- [ ] a precision manifest in `manifests/`
- [ ] a quantizer / builder change
- [ ] search or analysis tooling
- [ ] docs / fix

## Candidate (manifest PRs)

| | value |
|---|---|
| manifest | `manifests/<name>.yaml` |
| candidate id | `bittrellis manifest manifests/<name>.yaml` → `id=…` |
| summary | e.g. `gdn FP8×36 NVFP4×108 · mlp NVFP4×64 · attn NVFP4×64 · lm_head Q4_K` |

## Evidence (optional, the evaluator re-measures everything)

If you ran it on an RTX 5090, paste `bittrellis frontier` output for your artifact next to R0.
Self-reported numbers are never scored.

## Checklist

- [ ] `bittrellis manifest` passes and `pytest` is green
- [ ] no changes to `configs/`, `data/corpus/`, `bittrellis/eval/` or `bittrellis/frontier/` (evaluator paths) unless the PR is explicitly about them
- [ ] the hypothesis is stated in the manifest `description`
