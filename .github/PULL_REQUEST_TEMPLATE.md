<!-- One PR = one candidate (a manifest), optionally with the quantizer it introduces. -->

## Candidate

| | |
|---|---|
| manifest | `manifests/<name>.yaml` |
| candidate id | output of `bittrellis manifest manifests/<name>.yaml` (`id=…`) |
| summary | e.g. `gdn FP8×12 NVFP4×132 · mlp NVFP4@unsloth×56 NVFP4×8 · attn NVFP4×64 · lm_head NVFP4` |
| new quantizer? | no / `bittrellis/quantizers/<module>.py` (`name@vN`, lineage, replay mode) |

## Hypothesis

<!-- What should move (RP-KL, decode, 4K prefill, peak GPU memory), and why. -->

## Evidence (optional)

<!-- Never scored; the evaluator re-measures everything. If you ran it on an RTX 5090, paste:
     bittrellis compare results/feasibility/artifacts/V0-baseline-rebuild artifacts/<name> -->

## Checklist

- [ ] `bittrellis manifest manifests/<name>.yaml --against manifests/` passes (no duplicate)
- [ ] `ruff check .` and `pytest -q` pass
- [ ] no changes to protected evaluator paths (see CONTRIBUTING.md)
- [ ] quantizer PRs: deterministic, versioned, with a test; understood that evaluation waits for `eval-approved`
