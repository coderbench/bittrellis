# Submitted precision maps

> One file per candidate: `manifests/<name>.yaml`. One manifest per pull request.

Format: [precision_manifest.md](../docs/precision_manifest.md) · scoring: [miner_guide.md](../docs/miner_guide.md) · measured starting points: [`experiments/feasibility/variants/`](../experiments/feasibility/variants/).

This directory is intentionally empty of examples: anything committed here is compared against your
submission by the duplicate screen. Copy the skeleton below into a new file instead.

## A first candidate

```yaml
schema: bittrellis/manifest@2
track: HPC-01
name: my-first-recipe                   # lowercase, hyphenated; becomes the artifact name
description: >
  One sentence on what should move and why. This is your hypothesis, and the
  evaluator quotes it back in the score comment.
authors: [your-github-handle]
default: NVFP4                          # every one of the 273 units starts here
rules:                                  # applied top to bottom; later rules win
  - match: "L*.gdn.*"                   # glob over unit ids
    layers: "32-63"                     # optional: "a-b,c,d-e"
    format: FP8                         # FP8 executes on the recurrent path only
  - match: "L*.mlp"
    layers: "1-55"                      # layer 0 left alone: it picks the prefill path
    format: NVFP4
    quantizer: unsloth                  # calibrated bytes, attested
```

Check it before you open the PR — both of these run on a laptop, with no GPU:

```bash
bittrellis manifest manifests/my-first-recipe.yaml --expand      # what actually executes
bittrellis manifest manifests/my-first-recipe.yaml --against manifests/    # not a duplicate
```

## Things that cost you a tier

- **Layer 0 of each kind selects the batched-prefill path.** Changing it can cost 20-50% prefill.
- **Q4_K saves memory and costs prefill** — up to 49%. It has no encoder choice; it is a runtime fit.
- **Effects interact.** Two changes that each look free can be dominated together.
- **A public fidelity gain is not a gain.** It has to survive the private holdout; so far none has.

Read [`results/feasibility/`](../results/feasibility/feasibility_report.md) before spending GPU time.
