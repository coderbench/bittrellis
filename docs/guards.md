# Guards: screening and copies

Every pull request passes a **screen** before it costs any GPU time. The GPU measurement then runs
in **stages** and stops as soon as the outcome is known. Copies are judged by **what a submission
is**, not how its text is written, and a copy earns exactly what it adds.

Code: [`evaluator/guards.py`](../evaluator/guards.py), [`bittrellis/fingerprint.py`](../bittrellis/fingerprint.py),
[`evaluator/pr_bot.py`](../evaluator/pr_bot.py). Thresholds: `evaluation.screen` in
[`configs/hpc01.yaml`](../configs/hpc01.yaml).

## The whole flow

```text
 observe every open PR head (first-seen record)  →  queue in first-seen order
                                                          │
 SCREEN (CPU, seconds) ───────────────────────────────────┤
   1 queue share       an author's 4th waiting PR waits its turn        bt:queued
   2 manifest          valid, legal, right track                        bt:invalid-manifest
   3 duplicate         same expanded recipe as a seed, an accepted      bt:duplicate
                       result or an earlier PR
   4 near-copy         ≤ 2% of weights differ from an earlier PR by     bt:derivative (measured)
                       another author                                   bt:copy-review (3rd in an epoch)
   5 memory            predicted peak > 31.5 GiB                         bt:memory
   6 new quantizers    deterministic, and not an existing encoder       bt:nondeterministic
                       (probe bytes)                                    bt:same-encoder
                                                          │
 BUILD + AUDIT (CPU, ~3 min) ─────────────────────────────┤
   7 stored bytes      a new encoder's real bytes vs every existing     bt:same-encoder
                       encoder's bytes for the same tensors
                                                          │
 MEASURE (GPU) ───────────────────────────────────────────┤
   quality ~4.5 min → quality gates fail? stop                          bt:gate-fail
   2 speed runs ~1 min → dominated? stop, skip tasks + holdout          bt:dominated
   tasks ~4 min → private holdout → rank                                bt:frontier / bt:dominated / bt:gate-fail
```

Stage times are from the evaluator dry run on the pinned RTX 5090 (PR #1). A dominated result
therefore uses about 6 GPU minutes instead of about 15.

**Why skipping is safe.** Tasks and the holdout are gates: they can fail a result, never lift one.
A result that is already dominated on the four measured objectives stays dominated whatever they
say. If a later re-rank makes it non-dominated (for example, an earlier PR it was ranked against
closes), the bot rebuilds it and runs the skipped stages before labelling it `bt:frontier`.

## Who was first

The evaluator records every PR head the first time it sees it:

```text
<root>/observations/pr-000042-3f9c1a2b7d10.json   {pr, author, head, first_seen, candidate_id, keys, ...}
```

- The record is written once and lives outside every checkout. `first_seen`, `author` and `head`
  never change.
- **A force-push is a new head** with its own time. Opening an empty PR early and pushing copied
  content later does not make it the original.
- Every open head is observed **before** anything is evaluated, and the queue runs oldest first. An
  original is therefore always measured before a later near-copy of it.

## Recipes: judged by the expanded recipe

A manifest is expanded to all 273 units (`FORMAT@quantizer@version[+params]` per unit) before it is
compared. Rewritten rules, reordered rules and different layer ranges that expand to the same
recipe are the same recipe.

| Case | Test | Result |
|---|---|---|
| Duplicate | same candidate id as a seed, an accepted result, or an open or merged PR observed earlier | not measured, `bt:duplicate`, earns nothing |
| Near-copy | ≤ `near_copy_max_share` (2%) of searchable **weights** assigned differently from an earlier open or merged PR by **another** author | measured, `bt:derivative`, ranked with the original on the frontier |
| Repeated near-copies | the author's 3rd near-copy in this epoch | `bt:copy-review`, waits for a maintainer's `copy-cleared` |
| Your own earlier PR | never a copy | iterating is normal |
| A closed, unmerged PR | never a reference | it earns nothing, so its public idea is free to build on |

Distance is weighted by parameter count. Changing one attention projection moves far less than 2%;
changing a block of MLP layers moves far more.

## Credit: only what you add

Every PR is ranked against seeds, accepted results **and earlier measured PRs by other authors that
are still open**. Its FG-2 is the frontier space it adds on top of all of them.

- A pure copy adds nothing and earns 0, even if it slipped past the screen.
- A real improvement on someone's PR earns exactly the improvement.
- When an earlier PR merges or closes, the bot **re-ranks** every later PR that was measured against
  it and updates its label.

This is why nothing is blocked automatically. The search space is small, and the repository ships
a neighbour generator (`bittrellis search neighbors`). Honest miners will often find the same
recipe, and the credit rule already makes copying pointless.

## Quantizers: judged by the bytes they produce

Renaming a class or rewriting its loops changes code, not output. The audit already requires every
regenerable quantizer to be deterministic, so its bytes identify it.

1. **Probe (before build).** Each quantizer the PR adds encodes a tiny synthetic model whose weights
   come from a seed derived from the evaluator's secret. It runs twice, and the two runs must be
   byte-identical. Its output is compared with every quantizer on `main` and with earlier PRs' new
   quantizers. At ≥ `quantizer_same_bytes` (99%) identical bytes it is the same encoder.
2. **Stored bytes (after build, before GPU).** A probe can be recognised by its tiny shapes, so the
   real checkpoint is checked too. Bytes are sampled at secret offsets from up to four units the new
   quantizer encoded. They are compared with what every existing encoder stores for the same tensors,
   and with earlier PRs' stored bytes. `rtn` is run on the real weights; `baseline` and `unsloth` are
   read from their pinned sources, across both NVFP4 layouts. A "new" encoder that splices attested
   bytes is caught here.

A version bump that does not change the bytes (`rtn@v2` identical to `rtn@v1`) is also the same
encoder.

## Maintainer controls

| Label | Effect |
|---|---|
| `eval-approved` | lets a PR that runs contributed code be evaluated |
| `copy-cleared` | measures a PR held in `bt:copy-review` |

State is in `<root>/state.json`. Errors (`bt:eval-error`) are retried up to three times per head.

## Known limits

- **Contributed code runs in the evaluator's process space.** Until quantizer PRs run as an
  unprivileged user without access to tokens, `eval-approved` is a real trust decision.
- **Near-identical by accident.** Two miners may independently submit recipes within 2%. The later
  one is still measured and credited for what it adds, and one near-copy never triggers review.
- **Sketches compare samples, not whole tensors.** 65,536 bytes per tensor at secret offsets is
  enough to tell encoders apart (unrelated encoders share far fewer identical bytes), not a proof of
  identity.
