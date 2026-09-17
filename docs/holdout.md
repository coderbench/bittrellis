# Private holdout (validators)

Once quantizers are searchable, the public corpus becomes something miners can calibrate against.
The private holdout is the check that a fidelity gain is real rather than fitted to public tokens.

## Principles

- **Private text.** Documents that have not been published anywhere. A split of public datasets is
  not a holdout: anyone can rebuild it. The repository's `public-validation` split exists for local
  checks only.
- **Same structure** as the public corpus: five 4K category streams plus 8K, 16K and 32K needle
  streams, built by the same code (`assemble_streams`).
- **PASS/FAIL only.** A candidate's artifact receives `{"epoch": ..., "result": "PASS"}` and nothing
  else. Scores, positions and failing items stay in the private directory.
- **Rotation.** A new epoch (new text, new seed) replaces the old one on a declared schedule. The old
  epoch's text may be published after it retires, for audits and disputes.

## Setting up an epoch

```text
/secure/holdout-2026w38/            never inside the repository, never synced to public storage
  epoch.json                        {"epoch": "hpc01-h2026w38", "seed": "<long random secret>"}
  docs/general/*.txt                one document per file, scored as written
  docs/math/*.txt                   (chat-formatted Q&A where that matters)
  docs/code/*.txt
  docs/tools/*.txt
  docs/multilingual/*.txt
  docs/long/*.txt                   long filler for the 8K/16K/32K needle streams (≥ 60K tokens total)
```

```bash
bittrellis holdout build --private /secure/holdout-2026w38
bittrellis reference --corpus /secure/holdout-2026w38/corpus.json --out /secure/holdout-2026w38/reference
```

Each short category needs at least 4,096 tokens of documents. Building fails loudly otherwise.

## Checking a candidate

```bash
bittrellis holdout check models/candidates/<name> --private /secure/holdout-2026w38 \
    --artifact artifacts/<name> --incumbent-artifact results/feasibility/artifacts/V0-baseline-rebuild
```

The PR bot does this automatically when started with `--private`. The incumbent (V0, the shipped
checkpoint) is scored once per epoch and cached.

## PASS rule (epoch hpc01-e2)

A candidate PASSES if all of the following hold:

1. its audit passed;
2. holdout RP-KL ≤ `gates.rp_kl_max` (0.30);
3. it retrieves every holdout needle that BF16 retrieves;
4. **transfer**: if its public RP-KL gain over V0 is significant (paired 95% interval excludes 0),
   its holdout gain over V0 is at least `evaluation.holdout.min_gain_ratio` (0.5) of the public
   gain.

Ranking still uses the *public* RP-KL, which everyone can reproduce. The holdout does not rank. It
rejects results whose improvement does not carry over. That is a deliberate trade-off: miners can
optimize public fidelity only up to the point where it stops generalizing.

## Known limits

- **Unknown calibration data.** Attested sources (for example unsloth's checkpoint) were calibrated
  on data nobody here can inspect. Overlap with the holdout cannot be proven absent. The holdout
  is the mitigation, and the risk is accepted per source.
- **Probing.** Even PASS/FAIL leaks a little per submission. Evaluators should rate-limit
  submissions per miner and rotate epochs.
