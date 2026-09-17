# Private holdout (validators)

> Is a fidelity gain real, or fitted to the public corpus miners can calibrate against?

- **Private text:** never published. A public-dataset split is not a holdout (anyone can rebuild it); `public-validation` is for local checks only.
- **Same structure:** five 4K category streams plus 8K, 16K, 32K needle streams, same code (`assemble_streams`).
- **PASS/FAIL only:** the artifact gets `{"epoch": ..., "result": "PASS"}`; scores, positions, failing items stay private.
- **Rotation:** a new epoch (new text, seed) replaces the old on a declared schedule; retired text may be published for audits and disputes.

## Set up and check

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
bittrellis holdout check models/candidates/<name> --private /secure/holdout-2026w38 \
    --artifact artifacts/<name> --incumbent-artifact results/feasibility/artifacts/V0-baseline-rebuild
```

Build fails loudly if a short category has under 4,096 tokens. The PR bot runs the check when started
with `--private`; V0 is scored once per epoch, cached.

## PASS rule (epoch hpc01-e2)

1. audit passed;
2. holdout RP-KL ≤ `gates.rp_kl_max` (0.30);
3. keeps every holdout needle BF16 retrieves;
4. **transfer:** if public RP-KL gain over V0 is significant (paired 95% interval excludes 0), holdout gain over V0 is ≥ `evaluation.holdout.min_gain_ratio` (0.5) of it.

Ranking uses reproducible *public* RP-KL; the holdout only rejects gains that do not carry over, so
miners can optimize public fidelity only until it stops generalizing.

## Known limits

- **Unknown calibration data:** attested sources (e.g. unsloth's) used data nobody here can inspect; holdout overlap cannot be proven absent. The holdout mitigates; risk is accepted per source.
- **Probing:** even PASS/FAIL leaks a little per submission; evaluators should rate-limit per miner and rotate epochs.
