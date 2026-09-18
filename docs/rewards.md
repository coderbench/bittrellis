# Rewards: from a measured result to TAO

> [Gittensor](https://github.com/entrius/gittensor) (Bittensor subnet 74) pays a merged PR by one `eval:*` label. This page shows how your result earns that label and what it is worth.

## How Gittensor pays

```text
earned = fixed_base_score × label multiplier × time decay × review factor × open-PR spam factor
```

Source: [`scoring.py`](https://github.com/entrius/gittensor/blob/main/gittensor/validator/oss_contributions/scoring.py),
[`label_resolution.py`](https://github.com/entrius/gittensor/blob/main/gittensor/validator/oss_contributions/label_resolution.py).
Your emission share: your earned score against everyone else's.

| Factor | For a miner |
|---|---|
| **Merged only** | an open PR earns nothing; it reserves collateral (20% of its potential score) |
| **Label multiplier** | the tier below (`label_multipliers` in the registry entry); no tier label means ×0 |
| **Time decay** | half the score is gone about 10 days after merge |
| **Review factor** | each maintainer "changes requested" review lowers it |
| **Credibility gate** | merged ÷ (merged + closed) must stay ≥ 0.2: closed PRs count against you |
| **Spam factor** | too many open PRs in the repository sets your score there to 0 |

## Tiers

FG-2 ([key terms](../README.md#key-terms)) buckets into SparkInfer's tiers. Thresholds: `rewards.tiers_fg2` in [`configs/hpc01.yaml`](../configs/hpc01.yaml), calibrated to the seeds.

| Label | Noise-aware FG-2 | Proposed multiplier | Seed at this level |
|---|---|---:|---|
| ![eval:XL](https://img.shields.io/badge/eval%3AXL-0e8a16?style=flat-square) | ≥ 0.50% | ×4.0 | V0, the shipped map (2.25%) |
| ![eval:L](https://img.shields.io/badge/eval%3AL-2da44e?style=flat-square) | ≥ 0.25% | ×2.5 | — |
| ![eval:M](https://img.shields.io/badge/eval%3AM-4ac26b?style=flat-square) | ≥ 0.12% | ×1.5 | — |
| ![eval:S](https://img.shields.io/badge/eval%3AS-8ddb8c?style=flat-square) | ≥ 0.035% | ×1.0 | V4 (0.10%), V1 all Q4_K (0.09%) |
| ![eval:XS](https://img.shields.io/badge/eval%3AXS-c6efce?style=flat-square) | ≥ 0.005% | ×0.5 | V6 (0.02%) |
| ![eval:none](https://img.shields.io/badge/eval%3Anone-bfc5cc?style=flat-square) | below 0.005%, dominated, or a duplicate | ×0 | V7, V9; a copy of V0 within noise |
| ![eval:REJECT](https://img.shields.io/badge/eval%3AREJECT-b60205?style=flat-square) | failed a gate, the audit, the holdout or a screen | ×0 | V13 and V3 (holdout), V5 (task guard) |

FG-2 only counts gains beyond measurement noise ([frontier.md](frontier.md#fg-2)).

- A PR still waiting (queue, approval, maintainer) has **no** tier label.
- **No private holdout PASS, no paid tier.** A result measured without one is `bt:provisional`.
- FG-2 counts merged results and earlier open PRs by other authors: a near-copy earns only what it adds ([guards.md](guards.md)).

## Merging

Merging is payment, so a maintainer merges, never the bot.

1. Each pass the bot marks **one** open result `bt:merge-first`: highest tier, then largest FG-2, then first observed.
2. A maintainer reviews and merges it; it joins the frontier.
3. Next pass, other open results are re-ranked against it; one whose gain the merge covered drops to `eval:none`.
4. **A merged PR's tier is final**; the bot never relabels it.

The bot never closes PRs (closing costs credibility); authors close dominated or duplicate PRs. Close PRs you are not pursuing: they reserve collateral and count toward the spam limit.

## The public record

After every pass the evaluator writes and pushes a score record to [its own repository](https://github.com/coderbench/bittrellis-ledger)
([`evaluator/ledger.py`](../evaluator/ledger.py), [`publish_ledger.py`](../evaluator/publish_ledger.py)):
one write-once record per evaluated PR head (status, tier, FG-2, measured row, screen result), the
first-seen records, the current frontier, and the artifacts of merged results. The GPU box is rented;
the record outlives it. Re-derive any score yourself:

```bash
git clone https://github.com/coderbench/bittrellis-ledger && cd bittrellis-ledger
bittrellis frontier hpc01-e3/accepted <your artifact>
```

The private holdout never appears there: records carry PASS or FAIL only.

## Proposed registry entry

For Gittensor's `gittensor/validator/weights/master_repositories.json`. The Gittensor team sets `emission_share` and the final multipliers; the rest mirrors SparkInfer's entry:

```json
"coderbench/bittrellis": {
  "default_label_multiplier": 0.0,
  "eligibility": {"min_credibility": 0.2, "min_issue_credibility": 0.0, "min_valid_merged_prs": 0, "min_valid_solved_issues": 0},
  "emission_share": "<set by Gittensor>",
  "fixed_base_score": 1.0,
  "issue_discovery_share": 0.0,
  "label_multipliers": {"eval:XL": 4.0, "eval:L": 2.5, "eval:M": 1.5, "eval:S": 1.0, "eval:XS": 0.5, "eval:none": 0.0, "eval:REJECT": 0.0},
  "maintainer_cut": 0.4,
  "scoring": {"standard_issue_multiplier": 1.0, "maintainer_issue_multiplier": 1.0},
  "trusted_label_pipeline": true
}
```

`trusted_label_pipeline: true` accepts the evaluator account's labels whatever its GitHub association. Only maintainers and the evaluator can label here, so miners cannot label their own PRs.
