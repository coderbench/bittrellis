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

| Label | FG-2 | Proposed multiplier | Seed at this level |
|---|---|---:|---|
| ![eval:XL](https://img.shields.io/badge/eval%3AXL-0e8a16?style=flat-square) | ≥ 0.60% | ×4.0 | V0, the shipped map (0.66%) |
| ![eval:L](https://img.shields.io/badge/eval%3AL-2da44e?style=flat-square) | ≥ 0.30% | ×2.5 | V13, calibrated MLP bytes (0.43%) |
| ![eval:M](https://img.shields.io/badge/eval%3AM-4ac26b?style=flat-square) | ≥ 0.15% | ×1.5 | — |
| ![eval:S](https://img.shields.io/badge/eval%3AS-8ddb8c?style=flat-square) | ≥ 0.07% | ×1.0 | V1 all Q4_K (0.13%), V4 (0.10%) |
| ![eval:XS](https://img.shields.io/badge/eval%3AXS-c6efce?style=flat-square) | > 0, beyond noise | ×0.5 | V6 (0.05%), V5 (0.03%) |
| ![eval:none](https://img.shields.io/badge/eval%3Anone-bfc5cc?style=flat-square) | 0: dominated, or a duplicate | ×0 | V3, V7, V9 |
| ![eval:REJECT](https://img.shields.io/badge/eval%3AREJECT-b60205?style=flat-square) | failed a gate, the audit or a screen | ×0 | — |

- A PR still waiting (queue, approval, maintainer) has **no** tier label.
- FG-2 counts merged results and earlier open PRs by other authors: a near-copy earns only what it adds ([guards.md](guards.md)).

## Merging

Merging is payment, so a maintainer merges, never the bot.

1. Each pass the bot marks **one** open result `bt:merge-first`: highest tier, then largest FG-2, then first observed.
2. A maintainer reviews and merges it; it joins the frontier.
3. Next pass, other open results are re-ranked against it; one whose gain the merge covered drops to `eval:none`.
4. **A merged PR's tier is final**; the bot never relabels it.

The bot never closes PRs (closing costs credibility); authors close dominated or duplicate PRs. Close PRs you are not pursuing: they reserve collateral and count toward the spam limit.

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
