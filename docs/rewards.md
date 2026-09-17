# Rewards: from a measured result to TAO

BitTrellis is scored on [Gittensor](https://github.com/entrius/gittensor) (Bittensor subnet 74).
Gittensor does not read our frontier numbers. It pays a **merged** pull request by **one label**
that the evaluator applies. This page explains how a measured result becomes that label, and what
that label is worth.

## How Gittensor pays a pull request

For a repository with an evaluator, Gittensor's validators compute each merged PR's score as

```text
earned = fixed_base_score × label multiplier × time decay × review factor × open-PR spam factor
```

([`oss_contributions/scoring.py`](https://github.com/entrius/gittensor/blob/main/gittensor/validator/oss_contributions/scoring.py),
[`label_resolution.py`](https://github.com/entrius/gittensor/blob/main/gittensor/validator/oss_contributions/label_resolution.py)).
A miner's share of the repository's emission is their earned score against everyone else's.

| Factor | What it means for a BitTrellis miner |
|---|---|
| **Merged only** | an open PR earns nothing; it only reserves collateral (20% of its potential score) |
| **Label multiplier** | the `eval:*` tier below; no tier label means ×0 |
| **Time decay** | a merged PR loses half its score about 10 days after merge |
| **Review factor** | each maintainer "changes requested" review lowers the score |
| **Credibility gate** | merged ÷ (merged + closed) must stay ≥ 0.2, so closed PRs count against you |
| **Spam factor** | too many open PRs in the repository sets your score there to 0 |

## Tiers

The evaluator buckets the PR's measured frontier gain (FG-2: the quality × speed × memory space it
adds on top of every earlier result) into the same `eval:*` tiers SparkInfer uses. Thresholds live in
`rewards.tiers_fg2` in [`configs/hpc01.yaml`](../configs/hpc01.yaml) and are calibrated to the
measured seeds:

| Label | FG-2 | Proposed multiplier | Seed at this level |
|---|---|---:|---|
| ![eval:XL](https://img.shields.io/badge/eval%3AXL-0e8a16?style=flat-square) | ≥ 0.60% | ×4.0 | V0, the shipped map (0.66%) |
| ![eval:L](https://img.shields.io/badge/eval%3AL-2da44e?style=flat-square) | ≥ 0.30% | ×2.5 | V13, calibrated MLP bytes (0.43%) |
| ![eval:M](https://img.shields.io/badge/eval%3AM-4ac26b?style=flat-square) | ≥ 0.15% | ×1.5 | — |
| ![eval:S](https://img.shields.io/badge/eval%3AS-8ddb8c?style=flat-square) | ≥ 0.07% | ×1.0 | V1 all Q4_K (0.13%), V4 (0.10%) |
| ![eval:XS](https://img.shields.io/badge/eval%3AXS-c6efce?style=flat-square) | > 0, beyond noise | ×0.5 | V6 (0.05%), V5 (0.03%) |
| ![eval:none](https://img.shields.io/badge/eval%3Anone-bfc5cc?style=flat-square) | 0: dominated, or a duplicate | ×0 | V3, V7, V9 |
| ![eval:REJECT](https://img.shields.io/badge/eval%3AREJECT-b60205?style=flat-square) | failed a gate, the audit or a screen | ×0 | — |

- A PR waiting in the queue, for approval or for a maintainer has **no** tier label yet.
- The tier is computed against earlier open PRs by other authors and against merged results, so a
  near-copy earns only the tier of what it adds ([guards.md](guards.md)).
- The multipliers are a **proposal**: the Gittensor team sets them in its registry.

## Merging

Merging is the moment of payment, so it is a maintainer's decision, never the bot's.

1. Every pass the bot re-ranks open results and marks **one** `bt:merge-first`: the highest tier, then the
   largest FG-2, then the one observed first.
2. A maintainer reviews and merges it.
3. The merged result joins the frontier. On the next pass every other open result is re-ranked against it, and its
   tier label is updated. A PR whose gain the merge already covered drops to `eval:none`.
4. **A merged PR's tier is final.** The bot never relabels a merged PR.

The bot never closes pull requests. Closing costs the author credibility on Gittensor, so a
dominated or duplicate PR stays open until its author closes it. Authors should close PRs they are
not pursuing: open PRs reserve collateral and count toward the spam limit.

## Proposed registry entry

For the Gittensor team's `gittensor/validator/weights/master_repositories.json`. `emission_share` is
theirs to set; the rest mirrors the SparkInfer entry, with this repository's tier labels:

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

`trusted_label_pipeline: true` makes Gittensor accept labels from the evaluator's account whatever
its GitHub association. Only maintainers and the evaluator can apply labels in this repository, so
miners cannot label their own PRs.
