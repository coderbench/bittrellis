# Security and evaluation integrity

## Reporting

Report vulnerabilities in the evaluator, audit bypasses, or ways to move the frontier without a
real improvement **privately** to the maintainers (GitHub security advisory on this repository),
not in a public issue or PR. Credible reports that close a real gap are treated as high-value
contributions.

## What the evaluator protects

| Threat | Protection |
|---|---|
| Fine-tuned or substituted weights disguised as quantization | lineage audit: runtime units byte-identical to BF16, attested units identical to hash-pinned sources, regenerable units rebuilt byte-for-byte; anomaly bound on reconstruction error |
| Tampered source checkpoints on the evaluator host | every source file sha256-pinned in `configs/sources.lock.json` and verified before builds and audits |
| Runtime knobs faking a better checkpoint | every `SPARKINFER_*` variable cleared; only the track's pinned env applied and recorded |
| Overfitting the public corpus | private rotating holdout, PASS/FAIL only |
| Measurement noise producing fake wins | two-run spreads with floors, paired bootstrap on RP-KL, ε-dominance |
| Resubmitting known results | content-addressed candidate ids; duplicates of seeds or accepted results earn nothing |
| Malicious code in PRs | manifest-only PRs run trusted `main` code; code PRs require a maintainer `eval-approved` label; evaluator paths are never auto-evaluated |

## Evaluator host hygiene

- Run the PR bot under a dedicated user without access to the private holdout's parent directories
  other than the current epoch.
- Keep `GITHUB_TOKEN` scoped to pull-request read and issue comment/label write.
- Approved code PRs execute on the GPU host: review them as you would any code you run as root.
