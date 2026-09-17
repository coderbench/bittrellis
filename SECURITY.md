# Security and evaluation integrity

> How to report a gap, what the evaluator defends against, and how to run its host.

## Reporting

Report evaluator vulnerabilities, audit bypasses, or ways to move the frontier without a real
improvement **privately** (GitHub security advisory on this repository), never in a public issue or
PR. Credible reports that close a real gap are treated as high-value contributions.

## What the evaluator protects

| Threat | Protection |
|---|---|
| Fine-tuned or substituted weights disguised as quantization | lineage audit: runtime units byte-identical to BF16, attested units identical to hash-pinned sources, regenerable units rebuilt byte-for-byte; anomaly bound on reconstruction error |
| Tampered source checkpoints on the evaluator host | every source file sha256-pinned in `configs/sources.lock.json`, verified before builds and audits |
| Runtime knobs faking a better checkpoint | every `SPARKINFER_*` variable cleared; only the track's pinned env applied and recorded |
| Overfitting the public corpus | private rotating holdout, PASS/FAIL only |
| Measurement noise producing fake wins | two-run spreads with floors, paired bootstrap on RP-KL, ε-dominance |
| Resubmitting known results | content-addressed candidate ids over the expanded recipe; duplicates of seeds, accepted results or earlier PRs are not measured |
| Copying another miner's open PR | first-seen record per head commit; near-copies are ranked with the original and earn only what they add ([guards](docs/guards.md)) |
| Contributed code reading secrets or faking its audit | unprivileged account, allow-listed environment; regenerates secret samples without access to its checkpoint; trusted code compares ([guards](docs/guards.md#isolation-contributed-code-produces-trusted-code-judges)) |
| Renaming an existing encoder | quantizers compared by output bytes on seeded probes and the real checkpoint, not source text |
| Malicious code in PRs | manifest-only PRs run trusted `main` code; code PRs need a maintainer `eval-approved` label; evaluator paths are never auto-evaluated |

## Evaluator host hygiene

1. Run the PR bot as root on a dedicated host: it needs root to switch into the sandbox account. Set that account up once with [`evaluator/setup_sandbox.sh`](evaluator/setup_sandbox.sh); the bot refuses quantizer PRs until the account cannot read secrets or reach the network.
2. Scope `GITHUB_TOKEN` to pull-request read and issue comment/label write. Store it in a root-only file (`BT_TOKEN_FILE`), never in the repository or the sandbox's environment.
3. Keep the private holdout, `secret.txt` and `state.json` root-only (`chmod 700` / `600`). Stop any notebook or web server a rented image starts.
4. `eval-approved` is still a code review: the sandbox is an account on the same kernel, not a VM.
