#!/usr/bin/env bash
# Run the PR evaluator on the pinned GPU host. Configure once, then run under systemd, tmux or cron.
#   GITHUB_TOKEN   token with pull-request read, issues write (comments + labels)
#   BT_EVAL_ROOT   evaluator state/artifacts (large: checkpoints are built here, then deleted)
#   BT_PRIVATE     private holdout directory for the current epoch (never inside the repo)
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GITHUB_TOKEN:?set GITHUB_TOKEN}"
: "${BT_EVAL_ROOT:=/workspace/bt-eval}"
args=(--repo "${BT_REPO:-coderbench/bittrellis}" --root "$BT_EVAL_ROOT")
[ -n "${BT_PRIVATE:-}" ] && args+=(--private "$BT_PRIVATE")
exec python evaluator/pr_bot.py "${args[@]}" "$@"
