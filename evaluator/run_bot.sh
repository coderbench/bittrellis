#!/usr/bin/env bash
# Run the PR evaluator on the pinned GPU host. Configure once, then run under systemd, tmux or cron.
#   GITHUB_TOKEN   token with pull-request read, issues write (comments + labels)
#   BT_EVAL_ROOT   evaluator state/artifacts (large: checkpoints are built here, then deleted)
#   BT_PRIVATE     private holdout directory for the current epoch (never inside the repo)
#   BT_SANDBOX_USER  account that runs contributed code (default bt-sandbox; evaluator/setup_sandbox.sh)
#   BT_TOKEN_FILE  where the token is stored, checked to be unreadable by the sandbox
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GITHUB_TOKEN:?set GITHUB_TOKEN}"
: "${BT_EVAL_ROOT:=/workspace/bt-eval}"
args=(--repo "${BT_REPO:-coderbench/bittrellis}" --root "$BT_EVAL_ROOT")
[ -n "${BT_PRIVATE:-}" ] && args+=(--private "$BT_PRIVATE")
exec python evaluator/pr_bot.py "${args[@]}" "$@"
