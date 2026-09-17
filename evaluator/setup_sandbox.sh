#!/usr/bin/env bash
# One-time host setup for running contributed quantizer code in isolation (evaluator/sandbox.py).
# Run as root on the evaluation host:
#   BT_EVAL_ROOT=/workspace/bt-eval BT_PRIVATE=/secure/holdout-epoch evaluator/setup_sandbox.sh
set -euo pipefail
USER_NAME="${BT_SANDBOX_USER:-bt-sandbox}"
: "${BT_EVAL_ROOT:?set BT_EVAL_ROOT}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"

id "$USER_NAME" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "$USER_NAME"

# Evaluator state: traversable to reach each PR's untrusted/ directory, readable by nobody else.
mkdir -p "$BT_EVAL_ROOT/prs" "$BT_EVAL_ROOT/accepted" "$BT_EVAL_ROOT/observations"
chown root:root "$BT_EVAL_ROOT" "$BT_EVAL_ROOT/prs"
chmod 711 "$BT_EVAL_ROOT" "$BT_EVAL_ROOT/prs"
chmod 700 "$BT_EVAL_ROOT/accepted" "$BT_EVAL_ROOT/observations"
for f in state.json secret.txt; do [ -e "$BT_EVAL_ROOT/$f" ] && chmod 600 "$BT_EVAL_ROOT/$f"; done
[ -n "${BT_PRIVATE:-}" ] && chmod 700 "$BT_PRIVATE"

# Models, the Python environment and the repository must be readable (never writable) by the sandbox.
chmod -R o+rX "$REPO/models/" "$REPO/data/corpus" 2>/dev/null || true
echo "sandbox account: $USER_NAME"
echo "check: runuser -u $USER_NAME -- test -r $BT_EVAL_ROOT/secret.txt && echo UNSAFE || echo ok"
