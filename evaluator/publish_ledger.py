#!/usr/bin/env python3
"""Push the score record to its own repository, so it outlives the rented GPU box.

    BT_LEDGER_TOKEN=... evaluator/publish_ledger.py --ledger /workspace/bt-eval/ledger \
        --remote https://github.com/coderbench/bittrellis-ledger.git

- **Never forced.** Records are write-once on disk; a published history that could be force-pushed
  would not be evidence of anything. Protect the branch against force pushes too.
- **The token never reaches a command line or a file.** It goes to git as a request header through
  the environment, so no other account on the box can read it from `ps`.
- **A failed push never fails an evaluation pass.** The commit stays local and goes out next time.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import subprocess
import sys
from pathlib import Path

TOKEN_ENV, REMOTE_ENV = "BT_LEDGER_TOKEN", "BT_LEDGER_REMOTE"
AUTHOR = ("bittrellis evaluator", "evaluator@bittrellis.invalid")
_CREDENTIALS = re.compile(r"^https?://[^/\s@]+@")


class PublishError(RuntimeError):
    """The ledger could not be published; its local history is untouched."""


def git_env(token: str | None = None) -> dict:
    env = dict(os.environ)
    env.update(GIT_AUTHOR_NAME=AUTHOR[0], GIT_AUTHOR_EMAIL=AUTHOR[1], GIT_COMMITTER_NAME=AUTHOR[0],
               GIT_COMMITTER_EMAIL=AUTHOR[1], GIT_TERMINAL_PROMPT="0")
    env.pop(TOKEN_ENV, None)
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.extraHeader",
                   GIT_CONFIG_VALUE_0=f"Authorization: Basic {basic}")
    return env


def check_remote(remote: str) -> None:
    if _CREDENTIALS.match(remote):
        raise PublishError("the remote URL carries credentials; pass the token in " + TOKEN_ENV)


def git(args: list[str], cwd: Path, token: str | None = None, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *args], cwd=cwd, env=git_env(token), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise PublishError(f"git {' '.join(args)}: {(r.stderr or r.stdout).strip()[:400]}")
    return r


def publish(ledger: Path, remote: str, token: str | None, message: str, branch: str = "main") -> str | None:
    """Commit everything in `ledger` and push it. Returns the commit, or None when nothing changed."""
    check_remote(remote)
    ledger = Path(ledger)
    if not (ledger / ".git").exists():
        git(["init", "-q", "-b", branch], ledger)
    if git(["remote"], ledger).stdout.split() == []:
        git(["remote", "add", "origin", remote], ledger)
    else:
        git(["remote", "set-url", "origin", remote], ledger)
    git(["add", "-A"], ledger)
    if not git(["status", "--porcelain"], ledger).stdout.strip():
        return None
    git(["commit", "-q", "-m", message], ledger)
    git(["push", "--quiet", "origin", f"HEAD:{branch}"], ledger, token)   # never --force
    return git(["rev-parse", "HEAD"], ledger).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--remote", default=os.environ.get(REMOTE_ENV))
    ap.add_argument("--message", default="records: evaluator pass")
    ap.add_argument("--branch", default="main")
    a = ap.parse_args()
    if not a.remote:
        print(f"no remote: pass --remote or set {REMOTE_ENV}", file=sys.stderr)
        return 2
    commit = publish(Path(a.ledger), a.remote, os.environ.get(TOKEN_ENV), a.message, a.branch)
    print(f"published {commit}" if commit else "nothing to publish")
    return 0


if __name__ == "__main__":
    sys.exit(main())
