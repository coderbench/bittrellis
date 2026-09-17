# Contributing to BitTrellis

Thanks for helping find the quantization topology the hardware actually wants.

## Kinds of contributions

| Kind | Where | How it is evaluated |
|---|---|---|
| **Manifest** (topology and quantizer placement) | `manifests/<name>.yaml` | automatically, with trusted `main` code; FG-2 |
| **Quantizer** (better bytes for an executed format) | `bittrellis/quantizers/` plus one manifest using it | after maintainer review and the `eval-approved` label; FG-2 of its manifest |
| **Search tooling** (proposals, surrogates) | `bittrellis/search.py` | review; the manifests it finds are scored |
| **Docs, fixes, tests** | outside evaluator paths | review |
| **Evaluator changes** | protected paths below | maintainers only; each change bumps the evaluator epoch |

Start with the [miner guide](docs/miner_guide.md) and the
[specification](docs/specification.md).

## Ground rules

- **One PR, one candidate.** A manifest PR adds exactly one manifest.
- **Protected evaluator paths:** `configs/`, `data/`, `bittrellis/eval/`, `bittrellis/frontier/`,
  `bittrellis/validate.py`, `bittrellis/lineage.py`, `bittrellis/holdout.py`, `bittrellis/build.py`,
  `bittrellis/runtime.py`, `evaluator/`, `tools/`, `scripts/`, `.github/`, `pyproject.toml`. PRs that
  touch them are not evaluated automatically.
- **No self-reported scores.** Everything that matters is re-measured on the reference rig.
- **Duplicates earn nothing.** Candidate ids are content addresses.
- **Found a way to game the evaluator?** Report it privately ([SECURITY.md](SECURITY.md)).

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```

Tests build tiny synthetic Qwen3.8-shaped checkpoints. They cover build, audit, every lineage class
(including sequential replay and the replay cache), tamper detection, RP-KL, ε-dominance and the PR
classifier, without weights or a GPU.

Commit messages: one line with a conventional prefix, for example `feat(quantizers): …`,
`fix(audit): …`, `docs(miner-guide): …`, `test(manifest): …`.
