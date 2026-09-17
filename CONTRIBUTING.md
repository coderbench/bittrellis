# Contributing to BitTrellis

> PR kinds and rules. New? Read the [miner guide](docs/miner_guide.md) and [specification](docs/specification.md).

| Kind | Where | Evaluated |
|---|---|---|
| **Manifest** | exactly one `manifests/<name>.yaml`, nothing else | automatically, with trusted `main` code; FG-2 |
| **Quantizer / search code** | `bittrellis/quantizers/`, `bittrellis/search.py`, `tests/` (+ at most one manifest, + `.md` files) | after review and `eval-approved`; its manifest's FG-2 (search tooling: the manifests it finds) |
| **Docs, fixes** | outside evaluator paths | review |
| **Evaluator** | protected paths | maintainers only; bumps evaluator epoch |

## Rules

- **One PR, one candidate.**
- **Protected evaluator paths**, never evaluated automatically (`bt:touches-evaluator`): `configs/`, `data/`,
  `bittrellis/eval/`, `bittrellis/frontier/`, `bittrellis/{validate,lineage,holdout,runtime,build,fingerprint,synthetic,manifest,precision,cli}.py`,
  `evaluator/`, `tools/`, `.github/`, `scripts/`, `pyproject.toml`.
- **No self-reported scores:** the reference rig re-measures.
- **Duplicates earn nothing:** candidate ids are content addresses.
- **Found an evaluator exploit?** Report privately ([SECURITY.md](SECURITY.md)).

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```

Tests (tiny synthetic Qwen3.8-shaped checkpoints; no weights, no GPU) cover build, audit, every lineage
class incl. sequential replay and the replay cache, tamper detection, RP-KL, ε-dominance, PR classifier.

Commits: one line, conventional prefix: `feat(quantizers): …`, `fix(audit): …`, `docs(miner-guide): …`, `test(manifest): …`.
