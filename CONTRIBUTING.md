# Contributing to BitTrellis

Thanks for helping find the precision map the hardware actually wants.

## Kinds of contributions

| Kind | Where | Scored by |
|---|---|---|
| **Precision map** | `manifests/<name>.yaml` | Frontier Gain on the pinned RTX 5090 |
| **Quantizer** (better NVFP4/FP8 bytes, same format) | `bittrellis/quant/`, `bittrellis/build.py` | Frontier Gain of the maps it enables |
| **Search / analysis** (sensitivity scans, surrogates) | `bittrellis/search/` (Phase 3+) | the maps it finds |
| **Docs, fixes, tests** | anywhere outside evaluator paths | review |

Start with the [miner guide](docs/miner_guide.md).

## Ground rules

- **One PR, one idea.** A manifest PR adds exactly one manifest.
- **Evaluator paths are maintainer-owned:** `configs/`, `data/corpus/`, `bittrellis/eval/`,
  `bittrellis/frontier/`, `bittrellis/validate.py`, `.github/`. PRs that change them together
  with a manifest are closed.
- **No self-reported scores.** Every number that matters is re-measured on the reference rig.
- **Pins change only through a new track version,** never in a candidate PR.

## Development

```bash
pip install -e ".[dev]"
ruff check .
pytest -q
```

Tests build tiny synthetic Qwen3.8-shaped checkpoints, so the build → audit → tamper-detection
path is covered without weights or a GPU.

Commit messages: one line, conventional prefix. For example `feat(quant): …`, `fix(audit): …`,
`docs(readme): …`, `perf(build): …`, `test(manifest): …`.
