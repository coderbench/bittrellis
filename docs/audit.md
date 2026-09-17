# Audit

`bittrellis audit <checkpoint>` runs before any GPU time is spent. A candidate that fails is never
scored.

| # | Check | Fails when |
|---|---|---|
| 1 | **Sources** | the base, the shipped checkpoint, or any attested source used differs from `configs/sources.lock.json` (sha256 per LFS file, git blob id per small file; verification cached by path, size and mtime) |
| 2 | **Config** | `config.json` differs from the shipped config outside `quantization_config`, or has none (the loader would reject it) |
| 3 | **Tensor set** | a tensor is missing, or there is any tensor the manifest's quantizers and the frozen set do not account for |
| 4 | **Frozen tensors** | any non-searchable tensor (embeddings, norms, conv1d, `in_proj_a/b`, `A_log`, `dt_bias`, vision tower) is not byte-identical to the shipped checkpoint |
| 5 | **Execution** | a unit does not load, or the pinned loader would execute a different format than the manifest selects (see [precision_space.md](precision_space.md)) |
| 6 | **Lineage: runtime** | a Q4_K unit's stored BF16 is not byte-identical to the base |
| 6 | **Lineage: attested** | a unit's bytes differ from the verified source |
| 6 | **Lineage: regenerable** | a sampled unit's rebuild (in pipeline order for sequential quantizers) differs by one byte. The evaluator picks samples with a secret, and a contributed quantizer's samples are regenerated in the sandbox without access to the checkpoint, then compared by trusted code ([guards.md](guards.md#isolation-contributed-code-produces-trusted-code-judges)) |
| 7 | **Anomaly** | reconstruction error is more than 5× round-to-nearest (warning above 2×) |

`audit.json` records every error, the executed format of each unit, the lineage outcome per
quantizer (samples, replayed Linears, cache hits), and the worst anomaly ratios.

Runtime correctness is checked during evaluation, not the audit:

- every scoring and benchmark process loads and exits 0;
- no NaN or Inf log-probs;
- argmax ids are inside the vocabulary;
- the executed map matches the manifest;
- only pinned `SPARKINFER_*` variables reach the runtime.

The outcome is recorded in `correctness.json`.
