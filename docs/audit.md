# Audit

> Is the checkpoint a legal encoding of the pinned weights? `bittrellis audit <checkpoint>` runs before any GPU time; failures are never scored.

Fails on:

1. **Sources:** base, shipped checkpoint or used attested source differs from `configs/sources.lock.json` (sha256 per LFS file, git blob id per small file; cached by path, size, mtime).
2. **Config:** `config.json` differs from the shipped one outside `quantization_config`, or lacks it (the loader would reject it).
3. **Tensor set:** a tensor is missing, or unaccounted for by the manifest's quantizers and the frozen set.
4. **Frozen tensors:** a non-searchable tensor (embeddings, norms, conv1d, `in_proj_a/b`, `A_log`, `dt_bias`, vision tower) is not byte-identical to the shipped checkpoint.
5. **Execution:** a unit fails to load, or the pinned loader would execute a format other than the manifest's ([precision_space.md](precision_space.md)).
6. **Lineage:** *runtime*: a Q4_K unit's stored BF16 is not byte-identical to the base. *Attested*: a unit's bytes differ from the verified source. *Regenerable*: a sampled rebuild (pipeline order for sequential quantizers) differs by one byte; samples are secret-chosen, contributed quantizers regenerate them in the sandbox without checkpoint access, and trusted code compares ([guards.md](guards.md#isolation-contributed-code-produces-trusted-code-judges)).
7. **Anomaly:** reconstruction error over 5× round-to-nearest (warning above 2×).

`audit.json` records every error, each unit's executed format, lineage per quantizer (samples,
replayed tensors, cache hits), worst anomaly ratios.

**Runtime correctness** is checked during evaluation, into `correctness.json`: every scoring and
benchmark process loads and exits 0; no NaN or Inf log-probs; argmax ids in the vocabulary; executed
map matches the manifest; only pinned `SPARKINFER_*` variables reach the runtime.
