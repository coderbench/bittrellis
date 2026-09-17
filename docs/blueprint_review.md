# Blueprint review: what changed and why

> Which blueprint premises failed against the runtime, checkpoints or measurement noise, and what replaced them.

> [!NOTE]
> **Historical record of the first review; some details are superseded** (e.g. top-64 references).
> Authoritative: [specification.md](specification.md) v2.1, adding the second review:
> Reference-Partition KL, lineage classes, an internal frontier seeded with V0, a private holdout, and
> two performance runs with a 512-token decode window.

The intent holds: Qwen3.8-27B × RTX 5090 × SparkInfer, per-module precision search, feasibility first.
Terms: [README](../README.md#key-terms); GDN = Gated DeltaNet (recurrent layers).

## 1. "BF16 · FP8 · NVFP4 per module" is not deployable

- **Reality at SparkInfer `b1ed168`** ([precision_space.md](precision_space.md)): NVFP4 runs native everywhere, FP8 only on GDN, the rest as a Q4_K fit. BF16 never executes for a searchable Linear; FP8 attention/MLP/head silently becomes a Q4_K refit.
- **Fix:** manifests name what *runs*: `NVFP4` anywhere, `FP8` on GDN only, `Q4_K` anywhere (stored BF16 so the fit sees clean weights). Undeployable assignments are rejected, not relabeled, per the blueprint's own rule: *"If one format is not actually deployable in the pinned runtime: remove it. Do not fake support."*

## 2. The module names and search units were wrong

- **Blueprint:** `block.00.gdn.in_proj_a → FP8`, `block.58.mlp.down_proj → BF16`, attention in every block.
- **Reality:** tensors are `model.language_model.layers.{i}.linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`. `in_proj_a`/`in_proj_b` are 48×5120 gate scalars, BF16 by loader requirement. Attention: 16 of 64 layers (3, 7, …, 63). MLP `gate/up/down` share a format.
- **Fix:** `bittrellis inventory` derives 273 searchable units (`L{i}.gdn.qkv`, `L{i}.attn.o`, `L{i}.mlp`, `lm_head`) from config and loader.

## 3. The "most aggressive legal precision" variant *is* the baseline

- **Reality:** the baseline (`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`) runs NVFP4 on all 400 Linears. Nothing runs below 4.5 bits/weight and Q4_K costs the same, so "+5% smaller at equal quality" cannot come from fewer bits.
- **Fix:** [feasibility variants](feasibility_protocol.md) test real trade-offs: NVFP4 vs Q4_K kernels and error, GDN FP8, single head quantization, depth, interactions.

## 4. Quality was relative to a quantized baseline, with noisy benchmarks

- **Reality:** a 4-bit baseline makes "quality ≥ baseline − delta" reward reproducing its errors. On its model card, three NVFP4 builds score 79.0%, 80.2% and 78.7% over 328 items, 95% intervals overlapping: task accuracy cannot separate precision maps.
- **Fix:** primary metric **KL(BF16 ‖ candidate)**, teacher-forced through SparkInfer on a fixed 78K-token corpus (21,624 scored positions: general, math, code, tool-call, multilingual, 8K/16K/32K needle streams); deterministic (`SPARKINFER_DETERMINISTIC=1`), block-bootstrap intervals. Tasks only guard against regressions.

## 5. The BF16 reference cannot run on the target

- **Reality:** BF16 Qwen3.8-27B is 52 GiB; it fits neither a 32 GB card nor SparkInfer's loader.
- **Fix:** reference distributions computed **once** with Hugging Face transformers (GPU + CPU offload, ~25 s per 4K stream), stored as top-64 log-probabilities, hash-pinned, shared by every candidate, reference and llama.cpp run.

## 6. "Calibration set 512–2,000 samples" does not apply to v1

- **Reality:** the Q4_K fit is calibration-free; NVFP4/FP8 weight scales need no activation statistics; the loader never reads `input_scale`. The baseline's NVFP4 bytes are round-to-nearest with a wider global scale (block scales match 100%, nibbles 99.7–99.9%).
- **Fix:** v1 quantizers are `baseline` (shipped NVFP4 bytes) and `rtn`. Calibrated encoders (GPTQ-style rounding, scale search) are a planned miner lever; the audit's fidelity bound stops them smuggling in fine-tuned weights.

## 7. There was no external comparison target

A project that only beats itself can report any number. **Fix:** like SparkInfer vs llama.cpp, four references, same harness:

- **R0** gittensor NVFP4: what SparkInfer ships today.
- **R1** unsloth NVFP4: hand-tuned mixed NVFP4/FP8 map.
- **R2** llama.cpp + unsloth UD-Q4_K_M: imatrix-tuned mixed GGUF, same KL code via a libllama tool.
- **R3** NVIDIA NVFP4: recorded as **unloadable** (per-tensor FP8).

## 8. Runtime knobs could fake a better checkpoint

`SPARKINFER_Q38_GDN_NVFP4=0` and others change execution without touching the checkpoint. **Fix:** every `SPARKINFER_*` variable is cleared per run; only the pinned env applies.

## 9. Anti-gaming needed teeth

Forbidden actions had no enforcement. **Fix:** `bittrellis audit` runs before scoring and checks:
config identical to the baseline outside `quantization_config`; exact tensor set; non-searchable
tensors byte-identical to the baseline; BF16 Linears byte-identical to the base; NVFP4/FP8 Linears
within 2× of round-to-nearest error on sampled rows; manifest ≡ loader execution.

## 10. Three objectives let a map hide a large cost

- **Measured:** V6 (all MLPs Q4_K) matches the baseline's KL and decode, saves 1.2 GiB, dominating on quality × decode × VRAM, yet halves prefill (−45%).
- **Fix:** 4K prefill is a fourth objective; **ε-dominance** uses each objective's measured noise as tolerance, so reruns cannot move a result onto the frontier. Frontier Gain is versioned FG-2.

## 11. The quantizer is as big a lever as the topology

- **Measured:** unsloth's checkpoint beat BitTrellis's GDN-FP8 map on KL at equal GDN precision: its NVFP4 MLP bytes are calibrated (GPTQ-style `actorder`). Splicing them into the shipped map (V13) cut KL 12% at near-identical cost in e1 (e2: 11.5% at −4.4% prefill). No precision-only change did that.
- **Fix:** manifests assign `precision@quantizer` per unit. Quantizers are in-repo or pinned public checkpoints (`quantizer_sources`); manifests never carry bytes. The audit admits calibrated encoders (1.2–1.6× round-to-nearest weight error), rejects substitution (> 5×).

## 12. Smaller corrections

- **Repetitions:** 2, never more; decode noise = their spread.
- **MTP:** out of scope; the baseline strips the MTP head and the runtime never loads it.
- **Long context:** 8K/16K/32K, teacher-forced decode after batched prefill: recurrent-state error accumulates there, at affordable per-candidate cost.
- **Pinning:** every repo, revision, commit, dataset and tokenizer, by full hash, in [`configs/hpc01.yaml`](../configs/hpc01.yaml) and the corpus manifest.
