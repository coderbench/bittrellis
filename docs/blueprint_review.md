# Blueprint review: what changed and why

BitTrellis started from a design blueprint (Qwen3.8-27B × RTX 5090 × SparkInfer; search
BF16/FP8/NVFP4 per module; feasibility first). The intent holds. Several factual premises did
not survive contact with the pinned runtime, the published checkpoints, or measurement noise.
This page lists each one and what the repository does instead.

## 1. "BF16 · FP8 · NVFP4 per module" is not deployable

**Blueprint:** every module picks BF16, FP8 or NVFP4.

**Reality at SparkInfer `b1ed168`:** the loader keeps NVFP4 native everywhere, keeps FP8 native
only on Gated DeltaNet projections, and fits Q4_K at load time to everything else. BF16 never
executes for a searchable Linear. FP8 attention/MLP/head weights silently run as a Q4_K refit of
the FP8 bytes. See [precision_space.md](precision_space.md).

**Fix:** the manifest names the precision that *runs*: `NVFP4` everywhere, `FP8` on GDN only,
`Q4_K` everywhere (stored BF16 so the runtime's fit sees clean weights). Undeployable
assignments are rejected, not relabeled. The blueprint's own rule applied: *"If one format is
not actually deployable in the pinned runtime: remove it. Do not fake support."*

## 2. The module names and search units were wrong

**Blueprint:** `block.00.gdn.in_proj_a → FP8`, `block.58.mlp.down_proj → BF16`, attention in every block.

**Reality:**
- Tensors are `model.language_model.layers.{i}.linear_attn.{in_proj_qkv,in_proj_z,out_proj}`, `self_attn.{q,k,v,o}_proj`, and `mlp.{gate,up,down}_proj`.
- `in_proj_a` and `in_proj_b` are 48×5120 gate scalars. The loader requires them in BF16, so they are not searchable.
- Only 16 of 64 layers have attention (3, 7, …, 63).
- MLP `gate/up/down` must share a format.

**Fix:** 273 searchable units (`L{i}.gdn.qkv`, `L{i}.attn.o`, `L{i}.mlp`, `lm_head`), derived
from the config and the loader. `bittrellis inventory` lists them.

## 3. The "most aggressive legal precision" variant *is* the baseline

**Blueprint:** V1 = all eligible modules at the most aggressive precision, compared with V0 = baseline.

**Reality:** the baseline (`gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090`) already runs NVFP4
on all 400 Linears, and nothing in the runtime is smaller than 4.5 bits/weight. Q4_K costs the
same bits. So "+5% smaller at equal quality" cannot come from fewer bits.

**Fix:** the feasibility variants test the trade-offs that do exist (see
[feasibility_protocol.md](feasibility_protocol.md)):
- NVFP4 vs Q4_K kernels and error;
- FP8 on the recurrent path;
- avoiding double quantization of the head;
- depth and interaction effects.

## 4. Quality was defined relative to a quantized baseline, with noisy benchmarks

**Blueprint:** gate candidates on "quality ≥ baseline − delta" using task benchmarks.

**Reality:**
- The baseline is itself a 4-bit model, so "equal to baseline" rewards reproducing its errors.
- On the baseline's own model card, three NVFP4 builds score 79.0%, 80.2% and 78.7% over 328 items with overlapping 95% intervals. Task accuracy cannot separate precision maps.

**Fix:**
- The primary quality metric is **KL(BF16 ‖ candidate)**, teacher-forced through SparkInfer on a fixed 49K-token corpus. It covers general, math, code, tool-call and multilingual text, plus 8K/16K/32K long-context streams with planted needles.
- Scoring is deterministic (`SPARKINFER_DETERMINISTIC=1`). Uncertainty is a block-bootstrap interval over positions, not run-to-run noise.
- SparkInfer's own task suite remains as a guard against regressions, not as a ranking signal.

## 5. The BF16 reference cannot run on the target

**Blueprint:** implicitly compares against the original model.

**Reality:** BF16 Qwen3.8-27B is 52 GiB. It fits neither a 32 GB card nor SparkInfer's loader.

**Fix:** reference distributions are computed **once** with Hugging Face transformers (GPU + CPU
offload, ~25 s per 4K stream), stored as top-64 log-probabilities, and hash-pinned. Every
candidate, reference point and llama.cpp run is compared against the same files.

## 6. "Calibration set 512–2,000 samples" does not apply to v1

**Blueprint:** calibrate candidates on a corpus.

**Reality:** SparkInfer's Q4_K fit is calibration-free. NVFP4/FP8 weight scales need no activation
statistics, and the loader never reads `input_scale`. The baseline's NVFP4 bytes are
reproducible as round-to-nearest with a wider global scale: BitTrellis's encoder matches its
block scales 100% and nibbles 99.7–99.9%.

**Fix:** v1 quantizers are `baseline` (splice the shipped NVFP4 bytes) and `rtn`. Calibrated
encoders (GPTQ-style rounding, scale search) are a planned miner lever. They are allowed as long
as the audit's fidelity bound holds, so they cannot smuggle in fine-tuned weights.

## 7. There was no external comparison target

**Blueprint:** compare only against the project's own baseline.

**Reality:** a project that only beats itself can report any number. SparkInfer is measured against
llama.cpp; BitTrellis needs the same.

**Fix:** four reference points, measured with the same harness:

| | Reference | Why it matters |
|---|---|---|
| R0 | gittensor NVFP4 | what SparkInfer ships today |
| R1 | unsloth NVFP4 | a hand-tuned mixed NVFP4/FP8 map |
| R2 | **llama.cpp + unsloth UD-Q4_K_M** | the llama.cpp ecosystem's imatrix-tuned mixed-precision GGUF, scored by the same KL code through a libllama tool |
| R3 | NVIDIA NVFP4 | recorded as **unloadable** (per-tensor FP8) |

## 8. Runtime knobs could fake a better checkpoint

**Blueprint:** silent on runtime configuration.

**Reality:** `SPARKINFER_Q38_GDN_NVFP4=0` and similar variables change what executes without
touching the checkpoint.

**Fix:** every `SPARKINFER_*` variable is cleared before each run; only the track's pinned env is
applied.

## 9. Anti-gaming needed teeth

**Blueprint:** lists forbidden actions, with no enforcement.

**Fix:** `bittrellis audit` runs before anything is scored. It checks:

- config identical to the baseline outside `quantization_config`;
- exact tensor set, with no extras;
- non-searchable tensors byte-identical to the baseline;
- BF16 Linears byte-identical to the base model;
- NVFP4/FP8 Linears within 1.35× of round-to-nearest reconstruction error on sampled rows;
- manifest ≡ what the loader would execute.

## 10. Smaller corrections

- **Repetitions:** 2, never more. Decode noise is judged against the spread between those two runs.
- **MTP:** out of scope. The baseline strips the MTP head and the runtime never loads it.
- **Long context:** measured at 8K/16K/32K on the teacher-forced decode path after a batched prefill. That is where recurrent-state error accumulates and where it is affordable per candidate.
- **Pinning:** every repo, revision, commit, dataset and tokenizer is pinned by full hash in [`configs/hpc01.yaml`](../configs/hpc01.yaml) and the corpus manifest.
