# The HPC-01 precision space

> A precision is in the space only if the pinned runtime **executes** it. Storing bytes in a
> format is not the same as running them.

All line references are `runtime/src/models/qwen35.cpp` in
[gittensor-ai-lab/sparkinfer@`b1ed168`](https://github.com/gittensor-ai-lab/sparkinfer/tree/b1ed168e3931d22d66dfa0aed3bcc2684f2b8387)
(v0.5.8), loader `Qwen35Model::load_compressed_tensors`.

## What the loader does with each stored format

```text
                     stored in checkpoint
                ┌────────────┬──────────────────┬─────────────┐
 unit           │   NVFP4    │  FP8 (per row)   │    BF16     │
 ───────────────┼────────────┼──────────────────┼─────────────┤
 GDN qkv/z/out  │   NVFP4    │      FP8         │  Q4_K fit   │   keep_native()
 attention qkvo │   NVFP4    │  Q4_K fit (!)    │  Q4_K fit   │   attn_w()
 MLP block      │   NVFP4    │  Q4_K fit (!)    │  Q4_K fit   │   ffn_is_nvfp4(gate)
 lm_head        │ Q4_K fit + │  Q4_K fit        │  Q4_K fit   │   requant_q4k() always
                │ NVFP4 copy*│                  │             │
                └────────────┴──────────────────┴─────────────┘
   * the NVFP4 head copy serves wide packed batches only; batch-1 decode reads the Q4_K fit
```

`Q4_K fit` is `launch_proj_requant_q4k_lloyd` run at load time on the dequantized source.
Fitting it from BF16 is lossless in the source; fitting it from FP8 or NVFP4 quantizes twice.

## The space BitTrellis searches

| Unit kind | Units | Legal precisions | Stored as |
|---|---:|---|---|
| `L{i}.gdn.qkv`, `.z`, `.out` | 144 (48 layers) | `NVFP4` · `FP8` · `Q4_K` | NVFP4 · FP8 per-row · BF16 |
| `L{i}.attn.q`, `.k`, `.v`, `.o` | 64 (16 layers) | `NVFP4` · `Q4_K` | NVFP4 · BF16 |
| `L{i}.mlp` (gate+up+down) | 64 | `NVFP4` · `Q4_K` | NVFP4 · BF16 |
| `lm_head` | 1 | `NVFP4` · `Q4_K` | NVFP4 · BF16 |

**273 units**, roughly `3^144 · 2^129` maps. Full-attention layers are 3, 7, 11, …, 63
(`(i + 1) % 4 == 0`); the other 48 are Gated DeltaNet.

Not searchable, BF16-only in the loader: `embed_tokens`, all norms, `linear_attn.conv1d`,
`in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, and the vision tower.

### Hard constraints enforced by `bittrellis manifest` and `bittrellis audit`

- **FP8 is GDN-only.** FP8 attention/MLP/head weights load, but run as a Q4_K refit. The
  manifest rejects them rather than letting the label lie.
- **FP8 scales must be one BF16 value per output row.** Per-tensor or block FP8 fails the load.
  This is why NVIDIA's official Qwen3.8-27B-NVFP4 (per-tensor FP8 attention/GDN) cannot run.
- **MLP gate/up/down move together.** If `gate_proj` is NVFP4, `up_proj` and `down_proj` must be
  too or the load fails.
- **NVFP4 is group-16 with a UE4M3 block scale and one F32 tensor scale.** ModelOpt layout
  (`.weight`, `.weight_scale`, `.weight_scale_2`) and compressed-tensors layout
  (`.weight_packed`, `.weight_global_scale`) are both read; BitTrellis writes ModelOpt.
- **Runtime knobs are pinned.** `SPARKINFER_Q38_*` variables can move GDN/attention/FFN between
  NVFP4 and Q4_K without touching the checkpoint. They are part of the runtime, so BitTrellis
  clears every `SPARKINFER_*` variable before a run.

## What the formats cost

| Precision | Bits/weight (weights + scales) | Decode kernel | Prefill |
|---|---:|---|---|
| NVFP4 | 4.5 | `launch_gemv_nvfp4` (dequant to float) | CUTLASS block-scaled FP4 GEMM |
| FP8 | 8.0 + 16/row | `launch_gemv_fp8` | FP8 path |
| Q4_K | 4.5 | dp4a int8 MMVQ (heavily tuned) | Q4_K-in-GEMM row scales |

NVFP4 and Q4_K cost the **same bits**, so at this commit the useful trade-offs are not "fewer
bits". They are *which 4-bit format* (different error, different kernels), *where to spend 8
bits* (GDN only), and *how many times a weight is quantized*. The feasibility experiment measures
whether those choices move the real frontier. It does not assume they do.

## Stored format in the three published Qwen3.8 NVFP4 builds

`bittrellis describe <dir>` prints this for any checkpoint:

| Build | GDN | Attention | MLP | lm_head (batch 1) | Loads? |
|---|---|---|---|---|---|
| gittensor (R0) | NVFP4 | NVFP4 | NVFP4 | Q4_K(nvfp4) | yes |
| unsloth (R1) | FP8 | **Q4_K(fp8)** | NVFP4 ×56, **Q4_K(fp8)** ×8 | Q4_K(fp8) | yes |
| NVIDIA (R3) | per-tensor FP8 | per-tensor FP8 | NVFP4 | NVFP4 | **no** |

unsloth's FP8 attention and last eight MLPs look like "protected 8-bit" layers on disk, but at
this commit they run as 4-bit refits of 8-bit bytes.
