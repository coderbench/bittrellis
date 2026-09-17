# The HPC-01 precision space

> Which formats can each unit use? Only what the pinned runtime **executes** counts.

**NVFP4** is a 4-bit float with block scales, **FP8** an 8-bit float, **Q4_K** llama.cpp's 4-bit
k-quant fitted at load; GDN = Gated DeltaNet recurrent layers. Evidence cites loader functions in
`runtime/src/models/qwen35.cpp` (`Qwen35Model::load_compressed_tensors`),
[gittensor-ai-lab/sparkinfer@`b1ed168`](https://github.com/gittensor-ai-lab/sparkinfer/tree/b1ed168e3931d22d66dfa0aed3bcc2684f2b8387) (v0.5.8).

## What the loader executes

```text
stored ▸         NVFP4          FP8 (per row)   BF16        loader function
GDN qkv/z/out    NVFP4          FP8             Q4_K fit    keep_native()
attention qkvo   NVFP4          Q4_K fit (!)    Q4_K fit    attn_w()
MLP block        NVFP4          Q4_K fit (!)    Q4_K fit    ffn_is_nvfp4(gate)
lm_head          Q4_K fit +     Q4_K fit        Q4_K fit    requant_q4k() always
                 NVFP4 copy*
* wide packed batches only
```

`Q4_K fit` = `launch_proj_requant_q4k_lloyd` on the dequantized source at load: one quantization when
the stored bytes are BF16, a second one on top when they are FP8 or NVFP4.

## The searched space: 273 units, roughly `3^144 · 2^129` maps

- `L{i}.gdn.qkv/.z/.out`, 144 units (48 layers): NVFP4, FP8 (per-row), Q4_K (stored as BF16)
- `L{i}.attn.q/.k/.v/.o`, 64 units (16 layers): NVFP4, Q4_K (BF16)
- `L{i}.mlp` (gate+up+down), 64 units: NVFP4, Q4_K (BF16)
- `lm_head`, 1 unit: NVFP4, Q4_K (BF16)
- Full attention: layers 3, 7, 11, …, 63 (`(i + 1) % 4 == 0`); the other 48 are GDN.
- Not searchable, BF16-only in the loader: `embed_tokens`, all norms, `linear_attn.conv1d`, `in_proj_a`, `in_proj_b`, `A_log`, `dt_bias`, vision tower.
- Q4_K is always a runtime fit (quantizer `runtime`, no encoder choice) of frozen BF16; the head fits its stored BF16 or NVFP4 bytes (as in V0).
- Batch-1 decode runs the selected format, except **lm_head**: always its Q4_K fit (`Q4_K(nvfp4)` or `Q4_K`). Wide packed decode also uses the NVFP4 bytes as a block-scaled GEMM operand if free VRAM at load exceeds payload plus a 3 GiB reserve. Manifests select the *stored* format; the expanded manifest records both paths, so the label never claims NVFP4 runs at batch 1.

## Hard constraints (`bittrellis manifest`, `bittrellis audit`)

- **FP8 is GDN-only:** FP8 attention/MLP/head weights load but run as a Q4_K refit, so the manifest rejects them.
- **FP8 needs one BF16 scale per output row:** per-tensor or block FP8 fails the load, which blocks NVIDIA's official Qwen3.8-27B-NVFP4 (per-tensor FP8 attention/GDN).
- **MLP gate/up/down move together:** if `gate_proj` is NVFP4, `up_proj` and `down_proj` must be too, or the load fails.
- **NVFP4 is group-16, UE4M3 block scale, one F32 tensor scale:** ModelOpt (`.weight`, `.weight_scale`, `.weight_scale_2`) and compressed-tensors (`.weight_packed`, `.weight_global_scale`) layouts load; BitTrellis writes ModelOpt.
- **Runtime knobs are pinned:** `SPARKINFER_Q38_*` can move GDN/attention/FFN between NVFP4 and Q4_K without checkpoint changes; every `SPARKINFER_*` variable is cleared before a run.

## What the formats cost

| Precision | Bits/weight (weights + scales) | Decode kernel | Prefill |
|---|---:|---|---|
| NVFP4 | 4.5 | `launch_gemv_nvfp4` (dequant to float) | CUTLASS block-scaled FP4 GEMM |
| FP8 | 8.0 + 16/row | `launch_gemv_fp8` | FP8 path |
| Q4_K | 4.5 | dp4a int8 MMVQ (heavily tuned) | Q4_K-in-GEMM row scales |

Equal bits for NVFP4 and Q4_K: at this commit the trade-off is not "fewer bits" but *which 4-bit
format* (error, kernels), *where to spend 8 bits* (GDN only), *how often a weight is quantized*. The
feasibility experiment measures, not assumes, whether these move the real frontier.

## Published Qwen3.8 NVFP4 builds

`bittrellis describe <dir>` prints this for any checkpoint:

| Build | GDN | Attention | MLP | lm_head (batch 1) | Loads? |
|---|---|---|---|---|---|
| gittensor, shipped (V0) | NVFP4 | NVFP4 | NVFP4 | Q4_K(nvfp4) | yes |
| unsloth (R1) | FP8 | **Q4_K(fp8)** | NVFP4 ×56, **Q4_K(fp8)** ×8 | Q4_K(fp8) | yes |
| NVIDIA (R3) | per-tensor FP8 | per-tensor FP8 | NVFP4 | NVFP4 | **no** |

unsloth's FP8 attention and last eight MLPs look "protected 8-bit" on disk but at this commit
run as 4-bit refits of 8-bit bytes.
