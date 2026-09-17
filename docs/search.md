# Search

> How to out-search the built-in baseline.

## Baseline: greedy one-group moves, a floor to beat

```text
best valid internal candidate (initially V0)
  → every legal one-group change            bittrellis search neighbors <manifest> --out proposals/
  → public RP-KL for each (quality stage only, ~6 GPU minutes)
  → benchmark only those not dominated on RP-KL alone
  → keep what moves the ε-frontier; test pairs of winners
  → repeat around the new frontier points
```

**Group** = unit kind or GDN projection (`gdn-qkv`, `gdn-z`, `gdn-out`, `attn`, `mlp`) in one depth
half (`shallow` layers 0–31, `deep` 32–63), switched to every legal format/quantizer pair.

## What the [seeds](../results/feasibility/feasibility_report.md) say

- **Quantizer bytes:** V13's calibrated NVFP4: RP-KL −11.5%, equal decode and memory, 4K prefill
  −4.4%. Values or compressed-tensors layout? An in-repo calibrated quantizer writing ModelOpt layout would tell.
- **GDN FP8:** better fidelity; decode cost looks like a step, not a slope. Shallow and deep layers
  protect different capabilities.
- **Prefill:** the first layer of each kind picks the batched-prefill path; keep layer 0 NVFP4 or pay.
- **Interactions:** effects don't add (GDN × MLP Q4_K is significant): search pairs.

## Where better search wins

- **Surrogates:** predict RP-KL from assignment features, trained on the growing content-addressed
  (reusable) artifact set.
- **Per-category objectives:** long-context and math respond to different layers.
- **Cost models:** decode cost is non-linear in FP8 placement; prefill depends on first-layer formats.
- **Quantizer search:** GPTQ-style error feedback, scale and clipping search via the
  [quantizer contract](quantizer_contract.md), combined with topology.

Surrogates may guide; only measured results on the pinned RTX 5090 enter the frontier.
