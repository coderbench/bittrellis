# Search

BitTrellis ships a deliberately simple baseline search, so there is an obvious floor for miners to
beat.

## Baseline: greedy one-group moves

```text
start from the best valid internal candidate (initially V0)
  → propose every legal one-group change          bittrellis search neighbors <manifest> --out proposals/
  → score public RP-KL for each (quality stage only, cheap-ish: ~6 GPU minutes)
  → benchmark only proposals that are not dominated on RP-KL alone
  → keep what moves the ε-frontier; test pairs of the winners (interactions are real)
  → repeat around the new frontier points
```

A group is a unit kind or GDN projection in one depth half: `gdn-qkv`, `gdn-z`, `gdn-out`, `attn`,
`mlp` × `shallow` (layers 0–31) or `deep` (32–63). Each group is switched to every legal
format/quantizer pair.

## What the seed measurements already say

Start from [`results/feasibility/feasibility_report.md`](../results/feasibility/feasibility_report.md)
rather than rediscovering it:

- **Quantizer bytes.** Calibrated NVFP4 bytes (V13) cut RP-KL by 11.5% at the same decode speed and
  memory, but read 4K prompts 4.4% slower. Whether that cost comes from the byte values or from the
  compressed-tensors layout they ship in is open. An in-repo calibrated quantizer writing the
  ModelOpt layout would answer it.
- **GDN FP8.** It improves fidelity at a decode cost that looks like a step rather than a slope, and
  shallow and deep layers protect different capabilities.
- **Prefill path.** The first layer of each kind selects the batched-prefill path, so keep layer 0
  NVFP4 unless you intend to pay the prefill cost.
- **Interactions.** Effects do not add (the GDN × MLP Q4_K interaction is significant), so pairwise
  search matters.

## Where better search wins

- **Surrogates.** Predict RP-KL from assignment features, trained on the growing artifact set.
  Candidates are content-addressed, so results are reusable.
- **Per-category objectives.** Long-context and math respond to different layers.
- **Cost models.** Decode cost is non-linear in FP8 placement, and prefill depends on first-layer
  formats.
- **Quantizer search.** New encoders (GPTQ-style error feedback, scale and clipping search) behind the
  [quantizer contract](quantizer_contract.md), combined with topology.

Surrogates may guide search, but only measured results on the pinned RTX 5090 enter the frontier.
