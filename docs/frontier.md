# Frontier and Frontier Gain

## Objectives (HPC-01)

| | Objective | Direction | Normalization box |
|---|---|---|---|
| Quality | KL(BF16 ‖ candidate) | lower | 0.00 – 0.30 nats/token |
| Speed | decode tok/s @ 4K, batch 1 | higher | 60 – 120 tok/s |
| Footprint | peak VRAM during the 128/4K/16K sweep | lower | 20 – 32 GiB |

Prefill, the per-category KL, the tail and the task scores are reported but not ranked.

## Dominance

A result is **dominated** if another *valid* result has KL ≤, decode ≥ and VRAM ≤, with at least
one strict. The **frontier** is every valid result that nothing dominates.

```text
 KL ▲                          a trade-off is progress, not a loss:
    │   ×  dominated
    │                          ● R0 ships today
    │       ●R0                ◆ new map, faster but slightly less exact
    │              ◆           ★ new map, better on every axis
    │    ★
    └────────────────────▶ decode tok/s
```

## Frontier Gain (FG-1)

Map every valid result into the unit cube, with 1 as the best edge of each box axis. The **dominated
hypervolume** of a set is the volume of the union of boxes `[0, point]`.

```text
FG-1(candidate) = HV(frontier ∪ {candidate}) − HV(frontier)
```

- A dominated or invalid result earns exactly **0**.
- A result earns gain for the *new* space it opens, whichever axis that is.
- Gains add up: the frontier's total hypervolume only ever grows.
- It rewards effect, not effort. There are no XS/S/M/L/XL size labels.

`bittrellis frontier artifacts/*` prints each row's gates, frontier membership and its marginal
FG-1 against every other valid row. The box and the formula are versioned with the track;
changing either creates a new version.

## Reference points are on the same chart

R0 (SparkInfer's shipped checkpoint), R1 (unsloth's mixed map) and R2 (llama.cpp's
UD-Q4_K_M) are measured by the same harness. They sit in the frontier like any candidate. Beating
them is the point; a candidate that only beats another candidate has not moved anything.
