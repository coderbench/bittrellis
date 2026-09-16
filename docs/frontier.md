# Frontier and Frontier Gain

## Objectives (HPC-01)

| | Objective | Direction | Normalization box |
|---|---|---|---|
| Objective | Measure | Direction | Noise ε | Normalization box |
|---|---|---|---:|---|
| Quality | KL(BF16 ‖ candidate) | lower | 0.005 | 0.00 – 0.30 nats/token |
| Decode | tok/s @ 4K, batch 1 | higher | 1.0 | 60 – 120 tok/s |
| Prefill | tok/s @ 4K | higher | 450 | 2,000 – 20,000 tok/s |
| Footprint | peak VRAM over the 128/4K/16K sweep | lower | 0.1 GiB | 14 – 32 GiB |

Prefill is an objective, not a footnote. Several maps in the feasibility run save VRAM at equal
KL and decode by giving up 20–50% of prefill throughput. On three axes that looks like a free
win; on four it is the trade-off it really is.

The per-category KL, the tail and the task scores are reported but not ranked.

## Dominance

A result is **dominated** if another *valid* result is not worse than it by more than ε on any
objective, and better by more than ε on at least one. The **frontier** is every valid result that
nothing dominates. ε is each objective's measured noise: the two-rep spread of decode (±0.1%) and
prefill (±2%), and a KL margin below any significant paired difference. So a lucky rerun can't
push a result onto the frontier.

```text
 KL ▲                          a trade-off is progress, not a loss:
    │   ×  dominated
    │                          ● R0 ships today
    │       ●R0                ◆ new map, faster but slightly less exact
    │              ◆           ★ new map, better on every axis
    │    ★
    └────────────────────▶ decode tok/s
```

## Frontier Gain (FG-2)

Map every valid result into the 4-D unit box, with 1 as the best edge of each box axis. The **dominated
hypervolume** of a set is the volume of the union of boxes `[0, point]`.

```text
FG-2(candidate) = HV(frontier ∪ {candidate}) − HV(frontier)
```

- A dominated or invalid result earns exactly **0**.
- A result earns gain for the *new* space it opens, whichever axis that is.
- Gains add up: the frontier's total hypervolume only ever grows.
- It rewards effect, not effort. There are no XS/S/M/L/XL size labels.

`bittrellis frontier artifacts/*` prints each row's gates, frontier membership and its marginal
FG-2 against every other valid row. The box and the formula are versioned with the track;
changing either creates a new version.

## Reference points are on the same chart

R0 (SparkInfer's shipped checkpoint), R1 (unsloth's mixed map) and R2 (llama.cpp's
UD-Q4_K_M) are measured by the same harness. They sit in the frontier like any candidate. Beating
them is the point; a candidate that only beats another candidate has not moved anything.
