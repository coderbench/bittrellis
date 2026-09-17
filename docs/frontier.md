# Frontier and Frontier Gain (FG-2)

## Internal vs external

| | Participates in dominance and FG-2 | Examples |
|---|---|---|
| **Internal** | yes | legal BitTrellis manifests on the pinned SparkInfer: V0 (the incumbent), the seed candidates, accepted miner results |
| **External** | no, shown beside the frontier for context | R1 unsloth checkpoint · R2 llama.cpp UD-Q4_K_M · R3 NVIDIA (compatibility boundary) |

The frontier never starts empty. V0 is the shipped Gittensor checkpoint rebuilt byte for byte from
a manifest. Every seed candidate measured by maintainers (`results/feasibility/artifacts`) is also
part of it, so a miner result must beat those, and resubmitting one earns nothing.

## Objectives

| Objective | Measure | Direction | Materially better when | Box |
|---|---|---|---|---|
| Fidelity | RP-KL vs BF16 | lower | lower by more than 0.002 **and** the paired 95% interval excludes 0 | 0 – 0.30 |
| Decode | tok/s @ 4K, batch 1, mean of 2 runs | higher | higher by more than max(1%, either row's two-run spread) | 60 – 120 |
| Prefill | tok/s @ 4K, mean of 2 runs | higher | higher by more than max(3%, either row's two-run spread) | 2,000 – 20,000 |
| Memory | peak GPU GiB over a run | lower | lower by more than 0.1 GiB | 14 – 32 |

**Dominance.** A dominates B when A is materially better on at least one objective and materially
worse on none. The frontier is every valid internal row that no other valid internal row
dominates.

**Why prefill is an objective.** Several maps save memory at equal fidelity and decode by giving up
20–50% of prefill throughput. On three axes that looks like a free win; on four it is the trade-off
it really is.

## FG-2

Map each valid internal row into the 4-D unit box, with 1 as the best edge of each axis, and take
the dominated hypervolume, the volume of the union of boxes `[0, point]`:

```text
FG-2(candidate) = HV(frontier ∪ {candidate}) − HV(frontier)
```

- Dominated or invalid rows earn 0. External rows always earn 0.
- Gain rewards new operating room on any axis: a slower but much more faithful map earns gain just
  like a faster one.
- There are no manual size tiers. The formula, box and floors are versioned with the evaluator epoch.

```bash
bittrellis frontier --with-seeds artifacts/mine      # rank a result against the seeds
```
