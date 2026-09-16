#!/usr/bin/env bash
# Phase 1–2 feasibility experiment on the pinned GPU host. Resumable: finished stages are skipped.
#   prerequisites: scripts/setup_sparkinfer.sh, scripts/setup_models.sh base baseline r1 r2,
#                  scripts/setup_llamacpp.sh, bittrellis reference
set -euo pipefail
cd "$(dirname "$0")/../.."
A=${ARTIFACTS:-artifacts/feasibility}
M=${MODELS_DIR:-models}
mkdir -p "$A"

done_() { [ -f "$1/performance.json" ] && { [ "${SKIP_TASKS:-0}" = 1 ] || [ -f "$1/tasks.json" ]; }; }
stages() { [ "${SKIP_TASKS:-0}" = 1 ] && echo quality,performance || echo quality,performance,tasks; }

# 1. Baseline R0 and its determinism check.
done_ "$A/R0" || bittrellis evaluate "$M/Qwen3.8-27B-NVFP4-RTX5090" --reference-id R0 --out "$A/R0" --stages "$(stages)"
python experiments/feasibility/check_reproducible.py "$A/R0" "$M/Qwen3.8-27B-NVFP4-RTX5090" --out "$A/R0/reproducibility.json"

# 2. Variants: build (CPU) → audit + evaluate (GPU) → drop the weights, keep the artifacts.
for v in experiments/feasibility/variants/*.yaml; do
  name=$(basename "$v" .yaml)
  done_ "$A/$name" && continue
  out="$M/candidates/$name"
  [ -f "$out/bittrellis_build.json" ] || { rm -rf "$out"; bittrellis build "$v" --out "$out" --unsloth "$M/Qwen3.8-27B-NVFP4-unsloth"; }
  if [ "$name" = V0-baseline-rebuild ]; then
    python experiments/feasibility/check_reproducible.py --compare-tensors "$out" "$M/Qwen3.8-27B-NVFP4-RTX5090" --out "$A/V0-tensor-identity.json"
  fi
  bittrellis evaluate "$out" --out "$A/$name" --stages "$(stages)"
  [ "${KEEP_CANDIDATES:-0}" = 1 ] || rm -rf "$out"
done

# 3. External reference points.
done_ "$A/R1" || bittrellis evaluate "$M/Qwen3.8-27B-NVFP4-unsloth" --reference-id R1 --out "$A/R1" --stages "$(stages)"
if [ -x third_party/llama.cpp/build/bin/llamacpp_score ] && [ ! -f "$A/R2/performance.json" ]; then
  bittrellis evaluate-llamacpp --out "$A/R2"
fi

# 4. Frontier + report inputs.
bittrellis report "$A" --out results/feasibility
