#!/usr/bin/env bash
# Seed-frontier measurement on the pinned GPU host (evaluator epoch in configs/hpc01.yaml). Resumable.
#   prerequisites: scripts/setup_sparkinfer.sh · scripts/setup_llamacpp.sh · scripts/setup_models.sh base shipped unsloth gguf
#                  bittrellis reference
# CPU work (build + full audit) runs one candidate ahead of the GPU work (fidelity + performance).
set -euo pipefail
cd "$(dirname "$0")/../.."
A=${ARTIFACTS:-artifacts/seeds}
M=${MODELS_DIR:-models}
V=experiments/feasibility/variants
# Seeds: every candidate that was not dominated (or gate-failed) in epoch e1. V2, V10, V11, V12 were
# dominated by V13 and V8 failed the needle gate; they stay documented in the e1 report.
SEEDS=${SEEDS:-"V0-baseline-rebuild V1-all-q4k V3-gdn-fp8 V4-gdn-q4k V5-attn-q4k V6-mlp-q4k V7-mlp-q4k-early V9-gdn-q4k-mlp-q4k V13-mlp-unsloth-bytes"}
E1=${E1_ARTIFACTS:-results/feasibility-e1/artifacts}
mkdir -p "$A" "$A/_audit"

produce() {
  for name in $SEEDS; do
    out="$M/candidates/$name"
    [ -f "$A/_audit/$name.json" ] && continue
    [ -f "$out/bittrellis_build.json" ] || { rm -rf "$out"; bittrellis build "$V/$name.yaml" --out "$out"; }
    bittrellis audit "$out" --out "$A/_audit/$name.json.tmp" || true
    mv "$A/_audit/$name.json.tmp" "$A/_audit/$name.json"
  done
}
produce > "$A/_audit/producer.log" 2>&1 &
producer=$!

for name in $SEEDS; do
  [ -f "$A/$name/performance.json" ] && continue
  while [ ! -f "$A/_audit/$name.json" ]; do
    kill -0 $producer 2>/dev/null || { [ -f "$A/_audit/$name.json" ] || { echo "producer stopped before $name" >&2; exit 1; }; }
    sleep 10
  done
  out="$M/candidates/$name"
  stages=quality,performance,tasks
  # Reuse the e1 task guard when the checkpoint shards are byte-identical to the ones e1 measured.
  if python - "$E1/$name/candidate.json" "$out/bittrellis_build.json" <<'PY'
import json, sys
try:
    e1 = json.load(open(sys.argv[1]))["build"]["files"]
    now = json.load(open(sys.argv[2]))["files"]
except (OSError, KeyError, ValueError):
    sys.exit(1)
sys.exit(0 if e1 == now else 1)
PY
  then
    mkdir -p "$A/$name"
    python - "$E1/$name/tasks.json" "$A/$name/tasks.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
d["provenance"] = "measured in epoch hpc01-e1 on byte-identical checkpoint shards; task runner unchanged"
json.dump(d, open(sys.argv[2], "w"), indent=2)
PY
    stages=quality,performance
  fi
  bittrellis evaluate "$out" --out "$A/$name" --stages "$stages" --audit-json "$A/_audit/$name.json"
  if [ "$name" = V0-baseline-rebuild ]; then
    python experiments/feasibility/check_reproducible.py --compare-tensors "$out" "$M/Qwen3.8-27B-NVFP4-RTX5090" --out "$A/V0-tensor-identity.json"
    python experiments/feasibility/check_reproducible.py "$A/$name" "$out" --out "$A/$name/reproducibility.json"
  fi
  [ "${KEEP_CANDIDATES:-0}" = 1 ] || rm -rf "$out"
done
wait $producer || true

# External references: context only.
[ -f "$A/R1/performance.json" ] || bittrellis evaluate "$M/Qwen3.8-27B-NVFP4-unsloth" --external R1 --out "$A/R1" --stages quality,performance
[ -f "$A/R2/performance.json" ] || bittrellis evaluate-llamacpp --out "$A/R2"
for r in R1; do [ -f "$E1/$r/tasks.json" ] && [ ! -f "$A/$r/tasks.json" ] && cp "$E1/$r/tasks.json" "$A/$r/tasks.json"; done

python experiments/feasibility/collect.py "$A" results/feasibility/artifacts
python experiments/feasibility/analyze.py results/feasibility/artifacts results/feasibility
