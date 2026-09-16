#!/usr/bin/env bash
# Download the pinned checkpoints into $MODELS_DIR.
#   base      Qwen/Qwen3.8-27B (BF16, 52 GiB)            -- needed to build candidates and the reference
#   baseline  gittensor NVFP4 (17 GiB)                    -- needed to build candidates; reference R0
#   r1        unsloth NVFP4 (22 GiB)                      -- reference R1 (validators only)
#   r2        unsloth UD-Q4_K_M GGUF (15 GiB)             -- reference R2 (validators only)
# Usage: scripts/setup_models.sh [base] [baseline] [r1] [r2]    (default: base baseline)
set -euo pipefail
source "$(dirname "$0")/_pins.sh"
command -v hf >/dev/null || pip install -q "huggingface_hub[cli]>=0.34"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
want=("$@"); [ ${#want[@]} -eq 0 ] && want=(base baseline)
for w in "${want[@]}"; do
  case "$w" in
    base)     hf download "$PIN_BASE_REPO" --revision "$PIN_BASE_REVISION" --local-dir "$MODELS_DIR/Qwen3.8-27B" --max-workers 16 ;;
    baseline) hf download "$PIN_BASELINE_REPO" --revision "$PIN_BASELINE_REVISION" --local-dir "$MODELS_DIR/Qwen3.8-27B-NVFP4-RTX5090" --exclude "assets/*" ;;
    r1)       hf download "$PIN_R1_REPO" --revision "$PIN_R1_REVISION" --local-dir "$MODELS_DIR/Qwen3.8-27B-NVFP4-unsloth" ;;
    r2)       hf download "$PIN_R2_REPO" "$PIN_R2_FILE" --revision "$PIN_R2_REVISION" --local-dir "$MODELS_DIR/Qwen3.8-27B-GGUF" ;;
    *) echo "unknown model '$w'" >&2; exit 2 ;;
  esac
done
