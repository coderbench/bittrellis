#!/usr/bin/env bash
# Clone SparkInfer at the pinned commit and build the tools BitTrellis drives (RTX 5090, sm_120).
# Needs CUDA 12.8+, CMake >= 3.20, a C++17 compiler and rustc >= 1.79 (for the server's tokenizer).
set -euo pipefail
source "$(dirname "$0")/_pins.sh"
if [ ! -d "$SPARKINFER_DIR/.git" ]; then
  git clone "https://github.com/$PIN_SPARKINFER_REPO.git" "$SPARKINFER_DIR"
fi
git -C "$SPARKINFER_DIR" fetch --quiet origin
git -C "$SPARKINFER_DIR" checkout --quiet --detach "$PIN_SPARKINFER_COMMIT"
# shellcheck disable=SC2086
cmake -S "$SPARKINFER_DIR" -B "$SPARKINFER_DIR/build" $PIN_SPARKINFER_CMAKE_ARGS
# shellcheck disable=SC2086
cmake --build "$SPARKINFER_DIR/build" -j"$(nproc)" --target $PIN_SPARKINFER_TARGETS
echo "SparkInfer $PIN_SPARKINFER_COMMIT built in $SPARKINFER_DIR/build"
