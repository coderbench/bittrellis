#!/usr/bin/env bash
# Build llama.cpp at the pinned commit plus tools/llamacpp_score.cpp (external reference R2 only).
set -euo pipefail
source "$(dirname "$0")/_pins.sh"
if [ ! -d "$LLAMACPP_DIR/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$LLAMACPP_DIR"
fi
git -C "$LLAMACPP_DIR" fetch --quiet origin
git -C "$LLAMACPP_DIR" checkout --quiet --detach "$PIN_LLAMACPP_COMMIT"
cmake -S "$LLAMACPP_DIR" -B "$LLAMACPP_DIR/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_SERVER=ON
cmake --build "$LLAMACPP_DIR/build" -j"$(nproc)" --target llama llama-bench llama-server
g++ -O2 -std=c++17 "$REPO_ROOT/tools/llamacpp_score.cpp" -I "$LLAMACPP_DIR/include" -I "$LLAMACPP_DIR/ggml/include" \
  -L "$LLAMACPP_DIR/build/bin" -lllama -lggml -lggml-base -Wl,-rpath,"$LLAMACPP_DIR/build/bin" \
  -o "$LLAMACPP_DIR/build/bin/llamacpp_score"
echo "llama.cpp $PIN_LLAMACPP_COMMIT built in $LLAMACPP_DIR/build/bin"
