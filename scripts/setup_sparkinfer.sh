#!/usr/bin/env bash
# Clone SparkInfer at the pinned commit, build the targets BitTrellis drives, and compile
# tools/sparkinfer_refscore.cpp against the unmodified runtime library.
# Needs CUDA 12.8+, CMake >= 3.20, a C++17 compiler, and rustc >= 1.79 (server tokenizer).
# Tip: keep RUSTUP_HOME/CARGO_HOME off FUSE/encrypted mounts; cargo's archiver fails on some of them.
set -euo pipefail
source "$(dirname "$0")/_pins.sh"
if [ ! -d "$SPARKINFER_DIR/.git" ]; then
  git clone "https://github.com/$PIN_SPARKINFER_REPO.git" "$SPARKINFER_DIR"
fi
git -C "$SPARKINFER_DIR" fetch --quiet origin
git -C "$SPARKINFER_DIR" checkout --quiet --detach "$PIN_SPARKINFER_COMMIT"
if [ -n "$(git -C "$SPARKINFER_DIR" status --porcelain --untracked-files=no)" ]; then
  echo "SparkInfer checkout has local modifications; refusing to build" >&2; exit 1
fi
# shellcheck disable=SC2086
cmake -S "$SPARKINFER_DIR" -B "$SPARKINFER_DIR/build" $PIN_SPARKINFER_CMAKE_ARGS
# shellcheck disable=SC2086
cmake --build "$SPARKINFER_DIR/build" -j"$(nproc)" --target $PIN_SPARKINFER_TARGETS

B="$SPARKINFER_DIR/build"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# The toolkit is part of the pinned environment. Building against whatever the host happens to carry
# would change measurements without changing a pin, so say so rather than quietly produce numbers.
_cuda_want=$(python3 -c "import yaml;print(yaml.safe_load(open('$REPO_ROOT/configs/hpc01.yaml'))['runtime']['cuda'])" 2>/dev/null || echo "")
_cuda_have=$("$CUDA_HOME/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p')
if [ -n "$_cuda_want" ] && [ "$_cuda_have" != "$_cuda_want" ]; then
  echo "CUDA $_cuda_want is pinned but CUDA_HOME=$CUDA_HOME is ${_cuda_have:-not a CUDA toolkit}." >&2
  echo "Point CUDA_HOME at the pinned toolkit (e.g. /usr/local/cuda-$_cuda_want) or re-pin the track." >&2
  exit 1
fi
RT_LIB_DIR=$(dirname "$(find "$B" -name 'libsparkinfer_runtime.so' | head -1)")
MOE_LIB_DIR=$(dirname "$(find "$B" -name 'libsparkinfer_moe.so' | head -1)")
g++ -O2 -std=c++17 "$REPO_ROOT/tools/sparkinfer_refscore.cpp" \
  -I "$SPARKINFER_DIR/runtime/include" -I "$SPARKINFER_DIR/runtime/examples" \
  -I "$SPARKINFER_DIR/moe/include" -I "$SPARKINFER_DIR/kernels/include" \
  -I "$B/_deps/nlohmann_json-src/include" -I "$CUDA_HOME/include" \
  -L "$RT_LIB_DIR" -L "$MOE_LIB_DIR" -L "$CUDA_HOME/lib64" \
  -lsparkinfer_runtime -lsparkinfer_moe -lcudart -lpthread \
  -Wl,-rpath,"$RT_LIB_DIR:$MOE_LIB_DIR:$CUDA_HOME/lib64" \
  -o "$B/bittrellis_refscore"
echo "SparkInfer $PIN_SPARKINFER_COMMIT built; scorer at $B/bittrellis_refscore"
