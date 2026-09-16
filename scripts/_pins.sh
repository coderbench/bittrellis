# Sourced by the other scripts: exports pinned values from configs/hpc01.yaml.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python3}"
eval "$("$PY" - "$REPO_ROOT/configs/hpc01.yaml" <<'PYEOF'
import shlex, sys, yaml
t = yaml.safe_load(open(sys.argv[1]))
r2 = t["references"]["R2"]
pins = {
    "BASE_REPO": t["model"]["base"]["repo"], "BASE_REVISION": t["model"]["base"]["revision"],
    "BASELINE_REPO": t["model"]["baseline"]["repo"], "BASELINE_REVISION": t["model"]["baseline"]["revision"],
    "R1_REPO": t["references"]["R1"]["repo"], "R1_REVISION": t["references"]["R1"]["revision"],
    "R2_REPO": r2["repo"], "R2_REVISION": r2["revision"], "R2_FILE": r2["file"],
    "LLAMACPP_COMMIT": r2["llama_cpp_commit"],
    "SPARKINFER_REPO": t["runtime"]["repo"], "SPARKINFER_COMMIT": t["runtime"]["commit"],
    "SPARKINFER_CMAKE_ARGS": " ".join(t["runtime"]["cmake_args"]),
    "SPARKINFER_TARGETS": " ".join(t["runtime"]["targets"]),
}
for k, v in pins.items():
    print(f"export PIN_{k}={shlex.quote(str(v))}")
PYEOF
)"
export MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/models}"
export SPARKINFER_DIR="${BITTRELLIS_SPARKINFER:-$REPO_ROOT/third_party/sparkinfer}"
export LLAMACPP_DIR="${BITTRELLIS_LLAMACPP:-$REPO_ROOT/third_party/llama.cpp}"
