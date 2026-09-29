#!/usr/bin/env bash
# Build the frozen evaluator source (git b8c0c54, identity 14e72581...) on the pod and
# check it. Any failure exits non-zero: remote_start.sh then writes FAILED.json + DONE
# and no evaluation runs. No credentials used.
set -euo pipefail
K=/workspace/kit
R=/workspace/results
S=/workspace/src
V=/workspace/venv
mkdir -p "$S" "$R"
BASE=/venv/main/bin/python
[ -x "$BASE" ] || BASE=$(command -v python3)
read_expected() { "$BASE" -c "import json,sys; print(json.load(open('$K/expected.json'))[sys.argv[1]])" "$1"; }
TORCH_V=$(read_expected torch_version)
NUMPY_V=$(read_expected numpy_version)
SRC_SHA=$(read_expected source_sha256)
ENGINE=$(read_expected engine_digest)
FREEZE=$(read_expected freeze_pod_path)

tar -xzf "$K/source.tar.gz" -C "$S"

# Python: the local evaluation's torch/numpy versions (CPU wheels) in a fresh venv;
# if that cannot be installed, the image's own environment (recorded in env.json).
MATCHED=false
if "$BASE" -m venv "$V" > "$R/venv.log" 2>&1 \
   && "$V/bin/python" -m pip install --disable-pip-version-check -q "torch==$TORCH_V" \
        --index-url https://download.pytorch.org/whl/cpu >> "$R/venv.log" 2>&1 \
   && "$V/bin/python" -m pip install --disable-pip-version-check -q "numpy==$NUMPY_V" \
        "pybind11>=2.12" >> "$R/venv.log" 2>&1; then
  PY="$V/bin/python"; MATCHED=true
else
  echo "exact torch $TORCH_V / numpy $NUMPY_V venv failed; using the image environment" | tee -a "$R/venv.log"
  PY="$BASE"
  "$PY" -m pip install --disable-pip-version-check -q "pybind11>=2.12" >> "$R/venv.log" 2>&1
fi
echo "$PY" > "$R/python-path.txt"
if ! command -v g++ >/dev/null; then apt-get update -qq && apt-get install -y -qq --no-install-recommends build-essential; fi
need=()
command -v cmake >/dev/null || need+=("cmake>=3.24")
command -v ninja >/dev/null || need+=("ninja")
if [ ${#need[@]} -gt 0 ]; then "$BASE" -m pip install --disable-pip-version-check -q "${need[@]}"; fi
export PATH="$(dirname "$BASE"):$PATH"
"$PY" -m pip freeze > "$R/pip-freeze.txt"

cd "$S"
export PYTHONPATH="$S/python:$S/oracle:$S" CUDA_VISIBLE_DEVICES=""
# Engine digest and source identity before the build (receipt = source-identity.json).
"$PY" - "$SRC_SHA" "$ENGINE" <<'PYEOF'
import sys
from infra.history_artifacts import source_identity, engine_digest
s, e = source_identity(), engine_digest()
print("source", s["source_sha256"], "revision", s["revision"], "engine", e)
assert s["source_sha256"] == sys.argv[1], "source identity mismatch"
assert e == sys.argv[2], "engine digest mismatch with the locally recorded one"
PYEOF
JOBS=$("$PY" -c 'import os; print(max(1, min(16, len(os.sched_getaffinity(0)))))')
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PY" > "$R/cmake.log" 2>&1
cmake --build build -j "$JOBS" >> "$R/cmake.log" 2>&1
timeout 900 ./build/cpp/tests/gd_tests > "$R/gd-tests.log" 2>&1
echo "gd_tests ok"
sha256sum python/gd/*.so > "$R/engine-so-sha256.txt"

# Evaluation assets at the absolute paths freeze.json names (freeze sha unchanged).
"$PY" - "$K" <<'PYEOF'
import json, os, sys
from pathlib import Path
kit = Path(sys.argv[1])
expected = json.loads((kit / "expected.json").read_text())
for name, meta in expected["assets"].items():
    target = Path(meta["pod"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        target.unlink()
    target.symlink_to(kit / "assets" / name)
    print("placed", name, "->", target)
PYEOF

# Import, identity after the build, and one checkpoint load from the packaged layout.
cd /workspace
"$PY" - "$SRC_SHA" "$ENGINE" "$FREEZE" "$K" "$MATCHED" "$R/env.json" <<'PYEOF'
import json, platform, sys, torch, numpy
from pathlib import Path
import gd
from eval.history_frozen import evaluate  # noqa: F401  (import check)
from eval.policies import load_policy
from infra.history_artifacts import source_identity, engine_digest, sha256
src, eng, freeze, kit, matched, out = sys.argv[1:7]
s = source_identity()
assert s["source_sha256"] == src, "source identity changed by the build"
assert engine_digest() == eng
assert sha256(Path(freeze)) == json.loads((Path(kit) / "expected.json").read_text())["freeze_sha256"]
torch.set_num_threads(4)
policy = load_policy(str(Path(kit) / "ckpt" / "u2623.pt"), "cpu")
assert getattr(policy, "stage", None) == "history_ppo"
env = dict(python=sys.version, platform=platform.platform(), machine=platform.machine(),
           torch=torch.__version__, numpy=numpy.__version__, matched_local_versions=matched == "true",
           gd=gd.__file__, source_sha256=s["source_sha256"], revision=s.get("revision"),
           engine_digest=engine_digest(), torch_parallel_info=torch.__config__.parallel_info(),
           torch_config=torch.__config__.show())
Path(out).write_text(json.dumps(env, indent=2) + "\n")
print(json.dumps({k: env[k] for k in ("python", "machine", "torch", "numpy", "matched_local_versions")}))
PYEOF
echo "setup ok"
