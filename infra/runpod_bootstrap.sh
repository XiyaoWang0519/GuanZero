#!/usr/bin/env bash
# Prepare an uploaded source tree inside the official RunPod PyTorch image.
# Credentials are neither read nor installed by this script.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PY:-python}
export PYTHONPATH="$ROOT/python:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - <<'PY'
import json, os, sys, torch
if sys.version_info < (3, 11):
    raise SystemExit("Python 3.11+ is required")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing a CPU fallback")
device = torch.cuda.get_device_properties(0)
print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__,
                  "cuda_runtime": torch.version.cuda, "gpu": device.name,
                  "gpu_memory_gib": device.total_memory / 2**30,
                  "cpu_count": os.cpu_count(), "bf16": torch.cuda.is_bf16_supported()}))
PY
if ! command -v g++ >/dev/null || ! command -v rsync >/dev/null; then
  apt-get update
  apt-get install -y --no-install-recommends build-essential rsync
fi
"$PY" -m pip install --disable-pip-version-check -r requirements-dev.txt \
  -r requirements-train.txt 'cmake>=3.24' ninja
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v "$PY")" -DGD_BUILD_FUZZ=ON
cmake --build build -j "${BUILD_JOBS:-8}"
./build/cpp/tests/gd_tests
"$PY" -m pytest -q tests/test_training.py tests/test_training_env.py \
  tests/test_belief_probe.py tests/test_evaluation.py tests/test_watchdog.py
PY="$PY" ./scripts/preflight.sh cuda runs/gpu-preflight
