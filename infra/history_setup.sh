#!/usr/bin/env bash
# Uploaded source root; no credentials, datasets or legacy preflight.
set -euo pipefail
cd /workspace/GuanZero
mkdir -p /workspace/results
export PYTHONPATH="$PWD/python:$PWD/oracle:$PWD"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -c 'import torch; assert torch.cuda.is_available(), "CUDA required"'
if ! command -v g++ >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends build-essential
fi
python -m pip install --disable-pip-version-check numpy pybind11 pytest pytest-xdist hypothesis 'cmake>=3.24' ninja
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$(command -v python)"
cmake --build build -j 4
./build/cpp/tests/gd_tests
python -m infra.cpu_budget --facts /workspace/results/setup-host-facts.json
