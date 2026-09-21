#!/usr/bin/env sh
# Everything CI runs. Fails on the first error.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PY:-.venv/bin/python}

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$ROOT/.venv/bin/python" -DGD_BUILD_FUZZ=ON -DGD_BUILD_BENCH=ON >/dev/null
cmake --build build -j

./build/cpp/tests/gd_tests
"$PY" oracle/test_gd_reference.py
PYTHONPATH="$ROOT/python:$ROOT/oracle:$ROOT" "$PY" -m pytest -q tests
./build/cpp/fuzz/gd_fuzz --rounds 20000 --threads 4 --deep-every 1
./build/cpp/fuzz/gd_fuzz --rounds 20000 --threads 4 --deep-every 1 --full
echo "all checks passed"
