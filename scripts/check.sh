#!/usr/bin/env sh
# Everything CI runs. Fails on the first error.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PY:-.venv/bin/python}
PY_EXECUTABLE=$("$PY" -c 'import sys; print(sys.executable)')

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$PY_EXECUTABLE" -DGD_BUILD_FUZZ=ON -DGD_BUILD_BENCH=ON >/dev/null
cmake --build build -j

./build/cpp/tests/gd_tests
"$PY" oracle/test_gd_reference.py
# Bound worker and BLAS thread counts: -n auto oversubscribes smaller hosts
# and can starve the watchdog's real-time subprocess tests. Set 0 to debug.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH="$ROOT/python:$ROOT/oracle:$ROOT" \
  "$PY" -m pytest -q tests -n "${PYTEST_WORKERS:-4}" --dist=worksteal
./build/cpp/fuzz/gd_fuzz --rounds 20000 --threads 4 --deep-every 1
./build/cpp/fuzz/gd_fuzz --rounds 20000 --threads 4 --deep-every 1 --full
./build/cpp/fuzz/gd_fuzz --rounds 20000 --threads 4 --deep-every 1 --driver styled
echo "all checks passed"
