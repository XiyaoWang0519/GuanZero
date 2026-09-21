#!/usr/bin/env sh
# Everything CI runs. Fails on the first error.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PY:-.venv/bin/python}

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build -j
./build/cpp/tests/gd_tests
"$PY" oracle/test_gd_reference.py
"$PY" -m pytest -q tests
