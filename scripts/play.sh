#!/usr/bin/env sh
# Play Guandan in the browser against a history checkpoint (eval/play_vs_model.py).
#   ./scripts/play.sh [checkpoint.pt] [extra flags, e.g. --reveal --port 8800]
# Builds the gd_core extension first if it does not import.
set -eu
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
if [ -z "${PY:-}" ]; then
  if [ -x .venv/bin/python ]; then PY=.venv/bin/python; else PY=python3; fi
fi
CKPT=${1:-}
if [ -n "$CKPT" ] && [ "${CKPT#--}" = "$CKPT" ]; then shift; else
  CKPT=.work/longrun-u4902-2026-10-02/download/results/segments/main-u4902/latest.pt
fi
if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT" >&2
  echo "usage: ./scripts/play.sh path/to/checkpoint.pt" >&2
  exit 1
fi
export PYTHONPATH="$ROOT/python:$ROOT"
if ! "$PY" -c 'import gd' 2>/dev/null; then
  echo "building gd_core for $("$PY" -c 'import sys; print(sys.executable)') ..."
  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$("$PY" -c 'import sys; print(sys.executable)')" >/dev/null
  cmake --build build -j 4 --target _gd_core
fi
exec "$PY" -m eval.play_vs_model --checkpoint "$CKPT" "$@"
