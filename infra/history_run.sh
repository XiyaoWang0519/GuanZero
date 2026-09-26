#!/usr/bin/env bash
set -euo pipefail
cd /workspace/GuanZero
: "${POD_DEADLINE_EPOCH:?pass the owned provider manifest deadline}"
export PYTHONPATH="$PWD/python:$PWD/oracle:$PWD"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export NVIDIA_TF32_OVERRIDE=0
# Separate provider guard and local 60-second artifact sync must already run.
exec python -m infra.watchdog --budget-usd 1.2 --hourly-rate-usd 0.8 \
  --max-hours 0.75 --grace-seconds 30 \
  --log-file /workspace/results/watchdog.jsonl -- \
  python -m infra.history_pilot remote --manifest /workspace/payload/run-manifest.json \
  --output /workspace/results --deadline-epoch "$POD_DEADLINE_EPOCH"
