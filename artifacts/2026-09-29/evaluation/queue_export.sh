#!/bin/zsh
# Usage: queue_export.sh <name>...  (export-based evaluations, then refresh results)
cd /Users/xiyaowang/Developer/Projects/GuanZero
W=.work/longrun-batched-eval-2026-09-29
for n in "$@"; do
  $W/run_one_export.sh $n
  PYTHONPATH=python:oracle:. .work/external/danlm-venv/bin/python $W/summarize.py > $W/summarize-after-$n.log 2>&1
done
