#!/bin/zsh
# Usage: queue.sh <name>...   runs run_one.sh for each name in turn, then refreshes results.
cd /Users/xiyaowang/Developer/Projects/GuanZero
W=.work/longrun-batched-eval-2026-09-29
for n in "$@"; do
  $W/run_one.sh $n
  PYTHONPATH=python:oracle:. .work/external/danlm-venv/bin/python $W/summarize.py > $W/out/summarize-after-$n.log 2>&1
done
