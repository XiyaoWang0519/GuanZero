#!/bin/zsh
# Usage: run_one.sh <name>   (evaluates ckpt/<name>.pt with the unchanged reference run_point.py)
cd /Users/xiyaowang/Developer/Projects/GuanZero
W=.work/longrun-batched-eval-2026-09-29
F=.work/history-budget-2026-09-27/evaluation/freeze.json
s=$(date +%s)
echo "start $(date -u +%FT%TZ)" > $W/out/$1.time
PYTHONPATH=python:oracle:. nice -n 10 .work/external/danlm-venv/bin/python $W/run_point.py $F $W/ckpt/$1.pt $W/out/$1.json > $W/out/$1.log 2>&1
echo "exit $? end $(date -u +%FT%TZ) wall_seconds $(( $(date +%s) - s ))" >> $W/out/$1.time
