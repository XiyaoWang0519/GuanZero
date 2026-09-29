#!/bin/zsh
# Usage: run_one_export.sh <name> [outdir]
# Same as run_one.sh, but imports the evaluator from the read-only export of b8c0c54
# (source identity 14e72581...), which the first eight points used.
cd /Users/xiyaowang/Developer/Projects/GuanZero
W=.work/longrun-batched-eval-2026-09-29
E=$W/src-b8c0c54
O=${2:-$W/out}
F=.work/history-budget-2026-09-27/evaluation/freeze.json
s=$(date +%s)
echo "start $(date -u +%FT%TZ) source=export-b8c0c54" > $O/$1.time
PYTHONPATH=$E/python:$E/oracle:$E nice -n 10 .work/external/danlm-venv/bin/python $W/run_point.py $F $W/ckpt/$1.pt $O/$1.json > $O/$1.log 2>&1
echo "exit $? end $(date -u +%FT%TZ) wall_seconds $(( $(date +%s) - s ))" >> $O/$1.time
