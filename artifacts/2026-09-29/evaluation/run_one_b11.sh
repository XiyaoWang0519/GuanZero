#!/bin/zsh
# Usage: run_one_b11.sh <name> [concurrency_note]
# B11-only evaluation with the read-only export of b8c0c54 (source identity 14e72581...)
# and the derived freeze b11only/freeze.b11-only.json (b11-main only; otherwise identical).
# Claims each point with an atomic mkdir lock so several lanes never run the same point.
cd /Users/xiyaowang/Developer/Projects/GuanZero
W=.work/longrun-batched-eval-2026-09-29
E=$W/src-b8c0c54
O=$W/out-b11
F=$W/b11only/freeze.b11-only.json
mkdir -p $O/claims
if ! mkdir $O/claims/$1 2>/dev/null || [ -f $O/$1.time ]; then echo "skip $1 (already claimed)"; exit 0; fi
s=$(date +%s)
echo "start $(date -u +%FT%TZ) source=export-b8c0c54 freeze=b11-only concurrent=${2:-?}" > $O/$1.time
PYTHONPATH=$E/python:$E/oracle:$E nice -n 10 .work/external/danlm-venv/bin/python $W/run_point.py $F $W/ckpt/$1.pt $O/$1.json > $O/$1.log 2>&1
echo "exit $? end $(date -u +%FT%TZ) wall_seconds $(( $(date +%s) - s ))" >> $O/$1.time
