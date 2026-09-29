#!/bin/zsh
# Verify u2201 once it lands, then refresh results.v2 whenever a new point finishes.
setopt nullglob
W=/Users/xiyaowang/Developer/Projects/GuanZero/.work/longrun-batched-eval-2026-09-29
E=$W/src-b8c0c54
PY=/Users/xiyaowang/Developer/Projects/GuanZero/.work/external/danlm-venv/bin/python
last=""
while true; do
  if [ -f $W/out-b11/u2201.json ] && [ ! -f $W/b11only/VERDICT ]; then
    $PY $W/b11only/verify.py > $W/b11only/verify.log 2>&1 || echo FAIL-ERROR > $W/b11only/VERDICT
  fi
  cur="$(find $W/out-b11 $W/out-full -name 'u*.json' 2>/dev/null | sort) $(cat $W/b11only/VERDICT 2>/dev/null)"
  if [ "$cur" != "$last" ]; then
    PYTHONPATH=$E/python:$E/oracle:$E $PY $W/summarize_v2.py > $W/summarize_v2.log 2>&1
    last="$cur"
  fi
  n=$(find $W/out-b11 -name 'u*.json' | wc -l)
  [ $n -ge 6 ] && break
  sleep 30
done
echo done
