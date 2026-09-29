#!/bin/zsh
# Usage: lane_b11.sh <first> <rest...>
# Runs <first> B11-only, then waits for b11only/VERDICT (PASS -> B11-only, else full two-baseline freeze into out-full/).
W=/Users/xiyaowang/Developer/Projects/GuanZero/.work/longrun-batched-eval-2026-09-29
$W/run_one_b11.sh $1 2
shift
while [ ! -f $W/b11only/VERDICT ]; do sleep 20; done
for n in "$@"; do
  if grep -q PASS $W/b11only/VERDICT; then $W/run_one_b11.sh $n 2
  else mkdir -p $W/out-full; $W/run_one_export.sh $n $W/out-full; fi
done
