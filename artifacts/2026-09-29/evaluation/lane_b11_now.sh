#!/bin/zsh
# Usage: lane_b11_now.sh <names...>  B11-only, no verdict gate (provisional until u2201 verification passes).
W=/Users/xiyaowang/Developer/Projects/GuanZero/.work/longrun-batched-eval-2026-09-29
for n in "$@"; do $W/run_one_b11.sh $n 4; done
