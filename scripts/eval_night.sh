#!/bin/zsh
# Nightly evaluation of one training segment (October 3, 2026 yardsticks).
#   scripts/eval_night.sh SEGMENT_DIR NAME BASELINE.pt [CURVE_STRIDE=4] [CURVE_DEALS=512]
# SEGMENT_DIR holds update-*.pt and latest.pt. Writes .work/nightly-eval/NAME/.
#   endpoint: 2,000 deals vs BASELINE (the previous lineage endpoint; internal yardstick),
#             4,000 deals vs DanLM (external yardstick), 256 deals vs B11 (sanity line only);
#   curve:    every CURVE_STRIDE-th checkpoint, CURVE_DEALS deals vs BASELINE.
# Skips points whose output exists. Batched MPS backend for the Transformer-vs-Transformer
# points (65 s per 256 deals); DanLM arena 12 workers (~9 min).
set -u
cd /Users/xiyaowang/Developer/Projects/GuanZero
seg=$1; name=$2; base=$3; stride=${4:-4}; curve_deals=${5:-512}
PY=.work/external/danlm-venv/bin/python
W=.work/nightly-eval/$name; mkdir -p $W/curve
export PYTHONPATH=python:oracle:.
upd() { $PY -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['progress']['updates'])" $1 2>/dev/null; }
latest=$seg/latest.pt; u=$(upd $latest)
echo "$name endpoint u$u vs $(basename $(dirname $base))/$(basename $base)"
[ -f $W/endpoint-vs-baseline.json ] || $PY -m eval.lineage_eval --candidate $latest --baseline $base --deals 2000 --device mps --output $W/endpoint-vs-baseline.json 2>&1 | grep -v Warning | tail -1
[ -f $W/endpoint-vs-danlm.json ] || DANLM_ROOT=.work/external/DanLM $PY -m eval.danlm.arena duplicate --checkpoint $latest --deals 4000 --seed 20260929 --workers 12 --output $W/endpoint-vs-danlm.json > $W/endpoint-vs-danlm.log 2>&1
[ -f $W/endpoint-vs-b11.json ] || $PY -m eval.history_frozen --freeze .work/longrun-batched-eval-2026-09-29/b11only/freeze.b11-only.json --candidate $latest --output $W/endpoint-vs-b11.json --backend batched --device mps --threads 2 > $W/endpoint-vs-b11.log 2>&1
i=0
for ckpt in $(ls $seg/update-*.pt | sort); do
  i=$((i+1)); [ $((i % stride)) -eq 0 ] || continue
  cu=$(upd $ckpt); [ -n "$cu" ] || continue
  [ -f $W/curve/u$cu.json ] || $PY -m eval.lineage_eval --candidate $ckpt --baseline $base --deals $curve_deals --device mps --output $W/curve/u$cu.json 2>&1 | grep -v Warning | tail -1
done
$PY scripts/eval_night_summary.py $W
