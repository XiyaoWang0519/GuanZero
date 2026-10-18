#!/usr/bin/env bash
# Run inside a separate allocated CUDA session, not the production run directory.
# No sbatch/srun submission, package installation, checkpoint copying, or GPU reset.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != "--ppo" ) ]]; then
    echo "Usage: bash scripts/benchmark-paged-attention.sh NEW_OUTPUT_DIRECTORY [--ppo]" >&2
    exit 2
fi
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-python}
export PYTHONPATH="$ROOT/python:$ROOT/oracle:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
# This script deliberately uses logical CUDA device 0 from the existing allocation.
# Do not broaden the scheduler's CUDA_VISIBLE_DEVICES to reach other users' GPUs.
"$PYTHON" - <<'PY'
import torch
import gd._gd_core
from train.history_paged_attention import require_triton
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU. Use an allocated compute session, not a login node.")
require_triton(torch.device("cuda:0"), torch.float32, 16)
x = torch.ones(1, device="cuda:0")
torch.cuda.synchronize(0)
print("GPU:", torch.cuda.get_device_name(0), "PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
PY
# mkdir (without -p for the final component) atomically refuses existing outputs.
mkdir -p -- "$(dirname -- "$1")"
mkdir -- "$1"
OUT=$(cd -- "$1" && pwd)
"$PYTHON" -m pytest -q \
    tests/test_paged_attention_prototype.py \
    tests/test_paged_attention_integration.py \
    tests/test_paged_attention_benchmark.py \
    --junitxml="$OUT/gates.xml" 2>&1 | tee "$OUT/gates.log"
"$PYTHON" - "$OUT/gates.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
if not suites or any(int(s.attrib.get(k, 0)) for s in suites for k in ("skipped", "failures", "errors")):
    raise SystemExit("Acceptance gates must execute with zero skips/failures/errors before timing.")
PY
"$PYTHON" -m bench.paged_attention --output "$OUT/component.json" 2>&1 | tee "$OUT/component.log"
if [[ ${2:-} == "--ppo" ]]; then
    BLOCKS=${PPO_BLOCKS:-2}
    if [[ ! "$BLOCKS" =~ ^[1-9][0-9]*$ ]]; then
        echo "PPO_BLOCKS must be a positive integer" >&2
        exit 2
    fi
    for CASE in paged snapshots; do
        for ((BLOCK=0; BLOCK<BLOCKS; BLOCK++)); do
            ORDER=0
            for BACKEND in torch triton triton torch; do
                NAME=$(printf 'ppo-%s-%03d-%d-%s' "$CASE" "$BLOCK" "$ORDER" "$BACKEND")
                "$PYTHON" -m bench.paged_attention_ppo \
                    --backend "$BACKEND" --case "$CASE" --output "$OUT/$NAME" \
                    2>&1 | tee "$OUT/$NAME.log"
                ORDER=$((ORDER + 1))
            done
        done
    done
    "$PYTHON" -m bench.paged_attention_results "$OUT" --output "$OUT/ppo-summary.json"
fi
printf '\nResults: %s\n' "$OUT"
printf 'Prototype remains opt-in; successful synthetic gates do not authorize production adoption.\n'
