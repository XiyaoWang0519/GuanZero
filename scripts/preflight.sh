#!/usr/bin/env bash
# A bounded real learner run and resume on the selected device, before renting
# for a long run. Artifacts are preserved so the result can be inspected.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PY:-.venv/bin/python}
DEVICE=${1:-cpu}
CONFIG=train/configs/smoke.json
if [[ "$DEVICE" == cuda ]]; then
  CONFIG=train/configs/gpu-smoke.json
fi
RUN_DIR=${2:-$(mktemp -d "${TMPDIR:-/tmp}/guanzero-preflight.XXXXXX")}
export PYTHONPATH="$ROOT/python:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PY" - "$DEVICE" <<'PY'
import sys
import gd
import torch
device = sys.argv[1]
if device not in ("cpu", "cuda"):
    raise SystemExit("device must be cpu or cuda")
if device == "cuda" and not torch.cuda.is_available():
    raise SystemExit("CUDA was requested but is unavailable; refusing a CPU fallback")
x = torch.ones((8, 8), device=device)
assert float((x @ x).sum().cpu()) == 512
print(f"preflight: torch={torch.__version__}, device={device}, obs={gd.OBS_DIM}, action={gd.ACT_DIM}")
PY
"$PY" -m train.dmc --config "$CONFIG" --run-dir "$RUN_DIR" \
  --device "$DEVICE" --max-updates 2 --max-seconds 120
test -s "$RUN_DIR/latest.pt"
"$PY" - "$RUN_DIR/latest.pt" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert checkpoint["progress"]["updates"] == 2, "initial smoke run did not finish two updates"
PY
"$PY" -m train.dmc --config "$CONFIG" --run-dir "$RUN_DIR" \
  --device "$DEVICE" --resume "$RUN_DIR/latest.pt" --max-updates 4 --max-seconds 120
"$PY" - "$RUN_DIR/latest.pt" <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert checkpoint["progress"]["updates"] == 4, "resume did not reach four cumulative updates"
PY
echo "preflight completed; inspect artifacts: $RUN_DIR"
