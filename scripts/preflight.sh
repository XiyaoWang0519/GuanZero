#!/usr/bin/env bash
# A bounded real learner run and resume on the selected device, before renting
# for a long run. Artifacts are preserved so the result can be inspected.
#
#   scripts/preflight.sh [cpu|cuda] [RUN_DIR]        Stage A DMC learner
#   scripts/preflight.sh ppo [cpu|cuda] [RUN_DIR]    Stage B PPO learner (B5):
#     2 updates, checkpoint, resume to 4 updates, checkpoint again. Needs the
#     M1 final and the B2 critic under .work (found in the main checkout too).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PY:-.venv/bin/python}
if [[ "${1:-}" == ppo ]]; then
  DEVICE=${2:-cpu}
  RUN_DIR=${3:-$(mktemp -d "${TMPDIR:-/tmp}/guanzero-ppo-preflight.XXXXXX")}
  export PYTHONPATH="$ROOT/python:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
  CONFIG=train/configs/ppo-smoke.json
  "$PY" -m train.ppo --config "$CONFIG" --run-dir "$RUN_DIR" \
    --device "$DEVICE" --max-updates 2 --max-seconds 300
  "$PY" - "$RUN_DIR/latest.pt" 2 <<'PY'
import sys
import torch
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert checkpoint["stage"] == "ppo", "not a Stage B checkpoint"
assert checkpoint["progress"]["updates"] == int(sys.argv[2]), "wrong number of PPO updates"
PY
  "$PY" -m train.ppo --config "$CONFIG" --run-dir "$RUN_DIR" \
    --device "$DEVICE" --resume "$RUN_DIR/latest.pt" --max-updates 4 --max-seconds 300
  "$PY" - "$RUN_DIR/latest.pt" 4 <<'PY'
import sys
import torch
from eval.policies import load_policy
checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
assert checkpoint["progress"]["updates"] == int(sys.argv[2]), "resume did not reach four PPO updates"
assert checkpoint["progress"]["resumes"] == 1, "checkpoint was not resumed"
policy = load_policy(sys.argv[1])
print(f"preflight: {policy.name} loads through load_policy")
PY
  "$PY" - "$RUN_DIR/metrics.jsonl" <<'PY'
import json
import math
import sys
rows = [json.loads(line) for line in open(sys.argv[1])]
assert [r["updates"] for r in rows] == [1, 2, 3, 4], "metrics do not cover four updates"
for key in ("clip_fraction", "approx_kl", "explained_variance", "entropy", "kl_ref", "value_loss"):
    assert all(math.isfinite(r[key]) for r in rows), key
print("preflight: " + "; ".join(f"update {r['updates']} clip {r['clip_fraction']:.3f} "
      f"kl {r['approx_kl']:.2e} ev {r['explained_variance']:.3f} ent {r['entropy']:.3f}" for r in rows))
PY
  echo "ppo preflight completed; inspect artifacts: $RUN_DIR"
  exit 0
fi
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
