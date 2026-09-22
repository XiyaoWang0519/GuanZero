#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
PY=${PY:-.venv/bin/python}
: "${BUDGET_USD:?Set the remaining approved budget cap explicitly}"
: "${HOURLY_RATE_USD:?Set the actual all-in hourly estimate explicitly}"
: "${MAX_HOURS:?Set a maximum walltime explicitly}"
RUN_DIR=${RUN_DIR:-runs/m1-$(date -u +%Y%m%dT%H%M%SZ)}
CONFIG=${CONFIG:-train/configs/m1.json}
DEVICE=${DEVICE:-cuda}
export PYTHONPATH="$ROOT/python:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
watchdog=(--budget-usd "$BUDGET_USD" --hourly-rate-usd "$HOURLY_RATE_USD"
  --max-hours "$MAX_HOURS" --spent-usd "${SPENT_USD:-0}"
  --grace-seconds "${GRACE_SECONDS:-30}" --log-file "$RUN_DIR/watchdog.jsonl")
if [[ -n ${SYNC_DEST:-} ]]; then
  watchdog+=(--sync-source "$RUN_DIR" --sync-dest "$SYNC_DEST"
    --sync-interval-seconds "${SYNC_INTERVAL_SECONDS:-60}")
fi
if [[ -n ${TEARDOWN_COMMAND_JSON:-} ]]; then
  watchdog+=(--teardown-command-json "$TEARDOWN_COMMAND_JSON")
fi
if [[ ${ALLOW_TEARDOWN:-0} == 1 ]]; then
  watchdog+=(--allow-teardown)
fi
trainer=(-m train.dmc --config "$CONFIG" --run-dir "$RUN_DIR" --device "$DEVICE")
if [[ -n ${RESUME:-} ]]; then
  trainer+=(--resume "$RESUME")
fi
exec "$PY" -m infra.watchdog "${watchdog[@]}" -- "$PY" "${trainer[@]}" "$@"
