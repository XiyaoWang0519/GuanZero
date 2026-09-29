#!/usr/bin/env bash
# Pod entry, run by supervisord (survives the SSH session): verify kit, start the
# self-stop guard and GPU sampler, build + gates, the overnight run, manifest, DONE.
# The guard's verify is read-only; a PUT state=running restarts a running container.
set -u
K=/workspace/kit
R=/workspace/results
mkdir -p "$R"
source "$K/deadlines.env"
cp "$K/deadlines.env" "$R/deadlines.env"
# Which rented machine produced these results.
echo "{\"machine_id\": ${KIT_MACHINE_ID:-null}, \"offer_id\": ${KIT_OFFER_ID:-null}, \"dph_total\": ${KIT_DPH_TOTAL:-null}, \"mode\": \"${KIT_MODE:-}\", \"container_label\": \"${VAST_CONTAINERLABEL:-}\"}" > "$R/machine.json"
PY=/venv/main/bin/python
[ -x "$PY" ] || PY=python3
cd "$K"
if ! sha256sum -c kit-files.sha256 > "$R/kit-verify.txt" 2>&1; then
  echo '{"stage":"kit-verify","ok":false}' > "$R/FAILED.json"; touch "$R/DONE"; exit 1
fi
"$PY" "$K/remote_guard.py" >> "$R/remote-guard.log" 2>&1 < /dev/null &
"$PY" "$K/remote_guard.py" verify > "$R/remote-guard-verify.log" 2>&1 || echo "verify error $?" >> "$R/remote-guard-verify.log"
source /venv/main/bin/activate
python "$K/gpu_sampler.py" "$R/gpu-samples.jsonl" 5 "$REMOTE_STOP_EPOCH" > "$R/gpu-sampler.log" 2>&1 < /dev/null &
SAMPLER=$!
date -u +%FT%TZ >> "$R/setup-started.txt"
# A restart after a completed setup (gates.json and the built engine) skips the rebuild.
if { [ -f "$R/gates.json" ] && ls /workspace/src/python/gd/_gd_core*.so >/dev/null 2>&1; } || \
   timeout --signal=TERM --kill-after=30s 1500s bash "$K/remote_setup.sh" > "$R/setup.log" 2>&1; then
  date -u +%FT%TZ > "$R/train-started.txt"
  cd /workspace/src
  python "$K/longrun.py" --source-root /workspace/src --results "$R" --gates "$R/gates.json" \
    --init "$K/init.pt" --mode "$KIT_MODE" --train-hours "$TRAIN_HOURS" \
    --hard-deadline-epoch "$TRAIN_DEADLINE_EPOCH" > "$R/orchestrator.log" 2>&1
  echo "{\"orchestrator_returncode\": $?}" > "$R/orchestrator-exit.json"
else
  echo '{"stage":"setup","ok":false}' > "$R/FAILED.json"
fi
kill "$SAMPLER" 2>/dev/null
python - "$R" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
files = {}
for path in sorted(root.rglob("*")):
    if path.is_file() and path.name not in ("results-manifest.json", "DONE") \
            and not path.name.startswith("."):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        files[str(path.relative_to(root))] = dict(sha256=digest.hexdigest(),
                                                   bytes=path.stat().st_size)
(root / "results-manifest.json").write_text(json.dumps(files, indent=1) + "\n")
PYEOF
touch "$R/DONE"
