#!/usr/bin/env bash
# Pod entry, run by supervisord (survives the SSH session): verify every uploaded file by
# SHA256, start the self-stop guard (read-only verify), build + checks, evaluations until
# the deadline, results manifest, DONE.
set -u
K=/workspace/kit
R=/workspace/results
mkdir -p "$R"
source "$K/deadlines.env"
cp "$K/deadlines.env" "$R/deadlines.env"
echo "{\"machine_id\": ${KIT_MACHINE_ID:-null}, \"offer_id\": ${KIT_OFFER_ID:-null}, \"dph_total\": ${KIT_DPH_TOTAL:-null}, \"container_label\": \"${VAST_CONTAINERLABEL:-}\"}" > "$R/machine.json"
BASE=/venv/main/bin/python
[ -x "$BASE" ] || BASE=python3
cd "$K"
manifest() {
  "$BASE" - "$R" <<'PYEOF'
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
        files[str(path.relative_to(root))] = dict(sha256=digest.hexdigest(), bytes=path.stat().st_size)
(root / "results-manifest.json").write_text(json.dumps(files, indent=1) + "\n")
PYEOF
}
if ! { sha256sum -c kit-files.sha256 && sha256sum -c generated.sha256; } > "$R/kit-verify.txt" 2>&1; then
  echo '{"stage":"kit-verify","ok":false}' > "$R/FAILED.json"; manifest; touch "$R/DONE"; exit 1
fi
"$BASE" "$K/remote_guard.py" >> "$R/remote-guard.log" 2>&1 < /dev/null &
"$BASE" "$K/remote_guard.py" verify > "$R/remote-guard-verify.log" 2>&1 || echo "verify error $?" >> "$R/remote-guard-verify.log"
date -u +%FT%TZ > "$R/setup-started.txt"
if timeout --signal=TERM --kill-after=30s 1500s bash "$K/remote_setup.sh" > "$R/setup.log" 2>&1; then
  date -u +%FT%TZ > "$R/eval-started.txt"
  PY=$(cat "$R/python-path.txt")
  FREEZE=$("$BASE" -c "import json; print(json.load(open('$K/expected.json'))['freeze_pod_path'])")
  "$PY" "$K/eval_pod.py" --kit "$K" --src /workspace/src --results "$R" --freeze "$FREEZE" \
    --python "$PY" --deadline-epoch "$EVAL_DEADLINE_EPOCH" > "$R/eval-pod.log" 2>&1
  echo "{\"eval_pod_returncode\": $?}" > "$R/eval-pod-exit.json"
else
  echo '{"stage":"setup","ok":false}' > "$R/FAILED.json"
fi
manifest
touch "$R/DONE"
