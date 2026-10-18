#!/bin/sh
# Start the play server and, with --public, a Cloudflare quick tunnel so
# friends outside this network can join. Ctrl-C stops both.
#
#   play/share.sh            # http://localhost:18688 and the LAN address
#   play/share.sh --public   # also prints a https://*.trycloudflare.com link
set -eu
cd "$(dirname "$0")/.."
PORT="${PORT:-18688}"
CHECKPOINT="${CHECKPOINT:-.work/velocity-milestones/m7-u73780.pt}"
PY="${PY:-.work/external/danlm-venv/bin/python}"

PYTHONPATH=python:oracle:. "$PY" -m play.server --checkpoint "$CHECKPOINT" --port "$PORT" &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT INT TERM

LAN=$(ipconfig getifaddr en0 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || true)
echo "local:  http://localhost:$PORT/"
[ -n "$LAN" ] && echo "LAN:    http://$LAN:$PORT/   (same Wi-Fi)"

if [ "${1:-}" = "--public" ]; then
  echo "starting Cloudflare quick tunnel; the https link appears below"
  cloudflared tunnel --no-autoupdate --url "http://localhost:$PORT"
else
  wait $SERVER
fi
