#!/usr/bin/env sh
# Copy completed checkpoint artifacts; never remove files from the destination.
set -eu
if [ "$#" -ne 2 ]; then
  echo "usage: infra/sync.sh RUN_DIR DESTINATION (local path or rsync destination)" >&2
  exit 2
fi
SOURCE=$1
DEST=$2
if [ ! -d "$SOURCE" ]; then
  echo "checkpoint source is not a directory: $SOURCE" >&2
  exit 2
fi
case "$DEST" in
  -*) echo "destination must not start with a dash" >&2; exit 2 ;;
esac
# rsync transfers to a temporary file and renames it on completion. Avoid
# --inplace: latest.pt is replaced atomically by the trainer while we read it.
# Network destinations use the operator's existing SSH/rsync credentials.
exec rsync -a --exclude='*.tmp' --exclude='.*.tmp' --exclude='.*.pt.*' --exclude='watchdog.jsonl' \
  -- "$SOURCE/" "$DEST/"
