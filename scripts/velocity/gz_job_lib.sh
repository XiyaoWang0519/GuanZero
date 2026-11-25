#!/usr/bin/env bash
# Functions shared by gz_job_env.sh and build_shm_env.sh. Sourcing this file
# has no side effects. Every function is safe under the callers' `set -euo
# pipefail`: an expected empty match must never abort a batch script.

gz_die() { echo "gz_job_env FATAL: $*" >&2; exit 2; }
gz_log() { echo "gz_job_env: $*" >&2; }

# /dev/shm is world-writable: another account could pre-create our directory
# and plant bin/python, tools/uv or .staged_ok. Only a non-symlink directory
# owned by us with mode 0700 is trusted; nobody else can then write below it.
gz_secure_dir() {
  local d=$1 owner mode
  [ -L "$d" ] && gz_die "refusing $d: it is a symlink"
  if [ ! -e "$d" ]; then
    mkdir -p "$(dirname "$d")"
    mkdir -m 700 "$d" 2>/dev/null || true
  fi
  [ -d "$d" ] && [ ! -L "$d" ] || gz_die "refusing $d: not a directory"
  read -r owner mode < <(stat -c '%u %a' "$d")
  [ "$owner" = "$(id -u)" ] || gz_die "refusing $d: owned by uid $owner, not $(id -u)"
  [ "$mode" = 700 ] || gz_die "refusing $d: mode $mode, need 700 (chmod 700 $d if it is yours)"
}

# Expands an IDX list ("0-1,3") to space-separated device minors; dies if malformed.
gz_expand_idx() {
  local part out=""
  for part in ${1//,/ }; do
    [[ $part =~ ^([0-9]+)(-([0-9]+))?$ ]] || gz_die "malformed GPU allocation IDX='$1'"
    if [ -n "${BASH_REMATCH[3]}" ]; then
      [ "${BASH_REMATCH[3]}" -ge "${BASH_REMATCH[1]}" ] || gz_die "malformed GPU allocation IDX='$1'"
      out+=" $(seq -s ' ' "${BASH_REMATCH[1]}" "${BASH_REMATCH[3]}")"
    else
      out+=" $part"
    fi
  done
  printf '%s\n' "${out# }"
}

# Prints the SLURM allocation's GPU indices (e.g. "1" or "0-1,3"); prints
# nothing for a CPU-only job. Dies on a malformed or contradictory allocation.
# `scontrol show job` text is not trusted wholesale: JobName, Command and
# Comment are user-controlled and may contain newlines, so a comment can forge
# whole "AllocTRES=" or "Nodes=... GRES=" lines. The GPU count comes from the
# single computed squeue tres-alloc field (no gres/gpu there means zero GPUs,
# whatever IDX text appears elsewhere); IDX comes only from this node's
# per-node GRES line, which must be unique, and must agree with SLURM_JOB_GPUS
# (set by slurmd; a submitter's value is not exported into the job).
gz_alloc_gpu_idx() {
  local tres="" ngpu info="" idx="" node minors _
  local -a lines
  for _ in 1 2 3; do
    tres=$(squeue -h -j "$SLURM_JOB_ID" -O tres-alloc:4000 2>/dev/null || true)
    tres=${tres//[[:space:]]/}
    [ -n "$tres" ] && break
    sleep "${GZ_SLURM_RETRY_SECS:-2}"
  done
  [ -n "$tres" ] || gz_die "squeue cannot report AllocTRES for job $SLURM_JOB_ID; cannot tell the GPU allocation"
  ngpu=$(tr ',' '\n' <<< "$tres" | sed -n 's|^gres/gpu=\([0-9][0-9]*\)$|\1|p' | head -1)
  if [ -z "$ngpu" ] || [ "$ngpu" -eq 0 ]; then
    return 0
  fi
  node=${SLURMD_NODENAME:-$(hostname -s)}
  info=$(scontrol show job -d "$SLURM_JOB_ID" 2>/dev/null || true)
  mapfile -t lines < <(grep -E "^ +Nodes=${node//./\\.} CPU_IDs=[^ ]+ Mem=[0-9]+ GRES=" <<< "$info" || true)
  if [ "${#lines[@]}" -gt 1 ]; then
    gz_die "ambiguous GPU allocation: ${#lines[@]} per-node GRES lines for $node (forged by a job name/comment?)"
  elif [ "${#lines[@]}" -eq 1 ]; then
    idx=$(sed 's/.* GRES=//' <<< "${lines[0]}" | grep -o 'gpu[^ ,]*(IDX:[^)]*)' | sed 's/.*IDX://; s/)$//' \
      | paste -sd, || true)
  else
    idx="${SLURM_JOB_GPUS:-}"
  fi
  [ -n "$idx" ] || gz_die "AllocTRES has gres/gpu=$ngpu but no IDX for $node and SLURM_JOB_GPUS is unset"
  minors=$(gz_expand_idx "$idx") || exit 2
  if [ "$(wc -w <<< "$minors")" -ne "$ngpu" ]; then
    gz_die "GPU allocation contradicts itself: AllocTRES gres/gpu=$ngpu but IDX='$idx'"
  fi
  if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    [ "$(gz_expand_idx "$SLURM_JOB_GPUS" | tr ' ' '\n' | sort -n | paste -sd' ')" \
      = "$(tr ' ' '\n' <<< "$minors" | sort -n | paste -sd' ')" ] \
      || gz_die "GPU allocation contradicts itself: IDX='$idx' but SLURM_JOB_GPUS='$SLURM_JOB_GPUS'"
  fi
  printf '%s\n' "$idx"
}

# GPUs outside the SLURM allocation (e.g. a card left out of gres after repair)
# are still visible through the device cgroup on max-compute-11; never use them.
# gres.conf lists /dev/nvidiaN in order, so allocation IDX N is device minor N.
gz_restrict_gpus() {
  GZ_ALLOWED_GPU_UUIDS=""
  [ -n "${SLURM_JOB_ID:-}" ] || return 0
  local idx minors minor uuid want got
  idx=$(gz_alloc_gpu_idx) || exit 2
  if [ -z "$idx" ]; then
    export GZ_ALLOWED_GPU_UUIDS="" GZ_ALLOWED_GPU_COUNT=0 CUDA_VISIBLE_DEVICES=""
    gz_log "CPU-only allocation: all GPUs hidden (CUDA_VISIBLE_DEVICES='')"
    return 0
  fi
  command -v nvidia-smi >/dev/null || gz_die "GPU allocation IDX=$idx but no nvidia-smi to map it"
  minors=$(gz_expand_idx "$idx") || exit 2
  while read -r minor uuid; do
    if [[ " $minors " == *" $minor "* ]]; then
      GZ_ALLOWED_GPU_UUIDS+="${GZ_ALLOWED_GPU_UUIDS:+,}$uuid"
    fi
  done < <(nvidia-smi -q | awk '/GPU UUID/{u=$NF} /Minor Number/{print $NF, u}')
  want=$(wc -w <<< "$minors")
  got=$(tr ',' '\n' <<< "$GZ_ALLOWED_GPU_UUIDS" | grep -c . || true)
  [ "$got" -eq "$want" ] \
    || gz_die "GPU allocation IDX=$idx not matched by visible devices (allowed='$GZ_ALLOWED_GPU_UUIDS')"
  export GZ_ALLOWED_GPU_UUIDS GZ_ALLOWED_GPU_COUNT="$want" CUDA_VISIBLE_DEVICES="$GZ_ALLOWED_GPU_UUIDS"
  gz_log "GPUs restricted to SLURM allocation IDX=$idx: $GZ_ALLOWED_GPU_UUIDS"
}

# Run-dir sync. Active only after gz_job_env.sh staged R from R_HOME
# (GZ_SYNC_ENABLED=1); a kit that sets R itself keeps its own I/O.
GZ_SHM_SYNC_SECS="${GZ_SHM_SYNC_SECS:-300}"
GZ_SHM_SYNC_TIMEOUT="${GZ_SHM_SYNC_TIMEOUT:-600}"
GZ_SHM_SYNC_MAX_FAILS="${GZ_SHM_SYNC_MAX_FAILS:-3}"
GZ_SHM_SYNC_MAX_AGE="${GZ_SHM_SYNC_MAX_AGE:-1200}"

gz_sync_back() {
  [ "${GZ_SYNC_ENABLED:-0}" = 1 ] || return 0
  local t0=$SECONDS rc=0 d f newest
  local -a updates
  # Append-only logs: --append copies only the new tail (skips files already
  # as long on /home). Everything else: rsync writes a temp file in the
  # destination dir and renames it, so /home never holds a torn latest.pt.
  # The timeout bounds a hung NFS mount so the failure counter still advances.
  timeout -k 30 "$GZ_SHM_SYNC_TIMEOUT" rsync -a --append --include='*/' --include='*.jsonl' \
    --include='*.log' --exclude='*' "$R/" "$R_HOME/" || rc=$?
  timeout -k 30 "$GZ_SHM_SYNC_TIMEOUT" rsync -a --exclude STOP --exclude .gz-sync-failed --exclude '*.jsonl' \
    --exclude '*.log' --exclude 'slurm-*.out' "$R/" "$R_HOME/" || rc=$?
  if [ "$rc" -eq 0 ]; then
    # Keep only the newest periodic update-*.pt locally once /home has a copy.
    for d in "$R"/segments/*/; do
      [ -d "$d" ] || continue
      mapfile -t updates < <(compgen -G "${d}update-*.pt" | sort -V || true)
      [ "${#updates[@]}" -gt 1 ] || continue
      newest=${updates[-1]}
      for f in "${updates[@]}"; do
        [ "$f" != "$newest" ] || continue
        if [ "$(stat -c %s "$f")" = "$(stat -c %s "$R_HOME/${f#"$R"/}" 2>/dev/null || true)" ]; then
          rm -f "$f"
        fi
      done
    done
  fi
  gz_log "sync $R -> $R_HOME rc=$rc in $((SECONDS - t0)) s at $(date -u +%FT%TZ)"
  return "$rc"
}

# A sync that keeps failing means checkpoints exist only in RAM on this node.
# After GZ_SHM_SYNC_MAX_FAILS consecutive failures or GZ_SHM_SYNC_MAX_AGE
# seconds without a success, stop the job: TERM to the batch shell is the kits'
# "stop cleanly, do not resubmit" path, the EXIT trap then makes one last sync
# attempt and exits 4 so sacct shows FAILED. $R/.gz-sync-status always holds
# the last outcome for the training process and operators.
gz_sync_loop() {
  local fails=0 last_ok=$SECONDS sleeper=""
  trap '[ -n "$sleeper" ] && kill "$sleeper" 2>/dev/null; exit 0' TERM
  while :; do
    sleep "$GZ_SHM_SYNC_SECS" & sleeper=$!
    wait "$sleeper" || true
    sleeper=""
    if gz_sync_back; then
      fails=0
      last_ok=$SECONDS
      echo "ok $(date -u +%FT%TZ)" > "$R/.gz-sync-status"
      continue
    fi
    fails=$((fails + 1))
    echo "failing fails=$fails last_ok_age=$((SECONDS - last_ok))s at $(date -u +%FT%TZ)" > "$R/.gz-sync-status"
    echo "gz_job_env WARNING: sync to $R_HOME failed ($fails in a row, last success $((SECONDS - last_ok)) s ago)" >&2
    if [ "$fails" -ge "$GZ_SHM_SYNC_MAX_FAILS" ] || [ $((SECONDS - last_ok)) -ge "$GZ_SHM_SYNC_MAX_AGE" ]; then
      echo "$(date -u +%FT%TZ) sync to $R_HOME failed $fails times; last success $((SECONDS - last_ok)) s ago" \
        > "$R/.gz-sync-failed"
      echo "gz_job_env FATAL: checkpoints are not reaching $R_HOME; stopping job (TERM to pid $GZ_JOB_SHELL_PID)." \
        "Local copy kept in $R" >&2
      kill -TERM "$GZ_JOB_SHELL_PID" 2>/dev/null || true
      return 0
    fi
  done
}

gz_start_sync() {
  [ "${GZ_SYNC_ENABLED:-0}" = 1 ] || return 0
  [ -n "${GZ_SYNC_PID:-}" ] && kill -0 "$GZ_SYNC_PID" 2>/dev/null && return 0
  GZ_JOB_SHELL_PID=$$
  rm -f "$R/.gz-sync-failed"
  gz_sync_loop &
  GZ_SYNC_PID=$!
  gz_log "sync loop pid=$GZ_SYNC_PID every ${GZ_SHM_SYNC_SECS}s (stop after $GZ_SHM_SYNC_MAX_FAILS failures or ${GZ_SHM_SYNC_MAX_AGE}s without success)"
}

# Safe to call more than once (the EXIT trap repeats it after a kit's own
# call so late writes such as jobs.log still reach /home).
gz_final_sync() {
  if [ -n "${GZ_SYNC_PID:-}" ]; then
    kill "$GZ_SYNC_PID" 2>/dev/null || true
    wait "$GZ_SYNC_PID" 2>/dev/null || true
    GZ_SYNC_PID=""
  fi
  gz_sync_back || true
}

gz_on_exit() {
  local rc=$?
  gz_final_sync
  if [ "${GZ_SYNC_ENABLED:-0}" = 1 ] && [ -f "$R/.gz-sync-failed" ]; then
    echo "gz_job_env FATAL: job stopped because syncing $R to $R_HOME failed: $(cat "$R/.gz-sync-failed")" >&2
    exit 4
  fi
  exit "$rc"
}
