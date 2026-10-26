#!/usr/bin/env bash
# Single entry point for every GuanZero SLURM job on Velocity. Source it from a
# batch script after setting GZ (and RUN for training). It guarantees:
#  - python/torch/gd come from the node-local env in /dev/shm (built from the
#    internet by build_shm_env.sh); no fallback to the NFS conda env, ever;
#  - pycache, TMPDIR, torch/triton/inductor/CUDA/pip caches live in /dev/shm;
#  - the run dir lives in /dev/shm and is synced to $R_HOME every
#    GZ_SHM_SYNC_SECS (>= 300) plus once on exit;
#  - the job aborts if the interpreter would import anything from /home.
#
# Inputs: GZ (required); GZ_STAGE_RUN=1 with RUN and R_HOME (training; R_HOME
# defaults to $GZ/runs/$RUN) to stage the run dir; GZ_SRC_ARCHIVE + GZ_SRC_NAME
# (source tarball or directory, staged once per node; a source root S on /home
# is staged automatically); K (kit dir whose *.py/*.json are copied to shm).
# Outputs: PY, S, K, R (shm run dir), GZ_SHM_ROOT; functions gz_start_sync,
# gz_final_sync. Exits the job (status 2) on any staging or guard failure.

: "${GZ:?set GZ before sourcing gz_job_env.sh}"
GZ_KIT_LIB="${GZ_KIT_LIB:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export GZ_SHM_ROOT="${GZ_SHM_ROOT:-/dev/shm/$USER}"
export GZ_SHM_ENV="${GZ_SHM_ENV:-$GZ_SHM_ROOT/gz-env-local}"
export GZ_FORBID_PREFIX="${GZ_FORBID_PREFIX:-/home}"
GZ_SHM_SYNC_SECS="${GZ_SHM_SYNC_SECS:-300}"
[ "$GZ_SHM_SYNC_SECS" -ge 300 ] || GZ_SHM_SYNC_SECS=300

gz_die() { echo "gz_job_env FATAL: $*" >&2; exit 2; }
gz_log() { echo "gz_job_env: $*"; }

for p in "$GZ_SHM_ROOT" "$GZ_SHM_ENV"; do
  case "$(realpath -m "$p")/" in
    "$GZ_FORBID_PREFIX"/*) gz_die "guard: $p is under $GZ_FORBID_PREFIX (NFS); the env must be node-local" ;;
  esac
done
mkdir -p "$GZ_SHM_ROOT"/{tmp,pycache,cache} || gz_die "cannot create $GZ_SHM_ROOT"

# Never inherit an activated conda/NFS env from the submitting shell.
unset PYTHONHOME PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV VIRTUAL_ENV
export PATH="$GZ_SHM_ENV/bin:$(printf '%s' "$PATH" | tr ':' '\n' | grep -v -e '^/home/' -e 'anaconda' -e 'conda' | paste -sd:)"
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$GZ_SHM_ROOT/pycache"
export TMPDIR="$GZ_SHM_ROOT/tmp"
export XDG_CACHE_HOME="$GZ_SHM_ROOT/cache"
export TORCHINDUCTOR_CACHE_DIR="$GZ_SHM_ROOT/cache/inductor"
export TRITON_CACHE_DIR="$GZ_SHM_ROOT/cache/triton"
export CUDA_CACHE_PATH="$GZ_SHM_ROOT/cache/nv"
export PIP_CACHE_DIR="$GZ_SHM_ROOT/cache/pip"
export UV_CACHE_DIR="$GZ_SHM_ROOT/cache/uv"
export MPLCONFIGDIR="$GZ_SHM_ROOT/cache/matplotlib"
export GZ_REQUIRE_NODE_LOCAL=1

if [ ! -f "$GZ_SHM_ENV/.staged_ok" ]; then
  gz_log "no node-local env at $GZ_SHM_ENV; building from the internet"
  bash "$GZ_KIT_LIB/build_shm_env.sh" env || gz_die "node-local env build failed; NOT falling back to NFS"
fi
PY="$GZ_SHM_ENV/bin/python"
export PYTHON="$PY"
hash -r

if [ -z "${GZ_SRC_ARCHIVE:-}" ] && [ -n "${S:-}" ]; then
  case "$(realpath -m "$S")/" in
    "$GZ_FORBID_PREFIX"/*) GZ_SRC_ARCHIVE="$S"; GZ_SRC_NAME="${GZ_SRC_NAME:-$(basename "$S")}" ;;
  esac
fi
if [ -n "${GZ_SRC_ARCHIVE:-}" ]; then
  : "${GZ_SRC_NAME:?set GZ_SRC_NAME with GZ_SRC_ARCHIVE}"
  bash "$GZ_KIT_LIB/build_shm_env.sh" src "$GZ_SRC_ARCHIVE" "$GZ_SRC_NAME" || gz_die "source staging failed"
  S="$GZ_SHM_ROOT/src/$GZ_SRC_NAME"
fi

if [ -n "${K:-}" ] && [ "${K#"$GZ_SHM_ROOT"}" = "$K" ]; then
  kit_local="$GZ_SHM_ROOT/kits/$(basename "$K")-${SLURM_JOB_ID:-$$}"
  mkdir -p "$kit_local"
  find "$K" -maxdepth 1 -type f \( -name '*.py' -o -name '*.json' \) -exec cp {} "$kit_local/" \; \
    || gz_die "cannot copy kit $K"
  K="$kit_local"
fi

if [ "${GZ_STAGE_RUN:-0}" = 1 ]; then
  : "${RUN:?GZ_STAGE_RUN=1 needs RUN}"
  R_HOME="${R_HOME:-$GZ/runs/$RUN}"
  R="$GZ_SHM_ROOT/runs/$RUN"
  mkdir -p "$R" "$R_HOME"
  gz_log "seeding $R_HOME -> $R"
  # Large append-only logs and latest.pt are read once, sequentially.
  rsync -a --exclude 'slurm-*.out' --exclude '*.out' --exclude 'gpu-*.jsonl' --exclude STOP \
    --exclude 'segments/*/update-*.pt' --exclude 'segments/*/*-attempt-*.log' \
    "$R_HOME/" "$R/" || gz_die "seeding run dir from $R_HOME failed"
  gz_log "seeded $(du -sh "$R" | cut -f1)"
fi
cd "${R:-$GZ_SHM_ROOT/tmp}" || gz_die "cannot cd into shm"

# GPUs outside the SLURM allocation (e.g. a card left out of gres after repair)
# are still visible through the device cgroup on max-compute-11; never use them.
# gres.conf lists /dev/nvidiaN in order, so allocation IDX N is device minor N.
gz_restrict_gpus() {
  GZ_ALLOWED_GPU_UUIDS=""
  [ -n "${SLURM_JOB_ID:-}" ] && command -v nvidia-smi >/dev/null || return 0
  local idx minors="" part minor uuid
  idx=$(scontrol show job -d "$SLURM_JOB_ID" 2>/dev/null | grep -o 'gpu[^ ]*(IDX:[^)]*)' \
        | sed 's/.*IDX://; s/)$//' | paste -sd,)
  [ -n "$idx" ] || idx="${SLURM_JOB_GPUS:-${SLURM_STEP_GPUS:-}}"
  for part in ${idx//,/ }; do
    if [[ $part == *-* ]]; then minors+=" $(seq -s ' ' "${part%-*}" "${part#*-}")"; else minors+=" $part"; fi
  done
  while read -r minor uuid; do
    [[ " $minors " == *" $minor "* ]] && GZ_ALLOWED_GPU_UUIDS+="${GZ_ALLOWED_GPU_UUIDS:+,}$uuid"
  done < <(nvidia-smi -q | awk '/GPU UUID/{u=$NF} /Minor Number/{print $NF, u}')
  local want; want=$(wc -w <<< "$minors")
  [ "$(tr ',' '\n' <<< "$GZ_ALLOWED_GPU_UUIDS" | grep -c .)" -eq "$want" ] \
    || gz_die "GPU allocation IDX=$idx not matched by visible devices (allowed='$GZ_ALLOWED_GPU_UUIDS')"
  export GZ_ALLOWED_GPU_UUIDS GZ_ALLOWED_GPU_COUNT="$want"
  # Empty for a CPU-only job: hides every GPU from CUDA.
  export CUDA_VISIBLE_DEVICES="$GZ_ALLOWED_GPU_UUIDS"
  gz_log "GPUs restricted to SLURM allocation IDX=${idx:-none}: ${GZ_ALLOWED_GPU_UUIDS:-none}"
}
gz_restrict_gpus

guard_out="$("$PY" -c 'import sys, torch; print(sys.prefix, torch.__file__)' 2>&1)" \
  || gz_die "guard: interpreter check failed: $guard_out"
case " $guard_out" in
  *" $GZ_FORBID_PREFIX"/*) gz_die "guard: python/torch resolve under $GZ_FORBID_PREFIX: $guard_out" ;;
esac
gz_log "guard ok: $guard_out"
if [ -n "${GZ_ALLOWED_GPU_COUNT:-}" ]; then
  seen=$("$PY" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)
  [ "$seen" = "$GZ_ALLOWED_GPU_COUNT" ] \
    || gz_die "guard: torch sees $seen GPUs but SLURM allocated $GZ_ALLOWED_GPU_COUNT"
  gz_log "guard ok: torch sees $seen GPU(s) = allocation"
fi
if [ -n "${S:-}" ]; then
  case "$S" in "$GZ_FORBID_PREFIX"/*) gz_die "guard: source root $S is on NFS; set GZ_SRC_ARCHIVE" ;; esac
  # Native libraries count too: an RPATH into the NFS env maps libstdc++ from /home.
  maps_out="$(cd "$S" && PYTHONPATH="$S/python" "$PY" -c '
import os, torch, gd._gd_core
prefix = os.environ["GZ_FORBID_PREFIX"].rstrip("/") + "/"
bad = sorted({l.split()[-1] for l in open("/proc/self/maps") if l.split()[-1].startswith(prefix)})
print(" ".join(bad))' 2>&1)" || gz_die "guard: gd._gd_core import failed from $S: $maps_out"
  [ -z "$maps_out" ] || gz_die "guard: torch/gd map files from $GZ_FORBID_PREFIX: $maps_out"
  gz_log "guard ok: torch + gd._gd_core map nothing from $GZ_FORBID_PREFIX"
fi

gz_sync_back() {
  [ -n "${R:-}" ] || return 0
  local t0=$SECONDS rc=0
  # Append-only logs: --append copies only the new tail (skips files already
  # as long on /home). Everything else: rsync writes a temp file in the
  # destination dir and renames it, so /home never holds a torn latest.pt.
  rsync -a --append --include='*/' --include='*.jsonl' --include='*.log' --exclude='*' \
    "$R/" "$R_HOME/" || rc=$?
  rsync -a --exclude STOP --exclude '*.jsonl' --exclude '*.log' --exclude 'slurm-*.out' \
    "$R/" "$R_HOME/" || rc=$?
  if [ "$rc" -eq 0 ]; then
    # Keep only the newest periodic update-*.pt locally once /home has a copy.
    local f newest
    for d in "$R"/segments/*/; do
      newest=$(ls -1 "$d"update-*.pt 2>/dev/null | tail -1)
      for f in "$d"update-*.pt; do
        [ -f "$f" ] && [ "$f" != "$newest" ] || continue
        [ "$(stat -c %s "$f")" = "$(stat -c %s "$R_HOME/${f#"$R"/}" 2>/dev/null)" ] && rm -f "$f"
      done
    done
  fi
  gz_log "sync $R -> $R_HOME rc=$rc in $((SECONDS - t0)) s at $(date -u +%FT%TZ)"
  return "$rc"
}

gz_start_sync() {
  [ -n "${R:-}" ] || return 0
  [ -n "${GZ_SYNC_PID:-}" ] && kill -0 "$GZ_SYNC_PID" 2>/dev/null && return 0
  ( trap 'exit 0' TERM; while sleep "$GZ_SHM_SYNC_SECS"; do gz_sync_back; done ) &
  GZ_SYNC_PID=$!
  gz_log "sync loop pid=$GZ_SYNC_PID every ${GZ_SHM_SYNC_SECS}s"
}

# Safe to call more than once (the EXIT trap repeats it after a kit's own
# call so late writes such as jobs.log still reach /home).
gz_final_sync() {
  if [ -n "${GZ_SYNC_PID:-}" ]; then
    kill "$GZ_SYNC_PID" 2>/dev/null; wait "$GZ_SYNC_PID" 2>/dev/null
    GZ_SYNC_PID=""
  fi
  gz_sync_back
}
trap gz_final_sync EXIT
gz_start_sync

gz_log "env=$GZ_SHM_ENV src=${S:-none} kit=${K:-none} run=${R:-none} home=${R_HOME:-none} sync=${GZ_SHM_SYNC_SECS}s"
