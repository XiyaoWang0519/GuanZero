#!/usr/bin/env bash
# Single entry point for every GuanZero SLURM job on Velocity. Source it from a
# batch script after setting GZ (and RUN for training). It guarantees:
#  - python/torch/gd come from the node-local env in /dev/shm (built from the
#    internet by build_shm_env.sh); no fallback to the NFS conda env, ever;
#  - /dev/shm/$USER is a 0700 directory owned by this account (no planted env);
#  - pycache, TMPDIR, torch/triton/inductor/CUDA/pip caches live in /dev/shm;
#  - CUDA sees only the SLURM-allocated GPUs (none for a CPU-only job);
#  - a staged run dir lives in /dev/shm and is synced to $R_HOME every
#    GZ_SHM_SYNC_SECS (>= 300) plus once on exit; persistent sync failure stops
#    the job (exit 4) instead of training on with checkpoints only in RAM;
#  - the job aborts if the interpreter would import anything from /home.
#
# Inputs: GZ (required); GZ_STAGE_RUN=1 with RUN and R_HOME (training; R_HOME
# defaults to $GZ/runs/$RUN) to stage the run dir; GZ_SRC_ARCHIVE + GZ_SRC_NAME
# (source tarball or directory, staged once per node and source identity; a
# source root S on /home is staged automatically); K (kit dir whose *.py/*.json
# are copied to shm).
# Outputs: PY, S, K, R (shm run dir when staged), GZ_SHM_ROOT; functions
# gz_start_sync, gz_final_sync. Exits the job (status 2) on any staging or
# guard failure.

: "${GZ:?set GZ before sourcing gz_job_env.sh}"
GZ_KIT_LIB="${GZ_KIT_LIB:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=gz_job_lib.sh
source "$GZ_KIT_LIB/gz_job_lib.sh"
export GZ_SHM_ROOT="${GZ_SHM_ROOT:-/dev/shm/$USER}"
export GZ_SHM_ENV="${GZ_SHM_ENV:-$GZ_SHM_ROOT/gz-env-local}"
export GZ_FORBID_PREFIX="${GZ_FORBID_PREFIX:-/home}"
[ "$GZ_SHM_SYNC_SECS" -ge 300 ] || GZ_SHM_SYNC_SECS=300

for p in "$GZ_SHM_ROOT" "$GZ_SHM_ENV"; do
  case "$(realpath -m "$p")/" in
    "$GZ_FORBID_PREFIX"/*) gz_die "guard: $p is under $GZ_FORBID_PREFIX (NFS); the env must be node-local" ;;
  esac
done
case "$(realpath -m "$GZ_SHM_ENV")/" in
  "$(realpath -m "$GZ_SHM_ROOT")"/*) ;;
  *) gz_die "GZ_SHM_ENV=$GZ_SHM_ENV must live inside GZ_SHM_ROOT=$GZ_SHM_ROOT" ;;
esac
gz_secure_dir "$GZ_SHM_ROOT"
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
  # stdout carries only the staged path, keyed by the source's identity.
  S=$(bash "$GZ_KIT_LIB/build_shm_env.sh" src "$GZ_SRC_ARCHIVE" "$GZ_SRC_NAME") || gz_die "source staging failed"
  [ -d "$S" ] || gz_die "source staging returned no directory: '$S'"
fi

if [ -n "${K:-}" ] && [ "${K#"$GZ_SHM_ROOT"}" = "$K" ]; then
  kit_local="$GZ_SHM_ROOT/kits/$(basename "$K")-${SLURM_JOB_ID:-$$}"
  mkdir -p "$kit_local"
  find "$K" -maxdepth 1 -type f \( -name '*.py' -o -name '*.json' \) -exec cp {} "$kit_local/" \; \
    || gz_die "cannot copy kit $K"
  K="$kit_local"
fi

GZ_SYNC_ENABLED=0
if [ "${GZ_STAGE_RUN:-0}" = 1 ]; then
  : "${RUN:?GZ_STAGE_RUN=1 needs RUN}"
  R_HOME="${R_HOME:-$GZ/runs/$RUN}"
  R="$GZ_SHM_ROOT/runs/$RUN"
  mkdir -p "$R" "$R_HOME"
  gz_log "seeding $R_HOME -> $R"
  # Large append-only logs and latest.pt are read once, sequentially. --update
  # keeps a local file newer than /home (a previous job whose final sync
  # failed) instead of rolling it back to the stale /home copy.
  rsync -a --update --exclude 'slurm-*.out' --exclude '*.out' --exclude 'gpu-*.jsonl' --exclude STOP \
    --exclude 'segments/*/update-*.pt' --exclude 'segments/*/*-attempt-*.log' \
    "$R_HOME/" "$R/" || gz_die "seeding run dir from $R_HOME failed"
  gz_log "seeded $(du -sh "$R" | cut -f1)"
  GZ_SYNC_ENABLED=1
fi
cd "${R:-$GZ_SHM_ROOT/tmp}" || gz_die "cannot cd into ${R:-$GZ_SHM_ROOT/tmp}"

gz_restrict_gpus

guard_out="$("$PY" -c 'import sys, torch; print(sys.prefix, torch.__file__)' 2>&1)" \
  || gz_die "guard: interpreter check failed: $guard_out"
case " $guard_out" in
  *" $GZ_FORBID_PREFIX"/*) gz_die "guard: python/torch resolve under $GZ_FORBID_PREFIX: $guard_out" ;;
esac
gz_log "guard ok: $guard_out"
if [ -n "${GZ_ALLOWED_GPU_COUNT:-}" ]; then
  seen=$("$PY" -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || true)
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

trap gz_on_exit EXIT
gz_start_sync

gz_log "env=$GZ_SHM_ENV src=${S:-none} kit=${K:-none} run=${R:-none} home=${R_HOME:-none} sync=$([ "$GZ_SYNC_ENABLED" = 1 ] && echo "${GZ_SHM_SYNC_SECS}s" || echo off)"
