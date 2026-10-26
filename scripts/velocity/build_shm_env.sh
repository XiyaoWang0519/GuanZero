#!/usr/bin/env bash
# Build GuanZero's Python env and source tree in node-local /dev/shm from the
# internet, never by copying the NFS conda env (small-file reads from /home are
# what starves other users on the node). Idempotent; flock serializes builders.
#
#   build_shm_env.sh env                  # Python + wheels -> $GZ_SHM_ENV
#   build_shm_env.sh src ARCHIVE NAME     # source tarball -> $GZ_SHM_ROOT/src/NAME, builds gd._gd_core
#
# The env is complete only when $GZ_SHM_ENV/.staged_ok exists; a source tree only
# when .gd_built exists. Both are written last, so a crash leaves no marker.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GZ_SHM_ROOT="${GZ_SHM_ROOT:-/dev/shm/$USER}"
GZ_SHM_ENV="${GZ_SHM_ENV:-$GZ_SHM_ROOT/gz-env-local}"
GZ_PYTHON_VERSION="${GZ_PYTHON_VERSION:-3.12.14}"
GZ_REQUIREMENTS="${GZ_REQUIREMENTS:-$HERE/requirements-velocity.txt}"
GZ_TORCH_INDEX="${GZ_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
GZ_FORBID_PREFIX="${GZ_FORBID_PREFIX:-/home}"
TOOLS="$GZ_SHM_ROOT/tools"
for p in "$GZ_SHM_ROOT" "$GZ_SHM_ENV"; do
  case "$(realpath -m "$p")/" in
    "$GZ_FORBID_PREFIX"/*) echo "build_shm_env: refusing $p: under $GZ_FORBID_PREFIX (NFS)" >&2; exit 2 ;;
  esac
done

export UV_CACHE_DIR="$GZ_SHM_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$GZ_SHM_ROOT/uv-python"
export UV_NO_CONFIG=1
export PIP_CACHE_DIR="$GZ_SHM_ROOT/cache/pip"
export XDG_CACHE_HOME="$GZ_SHM_ROOT/cache"
export TMPDIR="$GZ_SHM_ROOT/tmp"
export PYTHONNOUSERSITE=1
mkdir -p "$GZ_SHM_ROOT" "$TOOLS" "$UV_CACHE_DIR" "$TMPDIR"
cd "$GZ_SHM_ROOT"

log() { echo "build_shm_env: $*"; }

ensure_uv() {
  [ -x "$TOOLS/uv" ] && return 0
  log "fetching uv from github"
  curl -fsSL --retry 3 https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz \
    | tar -xz -C "$TOOLS" --strip-components=1
  "$TOOLS/uv" --version
}

env_ok() {
  [ -f "$GZ_SHM_ENV/.staged_ok" ] && [ -x "$GZ_SHM_ENV/bin/python" ] \
    && "$GZ_SHM_ENV/bin/python" -c "import numpy, torch" >/dev/null 2>&1
}

write_sitecustomize() {
  # Runs in every interpreter of this env, including spawned DDP/actor workers.
  local site
  site="$("$GZ_SHM_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  cat > "$site/sitecustomize.py" <<PYEOF
"""Refuse to run if this interpreter would import from NFS (${GZ_FORBID_PREFIX})."""
import importlib.util
import os
import sys

_forbid = os.environ.get("GZ_FORBID_PREFIX", "${GZ_FORBID_PREFIX}").rstrip("/")


def _under(path):
    if not path:
        return False
    real = os.path.realpath(path)
    return real == _forbid or real.startswith(_forbid + "/")


def _check():
    bad = [f"sys.prefix={sys.prefix}" for p in (sys.prefix,) if _under(p)]
    bad += [f"sys.executable={sys.executable}" for p in (sys.executable,) if _under(p)]
    bad += [f"sys.path entry {p or os.getcwd()!r}" for p in sys.path if _under(p or os.getcwd())]
    for name in ("torch", "numpy"):
        spec = importlib.util.find_spec(name)
        if spec is not None and _under(spec.origin):
            bad.append(f"{name} resolves to {spec.origin}")
    if bad:
        raise SystemExit("GZ node-local guard: refusing to import from " + _forbid
                         + " (NFS); " + "; ".join(bad))


if _forbid and os.environ.get("GZ_ALLOW_NFS_IMPORTS") != "1":
    _check()
PYEOF
}

build_env() {
  if env_ok; then log "env ok at $GZ_SHM_ENV"; return 0; fi
  ensure_uv
  local t0=$SECONDS
  rm -rf "$GZ_SHM_ENV"
  "$TOOLS/uv" python install "$GZ_PYTHON_VERSION"
  "$TOOLS/uv" venv --seed --python "$GZ_PYTHON_VERSION" "$GZ_SHM_ENV"
  "$TOOLS/uv" pip install --python "$GZ_SHM_ENV/bin/python" \
    --index-url https://pypi.org/simple --extra-index-url "$GZ_TORCH_INDEX" \
    --index-strategy unsafe-best-match -r "$GZ_REQUIREMENTS"
  write_sitecustomize
  "$GZ_SHM_ENV/bin/python" - <<'PYEOF'
import sys, numpy, torch
print("env check", sys.version.split()[0], sys.prefix, "numpy", numpy.__version__,
      "torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
PYEOF
  date -u +%FT%TZ > "$GZ_SHM_ENV/.staged_ok"
  log "env built in $((SECONDS - t0)) s at $GZ_SHM_ENV"
}

src_ok() {
  [ -f "$1/.gd_built" ] && (cd "$1" && PYTHONPATH="$1/python" "$GZ_SHM_ENV/bin/python" -c "import gd._gd_core") >/dev/null 2>&1
}

verify_identity() {
  # source-identity.json (written by the campaign packer) lists per-file sha256.
  local dir=$1
  if [ ! -f "$dir/source-identity.json" ]; then
    # The trainer records git identity instead, so a tree without either fails at startup.
    [ -d "$dir/.git" ] && { log "git checkout; trainer records its revision"; return 0; }
    log "neither source-identity.json nor .git in the staged tree; the trainer cannot identify it"
    return 1
  fi
  "$GZ_SHM_ENV/bin/python" - "$dir" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
files = json.loads((root / "source-identity.json").read_text())["files"]
bad = [p for p, h in files.items()
       if not (root / p).is_file() or hashlib.sha256((root / p).read_bytes()).hexdigest() != h]
print(f"source identity: {len(files) - len(bad)}/{len(files)} files match", flush=True)
sys.exit(1 if bad else 0)
PYEOF
}

localize_rpath() {
  # An extension built inside the NFS conda env carries RPATH=<env>/lib, so the
  # loader would map libstdc++/libgcc_s from NFS. Copy those libs next to it and
  # point RPATH at them; rename into place so running processes keep their inode.
  local so=$1 rpath lib dir
  if [ ! -x "$GZ_SHM_ENV/bin/patchelf" ]; then
    ensure_uv
    "$TOOLS/uv" pip install --python "$GZ_SHM_ENV/bin/python" patchelf
  fi
  rpath=$("$GZ_SHM_ENV/bin/patchelf" --print-rpath "$so")
  [ -n "$rpath" ] || return 0
  case ":$rpath:" in *":$GZ_FORBID_PREFIX/"*) ;; *) return 0 ;; esac
  dir="$(dirname "$so")/.libs"
  mkdir -p "$dir"
  for lib in $("$GZ_SHM_ENV/bin/patchelf" --print-needed "$so"); do
    for p in ${rpath//:/ }; do
      [ -e "$p/$lib" ] && { cp -L "$p/$lib" "$dir/$lib"; break; }
    done
  done
  cp "$so" "$so.tmp"
  "$GZ_SHM_ENV/bin/patchelf" --set-rpath '$ORIGIN/.libs' "$so.tmp"
  mv "$so.tmp" "$so"
  log "rpath of $(basename "$so"): $rpath -> \$ORIGIN/.libs ($(ls "$dir" | paste -sd' '))"
}

build_src() {
  local archive=$1 name=$2
  local dest="$GZ_SHM_ROOT/src/$name"
  env_ok || { log "env missing; run '$0 env' first"; return 1; }
  if src_ok "$dest"; then
    for so in "$dest"/python/gd/_gd_core*.so; do localize_rpath "$so"; done
    log "source ok at $dest"; return 0
  fi
  local t0=$SECONDS
  rm -rf "$dest" "$dest.partial"
  mkdir -p "$dest.partial"
  if [ -d "$archive" ]; then
    tar -C "$archive" -cf - --exclude=./build --exclude=__pycache__ . | tar -C "$dest.partial" -xf -
  else
    # One sequential read of a single file: cheap on NFS, unlike a tree walk.
    cp "$archive" "$GZ_SHM_ROOT/tmp/src-$name.tar.gz"
    tar -xzf "$GZ_SHM_ROOT/tmp/src-$name.tar.gz" -C "$dest.partial"
    rm -f "$GZ_SHM_ROOT/tmp/src-$name.tar.gz"
  fi
  verify_identity "$dest.partial"
  mv "$dest.partial" "$dest"
  local py="$GZ_SHM_ENV/bin/python"
  # A staged tree that ships its extension (as the campaign snapshots do) keeps
  # it: rebuilding could change the engine the run was started with.
  if compgen -G "$dest/python/gd/_gd_core*.so" >/dev/null \
      && (cd "$dest" && PYTHONPATH="$dest/python" "$py" -c "import gd._gd_core") 2>/dev/null; then
    log "using the tree's prebuilt $(basename "$dest"/python/gd/_gd_core*.so)"
    for so in "$dest"/python/gd/_gd_core*.so; do localize_rpath "$so"; done
    date -u +%FT%TZ > "$dest/.gd_built"
    log "source $name staged in $((SECONDS - t0)) s at $dest"
    return 0
  fi
  "$GZ_SHM_ENV/bin/cmake" -S "$dest" -B "$dest/build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_MAKE_PROGRAM="$GZ_SHM_ENV/bin/ninja" -DPython_EXECUTABLE="$py" \
    -Dpybind11_DIR="$("$py" -m pybind11 --cmakedir)" >/dev/null
  "$GZ_SHM_ENV/bin/cmake" --build "$dest/build" --target _gd_core -j "${SLURM_CPUS_PER_TASK:-8}"
  (cd "$dest" && PYTHONPATH="$dest/python" "$py" -c "import gd._gd_core as m; print('gd._gd_core', m.__file__)")
  date -u +%FT%TZ > "$dest/.gd_built"
  log "source $name staged+built in $((SECONDS - t0)) s at $dest"
}

with_lock() {
  local lock="$GZ_SHM_ROOT/.build.lock"
  exec 9>"$lock"
  flock -w "${GZ_BUILD_LOCK_WAIT:-3600}" 9 || { log "timed out waiting for $lock"; return 1; }
  "$@"
}

case "${1:-}" in
  env) with_lock build_env ;;
  src) [ $# -eq 3 ] || { echo "usage: $0 src ARCHIVE|DIR NAME" >&2; exit 2; }
       with_lock build_src "$2" "$3" ;;
  *) echo "usage: $0 env | src ARCHIVE|DIR NAME" >&2; exit 2 ;;
esac
