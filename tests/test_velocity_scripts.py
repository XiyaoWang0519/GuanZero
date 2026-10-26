"""Shell-level tests for scripts/velocity (Velocity node-local staging).

The scripts run inside SLURM batch shells that use ``set -euo pipefail``, so
every case runs the real bash under that mode with stubbed scontrol/nvidia-smi.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VELOCITY = ROOT / "scripts" / "velocity"
LIB = VELOCITY / "gz_job_lib.sh"
BUILD = VELOCITY / "build_shm_env.sh"

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("rsync") is None or shutil.which("flock") is None,
    reason="Velocity scripts need Linux with rsync and flock",
)

NVIDIA_SMI_Q = "".join(
    f"GPU 0000:{i:02d}:00.0\n    GPU UUID : GPU-{'abcd'[i]}\n    Minor Number : {i}\n" for i in range(4)
)


def _write_exe(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)


def _stub_bin(tmp_path: Path, scontrol: str | None) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    if scontrol is None:
        _write_exe(bin_dir / "scontrol", "#!/bin/sh\nexit 1\n")
    else:
        (tmp_path / "scontrol.out").write_text(scontrol)
        _write_exe(bin_dir / "scontrol", f"#!/bin/sh\ncat '{tmp_path / 'scontrol.out'}'\n")
    (tmp_path / "nvidia-smi.out").write_text(NVIDIA_SMI_Q)
    _write_exe(bin_dir / "nvidia-smi", f"#!/bin/sh\ncat '{tmp_path / 'nvidia-smi.out'}'\n")
    return bin_dir


def _run_lib(body: str, env: dict[str, str] | None = None, path: Path | None = None,
             timeout: float = 30) -> subprocess.CompletedProcess[str]:
    full_env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "GZ_", "CUDA_"))}
    if path is not None:
        full_env["PATH"] = f"{path}:{full_env['PATH']}"
    full_env.update(env or {})
    script = f"set -euo pipefail\nsource '{LIB}'\n{body}\n"
    return subprocess.run(["bash", "-c", script], env=full_env, capture_output=True, text=True, timeout=timeout)


def _job(gres: str, tres_gpu: int | None) -> str:
    tres = "cpu=4,mem=8G,node=1,billing=4" + (f",gres/gpu={tres_gpu}" if tres_gpu is not None else "")
    return (f"JobId=7 JobName=t\n   AllocTRES={tres}\n"
            f"     Nodes=max-compute-11 CPU_IDs=0-3 Mem=8192 GRES={gres}\n")


SHOW_GPUS = 'gz_restrict_gpus\necho "CVD=[${CUDA_VISIBLE_DEVICES-unset}] N=$GZ_ALLOWED_GPU_COUNT"'


@pytest.mark.parametrize(
    ("gres", "tres", "expected"),
    [
        ("", None, "CVD=[] N=0"),
        ("gpu:l40s:1(IDX:2)", 1, "CVD=[GPU-c] N=1"),
        ("gpu:l40s:2(IDX:0-1)", 2, "CVD=[GPU-a,GPU-b] N=2"),
        ("gpu:l40s:2(IDX:1,3)", 2, "CVD=[GPU-b,GPU-d] N=2"),
    ],
)
def test_gpu_allocation_parsed(tmp_path: Path, gres: str, tres: int | None, expected: str) -> None:
    bin_dir = _stub_bin(tmp_path, _job(gres, tres))
    proc = _run_lib(SHOW_GPUS, {"SLURM_JOB_ID": "7"}, bin_dir)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == expected


def test_cpu_only_job_hides_gpus_without_nvidia_smi(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "scontrol.out").write_text(_job("", None))
    _write_exe(bin_dir / "scontrol", f"#!/bin/sh\ncat '{tmp_path / 'scontrol.out'}'\n")
    proc = _run_lib(SHOW_GPUS, {"SLURM_JOB_ID": "7", "PATH": f"{bin_dir}:/usr/bin:/bin"})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "CVD=[] N=0"


@pytest.mark.parametrize(
    ("gres", "tres", "message"),
    [
        ("gpu:l40s:1(IDX:x)", 1, "malformed"),
        ("gpu:l40s:1(IDX:3-1)", None, "malformed"),
        ("", 1, "contradicts"),
        ("gpu:l40s:1(IDX:0)", 2, "contradicts"),
    ],
)
def test_bad_gpu_allocation_rejected(tmp_path: Path, gres: str, tres: int | None, message: str) -> None:
    bin_dir = _stub_bin(tmp_path, _job(gres, tres))
    proc = _run_lib(SHOW_GPUS, {"SLURM_JOB_ID": "7"}, bin_dir)
    assert proc.returncode == 2
    assert message in proc.stderr
    assert "CVD=" not in proc.stdout


def test_scontrol_failure_falls_back_to_slurm_env(tmp_path: Path) -> None:
    bin_dir = _stub_bin(tmp_path, None)
    stub_sleep = 'sleep() { :; }\n'
    proc = _run_lib(stub_sleep + SHOW_GPUS, {"SLURM_JOB_ID": "7", "SLURM_JOB_GPUS": "1"}, bin_dir)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "CVD=[GPU-b] N=1"
    proc = _run_lib(stub_sleep + SHOW_GPUS, {"SLURM_JOB_ID": "7"}, bin_dir)
    assert proc.returncode == 2
    assert "cannot tell the GPU allocation" in proc.stderr


def _run_dirs(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    r, r_home = tmp_path / "shm-run", tmp_path / "home-run"
    (r / "segments" / "s1").mkdir(parents=True)
    r_home.mkdir()
    return r, r_home, {"GZ_SYNC_ENABLED": "1", "R": str(r), "R_HOME": str(r_home)}


def test_sync_without_periodic_checkpoints(tmp_path: Path) -> None:
    r, r_home, env = _run_dirs(tmp_path)
    (r / "segments" / "s1" / "latest.pt").write_bytes(b"ckpt")
    (r / "orchestrator.log").write_text("line\n")
    proc = _run_lib('gz_sync_back\necho "rc=$?"', env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "rc=0"
    assert (r_home / "segments" / "s1" / "latest.pt").read_bytes() == b"ckpt"
    assert (r_home / "orchestrator.log").read_text() == "line\n"


def test_sync_prunes_all_but_newest_update(tmp_path: Path) -> None:
    r, r_home, env = _run_dirs(tmp_path)
    for n in (9, 10, 11):
        (r / "segments" / "s1" / f"update-{n}.pt").write_bytes(b"x" * n)
    proc = _run_lib("gz_sync_back", env)
    assert proc.returncode == 0, proc.stderr
    assert sorted(p.name for p in (r / "segments" / "s1").iterdir()) == ["update-11.pt"]
    assert len(list((r_home / "segments" / "s1").glob("update-*.pt"))) == 3


def test_sync_is_off_unless_the_run_was_staged(tmp_path: Path) -> None:
    r = tmp_path / "kit-owned-run"
    r.mkdir()
    proc = _run_lib('gz_sync_back\ngz_start_sync\necho "pid=[${GZ_SYNC_PID:-}]"\ngz_final_sync\necho done',
                    {"R": str(r)})
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.split() == ["pid=[]", "done"]


def test_final_sync_does_not_wait_out_the_interval(tmp_path: Path) -> None:
    r, r_home, env = _run_dirs(tmp_path)
    (r / "late.json").write_text("{}")
    body = "GZ_SHM_SYNC_SECS=300\ngz_start_sync\nsleep 0.3\ngz_final_sync"
    t0 = time.monotonic()
    proc = _run_lib(body, env)
    assert proc.returncode == 0, proc.stderr
    assert time.monotonic() - t0 < 10
    assert (r_home / "late.json").exists()


def test_persistent_sync_failure_stops_the_job(tmp_path: Path) -> None:
    r, _, env = _run_dirs(tmp_path)
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    env["R_HOME"] = str(blocker / "run")
    body = """
GZ_SHM_SYNC_SECS=1 GZ_SHM_SYNC_MAX_FAILS=2
trap 'echo "kit got TERM"; exit 0' TERM
trap gz_on_exit EXIT
gz_start_sync
for _ in $(seq 300); do sleep 0.1; done
echo "kept training"
"""
    proc = _run_lib(body, env, timeout=60)
    assert proc.returncode == 4, proc.stderr
    assert proc.stdout.strip() == "kit got TERM"
    assert "checkpoints are not reaching" in proc.stderr
    assert (r / ".gz-sync-failed").exists()
    assert (r / ".gz-sync-status").read_text().startswith("failing fails=2")


def test_shm_root_must_be_private(tmp_path: Path) -> None:
    fresh = tmp_path / "fresh"
    proc = _run_lib(f"gz_secure_dir '{fresh}'")
    assert proc.returncode == 0, proc.stderr
    assert stat.S_IMODE(fresh.stat().st_mode) == 0o700

    loose = tmp_path / "loose"
    loose.mkdir(mode=0o755)
    loose.chmod(0o755)
    proc = _run_lib(f"gz_secure_dir '{loose}'")
    assert proc.returncode == 2
    assert "mode 755" in proc.stderr

    link = tmp_path / "link"
    link.symlink_to(fresh)
    proc = _run_lib(f"gz_secure_dir '{link}'")
    assert proc.returncode == 2
    assert "symlink" in proc.stderr


def _fake_env(root: Path) -> Path:
    env = root / "gz-env-local"
    (env / "bin").mkdir(parents=True)
    stubs = root / "stubs"
    stubs.mkdir()
    (stubs / "numpy.py").write_text("")
    (stubs / "torch.py").write_text("")
    _write_exe(env / "bin" / "python",
               f'#!/bin/sh\nPYTHONPATH="{stubs}${{PYTHONPATH:+:$PYTHONPATH}}" exec "{sys.executable}" "$@"\n')
    _write_exe(env / "bin" / "patchelf", "#!/bin/sh\nexit 0\n")
    (env / ".staged_ok").write_text("test\n")
    return env


def _source_tree(path: Path, version: str) -> Path:
    gd = path / "python" / "gd"
    gd.mkdir(parents=True)
    (gd / "__init__.py").write_text("")
    (gd / "_gd_core.py").write_text(f"VERSION = {version!r}\n")
    (gd / "_gd_core.prebuilt.so").write_bytes(b"\0")
    files = {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in path.rglob("*") if p.is_file()}
    (path / "source-identity.json").write_text(json.dumps({"files": files}))
    return path


def _tar(src: Path, out: Path) -> Path:
    with tarfile.open(out, "w:gz") as tf:
        tf.add(src, arcname=".")
    return out


def _stage(root: Path, source: Path, name: str) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GZ_", "PYTHON"))}
    env.update({"GZ_SHM_ROOT": str(root), "GZ_SHM_ENV": str(root / "gz-env-local"),
                "GZ_FORBID_PREFIX": "/nonexistent-nfs"})
    return subprocess.run(["bash", str(BUILD), "src", str(source), name], env=env,
                          capture_output=True, text=True, timeout=60)


@pytest.fixture
def shm_root(tmp_path: Path) -> Path:
    root = tmp_path / "shm"
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    _fake_env(root)
    return root


def test_source_stage_is_keyed_by_identity(tmp_path: Path, shm_root: Path) -> None:
    a = _tar(_source_tree(tmp_path / "a", "A"), tmp_path / "a.tar.gz")
    b = _tar(_source_tree(tmp_path / "b", "B"), tmp_path / "b.tar.gz")

    first = _stage(shm_root, a, "campaign")
    assert first.returncode == 0, first.stderr
    path_a = Path(first.stdout.strip())
    assert first.stdout.count("\n") == 1
    assert path_a.name.startswith("campaign-")
    assert "VERSION = 'A'" in (path_a / "python/gd/_gd_core.py").read_text()

    second = _stage(shm_root, b, "campaign")
    assert second.returncode == 0, second.stderr
    path_b = Path(second.stdout.strip())
    assert path_b != path_a
    assert "VERSION = 'B'" in (path_b / "python/gd/_gd_core.py").read_text()
    assert path_a.is_dir()

    again = _stage(shm_root, a, "campaign")
    assert again.returncode == 0, again.stderr
    assert Path(again.stdout.strip()) == path_a
    assert "source ok" in again.stderr

    from_dir = _stage(shm_root, tmp_path / "a", "campaign")
    assert from_dir.returncode == 0, from_dir.stderr
    assert "VERSION = 'A'" in (Path(from_dir.stdout.strip()) / "python/gd/_gd_core.py").read_text()


def test_source_without_identity_is_never_shared(tmp_path: Path, shm_root: Path) -> None:
    src = _source_tree(tmp_path / "plain", "P")
    (src / "source-identity.json").unlink()
    (src / ".git").mkdir()
    first, second = _stage(shm_root, src, "co"), _stage(shm_root, src, "co")
    assert first.returncode == 0 and second.returncode == 0, first.stderr + second.stderr
    assert first.stdout.strip() != second.stdout.strip()


def _hostile_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> Path:
    with tarfile.open(path, "w") as tf:
        for info, data in members:
            tf.addfile(info, io.BytesIO(data) if data is not None else None)
    return path


def _file_member(name: str, data: bytes = b"pwned") -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    return info, data


def _link_member(name: str, target: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info, None


@pytest.mark.parametrize(
    "kind", ["dotdot", "absolute-symlink", "relative-symlink", "write-through-symlink"])
def test_hostile_archive_cannot_escape(tmp_path: Path, shm_root: Path, kind: str) -> None:
    victim = tmp_path / "victim"
    victim.mkdir()
    members = {
        "dotdot": [_file_member("../../../victim/owned")],
        "absolute-symlink": [_link_member("out", str(victim))],
        "relative-symlink": [_link_member("out", "../../../victim")],
        "write-through-symlink": [_link_member("python", str(victim)), _file_member("python/owned")],
    }[kind]
    archive = _hostile_tar(tmp_path / f"{kind}.tar", members)
    proc = _stage(shm_root, archive, "hostile")
    assert proc.returncode != 0
    assert re.search(r"(AbsoluteLink|LinkOutsideDestination|OutsideDestination)Error", proc.stderr), proc.stderr
    assert proc.stdout == ""
    assert list(victim.iterdir()) == []
    assert not list((shm_root / "src").glob("hostile-*[0-9a-f]"))
