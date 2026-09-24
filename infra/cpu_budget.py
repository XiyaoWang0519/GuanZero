"""Usable CPUs of this host, a thread plan for concurrent PPO arms, host facts.

Neither `nproc` nor `os.cpu_count()` is safe on a pod. GNU `nproc` honours
OMP_NUM_THREADS (B11 phase 1 ran 4 engine threads per arm on 64 vCPUs because
its kit exported OMP_NUM_THREADS=4 first), and `os.cpu_count()` or
`nproc --all` ignore the cpuset and the CPU quota (a B8-type host showed 256
CPUs with 32 allocated). `usable_cpus` takes the smaller of the affinity mask
and the cgroup CPU quota, and ignores OMP_NUM_THREADS.

    python -m infra.cpu_budget --arms 3                  # plan, human-readable
    python -m infra.cpu_budget --arms 3 --field engine_threads   # one number
    python -m infra.cpu_budget --facts runs/host-facts.json      # provenance

Plan rule: one core per Python process (each arm's trainer, plus its actor
processes), the rest split evenly into engine threads per rollout process
(`num_threads`, which every actor process gets in full) up to 24, and torch
threads at most 4 (the CPU side of a GPU run is small).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

CGROUP = Path("/sys/fs/cgroup")
SELF_CGROUP = Path("/proc/self/cgroup")
# More engine threads per process were never measured to help (scan, B11).
MAX_ENGINE_THREADS = 24


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _v2_groups(root: Path, self_cgroup: Path) -> list[Path]:
    """This process's cgroup v2 directory and its ancestors up to `root`. A
    container with a private cgroup namespace sees its own group at the
    mount root; with the host's namespace the limit sits in a nested group
    named by /proc/self/cgroup."""
    groups = [root]
    for line in (_read(self_cgroup) or "").splitlines():
        if line.startswith("0::/"):
            path = root
            for part in Path(line[4:]).parts:
                path = path / part
                groups.append(path)
    return groups


def cgroup_quota(root: Path = CGROUP, self_cgroup: Path = SELF_CGROUP) -> float | None:
    """Tightest CPU quota in CPUs from cgroup v2 `cpu.max` (this group and its
    ancestors) or v1 `cfs_quota_us`, or None when unlimited or unknown."""
    quotas = []
    for group in _v2_groups(root, self_cgroup):
        quota, _, period = (_read(group / "cpu.max") or "").partition(" ")
        if quota and quota != "max" and period:
            quotas.append(int(quota) / int(period))
    for directory in (root / "cpu", root / "cpu,cpuacct", root):
        quota, period = _read(directory / "cpu.cfs_quota_us"), _read(directory / "cpu.cfs_period_us")
        if quota and period and int(quota) > 0:
            quotas.append(int(quota) / int(period))
    return min(quotas) if quotas else None


def affinity_cpus() -> int:
    """CPUs this process may run on (the cpuset); all logical CPUs where the
    platform has no affinity call (macOS)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def usable_cpus(root: Path = CGROUP, self_cgroup: Path = SELF_CGROUP) -> int:
    cpus = affinity_cpus()
    quota = cgroup_quota(root, self_cgroup)
    if quota is not None:
        cpus = min(cpus, max(1, math.floor(quota)))
    return cpus


def thread_plan(cpus: int, arms: int, actors: int = 0) -> dict[str, int]:
    """Threads per rollout process for `arms` concurrent trainers with
    `actors` actor processes each (0: in-process rollout)."""
    if cpus < 1 or arms < 1 or actors < 0:
        raise ValueError("cpus and arms must be positive and actors nonnegative")
    rollout = arms * max(1, actors)
    python = arms * (1 + actors)
    per_process = max(1, (cpus - python) // rollout)
    return {"cpus": cpus, "arms": arms, "actors": actors,
            "engine_threads": min(MAX_ENGINE_THREADS, per_process),
            "torch_threads": max(1, min(4, per_process // 4))}


def _run(command: list[str]) -> str | None:
    try:
        out = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _cpu_model() -> str | None:
    info = _read(Path("/proc/cpuinfo"))
    if info:
        for line in info.splitlines():
            if line.startswith("model name"):
                return line.partition(":")[2].strip()
    return _run(["sysctl", "-n", "machdep.cpu.brand_string"])


def host_facts(root: Path = CGROUP) -> dict[str, Any]:
    """What decides a run's speed on this host, for the run directory: every
    probe is optional and None when unavailable."""
    # Throttling counters: cgroup v2 at the root, v1 in the cpu controller.
    stat = next((text for text in (_read(root / "cpu.stat"), _read(root / "cpu" / "cpu.stat"),
                                   _read(root / "cpu,cpuacct" / "cpu.stat")) if text), None)
    return {
        "platform": platform.platform(), "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(), "affinity_cpus": affinity_cpus(),
        "cgroup_cpu_max": _read(root / "cpu.max"), "cgroup_quota_cpus": cgroup_quota(root),
        "cgroup_cpu_stat": dict(line.split(" ", 1) for line in stat.splitlines()) if stat else None,
        "usable_cpus": usable_cpus(root),
        "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "gpu": _run(["nvidia-smi", "--query-gpu=name,pcie.link.gen.current,"
                     "pcie.link.width.current,clocks.max.sm,power.limit",
                     "--format=csv,noheader"]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arms", type=int, default=1, help="concurrent trainers on this host")
    parser.add_argument("--actors", type=int, default=0, help="actor processes per trainer")
    parser.add_argument("--field", choices=("cpus", "engine_threads", "torch_threads"),
                        help="print only this number")
    parser.add_argument("--json", action="store_true", help="print the plan as JSON")
    parser.add_argument("--facts", type=Path, help="write host facts to this JSON file")
    parser.add_argument("--cgroup-root", type=Path, default=CGROUP, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.facts:
        args.facts.parent.mkdir(parents=True, exist_ok=True)
        args.facts.write_text(json.dumps(host_facts(args.cgroup_root), indent=2) + "\n")
    plan = thread_plan(usable_cpus(args.cgroup_root), args.arms, args.actors)
    if args.field:
        print(plan[args.field])
    elif args.json:
        print(json.dumps(plan))
    elif not args.facts:
        print(f"{plan['cpus']} usable CPUs, {plan['arms']} arm(s) x {plan['actors']} actor(s): "
              f"num_threads {plan['engine_threads']}, torch_threads {plan['torch_threads']}")


if __name__ == "__main__":
    main()
