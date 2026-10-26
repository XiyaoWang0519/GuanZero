"""Refuse to train from a Python environment on network storage.

On shared clusters (Velocity) the job wrapper ``scripts/velocity/gz_job_env.sh``
sets ``GZ_REQUIRE_NODE_LOCAL=1``. Spawned workers re-import torch, so a
worker started from an NFS interpreter or site-packages would multiply
small-file reads on the shared filesystem; each worker checks for itself.
"""

from __future__ import annotations

import os
import sys

import torch


def violations(prefix: str, paths: dict[str, str]) -> list[str]:
    """Names and values of ``paths`` that resolve under ``prefix``."""
    root = os.path.realpath(prefix).rstrip("/")
    bad = []
    for name, path in paths.items():
        real = os.path.realpath(path)
        if real == root or real.startswith(root + "/"):
            bad.append(f"{name}={path}")
    return bad


def mapped_files(maps: str = "/proc/self/maps") -> dict[str, str]:
    """Files mapped into this process (shared libraries included); empty off Linux."""
    try:
        with open(maps) as stream:
            paths = {line.split()[-1] for line in stream if line.split()[-1].startswith("/")}
    except OSError:
        return {}
    return {f"mapped {path}": path for path in sorted(paths)}


def require_node_local(where: str) -> None:
    """Raise if this process imports Python, torch or native libraries from
    ``GZ_FORBID_PREFIX``."""
    if os.environ.get("GZ_REQUIRE_NODE_LOCAL") != "1":
        return
    prefix = os.environ.get("GZ_FORBID_PREFIX", "/home")
    bad = violations(prefix, {"sys.prefix": sys.prefix, "sys.executable": sys.executable,
                              "torch.__file__": torch.__file__, **mapped_files()})
    if bad:
        raise RuntimeError(f"{where}: Python environment is on {prefix} (network storage), "
                           f"not node-local: {', '.join(bad)}")
