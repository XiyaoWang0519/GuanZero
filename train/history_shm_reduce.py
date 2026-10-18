"""Same-node shared-memory all-reduce for the data-parallel history trainer (opt-in).

``ShmAllReduce`` sums one float32 vector per rank through a single ``/dev/shm``
mapping instead of gloo's TCP-loopback ring. Every rank writes its vector into
its own slot, waits at a barrier, sums one contiguous chunk of the vector over
the slots in fixed rank order ``((s0 + s1) + s2) + ...`` into a shared result,
and waits at a second barrier; then every rank reads the same result bytes.
Every rank therefore holds a bitwise-identical sum. With two ranks the sum
equals gloo's bit for bit (a two-term IEEE sum commutes); with three or more
the FP32 summation order differs from gloo's ring, so ``ddp_reduce="shm"`` is a
tier-2 option (``HistoryPPOConfig.ddp_reduce``, default ``"gloo"``).

Layout (``mapping_bytes``): one 64-byte flag line per rank, then ``world``
padded float32 slots and two result buffers. The results alternate between
calls, so a rank that still reads call ``k``'s result cannot be overwritten by
call ``k + 1``: the returned view stays valid until the call after next.

Lifecycle: rank 0 checks the free space, creates the file (``O_EXCL``, mode
0600) and reserves its pages with ``posix_fallocate``, so a too-small
``/dev/shm`` fails at start-up instead of raising SIGBUS mid-run. Every rank
maps it; rank 0 unlinks it as soon as all ranks have mapped it, so nothing is
left in ``/dev/shm`` even if a rank is killed later. ``close`` unmaps.

Barriers spin on the flag lines (``sched_yield``, then 50 us sleeps) with a
deadline (``RuntimeError`` on timeout). The spin barrier relies on x86's
total store order; elsewhere ``dist.barrier`` (gloo) is used instead.
"""
from __future__ import annotations

import mmap
import os
import platform
import threading
import time
from typing import Callable
import uuid

import numpy as np
import torch
import torch.distributed as dist

SHM_DIRECTORY = "/dev/shm"
FILE_PREFIX = "guanzero-ddp-"
ALIGN = 16                  # float32 elements per 64-byte cache line
FLAG_WORDS = 8              # int64 words per rank flag: one cache line each
SPIN_YIELDS = 2000          # sched_yield polls before sleeping between polls
SPIN_SLEEP_SECONDS = 5e-5
DEFAULT_TIMEOUT_SECONDS = 1800.0   # gloo's default collective timeout
# An uncontended lock acquire/release is a locked instruction, a full fence on
# x86: it orders this rank's slot/result writes (including any non-temporal
# stores inside memcpy) before the flag store that publishes them.
_FENCE = threading.Lock()


def spin_barrier_supported() -> bool:
    """Whether the flag spin barrier's store-order assumption holds (x86)."""
    return platform.machine().lower() in ("x86_64", "amd64", "i386", "i686")


def layout(world: int, numel: int) -> tuple[int, int, int]:
    """``(chunk, stride, bytes)``: per-rank chunk and slot stride in floats (each
    a multiple of one cache line) and the mapping's size in bytes."""
    if world < 1 or numel < 1:
        raise ValueError("world and numel must be positive")
    chunk = -(-numel // world)
    chunk = -(-chunk // ALIGN) * ALIGN
    stride = chunk * world
    return chunk, stride, 8 * FLAG_WORDS * world + 4 * stride * (world + 2)


def mapping_bytes(world: int, numel: int) -> int:
    return layout(world, numel)[2]


def _create(directory: str, size: int) -> str:
    """Rank 0: a fresh mapping file of ``size`` reserved bytes; returns its path."""
    stat = os.statvfs(directory)
    free = stat.f_bavail * stat.f_frsize
    advice = (f"--ddp-reduce shm needs {size / 2**20:.1f} MiB in {directory} "
              f"({free / 2**20:.1f} MiB free); enlarge it (e.g. docker --shm-size) "
              "or use --ddp-reduce gloo")
    if free < size:
        raise RuntimeError(advice)
    path = os.path.join(directory, f"{FILE_PREFIX}{os.getpid()}-{uuid.uuid4().hex}")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        # Reserve the pages now: tmpfs would otherwise allocate on first touch
        # and a full /dev/shm would kill the run with SIGBUS mid-update.
        os.posix_fallocate(fd, 0, size)
    except OSError as error:
        os.close(fd)
        os.unlink(path)
        raise RuntimeError(f"{advice}: {error}") from error
    os.close(fd)
    return path


class ShmAllReduce:
    """Fixed rank-order sum of one float32 vector per rank over a shared mapping.

    Collective: every rank of the default process group constructs it with the
    same ``numel`` and calls ``all_reduce`` the same number of times.
    """

    def __init__(self, rank: int, world: int, numel: int, *,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS, spin: bool | None = None,
                 directory: str = SHM_DIRECTORY) -> None:
        self.rank, self.world, self.numel = int(rank), int(world), int(numel)
        self.chunk, self.stride, self.size = layout(self.world, self.numel)
        self.timeout = float(timeout)
        self.spin = spin_barrier_supported() if spin is None else bool(spin)
        message: list[str | None] = [None, None]
        if self.rank == 0:
            try:
                message = [_create(directory, self.size), None]
            except (OSError, RuntimeError) as error:
                message = [None, str(error)]
        dist.broadcast_object_list(message, src=0)
        path, error = message
        if error is not None:
            raise RuntimeError(error)
        self.path = str(path)
        self._mm: mmap.mmap | None = None
        failure = None
        try:
            try:
                fd = os.open(self.path, os.O_RDWR)
                try:
                    self._mm = mmap.mmap(fd, self.size)
                finally:
                    os.close(fd)
            except OSError as caught:
                failure = f"rank {self.rank} could not map {self.path}: {caught}"
            # Barrier and status in one: every rank has mapped the file (or failed).
            status = torch.tensor([0 if failure is None else 1], dtype=torch.long)
            dist.all_reduce(status, op=dist.ReduceOp.MAX)
        finally:
            if self.rank == 0 and os.path.exists(self.path):
                os.unlink(self.path)     # the mappings stay valid; nothing leaks
        if int(status):
            self.close()
            raise RuntimeError(failure or "shared-memory all-reduce: another rank could "
                               "not map the file (are all ranks on one node?)")
        self._flags = np.frombuffer(self._mm, np.int64, count=FLAG_WORDS * self.world)
        data = torch.from_numpy(np.frombuffer(self._mm, np.float32, offset=8 * FLAG_WORDS * self.world,
                                              count=self.stride * (self.world + 2)))
        self.slots = data[:self.world * self.stride].view(self.world, self.stride)
        self.results = data[self.world * self.stride:].view(2, self.stride)
        self.slots[self.rank].zero_()     # map this rank's pages before the first call
        self.calls = 0
        self._generation = 0

    def barrier(self) -> None:
        """All ranks reach this point; raises ``RuntimeError`` after ``timeout``."""
        if not self.spin:
            dist.barrier()
            return
        self._generation += 1
        target = self._generation
        with _FENCE:
            pass
        self._flags[FLAG_WORDS * self.rank] = target
        flags = self._flags[::FLAG_WORDS]
        spins = 0
        deadline = None
        while flags.min() < target:
            spins += 1
            if spins <= SPIN_YIELDS:
                os.sched_yield()
                continue
            now = time.monotonic()
            if deadline is None:
                deadline = now + self.timeout
            elif now > deadline:
                lagging = [r for r in range(self.world) if int(flags[r]) < target]
                raise RuntimeError(f"shared-memory all-reduce: ranks {lagging} did not reach "
                                   f"barrier {target} within {self.timeout:.0f} s")
            time.sleep(SPIN_SLEEP_SECONDS)

    def all_reduce(self, fill: Callable[[torch.Tensor], None]) -> torch.Tensor:
        """``fill(slot)`` writes this rank's ``numel`` values into ``slot``, a view
        of its slot in the mapping (a device-to-host copy can land there
        directly). Returns the rank-order sum as a view of the shared result,
        identical on every rank and valid until the call after next; callers
        copy it out and never write to it."""
        if self._mm is None:
            raise RuntimeError("shared-memory all-reduce is closed")
        fill(self.slots[self.rank, :self.numel])
        result = self.results[self.calls % 2]
        self.calls += 1
        self.barrier()
        begin = self.rank * self.chunk
        end = min(begin + self.chunk, self.numel)
        if begin < end:
            out, slots = result[begin:end], self.slots
            if self.world == 1:
                out.copy_(slots[0, begin:end])
            else:
                torch.add(slots[0, begin:end], slots[1, begin:end], out=out)
                for other in range(2, self.world):
                    out.add_(slots[other, begin:end])
        self.barrier()
        return result[:self.numel]

    def close(self) -> None:
        """Unmap. Idempotent; the file itself was unlinked at construction."""
        mm, self._mm = self._mm, None
        self.slots = self.results = self._flags = None
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                pass    # a caller still holds a view; the mapping goes with it
