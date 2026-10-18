"""Same-node shared-memory all-reduce for the data-parallel history trainer (opt-in).

``ShmAllReduce`` sums one float32 vector per rank through a single ``/dev/shm``
mapping instead of gloo's TCP-loopback ring. Every rank writes its vector into
its own slot and waits at a barrier; then rank ``r`` folds block ``r`` of the
vector over all slots into a shared result, in the order gloo's ring
all-reduce adds that block, and waits at a second barrier; every rank then
reads the same result bytes.

gloo's order (CPU ring, torch 2.14; derived and verified bitwise by probing):
with ``bytes = numel * 4``, ``segments = roundUp(max(2W, ceil(bytes / 1 MiB)), W)``,
``segment = roundUp(ceil(bytes / segments), 4)`` bytes and ``chunk = (segments / W)
* segment / 4`` elements, block ``b = [b * chunk, (b + 1) * chunk)`` (clipped) is
summed as ``((x_s + x_{s-1}) + x_{s-2}) + ...`` over all ranks with
``s = (b - 1) mod W`` (indices mod W). IEEE addition commutes and nothing is
contracted, so the fold is bitwise gloo's sum on every rank (as long as the
folding thread does not flush denormals, e.g. ``torch.set_flush_denormal(True)``,
while gloo's threads do not). The model is empirical; ``self_check`` compares
seeded wide-exponent probes of the real size with ``dist.all_reduce`` and the
trainer falls back to gloo on any difference.

Layout (``layout``): one 64-byte flag line per rank, then ``world`` float32
slots and two result buffers, each padded to a cache line. The results
alternate between calls, so the returned view stays valid until the call after
next.

Lifecycle: rank 0 checks the free space, creates the file (``O_EXCL``, mode
0600) and reserves its pages with ``posix_fallocate``, so a too-small
``/dev/shm`` fails at start-up (on every rank; the trainer then stays on gloo)
instead of raising SIGBUS mid-run. Every rank maps it; rank 0 unlinks it as soon as all ranks have mapped
it, so nothing is left in ``/dev/shm`` even if a rank is killed later. ``close``
unmaps.

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

GLOO_SEGMENT_BYTES = 1 << 20      # gloo ring all-reduce's segment cap (empirical)
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


def gloo_ring_chunk(numel: int, world: int, element: int = 4) -> int:
    """Elements per rank block of gloo's CPU ring all-reduce of ``numel`` elements
    of ``element`` bytes (see the module docstring)."""
    total = numel * element
    segments = -(-max(2 * world, -(-total // GLOO_SEGMENT_BYTES)) // world) * world
    segment = -(-(-(-total // segments)) // element) * element
    return (segments // world) * segment // element


def layout(world: int, numel: int) -> tuple[int, int, int]:
    """``(chunk, stride, bytes)``: gloo's per-rank block in elements, the slot
    stride in floats (a multiple of one cache line) and the mapping's size."""
    if world < 1 or numel < 1:
        raise ValueError("world and numel must be positive")
    stride = -(-numel // ALIGN) * ALIGN
    return gloo_ring_chunk(numel, world), stride, 8 * FLAG_WORDS * world + 4 * stride * (world + 2)


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
    """Sum of one float32 vector per rank over a shared mapping, in gloo's order.

    Collective: every rank of the default process group constructs it with the
    same ``numel`` and calls ``all_reduce`` (and ``self_check``) the same number
    of times.
    """

    def __init__(self, rank: int, world: int, numel: int, *,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS, spin: bool | None = None,
                 directory: str = SHM_DIRECTORY) -> None:
        self.rank, self.world, self.numel = int(rank), int(world), int(numel)
        self.chunk, self.stride, self.size = layout(self.world, self.numel)
        if self.chunk * self.world < self.numel:
            raise ValueError("the reduce blocks do not cover the vector")
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
        directly). Returns gloo's sum as a view of the shared result,
        identical on every rank and valid until the call after next; callers
        copy it out and never write to it."""
        if self._mm is None:
            raise RuntimeError("shared-memory all-reduce is closed")
        fill(self.slots[self.rank, :self.numel])
        result = self.results[self.calls % 2]
        self.calls += 1
        self.barrier()
        begin = min(self.rank * self.chunk, self.numel)
        end = min(begin + self.chunk, self.numel)
        if begin < end:
            out, slots, world = result[begin:end], self.slots, self.world
            if world == 1:
                out.copy_(slots[0, begin:end])
            else:
                # gloo's ring adds block b descending from rank (b - 1) mod W.
                first = (self.rank - 1) % world
                torch.add(slots[first, begin:end], slots[(first - 1) % world, begin:end], out=out)
                for step in range(2, world):
                    out.add_(slots[(first - step) % world, begin:end])
        self.barrier()
        return result[:self.numel]

    def self_check(self, trials: int = 3, seed: int = 20261018) -> int:
        """Collective: element mismatches, over all ranks, between this reduce and
        ``dist.all_reduce`` on ``trials`` seeded wide-exponent probes of the real
        size (zero means bitwise gloo's sum here). One rank contributes zeros in
        the last probe, as a rank without a real minibatch does."""
        mismatches = 0
        for trial in range(trials):
            g = torch.Generator().manual_seed(seed + 7919 * trial + self.rank)
            scale = torch.exp2(torch.randint(-24, 24, (self.numel,), generator=g).float())
            probe = torch.randn(self.numel, generator=g) * scale
            if trial == trials - 1 and self.rank == self.world - 1:
                probe.zero_()
            shared = self.all_reduce(lambda slot: slot.copy_(probe)).clone()
            dist.all_reduce(probe)
            mismatches += int((shared.view(torch.int32) != probe.view(torch.int32)).sum())
        total = torch.tensor([mismatches], dtype=torch.long)
        dist.all_reduce(total)
        return int(total)

    def close(self) -> None:
        """Unmap. Idempotent; the file itself was unlinked at construction."""
        mm, self._mm = self._mm, None
        self.slots = self.results = self._flags = None
        if mm is not None:
            try:
                mm.close()
            except BufferError:
                pass    # a caller still holds a view; the mapping goes with it
