"""Shared-memory all-reduce (train/history_shm_reduce.py): every rank gets the same
fixed rank-order sum, equal to gloo's at two ranks; the mapping is unlinked at
start-up; barriers time out; a too-small /dev/shm fails on every rank."""
import hashlib
import os
import socket
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train.history_shm_reduce import ShmAllReduce, layout


def _spawn(target, world: int, *args) -> list[dict]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=target, args=(r, world, port, queue, *args))
                 for r in range(world)]
    for p in processes:
        p.start()
    results = sorted((queue.get(timeout=300) for _ in processes), key=lambda r: r["rank"])
    for p in processes:
        p.join(timeout=60)
        assert p.exitcode == 0
    return results


def _init(rank: int, world: int, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_num_threads(1)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def _sum_worker(rank, world, port, queue, spin):
    _init(rank, world, port)
    try:
        n = 1000 + 37                       # not a multiple of the chunk alignment
        shm = ShmAllReduce(rank, world, n, spin=spin)
        unlinked = not os.path.exists(shm.path)
        g = torch.Generator().manual_seed(100 + rank)
        ordered = gloo = True
        digests = []
        for call in range(24):
            # Wide exponents and signed zeros, so the summation order shows in the bits.
            x = (torch.randn(n, generator=g) * torch.exp(4 * torch.randn(n, generator=g))).float()
            x[call % 5::17] = -0.0
            if call % world == rank:
                time.sleep(0.002 * (call % 4))       # skew the arrivals
            out = shm.all_reduce(lambda slot: slot.copy_(x)).clone()
            gathered = [torch.empty_like(x) for _ in range(world)]
            dist.all_gather(gathered, x)
            expected = gathered[0] + gathered[1]
            for other in gathered[2:]:
                expected = expected + other
            ordered &= torch.equal(_bits(out), _bits(expected))
            if world == 2:
                reduced = x.clone()
                dist.all_reduce(reduced)
                gloo &= torch.equal(_bits(out), _bits(reduced))
            digests.append(hashlib.sha256(_bits(out).numpy().tobytes()).hexdigest())
        shm.close()
        shm.close()                        # idempotent
        with pytest.raises(RuntimeError):
            shm.all_reduce(lambda slot: None)
        queue.put(dict(rank=rank, unlinked=unlinked, ordered=ordered, gloo=gloo,
                       digests=digests))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,spin", [(2, True), (3, True), (2, False)])
def test_rank_order_sum_identical_on_every_rank(world, spin):
    results = _spawn(_sum_worker, world, spin)
    assert all(r["unlinked"] for r in results)
    assert all(r["ordered"] for r in results)
    assert all(r["gloo"] for r in results)          # two ranks: equal to gloo's sum bitwise
    assert all(r["digests"] == results[0]["digests"] for r in results)


def _timeout_worker(rank, world, port, queue):
    _init(rank, world, port)
    try:
        shm = ShmAllReduce(rank, world, 64, timeout=0.3, spin=True)
        message = ""
        if rank == 0:                      # rank 1 never arrives
            begin = time.monotonic()
            try:
                shm.barrier()
            except RuntimeError as error:
                message = str(error)
            elapsed = time.monotonic() - begin
        else:
            elapsed = 0.0
        dist.barrier()
        shm.close()
        queue.put(dict(rank=rank, message=message, elapsed=elapsed))
    finally:
        dist.destroy_process_group()


def test_spin_barrier_times_out():
    first, _ = _spawn(_timeout_worker, 2)
    assert "ranks [1] did not reach barrier 1" in first["message"]
    assert 0.25 < first["elapsed"] < 30


def _small_shm_worker(rank, world, port, queue):
    _init(rank, world, port)
    try:
        if rank == 0:
            class Tiny:
                f_bavail, f_frsize = 1, 4096
            os.statvfs = lambda path: Tiny()
        try:
            ShmAllReduce(rank, world, 1 << 20)
            message = ""
        except RuntimeError as error:
            message = str(error)
        queue.put(dict(rank=rank, message=message))
    finally:
        dist.destroy_process_group()


def test_too_small_shm_fails_on_every_rank():
    results = _spawn(_small_shm_worker, 2)
    for r in results:
        assert "--ddp-reduce shm needs" in r["message"] and "MiB" in r["message"]


def test_layout_pads_chunks_to_cache_lines():
    chunk, stride, size = layout(3, 1000)
    assert chunk == 336 and stride == 1008 and chunk % 16 == 0
    assert size == 3 * 64 + 4 * 1008 * 5
