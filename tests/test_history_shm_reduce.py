"""Shared-memory all-reduce (train/history_shm_reduce.py): every rank gets
gloo's ring sum bit for bit; the self-check catches a wrong order model; the
mapping is unlinked at start-up; barriers time out; a too-small /dev/shm fails
on every rank."""
import hashlib
import os
import socket
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train import history_shm_reduce
from train.history_shm_reduce import ShmAllReduce, gloo_ring_chunk, layout


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
        report = dict(rank=rank, unlinked=True, mismatches=0, digests=[])
        for numel in (1, 7, 1037, 300_001, 2_156_784):
            shm = ShmAllReduce(rank, world, numel, spin=spin)
            dist.barrier()                      # rank 0 has returned and unlinked
            report["unlinked"] &= not os.path.exists(shm.path)
            g = torch.Generator().manual_seed(100 + rank)
            for call in range(4 if numel > 10**6 else 8):
                # Wide exponents and signed zeros, so the summation order shows in the bits.
                scale = torch.exp2(torch.randint(-20, 20, (numel,), generator=g).float())
                x = torch.randn(numel, generator=g) * scale
                x[call % 5::17] = -0.0
                if call % 3 == 2 and rank == world - 1:
                    x.zero_()                 # a rank without a real minibatch
                if call % world == rank:
                    time.sleep(0.002 * (call % 4))       # skew the arrivals
                out = shm.all_reduce(lambda slot: slot.copy_(x)).clone()
                dist.all_reduce(x)
                report["mismatches"] += int((_bits(out) != _bits(x)).sum())
                report["digests"].append(hashlib.sha256(_bits(out).numpy().tobytes()).hexdigest())
            shm.close()
            shm.close()                        # idempotent
        with pytest.raises(RuntimeError):
            shm.all_reduce(lambda slot: None)
        queue.put(report)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world,spin", [(2, True), (3, True), (4, True), (3, False)])
def test_sum_is_gloo_ring_sum_bitwise_on_every_rank(world, spin):
    results = _spawn(_sum_worker, world, spin)
    assert all(r["unlinked"] for r in results)
    assert all(r["mismatches"] == 0 for r in results)
    assert all(r["digests"] == results[0]["digests"] for r in results)


def _check_worker(rank, world, port, queue, corrupt):
    _init(rank, world, port)
    try:
        if corrupt:       # blocks still cover the vector, but at the wrong boundaries
            original = history_shm_reduce.gloo_ring_chunk
            history_shm_reduce.gloo_ring_chunk = lambda n, w, e=4: original(n, w, e) + 1000
        shm = ShmAllReduce(rank, world, 838_128)
        queue.put(dict(rank=rank, mismatches=shm.self_check()))
        shm.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("corrupt", [False, True])
def test_self_check_detects_a_wrong_order_model(corrupt):
    results = _spawn(_check_worker, 3, corrupt)
    assert len({r["mismatches"] for r in results}) == 1      # every rank sees the total
    assert (results[0]["mismatches"] > 0) == corrupt


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


def test_layout_follows_gloo_segments_and_pads_slots():
    # Small vectors: 2W segments; large ones: segments of at most 1 MiB, W | segments.
    assert gloo_ring_chunk(1000, 3) == 334 and gloo_ring_chunk(1, 3) == 2
    assert gloo_ring_chunk(2_156_784, 8) == 269_598      # 16 segments of 539,196 B
    for world in (2, 3, 4, 5, 8):
        for numel in (1, 7, 1001, 2_156_784, 5_756_528):
            assert world * gloo_ring_chunk(numel, world) >= numel
    chunk, stride, size = layout(3, 1000)
    assert chunk == 334 and stride == 1008
    assert size == 3 * 64 + 4 * 1008 * 5
