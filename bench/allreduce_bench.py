"""Gradient all-reduce transports of the data-parallel trainer: gloo vs /dev/shm.

Spawns ``--world`` local ranks and times one production-shaped gradient reduce
per call, the two transports interleaved (the order alternates every
iteration) so machine load affects both alike. Each timed call starts after an
alignment barrier, so rank arrival skew is excluded, and its time is the
maximum over ranks. Per call:

- gloo: ``flat.clone()`` (the fresh host tensor ``.cpu()`` makes on a GPU host),
  ``dist.all_reduce`` over gloo, then a copy of the sum into a persistent
  destination (standing in for the host-to-device copy; the division by the
  count runs on the device);
- shm: the copy into this rank's slot of the mapping (where the device-to-host
  copy lands), ``ShmAllReduce.all_reduce`` (barrier, this rank's block folded
  in gloo's ring order, barrier) and the same copy out. ``shm_fill`` is the
  slot copy alone and ``shm_reduce`` the rest of ``all_reduce`` (two barriers
  and the block fold).

The default vector is the main lineage's actor + critic gradient (2,156,783
float32) plus the count element: 8.63 MB. CPU only; the device copies of a GPU
host are not included. The untimed warmup calls also check that both transports
give the same bits. Example::

    PYTHONPATH=python:. python -m bench.allreduce_bench --world 4 --iters 30
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train.history_shm_reduce import ShmAllReduce

PRODUCTION_NUMEL = 1_509_870 + 646_913 + 1


def _worker(rank: int, world: int, port: int, args, queue) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        shm = ShmAllReduce(rank, world, args.numel, spin=not args.gloo_barrier)
        g = torch.Generator().manual_seed(rank)
        # Wide exponents, so a different summation order would show in the bits.
        flat = torch.randn(args.numel, generator=g) * torch.exp(3 * torch.randn(args.numel, generator=g))
        flat[-1] = 1.0
        destination = torch.empty(args.numel - 1)
        times: dict[str, list[float]] = {"gloo": [], "shm": [], "shm_fill": [],
                                         "shm_reduce": []}
        same = True

        def gloo_call() -> torch.Tensor:
            host = flat.clone()
            dist.all_reduce(host)
            assert float(host[-1]) == world
            return destination.copy_(host[:-1]).clone() if check else destination.copy_(host[:-1])

        def shm_call() -> torch.Tensor:
            begin = time.perf_counter()
            total = shm.all_reduce(lambda slot: slot.copy_(flat))
            shm_times.append(time.perf_counter() - begin)
            assert float(total[-1]) == world
            return destination.copy_(total[:-1]).clone() if check else destination.copy_(total[:-1])

        shm_times: list[float] = []
        for iteration in range(args.warmup + args.iters):
            record = iteration >= args.warmup
            check = iteration < args.warmup                     # bitwise check, untimed
            order = (("gloo", gloo_call), ("shm", shm_call))
            if iteration % 2:
                order = order[::-1]
            outputs = {}
            for name, call in order:
                dist.barrier()
                if name == "shm":
                    fill_begin = time.perf_counter()
                    shm.slots[rank, :args.numel].copy_(flat)   # fill cost alone
                    fill = time.perf_counter() - fill_begin
                    dist.barrier()
                begin = time.perf_counter()
                outputs[name] = call()
                elapsed = time.perf_counter() - begin
                if record:
                    times[name].append(elapsed)
                    if name == "shm":
                        times["shm_fill"].append(fill)
                        times["shm_reduce"].append(shm_times[-1] - fill)
            if check:
                same &= torch.equal(outputs["gloo"].view(torch.int32),
                                    outputs["shm"].view(torch.int32))
        shm.close()
        queue.put(dict(rank=rank, times=times, same=same))
    finally:
        dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--numel", type=int, default=PRODUCTION_NUMEL)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gloo-barrier", action="store_true",
                        help="shm barriers via dist.barrier instead of the flag spin")
    parser.add_argument("--json", default="", help="also write the summary here")
    args = parser.parse_args(argv)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    context = mp.get_context("spawn")
    queue = context.Queue()
    ranks = [context.Process(target=_worker, args=(r, args.world, port, args, queue))
             for r in range(args.world)]
    for process in ranks:
        process.start()
    results = sorted((queue.get(timeout=900) for _ in ranks), key=lambda r: r["rank"])
    for process in ranks:
        process.join(timeout=60)
    summary = dict(world=args.world, numel=args.numel, megabytes=4 * args.numel / 1e6,
                   iters=args.iters, cpus=os.cpu_count(), spin=not args.gloo_barrier,
                   load_average=os.getloadavg()[0])
    for name in results[0]["times"]:
        per_call = [max(r["times"][name][i] for r in results) for i in range(args.iters)]
        summary[f"{name}_median_ms"] = 1e3 * statistics.median(per_call)
        summary[f"{name}_min_ms"] = 1e3 * min(per_call)
    summary["speedup_median"] = summary["gloo_median_ms"] / summary["shm_median_ms"]
    summary["shm_equals_gloo_bitwise"] = all(r["same"] for r in results)
    print(json.dumps(summary, indent=1))
    if args.json:
        with open(args.json, "w") as stream:
            json.dump(summary, stream, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
