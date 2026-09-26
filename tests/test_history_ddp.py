"""Data-parallel history PPO: one rank is the base trainer; ranks stay identical,
average real-minibatch gradients, learn, and collect different trajectories."""
import os
import socket

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from train.history_ddp import HistoryDDPTrainer
from train.history_ppo import HistoryPPOConfig, HistoryTrainer


def small_config(**overrides) -> HistoryPPOConfig:
    values = dict(width=32, layers=1, heads=4, num_envs=4, steps_per_update=90, seed=3,
                  updates=2, epochs=1, minibatch_matches=2, snapshot_updates=1)
    values.update(overrides)
    return HistoryPPOConfig(**values)


def test_single_rank_matches_base_trainer(tmp_path):
    torch.use_deterministic_algorithms(True)
    try:
        base = HistoryTrainer(small_config(), tmp_path / "base")
        ddp = HistoryDDPTrainer(small_config(), tmp_path / "ddp", rank=0, world_size=1)
        for _ in range(2):
            a, b = base.update(), ddp.update()
            assert a["update_samples"] == b["update_samples"]
            assert a["policy_loss"] == b["policy_loss"]
        for p, q in zip(base.actor.state_dict().values(), ddp.actor.state_dict().values()):
            assert torch.equal(p, q)
    finally:
        torch.use_deterministic_algorithms(False)


def _worker(rank, world, port, root, queue):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_num_threads(1)
    try:
        trainer = HistoryDDPTrainer(small_config(), os.path.join(root, f"rank-{rank}"),
                                    rank=rank, world_size=world)
        # Gradient averaging over ranks that had a real minibatch.
        params = trainer.parameters_all()
        for p in params:
            p.grad = torch.full_like(p, float(rank + 1))
        count = trainer.reduce_gradients(real=rank == 0 or rank == 1)
        averaged = float(params[0].grad.flatten()[0])
        initial = [p.detach().clone() for p in trainer.actor.parameters()]
        lines = [trainer.update() for _ in range(2)]
        changed = any(not torch.equal(a, b) for a, b in zip(initial, trainer.actor.parameters()))
        checksum = float(sum(p.detach().double().sum() for p in trainer.parameters_all()))
        decisions = sum(int(l["step_decisions"]) for l in lines)
        first_obs = trainer.buffer.compact()["obs"][:1].copy()
        snapshots = sorted(trainer.population.metadata[i]["sha256"] for i in trainer.population.metadata)
        queue.put(dict(rank=rank, count=count, averaged=averaged, changed=changed,
                       checksum=checksum, decisions=decisions, lineage=trainer.lineage,
                       global_samples=lines[-1]["global_update_samples"],
                       samples=lines[-1]["update_samples"], first_obs=first_obs,
                       snapshots=snapshots, minibatches=[l["minibatches"] for l in lines]))
    finally:
        dist.destroy_process_group()


def test_two_ranks_stay_synchronized_and_learn(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_worker, args=(r, 2, port, str(tmp_path), queue))
                 for r in range(2)]
    for p in processes:
        p.start()
    results = sorted((queue.get(timeout=600) for _ in processes), key=lambda r: r["rank"])
    for p in processes:
        p.join(timeout=60)
        assert p.exitcode == 0
    a, b = results
    assert a["count"] == b["count"] == 2 and a["averaged"] == b["averaged"] == 1.5
    assert a["changed"] and b["changed"]
    assert a["checksum"] == b["checksum"]
    assert a["lineage"] == b["lineage"]
    assert a["snapshots"] == b["snapshots"] and a["snapshots"]
    assert a["minibatches"] == b["minibatches"]
    assert a["global_samples"] == b["global_samples"] == a["samples"] + b["samples"]
    # Different environments and seat RNG per rank.
    assert not np.array_equal(a["first_obs"], b["first_obs"])


def _resume_worker(rank, world, port, root, queue):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_num_threads(1)
    try:
        config = small_config(updates=4)
        first = HistoryDDPTrainer(config, os.path.join(root, f"a-{rank}"), rank=rank, world_size=world)
        for _ in range(2):
            first.update()
        path = os.path.join(root, "latest.pt")
        if rank == 0:
            first.save(path)
        dist.barrier()
        before = dict(first.progress)
        lineage = first.lineage
        second = HistoryDDPTrainer(config, os.path.join(root, f"b-{rank}"), rank=rank,
                                   world_size=world, resume=path)
        resumed_progress = dict(second.progress)
        line = second.update()
        queue.put(dict(rank=rank, lineage=second.lineage, same_lineage=second.lineage == lineage,
                       updates_before=before["updates"], updates_resumed=resumed_progress["updates"],
                       update_after=line["update"],
                       checksum=float(sum(p.detach().double().sum() for p in second.parameters_all())),
                       snapshots=sorted(m["sha256"] for m in second.population.metadata.values())))
    finally:
        dist.destroy_process_group()


def test_two_ranks_resume_one_lineage(tmp_path):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_resume_worker, args=(r, 2, port, str(tmp_path), queue))
                 for r in range(2)]
    for p in processes:
        p.start()
    results = sorted((queue.get(timeout=600) for _ in processes), key=lambda r: r["rank"])
    for p in processes:
        p.join(timeout=60)
        assert p.exitcode == 0
    a, b = results
    assert a["same_lineage"] and b["same_lineage"] and a["lineage"] == b["lineage"]
    assert a["updates_resumed"] == b["updates_resumed"] == a["updates_before"] == 2
    assert a["update_after"] == b["update_after"] == 3
    assert a["checksum"] == b["checksum"]
    assert a["snapshots"] == b["snapshots"] and a["snapshots"]
