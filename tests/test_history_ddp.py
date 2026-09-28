"""Data-parallel history PPO: one rank is the base trainer; ranks stay identical,
average real-minibatch gradients, learn, and collect different trajectories."""
import os
import socket

import numpy as np
import pytest
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
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
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
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


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


def test_cuda_index_accepts_bare_cuda():
    from train.history_ddp import cuda_index
    assert cuda_index("cpu") is None and cuda_index("cuda") is None
    assert cuda_index("cuda:1") == 1


@pytest.mark.parametrize("response_mode", ["none", "auxiliary"])
def test_global_minibatch_matches_one_union_minibatch(tmp_path, monkeypatch, response_mode):
    """Two rank minibatches with ddp_global_minibatch give the gradient of one
    minibatch over their union (emulated in one process: the all-reduce adds
    the other half's statistics, the gradient average divides by two)."""
    import train.history_ddp as history_ddp
    config = small_config(ddp_global_minibatch=True, num_envs=6, response_mode=response_mode)
    trainer = HistoryDDPTrainer(config, tmp_path / "t", rank=0, world_size=1)
    trainer.collect()
    values = trainer.refresh_values()
    assert trainer.buffer.finalize(values, config.gamma, config.gae_lambda) > 8
    rows = trainer.buffer.samples
    halves = (rows[: len(rows) // 3], rows[len(rows) // 3:])   # unequal sizes
    params = trainer.parameters_all()

    def gradient(parts):
        for p in params:
            p.grad = None
        for index, part in enumerate(parts):
            other = halves[1 - index]
            extra = np.array([len(other), trainer.buffer.advantage[other].astype(np.float64).sum(),
                              np.square(trainer.buffer.advantage[other].astype(np.float64)).sum(),
                              1.0])
            monkeypatch.setattr(history_ddp.dist, "all_reduce",
                                lambda t, op=None: t.add_(torch.from_numpy(extra)))
            scale = trainer.global_minibatch(part)
            terms = trainer.minibatch_loss(part)
            trainer.advantage_moments = None
            ((scale / 2) * terms["policy_total"]).backward()
            ((scale / 2) * terms["value_total"]).backward()
        return [p.grad.detach().clone() for p in params]

    emulated = gradient(halves)
    for p in params:
        p.grad = None
    terms = trainer.minibatch_loss(rows)
    terms["policy_total"].backward()
    terms["value_total"].backward()
    reference = [p.grad.detach().clone() for p in params]
    for a, b in zip(emulated, reference):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
    # Without the flag each rank normalizes its own half: a different gradient.
    trainer.config = small_config(num_envs=6, response_mode=response_mode)
    local = gradient(halves)
    assert any(not torch.allclose(a, b, rtol=1e-4, atol=1e-6) for a, b in zip(local, reference))


def test_launcher_runs_ranks_and_checkpoints_rank_zero(tmp_path):
    from train.history_ddp import main
    output = tmp_path / "run"
    code = main(["--output", str(output), "--world-size", "2", "--ddp-global-minibatch",
                 "--num-envs", "3", "--steps-per-update", "90", "--width", "32", "--layers", "1",
                 "--heads", "4", "--updates", "2", "--minibatch-matches", "1",
                 "--snapshot-updates", "1", "--seed", "5", "--torch-threads", "1"])
    assert code == 0
    assert (output / "latest.pt").exists() and (output / "update-000002.pt").exists()
    assert not (output / "rank-1" / "latest.pt").exists()
    lines = [__import__("json").loads(l) for l in (output / "metrics.jsonl").read_text().splitlines()]
    other = (output / "rank-1" / "metrics.jsonl").read_text().splitlines()
    assert [l["update"] for l in lines] == [1, 2] and len(other) == 2
    assert lines[-1]["global_step_decisions"] > lines[-1]["step_decisions"]
    manifest = __import__("json").loads((output / "manifest.json").read_text())
    assert manifest["config"]["ddp_global_minibatch"] is True


def test_launcher_sigterm_stops_all_ranks_and_saves(tmp_path):
    import subprocess
    import sys
    import time
    output = tmp_path / "run"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(["python", "oracle", "."]))
    process = subprocess.Popen(
        [sys.executable, "-m", "train.history_ddp", "--output", str(output), "--world-size", "2",
         "--num-envs", "2", "--steps-per-update", "40", "--width", "32", "--layers", "1",
         "--heads", "4", "--updates", "1000", "--minibatch-matches", "1", "--seed", "9",
         "--torch-threads", "1", "--checkpoint-updates", "1000"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.time() + 300
    metrics = output / "metrics.jsonl"
    while time.time() < deadline and not (metrics.exists() and metrics.read_text().count("\n") >= 1):
        time.sleep(0.2)
    process.send_signal(__import__("signal").SIGTERM)
    _, stderr = process.communicate(timeout=300)
    assert process.returncode == 0, stderr.decode()[-2000:]
    rank0 = metrics.read_text().splitlines()
    rank1 = (output / "rank-1" / "metrics.jsonl").read_text().splitlines()
    assert len(rank0) == len(rank1) < 1000
    assert (output / "latest.pt").exists()
