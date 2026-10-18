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


def test_resume_overrides_change_layout_and_are_recorded(tmp_path):
    from train.history_ppo import parse_resume_overrides
    first = HistoryTrainer(small_config(), tmp_path / "a")
    first.update()
    path = first.save(tmp_path / "a.pt")
    with pytest.raises(ValueError):
        HistoryTrainer(small_config(), tmp_path / "b", resume=path, resume_overrides={"lr": 1.0})
    with pytest.raises(ValueError):
        parse_resume_overrides(["entropy=0.1"])
    overrides = parse_resume_overrides(["num_envs=2", "minibatch-matches=1",
                                        "ddp_global_minibatch=true"])
    assert overrides == dict(num_envs=2, minibatch_matches=1, ddp_global_minibatch=True)
    second = HistoryTrainer(small_config(), tmp_path / "b", resume=path, resume_overrides=overrides)
    assert second.config.num_envs == 2 and second.env.num_envs == 2
    assert second.config_changes == [dict(at_update=1, changes=dict(
        ddp_global_minibatch=[False, True], minibatch_matches=[2, 1], num_envs=[4, 2]))]
    second.update()
    third = HistoryTrainer(small_config(), tmp_path / "c", resume=second.save(tmp_path / "b.pt"))
    assert third.config.num_envs == 2 and len(third.config_changes) == 1


def test_launcher_continues_single_process_lineage_with_global_counters(tmp_path):
    import json
    from train.history_ddp import main
    base = HistoryTrainer(small_config(num_envs=4), tmp_path / "base")
    base.update()
    path = base.save(tmp_path / "base.pt")
    before = dict(base.progress)
    output = tmp_path / "run"
    code = main(["--output", str(output), "--world-size", "2", "--resume", str(path),
                 "--updates", "3", "--resume-set", "num_envs=2",
                 "--resume-set", "minibatch_matches=1",
                 "--resume-set", "ddp_global_minibatch=true", "--torch-threads", "1"])
    assert code == 0
    lines = [json.loads(l) for l in (output / "metrics.jsonl").read_text().splitlines()]
    other = [json.loads(l) for l in (output / "rank-1" / "metrics.jsonl").read_text().splitlines()]
    assert [l["update"] for l in lines] == [2, 3]
    steps = sum(a["step_decisions"] + b["step_decisions"] for a, b in zip(lines, other))
    assert lines[-1]["decisions"] == other[-1]["decisions"] == before["decisions"] + steps
    payload = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
    assert payload["progress"]["decisions"] == lines[-1]["decisions"]
    assert payload["lineage"] == base.lineage
    assert payload["config"]["num_envs"] == 2 and payload["config"]["ddp_global_minibatch"]
    assert payload["config_changes"][0]["changes"]["num_envs"] == [4, 2]


def test_spread_gpus_assigns_ranks_round_robin_and_keeps_explicit_index():
    from train.history_ddp import rank_cuda_index
    assert [rank_cuda_index("cuda", r, True, 2) for r in range(4)] == [0, 1, 0, 1]
    assert [rank_cuda_index("cuda", r, True, 1) for r in range(3)] == [0, 0, 0]
    assert [rank_cuda_index("cuda", r, False, 2) for r in range(2)] == [None, None]
    assert [rank_cuda_index("cuda:1", r, True, 2) for r in range(2)] == [1, 1]
    assert [rank_cuda_index("cpu", r, True, 2) for r in range(2)] == [None, None]
    from train.history_ddp import build_parser
    assert build_parser().parse_args(["--output", "x", "--spread-gpus"]).spread_gpus is True


def _transport_worker(rank, world, port, root, transport, fault, queue):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_num_threads(1)
    try:
        if fault == "order":      # a wrong block model: the self-check must refuse it
            import train.history_shm_reduce as module
            original = module.gloo_ring_chunk
            module.gloo_ring_chunk = lambda n, w, e=4: original(n, w, e) + 1000
        elif fault == "space" and rank == 0:     # /dev/shm too small for the mapping
            os.statvfs = lambda path: type("Tiny", (), dict(f_bavail=1, f_frsize=4096))()
        config = small_config(epochs=2, minibatch_matches=1, ddp_global_minibatch=True,
                              response_mode="auxiliary", aux_heads="next,belief,outcome",
                              ddp_reduce=transport)
        output = os.path.join(root, f"{transport}-{fault}-{rank}")
        trainer = HistoryDDPTrainer(config, output, rank=rank, world_size=world)
        params = trainer.parameters_all()
        flat = lambda tensors: torch.cat([t.detach().reshape(-1) for t in tensors]).numpy().copy()
        # Direct reduce with random gradients; rank 1 has no real minibatch.
        g = torch.Generator().manual_seed(rank)
        local = [torch.randn(p.shape, generator=g) * torch.exp2(torch.randint(-12, 12, p.shape,
                                                                              generator=g))
                 for p in params]
        for p, x in zip(params, local):
            p.grad = x.clone()
        count = trainer.reduce_gradients(rank != 1)
        mixed = flat(p.grad for p in params)
        # No rank has a real minibatch: no step, gradients untouched.
        for p, x in zip(params, local):
            p.grad = x.clone()
        none = trainer.reduce_gradients(False)
        untouched = all(torch.equal(p.grad, x) for p, x in zip(params, local))
        lines, snapshots = [], []
        for _ in range(2):
            lines.append(trainer.update())
            snapshots.append(flat(params))
        active = trainer.gradient_reduce()
        trainer.close()
        import json
        manifest = json.loads((trainer.output / "manifest.json").read_text())["inference"]
        queue.put(dict(rank=rank, count=count, none=none, untouched=untouched, mixed=mixed,
                       params=snapshots, active=active,
                       manifest=manifest, metrics=[l["ddp_reduce"] for l in lines],
                       policy_loss=[l["policy_loss"] for l in lines],
                       minibatches=[l["minibatches"] for l in lines],
                       local_minibatches=[l["local_minibatches"] for l in lines]))
    finally:
        dist.destroy_process_group()


def _run_transport(tmp_path, transport: str, world: int, fault: str = "") -> list[dict]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_transport_worker,
                                 args=(r, world, port, str(tmp_path), transport, fault, queue))
                 for r in range(world)]
    for p in processes:
        p.start()
    results = sorted((queue.get(timeout=600) for _ in processes), key=lambda r: r["rank"])
    for p in processes:
        p.join(timeout=60)
        assert p.exitcode == 0
    return results


@pytest.mark.parametrize("world", [3, 4])
def test_shm_reduce_is_bitwise_gloo_at_three_and_four_ranks(tmp_path, world):
    """The shared-memory reduce adds in gloo's ring order, so the reduced gradients
    (including a rank without a real minibatch), losses and parameters after two
    updates equal gloo's bit for bit; a step with no real minibatch is skipped."""
    gloo = _run_transport(tmp_path, "gloo", world)
    shm = _run_transport(tmp_path, "shm", world)
    assert all(r["active"] == "shm" and r["metrics"] == ["shm", "shm"] for r in shm)
    assert shm[0]["manifest"]["ddp_reduce_active"] == "shm"
    assert shm[0]["manifest"]["ddp_reduce_fallback"] is None
    assert all(r["active"] == "gloo" for r in gloo)
    for a, b in zip(gloo, shm):
        assert a["count"] == b["count"] == world - 1 and a["none"] == b["none"] == 0
        assert a["untouched"] and b["untouched"]
        assert np.array_equal(a["mixed"].view(np.int32), b["mixed"].view(np.int32))
        assert a["policy_loss"] == b["policy_loss"] and a["minibatches"] == b["minibatches"]
        for update in range(2):
            assert np.array_equal(a["params"][update].view(np.int32),
                                  b["params"][update].view(np.int32))


@pytest.mark.parametrize("fault", ["order", "space"])
def test_shm_reduce_falls_back_to_gloo_and_records_why(tmp_path, fault):
    """A wrong order model fails the start-up self-check and a /dev/shm without
    room fails the mapping: on every rank the run continues on gloo (same
    numbers) and the manifest and metrics say so."""
    results = _run_transport(tmp_path, "shm", 3, fault)
    reason = "self-check" if fault == "order" else "unavailable: --ddp-reduce shm needs"
    for r in results:
        assert r["active"] == "gloo" and r["metrics"] == ["gloo", "gloo"]
        assert r["manifest"]["ddp_reduce"] == "shm"
        assert r["manifest"]["ddp_reduce_active"] == "gloo"
        assert r["manifest"]["ddp_reduce_fallback"].startswith(reason)
        assert np.array_equal(r["params"][1], results[0]["params"][1])


def test_launcher_shm_reduce_matches_gloo_bitwise_at_two_ranks(tmp_path):
    """Two ranks: a two-term sum commutes, so the shared-memory reduce gives
    gloo's parameters, optimizer state and losses bit for bit."""
    import json
    from train.history_ddp import main
    common = ["--world-size", "2", "--ddp-global-minibatch", "--num-envs", "3",
              "--steps-per-update", "90", "--width", "32", "--layers", "1", "--heads", "4",
              "--updates", "2", "--epochs", "2", "--minibatch-matches", "1",
              "--snapshot-updates", "1", "--seed", "5", "--torch-threads", "1",
              "--response-mode", "auxiliary", "--aux-heads", "next,belief,outcome"]
    payloads, metrics = {}, {}
    for transport in ("gloo", "shm"):
        output = tmp_path / transport
        assert main(["--output", str(output), "--ddp-reduce", transport, *common]) == 0
        payloads[transport] = torch.load(output / "latest.pt", map_location="cpu",
                                         weights_only=False)
        metrics[transport] = [json.loads(l) for l in
                              (output / "metrics.jsonl").read_text().splitlines()]
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["inference"]["ddp_reduce"] == transport
        assert manifest["inference"]["ddp_reduce_active"] == transport
    a, b = payloads["gloo"], payloads["shm"]
    for key in ("model", "critic"):
        assert a[key].keys() == b[key].keys()
        assert all(torch.equal(a[key][name], b[key][name]) for name in a[key])
    for name in ("actor", "critic"):
        for x, y in zip(a["optimizer"][name]["state"].values(),
                        b["optimizer"][name]["state"].values()):
            assert all(torch.equal(x[k], y[k]) for k in x)
    for key in ("policy_loss", "value_loss", "actor_grad_norm", "encoder_grad_norm"):
        assert [l[key] for l in metrics["gloo"]] == [l[key] for l in metrics["shm"]]
    assert [l["update"] for l in metrics["shm"]] == [1, 2]


class _BlockingMomentsTrainer(HistoryDDPTrainer):
    """The pre-async learn collectives: a MAX of the minibatch count per epoch and
    one blocking moment all-reduce per step (the synchronous fallback)."""

    def epoch_batches(self, samples):
        batches = list(self.minibatches()) if samples else []
        longest = torch.tensor([len(batches)], dtype=torch.long)
        dist.all_reduce(longest, op=dist.ReduceOp.MAX)
        return [batches[i] if i < len(batches) else None for i in range(int(longest))]


def _moments_worker(rank, world, port, root, queue):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_num_threads(1)
    try:
        calls = {"blocking": 0, "async": 0}
        for name in ("all_reduce", "all_gather", "barrier", "broadcast_object_list"):
            original = getattr(dist, name)

            def counted(*args, _original=original, **kwargs):
                calls["async" if kwargs.get("async_op") else "blocking"] += 1
                return _original(*args, **kwargs)
            setattr(dist, name, counted)
        report = dict(rank=rank)
        for label, cls in (("async", HistoryDDPTrainer), ("blocking", _BlockingMomentsTrainer)):
            # Unequal minibatch counts per rank, so some steps are padding (None).
            config = small_config(epochs=2, minibatch_matches=1 + rank, ddp_global_minibatch=True,
                                  response_mode="auxiliary", aux_heads="outcome")
            trainer = cls(config, os.path.join(root, f"{label}-{rank}"), rank=rank,
                          world_size=world)
            lines, counts = [], []
            for _ in range(2):
                before = dict(calls)
                lines.append(trainer.update())
                counts.append({k: calls[k] - before[k] for k in calls})
            report[label] = dict(
                params=torch.cat([p.detach().reshape(-1) for p in trainer.parameters_all()]
                                 ).numpy().copy(),
                policy_loss=[l["policy_loss"] for l in lines],
                minibatches=[l["minibatches"] for l in lines],
                local=[l["local_minibatches"] for l in lines], calls=counts)
        queue.put(report)
    finally:
        dist.destroy_process_group()


def test_async_moments_and_gathered_counts_match_blocking_collectives(tmp_path):
    """Issuing an epoch's per-step moment all-reduces at its start (async, same
    order) and gathering minibatch counts once per learn gives the same bits as
    one blocking moment all-reduce per step and a MAX per epoch, also when ranks
    have different minibatch counts."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=_moments_worker, args=(r, 3, port, str(tmp_path), queue))
                 for r in range(3)]
    for p in processes:
        p.start()
    results = sorted((queue.get(timeout=600) for _ in processes), key=lambda r: r["rank"])
    for p in processes:
        p.join(timeout=60)
        assert p.exitcode == 0
    for r in results:
        a, b = r["async"], r["blocking"]
        assert np.array_equal(a["params"].view(np.int32), b["params"].view(np.int32))
        assert a["policy_loss"] == b["policy_loss"] and a["minibatches"] == b["minibatches"]
        assert np.array_equal(a["params"], results[0]["async"]["params"])
        for update in range(2):
            # Per update: every step's moment all-reduce moves off the blocking
            # path, and two per-epoch MAXes become one all_gather.
            steps = a["minibatches"][update]
            assert a["calls"][update]["async"] == steps and b["calls"][update]["async"] == 0
            assert a["calls"][update]["blocking"] == b["calls"][update]["blocking"] - steps - 1
    # Padding steps happened: some rank had fewer local minibatches than steps.
    assert any(r["async"]["local"] != r["async"]["minibatches"] for r in results)
