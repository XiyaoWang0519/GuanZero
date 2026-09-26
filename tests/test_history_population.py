"""T4 population: pinned seats, learner-only rows, immutable snapshots and resume."""
import numpy as np
import pytest
import torch

from train.history_model import fresh_player, HistoryPolicyConfig
from train.history_population import HistoryPopulation, weights_digest
from train.history_ppo import HistoryPPOConfig, HistoryTrainer
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer
from test_history_rollout import make_env


def test_snapshot_seats_act_and_only_learner_trains():
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=1), 9)
    pool = HistoryPopulation(actor, "test", 12, snapshot_probability=1.0)
    identity = pool.snapshot(1)
    digest = weights_digest(pool.models[identity].state_dict())
    buffer, store, logs = SequenceRolloutBuffer(), MatchEventStore(), []
    collector = HistoryCollector(make_env(4), actor, store, buffer,
                                 torch.Generator().manual_seed(3),
                                 seat_policy=pool.assignment, resolve_policy=pool.resolve,
                                 assignment_log=logs.append)
    stats = collector.collect(100)
    data = buffer.compact()
    assignment = {(r["env"], r["match"]): r["seats"] for r in logs}
    assert len(logs) == len(assignment)  # called once per match
    assert all(seats.count(0) == 1 for seats in assignment.values())
    assert collector.policy_decisions[identity] > 0
    assert 0 < stats.learner_rows < stats.decisions
    for env, match, seat in zip(data["env"], data["match"], data["seat"]):
        assert assignment[(int(env), int(match))][int(seat)] == 0
    assert weights_digest(pool.models[identity].state_dict()) == digest
    assert all(p.grad is None and not p.requires_grad for p in pool.models[identity].parameters())
    buffer.finalize(np.zeros(len(buffer), np.float32))
    assert len(buffer.samples) > 0
    assert np.all(buffer.reward[~buffer.done] == 0)
    # Publishing and pool eviction cannot change already assigned seats/models.
    pinned = {k: v.copy() for k, v in collector.assignments.items()}
    for update in range(2, 9):
        pool.snapshot(update)
        pool.prune(collector.assignments)
    assert identity in pool.models
    for key, seats in pinned.items():
        assert np.array_equal(collector.assignment(*key), seats)
    pool.prune({})
    assert identity not in pool.models


def test_population_restore_and_lineage_validation():
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=1), 7)
    pool = HistoryPopulation(actor, "lineage", 12)
    pool.snapshot(2)
    pool.assignment(0, 0)
    state = pool.state_dict()
    restored = HistoryPopulation(actor, "lineage", 0)
    restored.load_state_dict(state)
    assert pool.assignment(1, 4) == restored.assignment(1, 4)
    assert restored.metadata == pool.metadata
    with pytest.raises(ValueError, match="lineage"):
        HistoryPopulation(actor, "other", 0).load_state_dict(state)
    state["snapshots"][1]["model"]["bos"].add_(1)
    with pytest.raises(ValueError, match="digest"):
        restored.load_state_dict(state)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_trainer_population_and_device_sampler_resume(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("remote CUDA acceptance required")
    cfg = HistoryPPOConfig(width=32, layers=1, num_envs=2, steps_per_update=90,
                           epochs=1, minibatch_matches=1, snapshot_updates=1,
                           snapshot_probability=1.0, seed=3)
    trainer = HistoryTrainer(cfg, tmp_path / "start", device=device)
    line = trainer.update()
    assert line["minibatches"] > 0
    assert trainer.generator.device.type == device
    assert len(trainer.population.models) == 1
    path = trainer.save()
    resumed = HistoryTrainer(cfg, tmp_path / "resume", device=device, resume=path)
    assert resumed.population.metadata == trainer.population.metadata
    assert len(resumed.store) == len(resumed.buffer) == 0
    assert resumed.collector.assignments == {}
    assert resumed.generator.get_state().equal(trainer.generator.get_state())
    resumed.collect()
    assert resumed.collector.policy_decisions[1] > 0
    rows = np.arange(len(resumed.buffer))
    with torch.no_grad():
        logp, _ = resumed.recompute_log_probs(rows)
    assert np.max(np.abs(logp.cpu().numpy() - resumed.buffer.compact()["logp"])) < 2e-5
    assert resumed.learn()["encoder_grad_norm"] > 0
