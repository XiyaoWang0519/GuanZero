"""T4 population: pinned seats, learner-only rows, immutable snapshots and resume."""
import json
import numpy as np
import pytest
import torch

from train.history_config import parse_resume_overrides
from train.history_model import (
    HistoryPolicyConfig,
    fresh_player,
    load_history_checkpoint,
    save_history_checkpoint,
)
from train.history_population import (
    HistoryPopulation,
    behavior_fingerprint,
    mean_behavior_tv,
    weights_digest,
)
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


def pre_archive_assignment(pool):
    """The sampler before the archive existed, copied verbatim."""
    seats = np.zeros(4, np.int64)
    eligible = sorted(pool.models)[-pool.recent:]
    if eligible:
        anchor = int(pool.rng.integers(4))
        for seat in range(4):
            if seat != anchor and pool.rng.random() < pool.probability:
                seats[seat] = pool.rng.choice(eligible)
    return seats.tolist()


def test_without_archive_the_sampler_and_stream_are_unchanged():
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=1), 9)
    new, old = (HistoryPopulation(actor, "t", 5) for _ in range(2))
    for update in range(2, 40, 2):
        new.snapshot(update)
        old.snapshot(update)
        for _ in range(7):
            assert new.assignment(0, 0) == pre_archive_assignment(old)
    assert new.rng.bit_generator.state == old.rng.bit_generator.state
    assert new.archive == []


def test_archive_stays_spread_and_is_sampled_resident_and_restored():
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=1), 9)
    pool = HistoryPopulation(actor, "t", 5, recent=2, snapshot_probability=1.0,
                             archive_every=4, archive_size=4, archive_share=0.5)
    for update in range(2, 66, 2):
        pool.snapshot(update)
        pool.prune({})
    updates = [pool.metadata[i]["update"] for i in pool.archive]
    assert len(updates) <= 4 and updates[-1] == 64 and updates[0] <= 16
    assert all(i in pool.models for i in pool.archive)
    recent = set(sorted(pool.models)[-2:])
    drawn = [s for _ in range(400) for s in pool.assignment(0, 0) if s]
    archived = [s for s in drawn if s in set(pool.archive) - recent]
    assert 0.3 < len(archived) / len(drawn) < 0.6
    state = pool.state_dict()
    again = HistoryPopulation(actor, "t", 5, recent=2, snapshot_probability=1.0,
                              archive_every=4, archive_size=4, archive_share=0.5)
    again.load_state_dict(state)
    assert again.archive == pool.archive
    assert again.assignment(0, 0) == pool.assignment(0, 0)
    with pytest.raises(ValueError, match="config mismatch"):
        HistoryPopulation(actor, "t", 5, recent=2, snapshot_probability=1.0).load_state_dict(state)


def test_old_population_state_loads_without_archive_fields():
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=1), 9)
    pool = HistoryPopulation(actor, "t", 5)
    pool.snapshot(2)
    state = pool.state_dict()
    for key in ("archive_every", "archive_size", "archive_share", "archive"):
        state.pop(key)
    again = HistoryPopulation(actor, "t", 5)
    again.load_state_dict(state)
    assert again.archive == [] and list(again.models) == list(pool.models)


def test_trainer_passes_archive_settings(tmp_path):
    trainer = HistoryTrainer(HistoryPPOConfig(width=32, layers=1, heads=4, num_envs=4,
                                              steps_per_update=16, seed=3, updates=4, epochs=1,
                                              minibatch_matches=2, snapshot_updates=1,
                                              population_archive_every=2,
                                              population_archive_size=3), tmp_path)
    trainer.run()
    assert trainer.population.archive_every == 2
    archived = [trainer.population.metadata[i]["update"] for i in trainer.population.archive]
    assert archived and all(u % 2 == 0 for u in archived)
    resumed = HistoryTrainer(HistoryPPOConfig(updates=5), tmp_path / "r", resume=tmp_path / "latest.pt")
    assert resumed.population.archive == trainer.population.archive


def _read_population_events(path):
    return [json.loads(line) for line in (path / "population.jsonl").read_text().splitlines() if line]


def test_trainer_nearest_pair_archive_probe_persistence_and_age_cap(tmp_path):
    cfg = HistoryPPOConfig(
        width=32,
        layers=1,
        heads=4,
        num_envs=4,
        steps_per_update=90,
        seed=3,
        updates=4,
        epochs=1,
        minibatch_matches=2,
        snapshot_updates=1,
        population_archive_every=2,
        population_archive_size=3,
        population_archive_filter="nearest_pair",
        population_archive_min_tv=0.0,
        population_archive_max_age_updates=6,
        population_behavior_probe_rows=16,
        population_behavior_probe_seed=20261117,
    )
    trainer = HistoryTrainer(cfg, tmp_path / "run1")
    manifest = json.loads((tmp_path / "run1" / "manifest.json").read_text())
    assert manifest["population"]["archive_filter"] == "nearest_pair"
    assert manifest["population"]["archive_max_age_updates"] == 6
    assert manifest["population"]["behavior_probe_rows"] == 16
    assert manifest["population"]["behavior_probe_seed"] == 20261117

    lines = [trainer.update() for _ in range(4)]
    pop_metrics = lines[-1]["population"]
    assert pop_metrics["archive_filter"] == "nearest_pair"
    assert pop_metrics["archive_max_age_updates"] == 6
    assert "archive_min_pair_tv" in pop_metrics
    assert "archive_mean_pair_tv" in pop_metrics
    assert "archive_age_span" in pop_metrics
    assert trainer.population.behavior_probe is not None
    assert trainer.population.behavior_probe.decisions == 16
    assert set(trainer.population.behaviors) == set(trainer.population.archive)

    events_1 = _read_population_events(tmp_path / "run1")
    assert len([e for e in events_1 if e["event"] == "behavior_probe_init"]) == 1
    assert len([e for e in events_1 if e["event"] == "behavior_probe_rebind"]) == 0

    ckpt = trainer.save()
    resumed = HistoryTrainer(HistoryPPOConfig(updates=10), tmp_path / "run2", resume=ckpt)
    assert resumed.population.behavior_probe is not None
    assert torch.equal(resumed.population.behavior_probe.obs, trainer.population.behavior_probe.obs)
    assert resumed.population.behaviors.keys() == trainer.population.behaviors.keys()

    resumed_lines = [resumed.update() for _ in range(6)]
    assert resumed_lines[-1]["population"]["archive_age_evictions"] >= 1
    assert set(resumed.population.behaviors) == set(resumed.population.archive)

    events_2 = _read_population_events(tmp_path / "run2")
    assert [e for e in events_2 if e["event"] == "behavior_probe_init"] == []
    assert [e for e in events_2 if e["event"] == "behavior_probe_rebind"] == []


def test_resume_override_switches_thin_to_nearest_pair(tmp_path):
    cfg = HistoryPPOConfig(
        width=32,
        layers=1,
        heads=4,
        num_envs=4,
        steps_per_update=90,
        seed=5,
        updates=4,
        epochs=1,
        minibatch_matches=2,
        snapshot_updates=1,
        population_archive_every=2,
        population_archive_size=3,
        population_archive_filter="thin",
    )
    trainer = HistoryTrainer(cfg, tmp_path / "thin")
    for _ in range(4):
        trainer.update()
    assert trainer.population.archive
    assert trainer.population.behavior_probe is None
    prior_models = set(trainer.population.models)
    prior_archive = list(trainer.population.archive)
    ckpt = trainer.save()

    overrides = parse_resume_overrides([
        "population-archive-filter=nearest_pair",
        "population-behavior-probe-rows=16",
        "population-archive-min-tv=0.01",
    ])
    resumed = HistoryTrainer(
        HistoryPPOConfig(updates=6),
        tmp_path / "nearest",
        resume=ckpt,
        resume_overrides=overrides,
    )
    assert prior_models <= set(resumed.population.models)
    assert resumed.population.archive == prior_archive
    assert resumed.population.behavior_probe is None

    events_pre = _read_population_events(tmp_path / "nearest")
    sched_events = [e for e in events_pre if e["event"] == "population_schedule_change"]
    assert len(sched_events) == 1
    assert sched_events[0]["previous"]["archive_filter"] == "thin"
    assert sched_events[0]["new"]["archive_filter"] == "nearest_pair"
    assert sched_events[0]["retained_archive"] == len(prior_archive)

    # Update 5 is a non-archive update; update 6 hits archive_every=2 and rebinds.
    resumed.update()
    assert resumed.population.behavior_probe is None
    resumed.update()
    assert resumed.population.behavior_probe is not None
    assert resumed.population.behavior_probe.decisions == 16
    assert set(resumed.population.behaviors) == set(resumed.population.archive)

    events_post = _read_population_events(tmp_path / "nearest")
    rebinds = [e for e in events_post if e["event"] == "behavior_probe_rebind"]
    assert len(rebinds) == 1
    assert rebinds[0]["archive_members"] == len(prior_archive)


def test_legacy_u195450_shaped_resume_rebinds_once_and_persists(tmp_path):
    cfg = HistoryPPOConfig(
        width=32,
        layers=1,
        heads=4,
        num_envs=4,
        steps_per_update=90,
        seed=7,
        updates=2,
        epochs=1,
        minibatch_matches=2,
        snapshot_updates=1,
        population_archive_every=1,
        population_archive_size=4,
        population_archive_filter="strength_behavior",
        population_archive_min_tv=0.0,
        population_behavior_probe_rows=16,
    )
    trainer = HistoryTrainer(cfg, tmp_path / "src")
    trainer.update()
    trainer.update()
    assert len(trainer.population.archive) >= 1
    assert trainer.population.behaviors

    src_ckpt = trainer.save()
    _, _, payload = load_history_checkpoint(src_ckpt, "cpu")
    payload["config"]["next_coef"] = 0.1
    for new_cfg_key in (
        "population_archive_max_age_updates",
        "population_behavior_probe_rows",
        "population_behavior_probe_seed",
    ):
        payload["config"].pop(new_cfg_key, None)
    pop = payload["population"]
    pop["archive_filter"] = "strength_behavior"
    pop["behavior_decisions"] = {}
    pop["behavior_probe"] = None
    pop["behavior_probe_meta"] = None
    pop.pop("archive_max_age_updates", None)
    legacy_ckpt = tmp_path / "u195450-legacy.pt"
    save_history_checkpoint(legacy_ckpt, payload)

    resumed = HistoryTrainer(HistoryPPOConfig(updates=4), tmp_path / "resumed1", resume=legacy_ckpt)
    assert resumed.population.behavior_probe is None
    assert resumed.population.archive
    resumed.update()
    assert resumed.population.behavior_probe is not None
    assert resumed.population.behavior_probe.decisions == 128
    assert set(resumed.population.behaviors) == set(resumed.population.archive)
    events_1 = _read_population_events(tmp_path / "resumed1")
    assert len([e for e in events_1 if e["event"] == "behavior_probe_rebind"]) == 1

    resumed.update()
    events_1_after = _read_population_events(tmp_path / "resumed1")
    assert len([e for e in events_1_after if e["event"] == "behavior_probe_rebind"]) == 1

    ckpt2 = resumed.save()
    resumed2 = HistoryTrainer(HistoryPPOConfig(updates=5), tmp_path / "resumed2", resume=ckpt2)
    assert resumed2.population.behavior_probe is not None
    assert torch.equal(resumed2.population.behavior_probe.obs, resumed.population.behavior_probe.obs)
    resumed2.update()
    events_2 = _read_population_events(tmp_path / "resumed2")
    assert [e for e in events_2 if e["event"] == "behavior_probe_rebind"] == []


def test_behavior_probe_and_fingerprint_deterministic_across_ranks(tmp_path):
    shared_kwargs = dict(
        width=32,
        layers=1,
        heads=4,
        steps_per_update=32,
        population_archive_every=1,
        population_archive_filter="nearest_pair",
        population_behavior_probe_rows=20,
        population_behavior_probe_seed=20261117,
    )
    t0 = HistoryTrainer(HistoryPPOConfig(seed=11, num_envs=2, **shared_kwargs), tmp_path / "rank0")
    t1 = HistoryTrainer(HistoryPPOConfig(seed=97, num_envs=5, **shared_kwargs), tmp_path / "rank1")
    t1.actor.load_state_dict(t0.actor.state_dict())

    probe0 = t0._ensure_behavior_probe()
    probe1 = t1._ensure_behavior_probe()

    saved0 = t0.population.behavior_probe
    saved1 = t1.population.behavior_probe
    assert saved0 is not None and saved1 is not None
    assert t0.population.behavior_probe_meta == t1.population.behavior_probe_meta
    assert torch.equal(saved0.obs, saved1.obs)
    assert torch.equal(saved0.cand, saved1.cand)
    assert torch.equal(saved0.offsets, saved1.offsets)
    assert torch.equal(saved0.streams.tokens, saved1.streams.tokens)
    assert torch.equal(saved0.prefix, saved1.prefix)
    assert torch.equal(saved0.round_index, saved1.round_index)

    fp0 = behavior_fingerprint(t0.population.resolve(0), probe0)
    fp1 = behavior_fingerprint(t1.population.resolve(0), probe1)
    assert np.array_equal(fp0, fp1)


def test_init_checkpoint_rebases_snapshot_updates_for_age_and_replacement(tmp_path):
    src = HistoryTrainer(
        HistoryPPOConfig(
            width=32,
            layers=1,
            heads=4,
            num_envs=4,
            steps_per_update=32,
            seed=13,
            snapshot_updates=1,
            population_archive_every=1,
            population_archive_size=4,
            population_archive_filter="thin",
        ),
        tmp_path / "src",
    )
    with torch.no_grad():
        src.actor.action_tower[0].weight.mul_(8.0)
        src.actor.fusion[-1].weight.mul_(200.0)
    snap1 = src.population.snapshot(9900)
    with torch.no_grad():
        src.actor.fusion[-1].weight.neg_()
    snap2 = src.population.snapshot(9964)
    src.progress["updates"] = 10000
    ckpt = src.save()

    t = HistoryTrainer(
        HistoryPPOConfig(
            width=32,
            layers=1,
            heads=4,
            num_envs=4,
            steps_per_update=32,
            seed=21,
            snapshot_updates=1,
            population_archive_every=2,
            population_archive_size=4,
            population_archive_filter="nearest_pair",
            population_archive_min_tv=0.05,
            population_archive_max_age_updates=200,
            population_behavior_probe_rows=16,
            init_checkpoint=str(ckpt),
        ),
        tmp_path / "run_a",
    )
    assert t.population.archive == [snap1, snap2]
    assert t.population.metadata[snap1]["update"] == -100
    assert t.population.metadata[snap1]["source_update"] == 9900
    assert t.population.metadata[snap2]["update"] == -36
    assert t.population.metadata[snap2]["source_update"] == 9964
    assert t.population.archive_diversity_stats()["archive_age_span"] == 64

    probe = t._ensure_behavior_probe()
    assert mean_behavior_tv(t.population.behaviors[snap1], t.population.behaviors[snap2], probe.decisions) >= 0.05

    # Actor matches snap2 (TV=0.0 < 0.05); new snapshot at update=2 must replace -36, not be rejected.
    t.progress["updates"] = 2
    id_new = t._snapshot_population()
    assert t.population.archive == [snap1, id_new]
    assert t.population.metadata[id_new]["update"] == 2
    assert t.population.archive_replacements == 1
    assert t.population.archive_rejects == 0
    assert t.population.archive_diversity_stats()["archive_age_span"] == 102

    # At update=102, age of snap1 (-100) is 202 > 200 -> evicted by population_archive_max_age_updates.
    with torch.no_grad():
        t.actor.fusion[-1].weight.neg_()
    t.progress["updates"] = 102
    id_later = t._snapshot_population()
    assert snap1 not in t.population.archive
    assert t.population.archive_age_evictions >= 1
    assert t.population.archive == [id_new, id_later]
    assert t.population.archive_diversity_stats()["archive_age_span"] == 100

    # Tight age cap on init_checkpoint immediately filters -100 (age 100 > 60) while keeping -36.
    t_tight = HistoryTrainer(
        HistoryPPOConfig(
            width=32,
            layers=1,
            heads=4,
            num_envs=4,
            steps_per_update=32,
            seed=22,
            snapshot_updates=1,
            population_archive_every=2,
            population_archive_size=4,
            population_archive_filter="nearest_pair",
            population_archive_min_tv=0.05,
            population_archive_max_age_updates=60,
            population_behavior_probe_rows=16,
            init_checkpoint=str(ckpt),
        ),
        tmp_path / "run_b",
    )
    assert t_tight.population.archive == [snap2]
    assert t_tight.population.metadata[snap2]["update"] == -36
