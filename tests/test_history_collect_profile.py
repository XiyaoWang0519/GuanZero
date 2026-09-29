"""Collection profiling only adds timings and counters; it never changes what is collected."""
from dataclasses import asdict

import numpy as np
import pytest
import torch

from train import history_rollout as rollout
from train.history_model import HistoryPolicyConfig, fresh_player
from train.history_ppo import HistoryPPOConfig, HistoryTrainer, parse_resume_overrides
from test_history_rollout import make_env

PHASES = {"env_pending", "events_and_rounds", "metadata_and_assignment", "input_indexing",
          "decision_upload", "public_cache_or_collation", "actor_and_sampling",
          "decision_download", "buffer_and_counters", "env_step"}
TIMING_KEYS = {"collection_phase_seconds", "collection_profile_synchronized",
               "collection_group_phase_seconds", "collection_policy_call_rows",
               "decisions_per_sec", "collect_seconds", "learn_seconds", "learn_decisions_per_sec",
               "learn_exposures_per_sec", "learner_collect_decisions_per_sec", "elapsed_seconds"}


@pytest.fixture(autouse=True)
def single_thread():
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1], warn_only=previous[2])


def player(seed):
    actor, _ = fresh_player(HistoryPolicyConfig(width=16, layers=1, action_width=16,
                                               fusion_width=16, critic_width=16), seed)
    return actor


@pytest.mark.parametrize("kv_cache", [False, True])
def test_profile_leaves_rows_choices_and_streams_unchanged(kv_cache):
    policies = {i: player(17 + i) for i in (0, 1, 2)}
    runs = []
    for profile in (False, True):
        collector = rollout.HistoryCollector(
            make_env(4, 23), policies[0], rollout.MatchEventStore(),
            rollout.SequenceRolloutBuffer(), torch.Generator().manual_seed(31),
            seat_policy=lambda env, match: [0, 1, 0, 2], resolve_policy=policies.__getitem__,
            record_choices=True, kv_cache=kv_cache, profile=profile,
            temperature=0.8, epsilon=0.2)
        runs.append((collector, collector.collect(24)))
    (plain, plain_stats), (profiled, profiled_stats) = runs
    assert len(plain.choice_log) == len(profiled.choice_log) == 24
    for before, after in zip(plain.choice_log, profiled.choice_log):
        assert before.tobytes() == after.tobytes()
    for name, before in plain.buffer.compact().items():
        after = profiled.buffer.compact()[name]
        assert before.dtype == after.dtype and before.shape == after.shape
        assert before.tobytes() == after.tobytes()
    assert torch.equal(plain.generator.get_state(), profiled.generator.get_state())
    assert plain.policy_decisions == profiled.policy_decisions
    assert plain.store.streams.keys() == profiled.store.streams.keys()
    for key, stream in plain.store.streams.items():
        for before, after in zip(stream.arrays(), profiled.store.streams[key].arrays()):
            assert np.array_equal(before, after)
    same = {k: v for k, v in asdict(plain_stats).items() if k not in rollout.PROFILE_STATS}
    assert same == {k: v for k, v in asdict(profiled_stats).items()
                    if k not in rollout.PROFILE_STATS}
    # Off: no timings or call shapes at all.
    assert all(not getattr(plain_stats, k) for k in rollout.PROFILE_STATS)
    # On: every phase, the per-group split of the per-call phases, exact call counts.
    assert set(profiled_stats.phase_seconds) == PHASES
    assert set(profiled_stats.group_phase_seconds) == {"learner", "snapshot"}
    for split in profiled_stats.group_phase_seconds.values():
        assert set(split) == {"public_cache_or_collation", "actor_and_sampling"}
    for name in ("public_cache_or_collation", "actor_and_sampling"):
        total = sum(split[name] for split in profiled_stats.group_phase_seconds.values())
        assert total == pytest.approx(profiled_stats.phase_seconds[name], rel=1e-9, abs=1e-12)
    calls = profiled_stats.policy_call_rows
    assert sum(sum(sizes.values()) for sizes in calls.values()) == profiled_stats.policy_batches
    rows = sum(size * count for sizes in calls.values() for size, count in sizes.items())
    assert rows == sum(profiled.policy_decisions.values())
    assert profiled_stats.policy_batches > profiled_stats.steps


def test_trainer_warmup_switches_profiling_on_without_changing_training(tmp_path):
    base = dict(width=16, layers=1, heads=4, num_envs=2, steps_per_update=40, epochs=1,
                minibatch_matches=1, snapshot_updates=1, seed=5, causal_sdpa=True,
                rollout_kv_cache=True)
    lines = {}
    trainers = {}
    for name, extra in (("off", {}), ("warm", dict(profile_collection=True,
                                                   profile_collection_warmup=1))):
        trainer = HistoryTrainer(HistoryPPOConfig(**base, **extra), tmp_path / name)
        lines[name] = [trainer.update() for _ in range(3)]
        trainers[name] = trainer
    off, warm = lines["off"], lines["warm"]
    assert [l["collection_profile_synchronized"] for l in off] == [False] * 3
    assert [l["collection_profile_synchronized"] for l in warm] == [False, True, True]
    assert not warm[0]["collection_phase_seconds"] and not warm[0]["collection_policy_call_rows"]
    for line in warm[1:]:
        assert set(line["collection_phase_seconds"]) == PHASES
        assert "learner" in line["collection_group_phase_seconds"]
        calls = line["collection_policy_call_rows"]
        assert sum(sum(c.values()) for c in calls.values()) == line["collection_policy_batches"]
        assert line["collection_steps"] == 40
    for before, after in zip(off, warm):
        assert ({k: v for k, v in before.items() if k not in TIMING_KEYS}
                == {k: v for k, v in after.items() if k not in TIMING_KEYS})
    for a, b in zip(trainers["off"].actor.parameters(), trainers["warm"].actor.parameters()):
        assert torch.equal(a, b)
    assert torch.equal(trainers["off"].generator.get_state(),
                       trainers["warm"].generator.get_state())


def test_resume_can_switch_profiling_on_and_records_it(tmp_path):
    cfg = HistoryPPOConfig(width=16, layers=1, heads=4, num_envs=2, steps_per_update=20,
                           epochs=1, minibatch_matches=1, snapshot_updates=1, seed=9)
    trainer = HistoryTrainer(cfg, tmp_path / "start")
    trainer.update()
    checkpoint = trainer.save()
    overrides = parse_resume_overrides(["profile_collection=true",
                                        "profile_collection_warmup=1"])
    resumed = HistoryTrainer(HistoryPPOConfig(updates=3), tmp_path / "diag", resume=checkpoint,
                             resume_overrides=overrides)
    assert resumed.config.profile_collection and resumed.config.profile_collection_warmup == 1
    assert resumed.config_changes[-1] == dict(
        at_update=1, changes={"profile_collection": [False, True],
                              "profile_collection_warmup": [0, 1]})
    first, second = resumed.update(), resumed.update()
    assert not first["collection_profile_synchronized"]
    assert second["collection_profile_synchronized"] and second["collection_phase_seconds"]
    with pytest.raises(ValueError, match="warmup"):
        HistoryPPOConfig(profile_collection_warmup=-1)
