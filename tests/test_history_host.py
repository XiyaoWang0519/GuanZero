"""The diagnostic hashes must detect state changes but ignore wall-clock noise."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from bench.history_host import ExactDigest, chunk_hashes, make_population
from train.history_model import HistoryPolicyConfig
from train.history_rollout import CollectStats, MatchEventStore, SequenceRolloutBuffer


def test_exact_digest_distinguishes_dtype_shape_and_bit_pattern():
    values = [np.zeros((1,), np.float32), np.zeros((1, 1), np.float32),
              np.zeros((1,), np.int32), np.asarray([-0.0], np.float32)]
    hashes = []
    for value in values:
        digest = ExactDigest()
        digest.array("value", value)
        hashes.append(digest.hexdigest())
    assert len(set(hashes)) == len(values)


@pytest.fixture
def replay():
    collector = SimpleNamespace(
        choice_log=[np.asarray([0, 1], np.int32)], buffer=SequenceRolloutBuffer(),
        store=MatchEventStore(), generator=torch.Generator().manual_seed(7),
        assignments={(0, 0): np.asarray([0, 1, 0, 1])}, policy_decisions={0: 1, 1: 1})
    population = SimpleNamespace(rng=np.random.default_rng(8), seat_matches={0: 2, 1: 2})
    return collector, CollectStats(decisions=2, policy_batches=2), population


def test_chunk_hash_ignores_only_phase_timing(replay):
    collector, stats, population = replay
    before = chunk_hashes(collector, stats, population)
    stats.phase_seconds["actor_and_sampling"] = 123.456
    assert chunk_hashes(collector, stats, population) == before
    stats.behaviour_entropy_sum = -0.0
    after = chunk_hashes(collector, stats, population)
    assert after["sha256"] != before["sha256"]
    assert after["parts"]["stats"] != before["parts"]["stats"]


@pytest.mark.parametrize("part", ["choices", "buffer", "public_streams", "generator", "population"])
def test_chunk_hash_detects_replay_state_changes(replay, part):
    collector, stats, population = replay
    before = chunk_hashes(collector, stats, population)
    if part == "choices":
        collector.choice_log[0][0] = 1
    elif part == "buffer":
        collector.buffer.compact()["obs"] = np.zeros((1, collector.buffer.obs_dim), np.uint8)
    elif part == "public_streams":
        collector.store.stream(0, 0)
    elif part == "generator":
        torch.rand(1, generator=collector.generator)
    else:
        collector.assignments[(0, 0)][0] = 1
    after = chunk_hashes(collector, stats, population)
    assert after["sha256"] != before["sha256"]
    assert after["parts"][part] != before["parts"][part]


def test_population_has_distinct_frozen_weights_and_production_assignments():
    args = SimpleNamespace(seed=123, causal_sdpa=True, identities=3, case="wide",
                           recent=1, snapshot_probability=0.5, archive_share=0.5)
    config = HistoryPolicyConfig(width=16, layers=1, heads=4, response_mode="auxiliary")
    actor, population, hashes = make_population(args, config)
    assert len(set(hashes.values())) == 4
    assert population.archive == [1, 2, 3]
    assert all(not parameter.requires_grad for model in [actor, *population.models.values()]
               for parameter in model.parameters())
    _, repeat, repeated_hashes = make_population(args, config)
    assert hashes == repeated_hashes
    for env in range(32):
        seats = population.assignment(env, 0)
        assert seats == repeat.assignment(env, 0)
        assert 0 in seats
        assert set(seats) <= set(hashes)
