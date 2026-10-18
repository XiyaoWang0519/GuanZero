"""Public event batching keeps the scalar stream, privacy and replay contracts."""
from dataclasses import asdict

import gd
import numpy as np
import pytest
import torch

from train.history_habit import RoundEventStore
from train.history_model import HistoryPolicyConfig, PublicStream, fresh_player
from train.history_ppo import HistoryPPOConfig, HistoryTrainer
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer
from train.logs import TOKEN_DIM, public_token


class ScalarEvents:
    """An older/custom environment exposing only the scalar public event API."""
    drain_public_batch = None

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)


def environment(seed=3, threads=1, limit=-1):
    return gd.VecEnv(num_envs=4, num_threads=threads, seed=seed,
                     log_public_actions=True, log_env_limit=limit)


def scalar_batch(events):
    """Independent legacy conversion, including public_token's flag redaction."""
    batch = {name: np.asarray([int(getattr(e, name)) for e in events], np.int64)
             for name in ("env_id", "match_id", "round_index", "phase")}
    batch["tokens"] = np.asarray([public_token(e) for e in events], np.uint8).reshape(-1, TOKEN_DIM)
    return batch


def assert_batch(actual, expected):
    assert actual.keys() == expected.keys()
    for name, array in actual.items():
        np.testing.assert_array_equal(array, expected[name], err_msg=name)
        assert array.dtype == expected[name].dtype
        assert array.flags.c_contiguous and array.flags.owndata
    assert not actual["tokens"][:, 150:158].any()


def assert_stores(actual, expected):
    assert actual.current == expected.current
    assert actual.streams.keys() == expected.streams.keys()
    for key, stream in actual.streams.items():
        other = expected.streams[key]
        assert (stream.match_id, stream.generation) == (other.match_id, other.generation)
        for a, b in zip(stream.arrays(), other.arrays()):
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("threads", [1, 2])
def test_binding_matches_scalar_across_forced_passes_exchanges_and_matches(threads):
    scalar, batched = environment(threads=threads), environment(threads=threads)
    scalar.reset()
    batched.reset()
    legacy, fast = MatchEventStore(), MatchEventStore()
    forced, phases, generations = False, set(), set()
    for _ in range(1000):
        a, b = scalar.pending(), batched.pending()
        expected_events = scalar.drain_public_actions()
        actual = batched.drain_public_batch()
        assert_batch(actual, scalar_batch(expected_events))
        legacy.ingest(expected_events)
        assert fast.ingest_batch(actual) == len(expected_events)
        forced |= any(event.forced for event in expected_events)
        phases.update(actual["phase"].tolist())
        generations.update(actual["match_id"].tolist())
        choices = np.array(a.greedy_choice, np.int32)
        np.testing.assert_array_equal(choices, b.greedy_choice)
        scalar.step(choices)
        batched.step(choices)
    assert forced
    assert {int(gd.Phase.Play), int(gd.Phase.Tribute), int(gd.Phase.BackTribute)} <= phases
    assert max(generations) >= 1, "the fixture must cross a match boundary"
    assert_stores(fast, legacy)


@pytest.mark.parametrize("limit", [0, 1, -1])
def test_batch_empty_drain_log_limit_and_owned_lifetime(limit):
    env = environment(limit=limit)
    env.reset()
    empty = env.drain_public_batch()
    assert_batch(empty, scalar_batch([]))
    for _ in range(15):
        pending = env.pending()
        env.step(np.array(pending.greedy_choice, np.int32))
    batch = env.drain_public_batch()
    assert all(a.flags.owndata for a in batch.values())
    if limit == 0:
        assert len(batch["tokens"]) == 0
    else:
        assert len(batch["tokens"]) > 0
        if limit > 0:
            assert (batch["env_id"] < limit).all()
    retained = {name: array.copy() for name, array in batch.items()}
    assert_batch(env.drain_public_batch(), scalar_batch([]))
    assert env.drain_public_actions() == []
    env.reset()
    del env
    assert_batch(batch, retained)


def token_arrays(n):
    tokens = np.zeros((n, TOKEN_DIM), np.uint8)
    tokens[np.arange(n), np.arange(n) % 4] = 1
    tokens[:, 4 + 11] = 1
    tokens[np.arange(n), 158 + np.arange(n) % 28] = 1
    return tokens, np.arange(n, dtype=np.int64) // 20, np.full(n, int(gd.Phase.Play), np.int64)


def test_bulk_stream_append_preserves_inputs_prior_views_growth_and_reset():
    tokens, rounds, phases = token_arrays(137)
    stream, legacy = PublicStream(3), PublicStream(3)
    views = []
    for start, end in [(0, 0), (0, 63), (63, 65), (65, 137)]:
        stream.extend_tokens(tokens[start:end], rounds[start:end], phases[start:end])
        for i in range(start, end):
            legacy.append_token(tokens[i], int(rounds[i]), int(phases[i]))
        for a, b in zip(stream.arrays(), legacy.arrays()):
            np.testing.assert_array_equal(a, b)
        views.append((stream.arrays(), tuple(a.copy() for a in stream.arrays())))
    tokens[:] = 0
    rounds[:] = 99
    phases[:] = 99
    stream.reset(4)
    stream.extend_tokens(*token_arrays(2))
    assert stream.match_id == 4 and stream.generation == 1 and stream.prefix == 2
    for arrays, expected in views:
        for a, b in zip(arrays, expected):
            assert not a.flags.writeable
            np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("invalid", ["shape", "round_shape", "phase_shape", "seat", "count",
                                    "private", "round_negative", "phase_negative"])
def test_bulk_stream_validates_the_complete_input_before_appending(invalid):
    tokens, rounds, phases = token_arrays(3)
    if invalid == "shape":
        tokens = tokens[:, :-1]
    elif invalid == "round_shape":
        rounds = rounds[:-1]
    elif invalid == "phase_shape":
        phases = phases[:, None]
    elif invalid == "seat":
        tokens[-1, :4] = 0
    elif invalid == "count":
        tokens[-1, 158:] = 1
    elif invalid == "private":
        tokens[-1, 153] = 1
    elif invalid == "round_negative":
        rounds[-1] = -1
    else:
        phases[-1] = -1
    stream = PublicStream()
    stream.extend_tokens(*token_arrays(1))
    with pytest.raises(ValueError):
        stream.extend_tokens(tokens, rounds, phases)
    assert stream.prefix == 1


def test_store_batches_preserve_interleaved_keys_pruning_and_validate_ids():
    tokens, rounds, phases = token_arrays(6)
    batch = dict(tokens=tokens, round_index=rounds, phase=phases,
                 env_id=np.array([0, 0, 1, 0, 0, 0]), match_id=np.array([0, 0, 0, 0, 1, 1]))
    legacy, fast = MatchEventStore(), MatchEventStore()
    for i in range(6):
        legacy.stream(batch["env_id"][i], batch["match_id"][i]).append_token(
            tokens[i], int(rounds[i]), int(phases[i]))
    assert fast.ingest_batch(batch) == 6
    assert_stores(fast, legacy)
    assert fast.prune({(0, 0)}) == 0
    assert fast.prune(set()) == 1
    assert (0, 0) not in fast.streams
    with pytest.raises(ValueError, match="went backwards"):
        fast.ingest_batch({name: array[:1] for name, array in batch.items()})
    for name in ("env_id", "match_id"):
        with pytest.raises(ValueError, match="one entry per public token"):
            MatchEventStore().ingest_batch({**batch, name: batch[name][:-1]})


@pytest.mark.parametrize("first", [(0, 0), (1, 5), (1, 6)])
def test_invalid_final_match_run_leaves_every_stream_unchanged(first):
    store = MatchEventStore()
    store.stream(1, 2).extend_tokens(*token_arrays(1))
    store.stream(1, 5).extend_tokens(*token_arrays(1))
    current = dict(store.current)
    retained = {key: (stream, tuple(array.copy() for array in stream.arrays()))
                for key, stream in store.streams.items()}
    tokens, rounds, phases = token_arrays(2)
    batch = dict(tokens=tokens, round_index=rounds, phase=phases,
                 env_id=np.array([first[0], 1]), match_id=np.array([first[1], 4]))
    with pytest.raises(ValueError, match="went backwards"):
        store.ingest_batch(batch)
    assert store.current == current and store.streams.keys() == retained.keys()
    for key, (stream, arrays) in retained.items():
        assert store.streams[key] is stream
        for actual, expected in zip(stream.arrays(), arrays):
            np.testing.assert_array_equal(actual, expected)
    # An older match that already exists remains legal, just as stream() allows.
    assert store.ingest_batch({**batch, "match_id": np.array([5, 2]),
                               "env_id": np.array([1, 1])}) == 2
    assert store.current == current


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def player(seed=17):
    return fresh_player(HistoryPolicyConfig(width=16, layers=1, heads=2,
                        action_width=16, fusion_width=16, critic_width=16), seed)[0]


@pytest.mark.parametrize("variant", ["plain", "paged", "merged"])
def test_complete_collection_and_sampler_are_bitwise_equal(variant):
    policies = {i: player(17 + i) for i in (0, 1, 2)}
    runs = []
    for batching in (False, True):
        env = environment(seed=23)
        collector = HistoryCollector(
            env if batching else ScalarEvents(env), policies[0], MatchEventStore(),
            SequenceRolloutBuffer(), torch.Generator().manual_seed(31),
            seat_policy=lambda env, match: [0, 1, 0, 2], resolve_policy=policies.__getitem__,
            record_choices=True, kv_cache=True, temperature=0.8, epsilon=0.2,
            paged_cache=variant != "plain", batch_snapshot_policies=variant == "merged",
            batch_snapshot_encoder=variant == "merged")
        stats = []
        for chunk in range(2):
            collector.invalidate_learner_cache()
            with collector.frozen_weights():
                stats.append(asdict(collector.collect(72, version=chunk)))
        runs.append((collector, stats))
    (legacy, a), (fast, b) = runs
    assert a == b
    assert legacy.policy_decisions == fast.policy_decisions
    assert torch.equal(legacy.generator.get_state(), fast.generator.get_state())
    for a, b in zip(legacy.choice_log, fast.choice_log):
        np.testing.assert_array_equal(a, b)
    for name, a in legacy.buffer.compact().items():
        np.testing.assert_array_equal(a, fast.buffer.compact()[name], err_msg=name)
    assert_stores(fast.store, legacy.store)


def test_collector_preserves_subclass_ingest_and_round_streams():
    class TrackedRoundStore(RoundEventStore):
        calls = 0

        def ingest(self, events):
            self.calls += 1
            return super().ingest(events)

    class ScalarOnly(ScalarEvents):
        def drain_public_batch(self):
            raise AssertionError("a specialized store must retain its ingest override")

    env = ScalarOnly(environment())
    store = TrackedRoundStore()
    collector = HistoryCollector(env, player(), store, SequenceRolloutBuffer(),
                                 torch.Generator().manual_seed(31), round_streams=True)
    collector.collect(120)
    assert store.calls == 120 and store.rounds
    for (env_id, match_id, round_index), stream in store.rounds.items():
        full = store.streams[(env_id, match_id)]
        selected = full.rounds == round_index
        for actual, expected in zip(stream.arrays(), full.arrays()):
            np.testing.assert_array_equal(actual, expected[selected])


def test_complete_ppo_updates_preserve_weights_optimizer_replay_and_losses(tmp_path):
    from test_history_trim import EXCLUDED_KEYS, record

    config = HistoryPPOConfig(width=16, layers=1, heads=2, num_envs=4,
                              steps_per_update=90, epochs=1, minibatch_matches=2,
                              snapshot_updates=1, snapshot_probability=1.0,
                              rollout_kv_cache=True, seed=3)
    records = []
    for batching in (False, True):
        trainer = HistoryTrainer(config, tmp_path / str(batching))
        if not batching:
            trainer.collector.env = ScalarEvents(trainer.collector.env)
        trainer.collector.choice_log = []
        lines = [trainer.update() for _ in range(2)]
        assert all(line["update_samples"] > 0 for line in lines)
        result = record(trainer, lines)
        result["progress"].pop("elapsed_seconds")
        result["lines"] = [{k: v for k, v in line.items() if k not in EXCLUDED_KEYS}
                           for line in lines]
        records.append(result)

    def identical(actual, expected):
        if isinstance(actual, torch.Tensor):
            assert torch.equal(actual, expected)
        elif isinstance(actual, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(actual, dict):
            assert actual.keys() == expected.keys()
            for key in actual:
                identical(actual[key], expected[key])
        elif isinstance(actual, (list, tuple)):
            assert len(actual) == len(expected)
            for a, b in zip(actual, expected):
                identical(a, b)
        else:
            assert actual == expected

    identical(*records)
