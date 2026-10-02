"""Trainer-owned frozen scopes preserve replay and enforce their weight contract."""
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import pytest
import torch

from train.history_inference import BatchedHistoryCache
from train.history_model import HistoryPolicyConfig, PublicStream, fresh_player
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer
from test_history_host_cache import append
from test_history_rollout import make_env


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


def test_cache_scans_only_boundaries_and_rebuilds_after_update(monkeypatch):
    actor = player()
    cache = BatchedHistoryCache(actor)
    stream = PublicStream(0)
    append(stream, 5)
    calls = []
    original = cache._signature
    monkeypatch.setattr(cache, '_signature', lambda: (calls.append(1), original())[1])
    with cache.frozen_weights():
        for _ in range(5):
            _, actual = cache.encode([(0, 0)], [stream])
        assert len(calls) == 1
    assert len(calls) == 2
    assert torch.equal(actual, BatchedHistoryCache(actor).encode([(0, 0)], [stream])[1])
    old = cache.entries[(0, 0)]
    with torch.no_grad():
        actor.public.weight.add_(0.1)
    with cache.frozen_weights():
        _, actual = cache.encode([(0, 0)], [stream])
    assert cache.entries[(0, 0)] is not old
    assert torch.equal(actual, BatchedHistoryCache(actor).encode([(0, 0)], [stream])[1])


@pytest.mark.parametrize('change', ['inplace', 'parameter', 'load', 'dtype'])
def test_scope_rejects_weight_mutations_and_clears_stale_entries(change):
    actor = player()
    cache = BatchedHistoryCache(actor)
    stream = PublicStream(0)
    with pytest.raises(RuntimeError, match='weights changed'):
        with cache.frozen_weights():
            cache.encode([(0, 0)], [stream])
            with torch.no_grad():
                if change == 'inplace':
                    actor.public.weight.add_(0.1)
                elif change == 'parameter':
                    actor.public.weight = torch.nn.Parameter(actor.public.weight.clone())
                elif change == 'load':
                    actor.load_state_dict(actor.state_dict())
                else:
                    actor.double()
    assert not cache.entries
    assert cache._frozen_signature is None
    with cache.frozen_weights():
        cache.encode([(0, 0)], [stream])


def collector(policies, **extra):
    return HistoryCollector(make_env(4, 23), policies[0], MatchEventStore(),
                            SequenceRolloutBuffer(), torch.Generator().manual_seed(31),
                            seat_policy=lambda env, match: [0, 1, 0, 2],
                            resolve_policy=policies.__getitem__, record_choices=True,
                            kv_cache=True, temperature=0.8, epsilon=0.2, **extra)


@pytest.mark.parametrize('variant', ['plain', 'paged', 'merged'])
def test_collection_replay_is_exact_with_frozen_scopes(variant):
    policies = {i: player(17 + i) for i in (0, 1, 2)}
    for actor in policies.values():
        actor.batched_private_attention = True
        actor.wide_private_projection = True
    extra = dict(paged_cache=variant != 'plain', batch_snapshot_policies=variant == 'merged',
                 batch_snapshot_encoder=variant == 'merged')
    runs = []
    for frozen in (False, True):
        run = collector(policies, **extra)
        stats = []
        for _ in range(2):
            with run.frozen_weights() if frozen else nullcontext():
                stats.append(asdict(run.collect(24)))
        assert run._frozen_stack is None
        assert not run._frozen_caches
        runs.append((run, stats))
    (plain, a), (fast, b) = runs
    assert a == b
    assert plain.policy_decisions == fast.policy_decisions
    assert torch.equal(plain.generator.get_state(), fast.generator.get_state())
    for x, y in zip(plain.choice_log, fast.choice_log):
        np.testing.assert_array_equal(x, y)
    for name, x in plain.buffer.compact().items():
        np.testing.assert_array_equal(x, fast.buffer.compact()[name])
    for key, stream in plain.store.streams.items():
        for x, y in zip(stream.arrays(), fast.store.streams[key].arrays()):
            np.testing.assert_array_equal(x, y)


def test_lazy_replaced_and_removed_caches_exit_even_on_failure():
    policies = {i: player(17 + i) for i in (0, 1, 2)}
    run = collector(policies)
    touched = []
    with pytest.raises(ValueError, match='test interruption'):
        with run.frozen_weights():
            run.collect(2)
            touched.extend(run.caches.values())
            assert touched and all(c._frozen_signature is not None for c in touched)
            replacement = player(99)
            touched.append(run._cache(1, replacement))
            run.caches.clear()
            raise ValueError('test interruption')
    assert run._frozen_stack is None and not run._frozen_caches
    assert all(c._frozen_signature is None for c in touched)
    with run.frozen_weights():
        run.collect(2)


def test_removed_cache_is_still_checked_and_nested_scopes_fail():
    policies = {i: player(17 + i) for i in (0, 1, 2)}
    run = collector(policies)
    with pytest.raises(RuntimeError, match='weights changed'):
        with run.frozen_weights():
            cache = run._cache(1, policies[1])
            with pytest.raises(RuntimeError, match='nested'):
                with run.frozen_weights():
                    pass
            with pytest.raises(RuntimeError, match='nested'):
                with cache.frozen_weights():
                    pass
            run.caches.clear()
            with torch.no_grad():
                policies[1].public.weight.add_(1)
    assert run._frozen_stack is None and cache._frozen_signature is None
