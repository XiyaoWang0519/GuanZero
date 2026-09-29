"""Paged KV cache: bitwise equal to the per-entry cache, page accounting, collector wiring."""
import copy

import numpy as np
import pytest
import torch

from train.history_inference import BatchedHistoryCache
from train.history_model import HistoryPolicyConfig, PublicStream, fresh_player
from train.history_paged_cache import KVPagePool, PagedHistoryCache
from test_history_inference import append


def devices():
    return ["cpu", "cuda"]


def skip_without(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA paged-cache acceptance")


def check_pool(pool, caches):
    used = sum(len(e.pages) for cache in caches for e in cache.entries.values())
    assert used == pool.used_pages
    owned = [p for cache in caches for e in cache.entries.values() for p in e.pages]
    assert len(set(owned)) == len(owned) and 0 not in owned and not set(owned) & set(pool.free)
    P = pool.page_tokens
    assert not pool.memory[:P].any() and not any(kv[:P].any() for kv in pool.kv)


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("chunk", [16, 128])
def test_paged_cache_matches_per_entry_cache_bitwise(device, chunk):
    skip_without(device)
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=2, heads=4), 7)
    actor.to(device).train()
    reference = BatchedHistoryCache(actor, chunk_size=chunk)
    pool = KVPagePool.for_actor(actor, page_tokens=8, pages=2)   # forces growth
    paged = PagedHistoryCache(actor, pool, chunk_size=chunk)
    rng = np.random.default_rng(3)
    streams = {i: PublicStream(i) for i in range(6)}
    for s, n in zip(streams.values(), [0, 3, 40, 150, 7, 90]):
        append(s, n)
    for step in range(30):
        # A varying subset in varying order, as a vector step's identity groups are.
        picked = [int(i) for i in rng.permutation(6)[:rng.integers(1, 7)]]
        keys = [(i, streams[i].match_id) for i in picked]
        batch = [streams[i] for i in picked]
        _, expected = reference.encode(keys, batch)
        _, actual = paged.encode(keys, batch)
        assert actual.shape == expected.shape and torch.equal(actual, expected), step
        assert paged.encoded_tokens == reference.encoded_tokens
        assert paged.rebuilds == reference.rebuilds
        assert {k: e.length for k, e in paged.entries.items()} == {
            k: e.length for k, e in reference.entries.items()}
        check_pool(pool, [paged])
        for i in picked:
            append(streams[i], int(rng.integers(0, 5)), offset=step)
        if step == 10:      # a new match on environment 2 resets its stream
            streams[2].reset(match_id=2)
            append(streams[2], 4)
        if step == 20:      # an assignment change prunes two matches
            active = {(i, streams[i].match_id) for i in (0, 1, 2, 3)}
            reference.prune(active)
            paged.prune(active)
            check_pool(pool, [paged])
        if step == 25:      # a weight update invalidates the whole policy cache
            with torch.no_grad():
                actor.public.weight.add_(1e-3)
    assert pool.grows > 1
    paged.clear()
    assert pool.used_pages == 0 and paged.bytes == 0


@pytest.mark.parametrize("device", devices())
def test_identities_share_one_pool(device):
    skip_without(device)
    config = HistoryPolicyConfig(width=32, layers=2, heads=4)
    first, _ = fresh_player(config, 1)
    second, _ = fresh_player(config, 2)
    first.to(device).train()
    second.to(device).train()
    pool = KVPagePool.for_actor(first, page_tokens=16, pages=2)
    caches = [PagedHistoryCache(actor, pool) for actor in (first, second)]
    references = [BatchedHistoryCache(actor) for actor in (first, second)]
    streams = [PublicStream(i) for i in range(4)]
    for s, n in zip(streams, [5, 30, 64, 100]):
        append(s, n)
    for rounds in range(3):
        for cache, reference, picked in zip(caches, references, ([0, 1, 3], [1, 2, 3])):
            keys = [(i, i) for i in picked]
            _, actual = cache.encode(keys, [streams[i] for i in picked])
            _, expected = reference.encode(keys, [streams[i] for i in picked])
            assert torch.equal(actual, expected)
        check_pool(pool, caches)
        for s in streams:
            append(s, 3, offset=rounds)
    caches[0].clear()
    check_pool(pool, caches)
    assert pool.used_pages == sum(len(e.pages) for e in caches[1].entries.values()) > 0
    with pytest.raises(ValueError, match="one architecture"):
        PagedHistoryCache(fresh_player(HistoryPolicyConfig(width=64, layers=2, heads=4), 3)[0]
                          .to(device), pool)


def test_returned_memory_is_independent_of_later_appends():
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=2, heads=4), 5)
    cache = PagedHistoryCache(actor.train())
    stream = PublicStream(0)
    append(stream, 20)
    _, before = cache.encode([(0, 0)], [stream])
    snapshot = before.clone()
    append(stream, 50)
    cache.encode([(0, 0)], [stream])
    assert torch.equal(before, snapshot)


@pytest.mark.parametrize("device", devices())
def test_merged_snapshot_encode_matches_per_identity_encodes(device):
    """One merged pass over three identities equals their separate encodes up to
    FP32 reduction order, including prefill chunks, fresh BOS rows, uneven row
    counts per slot and a slot gap; counters and pages stay per identity."""
    skip_without(device)
    from train.history_paged_cache import merged_encode
    from train.history_snapshot_batch import SnapshotHeads
    config = HistoryPolicyConfig(width=32, layers=2, heads=4)
    actors = [fresh_player(config, seed)[0].to(device).train() for seed in (1, 2, 3)]
    pool = KVPagePool.for_actor(actors[0], page_tokens=16, pages=2)
    merged = [PagedHistoryCache(actor, pool, chunk_size=32) for actor in actors]
    separate = [PagedHistoryCache(actor, chunk_size=32) for actor in actors]
    heads = SnapshotHeads(encoder=True)
    slots = [heads.slot(identity, actor) for identity, actor in zip((5, 9, 11), actors)]
    heads.retain({5, 11})               # slot 1 unused: a gap in the slot layout
    slots = [slots[0], slots[2]]
    use = [0, 2]
    streams = {i: PublicStream(i) for i in range(7)}
    for s, n in zip(streams.values(), [0, 5, 70, 33, 1, 100, 12]):
        append(s, n)
    picks = [[0, 2, 3, 5], [1, 4]]
    rng = np.random.default_rng(4)
    for step in range(8):
        parts = [(merged[a], [(i, i) for i in pick], [streams[i] for i in pick], slot)
                 for a, pick, slot in zip(use, picks, slots)]
        size = max(streams[i].prefix for pick in picks for i in pick) + 1
        actual = merged_encode(parts, heads, size)
        expected = []
        for a, pick in zip(use, picks):
            _, memory = separate[a].encode([(i, i) for i in pick], [streams[i] for i in pick])
            out = memory.new_zeros(len(pick), size, config.width)
            span = min(size, memory.shape[1])
            out[:, :span] = memory[:, :span]
            expected.append(out)
        expected = torch.cat(expected)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        for a in use:
            assert merged[a].encoded_tokens == separate[a].encoded_tokens
            assert merged[a].rebuilds == separate[a].rebuilds
        check_pool(pool, [merged[a] for a in use])
        for pick in picks:
            for i in pick:
                append(streams[i], step % 3 + 1, offset=step)
        order = [int(i) for i in rng.permutation(7)]
        cut = int(rng.integers(1, 6))
        picks = [order[:cut], order[cut:cut + int(rng.integers(1, 7 - cut + 1))]]


# ---- collector wiring ------------------------------------------------------------

@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("encoder", [False, True])
def test_collector_paged_and_merged_encoder_match_the_plain_rollout(device, encoder):
    """Paged cache: bitwise. Merged snapshot encoding: tier 2 on snapshot seats
    only; on these seeds the float noise flips no choice, so the whole rollout
    (learner rows, sampler state, snapshot choices) is identical."""
    skip_without(device)
    from test_history_snapshot_batch import assert_same_collection, run_collector
    plain = run_collector(device, False, kv_cache=True)[0]
    fast = run_collector(device, True, kv_cache=True, paged_cache=True,
                         batch_snapshot_encoder=encoder)[0]
    assert_same_collection(plain, fast)
    assert fast.kv_pool is not None and fast.snapshot_heads.encoder is encoder
    assert fast.cache_metrics()["pool_used_pages"] > 0


@pytest.mark.parametrize("device", devices())
def test_merged_encoder_refreshes_when_snapshots_change(device):
    skip_without(device)
    from test_history_snapshot_batch import (assert_same_collection, config, players,
                                             run_collector)

    def seats(env, match):
        base = 1 + (env + 2 * match) % 5
        return [0, base, 1 + (base % 5), 1 + ((base + 2) % 5)]

    def between(chunk, policies, collector):
        with torch.no_grad():
            if chunk == 0:      # identity 2 replaced by a new actor object
                policies[2] = fresh_player(config(), 77)[0].to(device)
            if chunk == 1:      # identity 1's ENCODER updated in place
                policies[1].public.weight.mul_(-4.0)

    policies = [players(range(6), device) for _ in range(2)]
    runs = [run_collector(device, merge, kv_cache=True, steps=60, seat_policy=seats,
                          between=between, policies=pols, **extra)[0]
            for pols, (merge, extra) in zip(policies, (
                (False, {}), (True, dict(paged_cache=True, batch_snapshot_encoder=True))))]
    assert_same_collection(*runs)
    # Random snapshots barely read their memory, so choices alone cannot show a
    # stale encoder: every live slot must hold its actor's current weights.
    from train.history_snapshot_batch import _encoder_values
    heads = runs[1].snapshot_heads
    assert 1 in heads.slots and 2 in heads.slots
    for identity, record in heads.slots.items():
        for name, value in _encoder_values(policies[1][identity]).items():
            assert torch.equal(heads.tensors[name][record.index], value), (identity, name)


def test_merged_encoder_requires_the_paged_cache():
    from train.history_ppo import HistoryPPOConfig
    with pytest.raises(ValueError, match="paged"):
        HistoryPPOConfig(rollout_kv_cache=True, batch_snapshot_policies=True,
                         batch_snapshot_encoder=True)
    HistoryPPOConfig(rollout_kv_cache=True, batch_snapshot_policies=True,
                     rollout_paged_cache=True, batch_snapshot_encoder=True)


@pytest.mark.parametrize("device", devices())
def test_page_span_matches_within_fp32_order(device):
    """Page-padded spans: same values on every real position (tier 2), memory
    padded to whole pages rather than powers of two."""
    skip_without(device)
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=2, heads=4), 7)
    actor.to(device).train()
    reference = BatchedHistoryCache(actor, chunk_size=32)
    tight = PagedHistoryCache(actor, KVPagePool.for_actor(actor, page_tokens=16),
                              chunk_size=32, page_span=True)
    streams = [PublicStream(i) for i in range(4)]
    for s, n in zip(streams, [0, 20, 70, 150]):
        append(s, n)
    for step in range(5):
        keys = [(i, i) for i in range(4)]
        _, expected = reference.encode(keys, streams)
        _, actual = tight.encode(keys, streams)
        longest = max(s.prefix for s in streams) + 1
        assert actual.shape[1] == -(-longest // 16) * 16 <= expected.shape[1]
        torch.testing.assert_close(actual, expected[:, :actual.shape[1]], rtol=1e-5, atol=1e-6)
        assert not expected[:, actual.shape[1]:].any()
        for s in streams:
            append(s, step + 1, offset=step)


def test_page_span_configuration():
    from train.history_ppo import HistoryPPOConfig
    with pytest.raises(ValueError, match="page_span"):
        HistoryPPOConfig(rollout_kv_cache=True, rollout_page_span=True)
    HistoryPPOConfig(rollout_kv_cache=True, rollout_paged_cache=True, rollout_page_span=True)
