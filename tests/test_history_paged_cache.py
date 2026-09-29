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
