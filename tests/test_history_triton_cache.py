"""Exact byte copies, stream ownership and eager fallback for optional Triton KV."""
import copy

import numpy as np
import pytest
import torch

from train.history_inference import BatchedHistoryCache, Entry
from train.history_model import HistoryActor, HistoryPolicyConfig, PublicStream
from train.history_triton_cache import INT32_LIMIT, TritonHistoryCache, rows_per_launch, triton
from test_history_host_cache import append


def bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def pattern(shape, device, offset=0):
    values = np.asarray([0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001,
                         0x7f800001, 0xffc12345, 1, 0x80000001, 0x3f800000,
                         0xdeadbeef, 0x7f7fffff, 0xffffffff], dtype=np.uint32)
    indices = (np.arange(int(np.prod(shape))) + offset) % len(values)
    return torch.from_numpy(values[indices].reshape(shape).view(np.float32).copy()).to(device)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    device = request.param
    if device == "cuda" and (not torch.cuda.is_available() or triton is None):
        pytest.skip("CUDA and Triton acceptance")
    old = (torch.are_deterministic_algorithms_enabled(),
           torch.is_deterministic_algorithms_warn_only_enabled(),
           torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield device
    torch.use_deterministic_algorithms(old[0], warn_only=old[1])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = old[2:]


def actor_on(device):
    return HistoryActor(HistoryPolicyConfig(width=32, layers=2, heads=4)).float().to(device)


@pytest.mark.parametrize("forced_rows", [None, 1, 2])
@pytest.mark.parametrize("capacities,starts,counts,capacity", [
    ([32], [0], [1], 32), ([64], [17], [7], 32),
    ([32, 64, 128], [10, 31, 69], [3, 5, 11], 128),
    ([64, 64, 64], [25, 20, 9], [7, 9, 11], 32),
    ([128, 256], [61, 120], [65, 9], 256),
])
def test_packing_copies_arbitrary_bits_and_strides(device, capacities, starts, counts, capacity,
                                                   forced_rows):
    actor = actor_on(device)
    fast, eager = TritonHistoryCache(actor), BatchedHistoryCache(actor)
    fast.max_rows_per_launch = forced_rows
    entries, originals = [], []
    for row, (own, start) in enumerate(zip(capacities, starts)):
        shape = (4, own, 8)
        keys = [pattern(shape, device, row + layer) for layer in range(2)]
        values = [pattern(shape, device, row + layer + 2) for layer in range(2)]
        entry = Entry(PublicStream(row), 0, start, own, keys, values,
                      actor.bos.new_zeros(own, 32))
        entries.append(entry)
        originals.append(Entry(entry.stream, 0, start, own, [t.clone() for t in keys],
                               [t.clone() for t in values], entry.memory.clone()))
    for layer in range(2):
        batch, width = len(entries), max(counts)
        # Distinct strides and offsets, including non-contiguous projection views.
        new_k = pattern((batch, width + 2, 3, 4, 16), device, 5)[:, 1:1+width, 1, :, 1::2].transpose(1, 2)
        new_v = pattern((batch, width + 3, 2, 4, 24), device, 9)[:, 2:2+width, 0, :, 2::3].transpose(1, 2)
        actual = fast._pack_keys_values(entries, counts, layer, new_k, new_v, capacity)
        expected = eager._pack_keys_values(originals, counts, layer, new_k, new_v, capacity)
        for a, b in zip(actual, expected):
            bits(a, b)
            assert a.stride() == b.stride() and a.is_contiguous()
        for a, b in zip(entries, originals):
            for x, y in zip(a.keys + a.values, b.keys + b.values):
                bits(x, y)
    if device == "cuda":
        assert fast.copy_stats["triton_packs"] == 2
        assert fast.copy_stats["pointer_reuses"] == 1
        old_table = fast._pointer_table
        old_address = entries[0].keys[0].data_ptr()
        entries[0].keys[0].data = entries[0].keys[0].data.clone()
        assert old_table.references[fast._columns[id(entries[0])]][0].data_ptr() == old_address
        assert entries[0].keys[0].data_ptr() != old_address
        actual = fast._pack_keys_values(entries, counts, 0, new_k, new_v, capacity)
        expected = eager._pack_keys_values(originals, counts, 0, new_k, new_v, capacity)
        for a, b in zip(actual, expected):
            bits(a, b)
        assert fast._pointer_table is not old_table
        assert fast.copy_stats["pointer_uploads"] == fast.copy_stats["table_rebuilds"] == 2
    else:
        assert not fast.metadata_bytes and fast.copy_stats["triton_packs"] == 0
    fast.clear()
    assert not fast.metadata_bytes


@pytest.mark.parametrize("min_batch", [1, 2, 4])
def test_encode_lifecycle_and_minimum_batch(device, min_batch):
    actor = actor_on(device)
    reference_actor = copy.deepcopy(actor)
    fast = TritonHistoryCache(actor, chunk_size=8, min_batch=min_batch)
    eager = BatchedHistoryCache(reference_actor, chunk_size=8)
    streams = [PublicStream(i) for i in range(4)]
    saved = None

    def compare(order):
        nonlocal saved
        keys, selected = [(i, i) for i in order], [streams[i] for i in order]
        meta, actual = fast.encode(keys, selected)
        expected_meta, expected = eager.encode(keys, selected)
        bits(actual, expected)
        bits(meta.lengths, expected_meta.lengths)
        if saved is not None:
            bits(*saved)
        saved = actual, actual.clone()
        for key in keys:
            a, b = fast.entries[key], eager.entries[key]
            for x, y in zip(a.keys + a.values + [a.memory], b.keys + b.values + [b.memory]):
                bits(x, y)

    compare([0])
    if min_batch > 1:
        assert fast.copy_stats["triton_packs"] == 0 and fast.metadata_bytes == 0
    for stream, count in zip(streams, [2, 11, 33, 7]):
        append(stream, count)
    compare([3, 1, 0, 2])
    streams[1] = copy.deepcopy(streams[1])
    streams[2].reset(2)
    append(streams[2], 3)
    compare([0, 1, 2, 3])
    with torch.no_grad():
        actor.bos.add_(0.125)
        reference_actor.bos.add_(0.125)
    compare([0, 1, 2, 3])
    state = copy.deepcopy(actor.state_dict())
    actor.load_state_dict(state)
    reference_actor.load_state_dict(state)
    compare([0, 1, 2, 3])
    actor.public = copy.deepcopy(actor.public)
    reference_actor.public = copy.deepcopy(reference_actor.public)
    compare([0, 1, 2, 3])
    fast.prune(set())
    assert fast._pointer_table is None and not fast.entries
    fast.clear()
    assert fast.bytes == 0
    if device == "cuda":
        assert fast.copy_stats["triton_packs"] > 0


@pytest.mark.parametrize("enabled,min_batch", [(True, 1), (True, 4), (False, 1)])
def test_encode_rejects_other_stream_before_unchanged_or_growing_entry(device, monkeypatch, enabled, min_batch):
    if device != "cuda":
        pytest.skip("CUDA stream ownership")
    fast = TritonHistoryCache(actor_on(device), enabled=enabled, min_batch=min_batch)
    streams = [PublicStream(0), PublicStream(1)]
    keys = [(0, 0), (1, 1)]
    fast.encode(keys, streams)
    previous = [(e.length, e.capacity, e.memory.data_ptr()) for e in fast.entries.values()]
    calls = []
    original = fast._entry

    def observed(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(fast, "_entry", observed)
    owner = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(owner)
    for grows in (False, True):
        if grows:
            append(streams[0], 40)
        with torch.cuda.stream(side), pytest.raises(RuntimeError, match="owning CUDA stream"):
            fast.encode(keys, streams)
        assert not calls
        assert previous == [(e.length, e.capacity, e.memory.data_ptr()) for e in fast.entries.values()]
    owner.wait_stream(side)
    fast.clear()


def test_limits_are_validated():
    actor = actor_on("cpu")
    for kwargs in ({"min_batch": 0}, {"min_batch": 3, "max_batch": 2}, {"max_table_bytes": 0},
                   {"max_packed_bytes": 0}):
        with pytest.raises(ValueError):
            TritonHistoryCache(actor, **kwargs)
    # Defaults no longer cap batch, table or packed size (large rollouts stay on Triton).
    cache = TritonHistoryCache(actor)
    assert cache.max_batch is cache.max_table_bytes is cache.max_packed_bytes is None


def test_rows_per_launch_keeps_int32_offsets():
    # Small problems: one launch.
    assert rows_per_launch(4096, 64 * 2048, [(3 * 64 * 128, 64 * 128)]) == 4096
    # Large model, 4096 streams, capacity 4096: one row short of the int32 bound.
    span = 128 * 4096
    rows = rows_per_launch(4096, span, [(span, span)])
    assert rows == 4095 and (rows - 1) * span + span <= INT32_LIMIT < rows * span + span
    # A wide row stride in the new K/V view tightens the bound.
    rows = rows_per_launch(4096, 1024, [(1 << 20, 1 << 19)])
    assert (rows - 1) * (1 << 20) + (1 << 19) <= INT32_LIMIT < rows * (1 << 20) + (1 << 19)
    # Grid y is capped; a single row beyond int32 cannot launch (eager fallback).
    assert rows_per_launch(100000, 32, [(32, 31)]) == 65535
    assert rows_per_launch(8, INT32_LIMIT + 1, []) == 0


def test_fallback_reasons_are_counted_on_cpu():
    actor = actor_on("cpu")
    cache = TritonHistoryCache(actor, min_batch=4)
    stream = PublicStream(0)
    append(stream, 5)
    cache.encode([(0, 0)], [stream])
    assert cache.copy_stats["fallback_disabled"] >= 1   # CPU: never Triton
    assert cache.copy_stats["triton_packs"] == 0


@pytest.mark.parametrize("streams,forced_rows", [(520, None), (520, 97), (1100, None)])
def test_large_batch_encode_matches_eager_bits(device, streams, forced_rows):
    """>= 512 streams (the old max_batch was 128), with and without forced chunking."""
    if device != "cuda":
        pytest.skip("Triton launches are CUDA-only")
    actor = actor_on(device)
    reference_actor = copy.deepcopy(actor)
    fast = TritonHistoryCache(actor, chunk_size=64, min_batch=1)
    fast.max_rows_per_launch = forced_rows
    eager = BatchedHistoryCache(reference_actor, chunk_size=64)
    rng = np.random.default_rng(streams)
    histories = [PublicStream(i) for i in range(streams)]
    keys = [(i, i) for i in range(streams)]
    for round_index in range(3):
        for stream in histories:
            append(stream, int(rng.integers(0, 90)))
        meta, actual = fast.encode(keys, histories)
        expected_meta, expected = eager.encode(keys, histories)
        bits(actual, expected)
        bits(meta.lengths, expected_meta.lengths)
    for key in keys[:: max(1, streams // 64)]:
        a, b = fast.entries[key], eager.entries[key]
        for x, y in zip(a.keys + a.values + [a.memory], b.keys + b.values + [b.memory]):
            bits(x, y)
    assert fast.copy_stats["triton_packs"] > 0
    assert fast.copy_stats["fallback_above_max_batch"] == 0 and fast.copy_stats["eager_packs"] == 0
    if forced_rows:
        assert fast.copy_stats["chunked_packs"] > 0
        assert fast.copy_stats["triton_launches"] > fast.copy_stats["triton_packs"]


def test_explicit_cuda_backend_requires_triton(device, monkeypatch):
    import train.history_triton_cache as cache_module
    monkeypatch.setattr(cache_module, "triton", None)
    actor = actor_on(device)
    if device == "cuda":
        with pytest.raises(RuntimeError, match="Triton is required"):
            TritonHistoryCache(actor)
    else:
        assert TritonHistoryCache(actor).bytes == 0
    assert TritonHistoryCache(actor, enabled=False).bytes == 0


# -- cache-wide pointer registry ---------------------------------------------------

def entry_on(actor, capacity, length, row=0):
    shape = (4, capacity, 8)
    return Entry(PublicStream(row), 0, length, capacity,
                 [actor.bos.new_zeros(shape) for _ in range(2)],
                 [actor.bos.new_zeros(shape) for _ in range(2)],
                 actor.bos.new_zeros(capacity, 32))


def registry_call(cache, entries, counts=None):
    """``_metadata`` with the eligibility gate open: registry bookkeeping and
    metadata arrays without a kernel launch (usable on CPU)."""
    cache._supported_entries = lambda entries, counts: True
    try:
        return cache._metadata(entries, list(counts or [1] * len(entries)))
    finally:
        del cache._supported_entries


def assert_columns(cache, context, entries, counts=None):
    table = context.table.pointers.cpu().numpy()
    ranges = context.ranges.cpu().numpy()
    columns = [cache._columns[id(entry)] for entry in entries]
    assert ranges.shape == (3, len(entries)) and ranges.dtype == np.int64
    assert ranges[0].tolist() == columns
    assert ranges[1].tolist() == [entry.length for entry in entries]
    assert ranges[2].tolist() == list(counts or [1] * len(entries))
    assert table.shape == (5, len(cache._column_entries))
    for entry, column in zip(entries, columns):
        assert context.table.entries[column] is entry
        assert table[0, column] == entry.capacity
        for layer in range(2):
            assert table[1 + 2 * layer, column] == entry.keys[layer].data_ptr()
            assert table[2 + 2 * layer, column] == entry.values[layer].data_ptr()


def test_registry_rebuilds_only_on_create_growth_prune_and_clear():
    actor = actor_on("cpu")
    cache = TritonHistoryCache(actor, chunk_size=8)
    streams = [PublicStream(i) for i in range(6)]
    keys = [(i, i) for i in range(6)]
    for stream, count in zip(streams, [5, 9, 3, 12, 7, 2]):
        append(stream, count)
    cache.encode(keys, streams)                     # CPU: eager, nothing registered
    assert not cache._columns and cache.copy_stats["table_rebuilds"] == 0
    entries = [cache.entries[key] for key in keys]

    def stats():
        return cache.copy_stats["table_rebuilds"], cache.copy_stats["table_reuses"]

    context = registry_call(cache, entries[:4], [1, 2, 3, 4])
    assert stats() == (1, 0)
    assert_columns(cache, context, entries[:4], [1, 2, 3, 4])
    first = cache._pointer_table
    # Any ordering or subset of registered entries reuses the uploaded table.
    for subset in ([entries[2], entries[0]], [entries[3], entries[1]], entries[:4][::-1]):
        context = registry_call(cache, subset)
        assert_columns(cache, context, subset)
        assert context.table is first
    assert stats() == (1, 3)
    assert cache.copy_stats["pointer_uploads"] == 1 and cache.copy_stats["pointer_reuses"] == 3
    # A new entry is created: one rebuild, then reuse again.
    context = registry_call(cache, [entries[4], entries[0]])
    assert stats() == (2, 3) and cache._columns[id(entries[4])] == 4
    assert_columns(cache, context, [entries[4], entries[0]])
    registry_call(cache, [entries[0], entries[4]])
    assert stats() == (2, 4)
    # Growth replaces the Entry object: its column is freed at once and reused.
    grown_column = cache._columns[id(entries[1])]
    append(streams[1], 40)
    cache.encode([keys[1]], [streams[1]])
    assert cache.entries[keys[1]] is not entries[1] and cache._pointer_table is None
    assert id(entries[1]) not in cache._columns and cache._free == [grown_column]
    assert cache.copy_stats["table_replacements"] == 1
    grown = cache.entries[keys[1]]
    context = registry_call(cache, [grown, entries[0]])
    assert stats() == (3, 4) and cache._columns[id(grown)] == grown_column
    assert_columns(cache, context, [grown, entries[0]])
    assert context.table.pointers[0, grown_column] == grown.capacity == 64
    # An unchanged prefix neither replaces nor rebuilds.
    cache.encode([keys[0]], [streams[0]])
    registry_call(cache, [entries[0]])
    assert stats() == (3, 5)
    # A stream reset replaces the Entry as growth does.
    streams[3].reset(9)
    append(streams[3], 4)
    cache.encode([keys[3]], [streams[3]])
    assert id(entries[3]) not in cache._columns and cache._pointer_table is None
    assert cache.copy_stats["table_replacements"] == 2
    registry_call(cache, [entries[0]])
    assert stats() == (4, 5)
    # Prune retires the dropped entry's column; pruning nothing keeps the table.
    cache.prune({key for key in keys if key != keys[2]})
    assert id(entries[2]) not in cache._columns and cache._pointer_table is None
    assert cache.copy_stats["table_prunes"] == 1
    context = registry_call(cache, [entries[0], entries[4]])
    assert stats() == (5, 5)
    assert context.table.entries[2] is None and context.table.references[2] == ()
    assert not context.table.pointers[:, 2].any()
    cache.prune(set(keys))
    assert cache.copy_stats["table_prunes"] == 1 and cache._pointer_table is context.table
    registry_call(cache, [entries[4]])
    assert stats() == (5, 6)
    # clear() empties the registry and releases the table.
    cache.clear()
    assert not cache._columns and not cache._bases and cache._pointer_table is None
    assert cache.metadata_bytes == 0 and cache.bytes == 0
    fresh = entry_on(actor, 32, 3)
    context = registry_call(cache, [fresh])
    assert stats() == (6, 6) and cache._columns[id(fresh)] == 0
    assert_columns(cache, context, [fresh])


def test_registry_reregisters_rebound_tensor_and_keeps_old_storage():
    actor = actor_on("cpu")
    cache = TritonHistoryCache(actor)
    entries = [entry_on(actor, 32, 4, 0), entry_on(actor, 64, 9, 1)]
    registry_call(cache, entries)
    old_table = cache._pointer_table
    old_address = entries[1].values[1].data_ptr()
    entries[1].values[1].data = entries[1].values[1].data.clone()
    context = registry_call(cache, entries)
    assert context.table is not old_table and cache.copy_stats["table_rebuilds"] == 2
    assert_columns(cache, context, entries)
    column = cache._columns[id(entries[1])]
    assert old_table.references[column][3].data_ptr() == old_address
    # A capacity change on a registered Entry is caught the same way.
    entries[0].capacity = 16
    assert registry_call(cache, entries) is None
    assert cache.copy_stats["fallback_entry_tensors"] == 1 and id(entries[0]) not in cache._columns


def test_registry_fallbacks_are_evaluated_at_registration():
    actor = actor_on("cpu")
    cache = TritonHistoryCache(actor)
    good = entry_on(actor, 32, 2, 0)
    cases = []
    wrong_dtype = entry_on(actor, 32, 2, 1)
    wrong_dtype.keys[1] = wrong_dtype.keys[1].double()
    cases.append((wrong_dtype, "fallback_entry_tensors"))
    strided = entry_on(actor, 32, 2, 2)
    strided.values[0] = torch.zeros(4, 8, 32).transpose(1, 2)
    cases.append((strided, "fallback_entry_tensors"))
    wrong_shape = entry_on(actor, 32, 2, 3)
    wrong_shape.keys[0] = torch.zeros(4, 64, 8)
    cases.append((wrong_shape, "fallback_entry_tensors"))
    duplicate = entry_on(actor, 32, 2, 4)
    duplicate.values[1] = duplicate.keys[1]
    cases.append((duplicate, "fallback_entry_tensors"))
    shared = torch.zeros(2, 4, 32, 8)
    inner = entry_on(actor, 32, 2, 5)
    inner.keys[0], inner.values[0] = shared[0], shared[1]
    cases.append((inner, "fallback_aliased_storage"))
    registry_call(cache, [good])
    across = entry_on(actor, 32, 2, 6)
    across.keys[1] = good.keys[1].view(4, 32, 8)   # same storage as a registered Entry
    cases.append((across, "fallback_aliased_storage"))
    for bad, reason in cases:
        before = cache.copy_stats[reason]
        rebuilds = cache.copy_stats["table_rebuilds"]
        assert registry_call(cache, [good, bad]) is None
        assert cache.copy_stats[reason] == before + 1
        assert id(bad) not in cache._columns and cache.copy_stats["table_rebuilds"] == rebuilds
    # The registry is unchanged by the rejections; good entries keep their table.
    context = registry_call(cache, [good])
    assert context.table is cache._pointer_table and list(cache._columns.values()) == [0]
    assert len(cache._bases) == 4
    # Per-call layout checks still run first and leave the registry untouched.
    assert cache._metadata([good], [1]) is None     # CPU: never Triton
    assert cache.copy_stats["fallback_disabled"] == 1 and cache._pointer_table is context.table


@pytest.mark.parametrize("forced_rows", [None, 3])
def test_changing_subsets_reuse_registry_and_match_eager_bits(device, forced_rows):
    """Learner-like pending subsets: a different ordered subset every call."""
    if device != "cuda":
        pytest.skip("Triton launches are CUDA-only")
    actor = actor_on(device)
    reference_actor = copy.deepcopy(actor)
    fast = TritonHistoryCache(actor, chunk_size=16)
    fast.max_rows_per_launch = forced_rows
    eager = BatchedHistoryCache(reference_actor, chunk_size=16)
    rng = np.random.default_rng(7)
    histories = [PublicStream(i) for i in range(24)]
    keys = [(i, i) for i in range(24)]
    for step in range(40):
        picked = sorted(rng.choice(24, int(rng.integers(3, 18)), replace=False).tolist(),
                        key=lambda _: rng.random())
        for i in picked:
            append(histories[i], int(rng.integers(1, 6)))
        if step % 13 == 12:
            fast.prune({keys[i] for i in picked})
            eager.prune({keys[i] for i in picked})
        meta, actual = fast.encode([keys[i] for i in picked], [histories[i] for i in picked])
        expected_meta, expected = eager.encode([keys[i] for i in picked],
                                               [histories[i] for i in picked])
        bits(actual, expected)
        bits(meta.lengths, expected_meta.lengths)
        for i in picked:
            a, b = fast.entries[keys[i]], eager.entries[keys[i]]
            for x, y in zip(a.keys + a.values + [a.memory], b.keys + b.values + [b.memory]):
                bits(x, y)
    stats = fast.copy_stats
    assert stats["triton_packs"] > 0 and stats["eager_packs"] == 0
    assert stats["table_reuses"] > 0 and stats["table_replacements"] > 0
    assert stats["table_rebuilds"] + stats["table_reuses"] == stats["metadata_uploads"]
    assert len(fast._columns) <= len(fast.entries)
