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
        assert old_table.references[0].data_ptr() == old_address
        assert entries[0].keys[0].data_ptr() != old_address
        actual = fast._pack_keys_values(entries, counts, 0, new_k, new_v, capacity)
        expected = eager._pack_keys_values(originals, counts, 0, new_k, new_v, capacity)
        for a, b in zip(actual, expected):
            bits(a, b)
        assert fast._pointer_table is not old_table
        assert fast.copy_stats["pointer_uploads"] == 2
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
