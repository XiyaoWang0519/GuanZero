"""Exact parity and bounded storage for host-side KV cache optimizations."""
import copy

import pytest
import torch

from train import history_inference as inference
from train.history_inference import BatchedHistoryCache
from train.history_model import HistoryPolicyConfig, PublicStream, fresh_player, sinusoidal
from test_history_model import valid_token


class ReferenceCache(BatchedHistoryCache):
    def _signature(self):
        return tuple((id(p), p._version, p.device, p.dtype) for p in self.actor.parameters())

    def _position_buffers(self, capacity):
        prototype = self.actor.bos
        return (sinusoidal(capacity, self.actor.config.width, prototype.device, prototype.dtype),
                torch.arange(capacity, device=prototype.device))

    def _pack_keys_values(self, entries, counts, layer_index, new_k, new_v, capacity):
        cfg = self.actor.config
        packed_k = new_k.new_zeros(len(entries), cfg.heads, capacity, cfg.width // cfg.heads)
        packed_v = torch.zeros_like(packed_k)
        for i, (entry, count) in enumerate(zip(entries, counts)):
            start, end = entry.length, entry.length + count
            entry.keys[layer_index][:, start:end].copy_(new_k[i, :, :count])
            entry.values[layer_index][:, start:end].copy_(new_v[i, :, :count])
            packed_k[i, :, :end].copy_(entry.keys[layer_index][:, :end])
            packed_v[i, :, :end].copy_(entry.values[layer_index][:, :end])
        return packed_k, packed_v


def append(stream, count):
    start = stream.prefix
    for i in range(start, start + count):
        stream.append_token(valid_token(i % 4, 7 + i % 90, 27 - i % 28), i // 17, i % 4)


def player(device="cpu"):
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=2, action_width=16,
                                               fusion_width=16, critic_width=16), 17)
    return actor.to(device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_position_reuse_matches_uncached_append_exactly(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA position-cache acceptance")
    actor = player(device)
    reference_actor = copy.deepcopy(actor)
    cached = BatchedHistoryCache(actor, chunk_size=8)
    reference = ReferenceCache(reference_actor, chunk_size=8)
    streams = [PublicStream(i) for i in range(3)]
    for stream, count in zip(streams, [0, 11, 29]):
        append(stream, count)

    def compare(order):
        keys = [(i, i) for i in order]
        selected = [streams[i] for i in order]
        actual_metadata, actual = cached.encode(keys, selected)
        expected_metadata, expected = reference.encode(keys, selected)
        assert torch.equal(actual, expected)
        assert torch.equal(actual_metadata.lengths, expected_metadata.lengths)
        for key in keys:
            a, b = cached.entries[key], reference.entries[key]
            assert a.length == b.length and a.capacity == b.capacity
            for x, y in zip(a.keys + a.values, b.keys + b.values):
                assert torch.equal(x[:, :a.length], y[:, :b.length])
            assert torch.equal(a.memory[:a.length], b.memory[:b.length])
        assert cached.encoded_tokens == reference.encoded_tokens
        assert cached.rebuilds == reference.rebuilds

    compare([0])
    compare([0, 1, 2])
    previous = cached._position_cache
    compare([2, 0])
    assert cached._position_cache is previous  # unchanged prefixes do no work
    for stream, count in zip(streams, [2, 19, 40]):
        append(stream, count)
    compare([2])  # singleton prefill uses both shorter and full-capacity views
    compare([2, 1, 0])  # ragged appends, reorder, and growth through 64/128
    assert cached._position_cache is not previous

    previous_entry = cached.entries[(2, 2)]
    streams[2].reset(2)
    append(streams[2], 7)
    compare([2, 0])
    assert cached.entries[(2, 2)] is not previous_entry
    previous = cached._position_cache
    with torch.no_grad():
        actor.public.weight.add_(0.05)
        reference_actor.public.weight.add_(0.05)
    compare([2, 0])
    assert cached._position_cache is not previous
    assert set(cached.entries) == {(2, 2), (0, 0)}

    cached.clear()
    reference.clear()
    assert cached.bytes == 0 and cached._position_cache is None
    compare([0, 2, 1])


def test_position_buffers_keep_one_exact_shape_and_account_for_bytes(monkeypatch):
    actor = player()
    cache = BatchedHistoryCache(actor)
    calls = []

    def counted(length, width, device, dtype):
        calls.append((length, width, device, dtype))
        return sinusoidal(length, width, device, dtype)

    monkeypatch.setattr(inference, "sinusoidal", counted)
    last = None
    for capacity in [32, 32, 64, 64, 32]:
        table, indices = cache._position_buffers(capacity)
        if last is not None and last[0] == capacity:
            assert table is last[1] and indices is last[2]
        assert torch.equal(table, sinusoidal(capacity, actor.config.width, "cpu", torch.float32))
        for length in [1, capacity // 2, capacity]:
            assert torch.equal(indices[:length], torch.arange(length))
        assert cache.bytes == capacity * (actor.config.width * 4 + 8)
        last = capacity, table, indices
    assert [call[0] for call in calls] == [32, 64, 32]

    stream = PublicStream(0)
    append(stream, 4)
    cache.encode([(0, 0)], [stream])
    entry = cache.entries[(0, 0)]
    entry_bytes = sum(t.numel() * t.element_size()
                      for t in (*entry.keys, *entry.values, entry.memory))
    assert cache.bytes == entry_bytes + 32 * (actor.config.width * 4 + 8)
    cache.prune(set())
    assert cache.bytes == 32 * (actor.config.width * 4 + 8)
    cache.clear()
    assert cache.bytes == 0


def test_parameter_dtype_change_replaces_position_buffers():
    actor = player()
    cache = BatchedHistoryCache(actor)
    reference = ReferenceCache(actor)
    stream = PublicStream(0)
    append(stream, 5)
    cache.encode([(0, 0)], [stream])
    previous = cache._position_cache
    actor.double()  # invalidation test only; production precision is unchanged
    _, actual = cache.encode([(0, 0)], [stream])
    _, expected = reference.encode([(0, 0)], [stream])
    assert torch.equal(actual, expected)
    assert cache._position_cache is not previous
    assert cache._position_cache[1].dtype == torch.float64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA migration acceptance")
def test_parameter_device_change_replaces_position_buffers():
    actor = player()
    cache = BatchedHistoryCache(actor)
    stream = PublicStream(0)
    append(stream, 5)
    cache.encode([(0, 0)], [stream])
    previous = cache._position_cache
    actor.cuda()
    _, actual = cache.encode([(0, 0)], [stream])
    _, expected = ReferenceCache(actor).encode([(0, 0)], [stream])
    assert torch.equal(actual, expected)
    assert cache._position_cache is not previous
    assert cache._position_cache[1].device.type == "cuda"


@pytest.mark.parametrize("lengths,counts,capacity", [
    ([9], [10], 32), ([80], [8], 32), ([20, 22], [21, 23], 32),
    ([2, 80], [3, 8], 32), ([2, 80], [3, 70], 128),
])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_kv_packing_preserves_bits_shape_and_stride(lengths, counts, capacity, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA KV packing acceptance")
    actor = player(device)
    cached, reference = BatchedHistoryCache(actor), ReferenceCache(actor)
    streams = [PublicStream(i) for i in range(len(lengths))]
    for stream, length in zip(streams, lengths):
        append(stream, length)
    actual_entries = [cached._entry((i, i), stream) for i, stream in enumerate(streams)]
    expected_entries = [reference._entry((i, i), stream) for i, stream in enumerate(streams)]
    shape = (len(lengths), actor.config.heads, max(counts),
             actor.config.width // actor.config.heads)
    generator = torch.Generator(device=device).manual_seed(42)
    keys = torch.randn(shape, device=device, generator=generator)
    values = torch.randn(shape, device=device, generator=generator)
    actual = cached._pack_keys_values(actual_entries, counts, 0, keys, values, capacity)
    expected = reference._pack_keys_values(expected_entries, counts, 0, keys, values, capacity)
    for a, b in zip(actual, expected):
        assert a.shape == b.shape and a.stride() == b.stride()
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))


@pytest.mark.parametrize("change", ["load_state_dict", "parameter", "submodule",
                                    "parameter_sharing", "sharing"])
def test_live_signature_detects_replacement_and_sharing(change):
    actor = player()
    cache = BatchedHistoryCache(actor)
    stream = PublicStream(0)
    append(stream, 5)
    cache.encode([(0, 0)], [stream])
    old_entry, old_signature = cache.entries[(0, 0)], cache.signature
    if change == "load_state_dict":
        actor.load_state_dict(copy.deepcopy(actor.state_dict()))
    elif change == "parameter":
        actor.public.weight = torch.nn.Parameter(actor.public.weight.detach().clone())
    elif change == "submodule":
        actor.public = copy.deepcopy(actor.public)
    elif change == "parameter_sharing":
        actor.out_proj.weight = actor.q_proj.weight
        actor.out_proj.bias = actor.q_proj.bias
    else:
        actor.out_proj = actor.q_proj
        actor.private_alias = actor.private
        actor.register_parameter("optional_parameter", None)
        actor.add_module("optional_module", None)
    expected_signature = tuple((id(p), p._version, p.device, p.dtype) for p in actor.parameters())
    assert cache._signature() == expected_signature
    assert cache._signature() != old_signature
    _, actual = cache.encode([(0, 0)], [stream])
    _, expected = ReferenceCache(actor).encode([(0, 0)], [stream])
    assert torch.equal(actual, expected)
    assert cache.entries[(0, 0)] is not old_entry


def test_packed_raw_upload_keeps_cached_encoding_exact(monkeypatch):
    from train.history_transfers import upload_arrays

    actor = player()
    streams = [PublicStream(0), PublicStream(1)]
    for stream, count in zip(streams, [19, 40]):
        append(stream, count)
    keys = [(0, 0), (1, 1)]
    _, expected = ReferenceCache(actor).encode(keys, streams)
    monkeypatch.setattr(inference, "upload_arrays",
                        lambda arrays, device: upload_arrays(arrays, device, packed=True))
    _, actual = BatchedHistoryCache(actor).encode(keys, streams)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("lengths", [[0], [2, 7], [2, 70]])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_encoded_memory_matches_original_packing_and_owns_snapshot(lengths, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA encoded-memory packing acceptance")
    actor = player(device)
    cache = BatchedHistoryCache(actor, chunk_size=8)
    streams = [PublicStream(i) for i in range(len(lengths))]
    keys = [(i, i) for i in range(len(lengths))]
    for stream, count in zip(streams, lengths):
        append(stream, count)

    def encode_and_compare():
        _, output = cache.encode(keys, streams)
        shape = (len(streams), inference.bucket(max(s.prefix + 1 for s in streams)),
                 actor.config.width)
        expected = actor.bos.new_zeros(shape)
        for i, key in enumerate(keys):
            entry = cache.entries[key]
            expected[i, :entry.length].copy_(entry.memory[:entry.length])
            assert output.untyped_storage().data_ptr() != entry.memory.untyped_storage().data_ptr()
        assert output.shape == expected.shape and output.stride() == expected.stride()
        assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
        return output

    original = encode_and_compare()
    saved = original.clone()
    for stream in streams:
        append(stream, 1)
    encode_and_compare()  # an append must not overwrite a prior output's zero padding
    assert torch.equal(original, saved)
    original.fill_(42)
    before_growth = encode_and_compare()  # output writes must not corrupt cached history
    saved = before_growth.clone()
    for stream in streams:
        append(stream, 65)
    encode_and_compare()
    assert torch.equal(before_growth, saved)
    streams[0].reset(0)
    encode_and_compare()
