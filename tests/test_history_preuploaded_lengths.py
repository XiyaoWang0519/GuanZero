"""Cache metadata can reuse trusted rollout uploads without another H2D copy."""
from dataclasses import asdict

import pytest
import torch

from train import history_inference as inference
from train import history_rollout as rollout
from train.history_inference import BatchedHistoryCache
from train.history_model import HistoryPolicyConfig, PublicStream, fresh_player
from test_history_host_cache import append
from test_history_rollout import make_env


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA lengths-upload acceptance")
    return request.param


@pytest.fixture(autouse=True)
def single_thread():
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1], warn_only=previous[2])
        torch.backends.cuda.matmul.allow_tf32 = previous[3]
        torch.backends.cudnn.allow_tf32 = previous[4]


def player(device="cpu", seed=7):
    actor, _ = fresh_player(HistoryPolicyConfig(width=16, layers=1, action_width=16,
                                               fusion_width=16, critic_width=16), seed)
    return actor.to(device)


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8))


def test_reused_lengths_preserve_reorder_growth_reset_and_invalidation(device, monkeypatch):
    actor = player(device)
    reused, original = BatchedHistoryCache(actor, 8), BatchedHistoryCache(actor, 8)
    streams = [PublicStream(i) for i in range(3)]
    for stream, count in zip(streams, [0, 11, 29]):
        append(stream, count)

    def compare(order):
        selected = [streams[i] for i in order]
        keys = [(i, i) for i in order]
        host_lengths = tuple(stream.prefix for stream in selected)
        uploaded = torch.tensor(host_lengths, dtype=torch.long, device=device)
        calls = []
        tensor = torch.tensor

        def record_tensor(*args, **kwargs):
            calls.append((args, kwargs))
            return tensor(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(inference.torch, "tensor", record_tensor)
            actual_metadata, actual = reused.encode(
                keys, selected, preuploaded_lengths=uploaded, host_lengths=host_lengths)
            assert not calls
            expected_metadata, expected = original.encode(keys, selected)
            assert len(calls) == 1
        assert actual_metadata.lengths is uploaded
        assert_bits(actual_metadata.lengths, expected_metadata.lengths)
        assert_bits(actual, expected)
        assert reused.encoded_tokens == original.encoded_tokens
        assert reused.rebuilds == original.rebuilds

    compare([2, 0, 1])
    compare([1, 2])  # No new public events still needs no metadata upload.
    append(streams[0], 65)
    compare([0, 2, 1])
    streams[2].reset(2)
    append(streams[2], 4)
    compare([2, 0])
    with torch.no_grad():
        actor.bos.add_(0.125)
    compare([0, 1, 2])


@pytest.mark.parametrize("case", ["missing_host", "stale", "reordered", "shape", "dtype", "device"])
def test_reused_lengths_reject_invalid_host_proof_or_tensor_metadata(case):
    cache = BatchedHistoryCache(player())
    streams = [PublicStream(0), PublicStream(1)]
    append(streams[0], 2)
    append(streams[1], 5)
    host_lengths = (2, 5)
    uploaded = torch.tensor(host_lengths)
    if case == "missing_host":
        host_lengths = None
    elif case == "stale":
        host_lengths = (2, 4)
    elif case == "reordered":
        host_lengths = (5, 2)
    elif case == "shape":
        uploaded = uploaded[:, None]
    elif case == "dtype":
        uploaded = uploaded.int()
    elif case == "device":
        uploaded = torch.empty(2, dtype=torch.long, device="meta")
    with pytest.raises(ValueError, match="uploaded"):
        cache.encode([(0, 0), (1, 1)], streams, preuploaded_lengths=uploaded,
                     host_lengths=host_lengths)
    assert not cache.entries  # Reject before cache allocation or computation.


def test_host_lengths_require_the_uploaded_tensor():
    cache = BatchedHistoryCache(player())
    with pytest.raises(ValueError, match="require an uploaded"):
        cache.encode([(0, 0)], [PublicStream(0)], host_lengths=(0,))


@pytest.mark.parametrize("generic_layout", [False, True])
def test_collector_lengths_toggle_preserves_population_replay(device, generic_layout, monkeypatch):
    policies = {i: player(device, 17 + i) for i in (0, 1, 2)}
    original_encode = BatchedHistoryCache.encode
    original_group = rollout._PolicyBatch
    calls = []

    def record_encode(cache, keys, streams, **kwargs):
        calls.append(bool(kwargs))
        if kwargs:
            assert kwargs["host_lengths"] == tuple(stream.prefix for stream in streams)
        return original_encode(cache, keys, streams, **kwargs)

    def group_without_layout_hint(*args, **kwargs):
        group = original_group(*args, **kwargs)
        group.one_decision_per_stream = False
        return group

    monkeypatch.setattr(BatchedHistoryCache, "encode", record_encode)
    if generic_layout:
        monkeypatch.setattr(rollout, "_PolicyBatch", group_without_layout_hint)
    results = []
    for enabled in (False, True):
        calls.clear()
        options = {} if enabled else {"reuse_cache_lengths": False}
        collector = rollout.HistoryCollector(
            make_env(4, 23), policies[0], rollout.MatchEventStore(),
            rollout.SequenceRolloutBuffer(), torch.Generator(device=device).manual_seed(31),
            device=device, seat_policy=lambda env, match: [0, 1, 0, 2],
            resolve_policy=policies.__getitem__, record_choices=True, kv_cache=True,
            temperature=0.8, epsilon=0.2, **options)
        stats = collector.collect(24)
        assert calls and all(value == (enabled and not generic_layout) for value in calls)
        assert stats.policy_batches > stats.steps
        results.append((collector, asdict(stats)))
    (original, original_stats), (reused, reused_stats) = results
    assert original_stats == reused_stats
    assert_bits(reused.generator.get_state(), original.generator.get_state())
    for before, after in zip(original.choice_log, reused.choice_log):
        assert before.tobytes() == after.tobytes()
    for name, before in original.buffer.compact().items():
        after = reused.buffer.compact()[name]
        assert before.dtype == after.dtype and before.shape == after.shape
        assert before.tobytes() == after.tobytes()
