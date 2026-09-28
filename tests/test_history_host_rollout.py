"""Bitwise acceptance for host-side rollout indexing and decision transfers.

These use frozen inference only. The reference disables the new host metadata
and downloads each tensor separately, preserving the original model operations
and RNG consumption. Cache-vs-dense tolerances are tested separately in
``test_history_inference.py``; each comparison here uses the same cache backend.
"""
from dataclasses import asdict, fields, replace

import numpy as np
import pytest
import torch

from train import history_rollout as rollout
from train.history_model import DecisionInputs, HistoryActor, HistoryPolicyConfig, fresh_player
from test_history_model import Rollout
from test_history_rollout import make_env


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA bitwise rollout acceptance")
    return request.param


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    # Strict CUDA byte comparisons require deterministic reductions and FP32.
    # Keep native pytest independent of the external GPU acceptance runner.
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


@pytest.fixture(scope="module")
def snapshot():
    return Rollout(num_envs=4, steps=12, seed=19, capture=(0, 4, 11)).last


def assert_tensor_bits(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8))


def assert_array_bits(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("response_mode", ["none", "auxiliary", "explicit"])
@pytest.mark.parametrize("batched_attention", [False, True])
def test_host_indices_preserve_logits_sampling_and_rng(snapshot, device, response_mode,
                                                      batched_attention):
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=1, action_width=16,
                                               fusion_width=16, response_mode=response_mode), 7)
    actor.to(device)
    actor.batched_private_attention = batched_attention
    original = snapshot.inputs(device)
    fast = replace(original, candidate_rows=original.rows, one_decision_per_stream=True)
    assert torch.equal(original.match_index, torch.arange(original.decisions, device=device))
    assert fast.rows is fast.candidate_rows
    assert_tensor_bits(fast.rows, original.rows)
    with torch.no_grad():
        encoded = actor.encode_batch(original.streams)
        before = actor.decision_states(encoded, original)
        after = actor.decision_states(encoded, fast)
        assert_tensor_bits(after, before)
        old_outputs = actor.candidate_outputs(before, original.cand, original.offsets, predict=True)
        new_outputs = actor.candidate_outputs(after, fast.cand, fast.offsets, predict=True,
                                              rows=fast.rows)
        for actual, expected in zip(new_outputs, old_outputs):
            if expected is None:
                assert actual is None
            else:
                assert_tensor_bits(actual, expected)
        assert_tensor_bits(actor.candidate_log_probs(fast, encoded),
                           actor.candidate_log_probs(original, encoded))
        for temperature, epsilon in [(1.0, 0.0), (0.7, 0.0), (1.0, 0.2),
                                     (1.7, 0.2), (1.0, 1.0)]:
            generators = [torch.Generator(device=device).manual_seed(29) for _ in range(2)]
            samples = [actor.explore(inputs, generator, encoded=encoded,
                                     max_candidates=int(snapshot.counts.max()),
                                     temperature=temperature, epsilon=epsilon)
                       for inputs, generator in zip((original, fast), generators)]
            for field in fields(samples[0]):
                assert_tensor_bits(getattr(samples[1], field.name),
                                   getattr(samples[0], field.name))
            assert_tensor_bits(generators[1].get_state(), generators[0].get_state())


def legacy_inputs(*args, **kwargs):
    kwargs.pop("candidate_rows", None)
    kwargs.pop("one_decision_per_stream", None)
    return DecisionInputs(*args, **kwargs)


@pytest.mark.parametrize("kv_cache", [False, True])
@pytest.mark.parametrize("temperature,epsilon", [(1.0, 0.0), (0.8, 0.2), (1.6, 0.0)])
@pytest.mark.parametrize("batched_attention", [False, True])
def test_population_rollout_is_bitwise_unchanged(monkeypatch, device, kv_cache,
                                                temperature, epsilon, batched_attention):
    config = HistoryPolicyConfig(width=32, layers=1, action_width=16, fusion_width=16)
    policies = {identity: fresh_player(config, 7 + identity)[0].to(device)
                for identity in (0, 1, 2)}
    for actor in policies.values():
        actor.batched_private_attention = batched_attention

    def collect():
        collector = rollout.HistoryCollector(
            make_env(4, 31), policies[0], rollout.MatchEventStore(),
            rollout.SequenceRolloutBuffer(), torch.Generator(device=device).manual_seed(13),
            device=device, seat_policy=lambda env, match: [0, 1, 0, 2],
            resolve_policy=policies.__getitem__, record_choices=True, kv_cache=kv_cache,
            temperature=temperature, epsilon=epsilon)
        stats = collector.collect(100)
        assert stats.rounds > 0
        assert set(collector.policy_decisions) == {0, 1, 2}
        return collector, stats

    optimized, optimized_stats = collect()
    with monkeypatch.context() as reference:
        reference.setattr(rollout, "DecisionInputs", legacy_inputs)
        reference.setattr(rollout, "_download_tensors",
                          lambda values, **kwargs: tuple(t.detach().cpu().numpy() for t in values))
        original, original_stats = collect()
    assert asdict(optimized_stats) == asdict(original_stats)
    assert optimized.policy_decisions == original.policy_decisions
    assert_tensor_bits(optimized.generator.get_state(), original.generator.get_state())
    assert len(optimized.choice_log) == len(original.choice_log)
    for actual, expected in zip(optimized.choice_log, original.choice_log):
        assert_array_bits(actual, expected)
    current, before = optimized.buffer.compact(), original.buffer.compact()
    assert current.keys() == before.keys()
    for name in current:
        assert_array_bits(current[name], before[name])
    assert optimized.store.streams.keys() == original.store.streams.keys()
    for key, stream in optimized.store.streams.items():
        for actual, expected in zip(stream.arrays(), original.store.streams[key].arrays()):
            assert_array_bits(actual, expected)


@pytest.mark.parametrize("packed", [False, True])
def test_download_preserves_mixed_dtype_shape_and_every_bit(device, packed):
    # Include signed zero, the smallest subnormal, infinities and a NaN payload:
    # numeric equality alone cannot prove these bytes survived the transfer.
    fp32 = np.array([0, 0x80000000, 1, 0x7F800000, 0xFF800000, 0x7FC00123],
                    dtype=np.uint32).view(np.float32)
    fp64 = np.array([0, 0x8000000000000000, 1, 0x7FF0000000000000,
                     0xFFF0000000000000, 0x7FF8000000000123], dtype=np.uint64).view(np.float64)
    strided = torch.from_numpy(fp32).to(device).reshape(2, 3).t().requires_grad_()
    assert not strided.is_contiguous()
    tensors = (
        # A bool at the front also tests byte offsets that are not dtype-aligned.
        torch.tensor([True, False, True], device=device),
        torch.tensor([-(2**63), 2**63 - 1, 2**53 + 17], dtype=torch.int64, device=device),
        strided,
        torch.from_numpy(fp64).to(device),
        torch.tensor([-0.17, 0.33, 1.123456789], dtype=torch.float64, device=device).sum(),
        torch.tensor(17, dtype=torch.int64, device=device),
        torch.empty((0, 3), dtype=torch.float32, device=device),
    )
    expected = tuple(t.detach().cpu().numpy() for t in tensors)
    actual = rollout._download_tensors(tensors, packed=packed)
    assert len(actual) == len(expected)
    for result, reference in zip(actual, expected):
        assert_array_bits(result, reference)


@pytest.mark.parametrize("width", [32, 64, 128])
@pytest.mark.parametrize("batch,length", [(1, 1), (4, 127), (32, 513), (16, 2403)])
@pytest.mark.parametrize("seed", [0, 7, 29])
def test_independent_attention_is_bitwise_across_batch_shapes(device, width, batch, length, seed):
    actor, _ = fresh_player(HistoryPolicyConfig(width=width, layers=1, heads=4), seed)
    actor.to(device)
    generator = torch.Generator(device=device).manual_seed(seed + 100)
    with torch.no_grad():
        query = torch.randn(batch, width, generator=generator, device=device)
        encoded = torch.randn(batch, length, width, generator=generator, device=device)
        keys, values = actor.kv_proj(encoded).chunk(2, dim=-1)
        prefix = torch.randint(length, (batch,), generator=generator, device=device)
        allowed = torch.arange(length, device=device)[None] <= prefix[:, None]
        reference = torch.cat([actor._attend(query[b:b + 1], keys[b:b + 1],
                                             values[b:b + 1], allowed[b:b + 1])
                               for b in range(batch)])
        actual = actor._attend_independent(query, keys, values, allowed)
    assert_tensor_bits(actual, reference)


@pytest.mark.parametrize("width", [32, 64, 128])
@pytest.mark.parametrize("batch", [1, 4, 16, 32])
@pytest.mark.parametrize("seed", [0, 7, 29])
def test_batched_one_row_projection_matches_linear_bits(device, width, batch, seed):
    torch.manual_seed(seed)
    layer = torch.nn.Linear(width, width).to(device)
    generator = torch.Generator(device=device).manual_seed(seed + 100)
    with torch.no_grad():
        for layout in ("normal", "strided", "zero", "mixed_scale"):
            value = torch.randn(batch, width * 2 if layout == "strided" else width,
                                generator=generator, device=device)
            if layout == "strided":
                value = value[:, ::2]
            elif layout == "zero":
                value.zero_()
            elif layout == "mixed_scale":
                exponent = torch.randint(-12, 13, value.shape, generator=generator, device=device)
                value *= torch.pow(2.0, exponent)
            reference = torch.cat([layer(value[b:b + 1]) for b in range(batch)])
            actual = HistoryActor._project_independent(layer, value)
            assert_tensor_bits(actual, reference)
