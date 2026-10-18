"""Real cache and self-play CUDA gates; requires a complete GuanZero checkout.

These tests compare both arms using the SAME projection/SDPA configuration.
Only the paged rollout-attention implementation changes. They are NOT replaced
by the CPU recurrence tests. A random checkpoint is a plumbing gate, not the
final mature-checkpoint acceptance required before production adoption.
"""
import copy

import numpy as np
import pytest
import torch

pytest.importorskip("gd._gd_core", reason="requires the built GuanZero native engine")

from bench.paged_attention import strict_fp32
from train.history_model import HistoryPolicyConfig, PublicStream, fresh_player
from train.history_paged_attention import experimental_paged_attention
from train.history_paged_cache import KVPagePool, PagedHistoryCache, merged_encode
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer
from train.history_snapshot_batch import SnapshotHeads
from test_history_inference import append
from test_history_rollout import make_env


@pytest.fixture
def cuda_precision():
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA kernel execution, not CPU emulation")
    pytest.importorskip("triton")
    with strict_fp32():
        yield


def check_live(reference, actual, streams):
    assert actual.shape == reference.shape and not actual.requires_grad
    assert torch.isfinite(actual).all()
    for row, stream in enumerate(streams):
        end = stream.prefix + 1
        torch.testing.assert_close(actual[row, :end], reference[row, :end], rtol=1e-4, atol=1e-5)
        assert not torch.count_nonzero(actual[row, end:])


def test_default_and_explicit_backend_remain_cache_local():
    actor = fresh_player(HistoryPolicyConfig(width=64, layers=2, heads=4), 4)[0]
    existing = PagedHistoryCache(actor)
    assert existing.attention_backend == "torch"
    with experimental_paged_attention("triton"):
        explicit = PagedHistoryCache(actor, attention_backend="torch")
        assert explicit.attention_backend == "torch"
        with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
            PagedHistoryCache(actor)
        assert existing.attention_backend == "torch"
    assert PagedHistoryCache(actor).attention_backend == "torch"
    with pytest.raises(AttributeError):
        existing.attention_backend = "triton"


@pytest.mark.parametrize("chunk,page_span", [(17, False), (128, True)])
def test_cuda_ordinary_prefill_append_reorder_reset_prune_and_weights(cuda_precision, chunk, page_span):
    actor = fresh_player(HistoryPolicyConfig(width=64, layers=2, heads=4), 4)[0].cuda().train()
    caches = [PagedHistoryCache(actor, KVPagePool.for_actor(actor, pages=2), chunk_size=chunk,
                               page_span=page_span, attention_backend=b) for b in ("torch", "triton")]
    streams = [PublicStream(i) for i in range(4)]
    for stream, count in zip(streams, (0, 64, 133, 257)):
        append(stream, count)
    for step in range(8):
        order = (3, 0, 2, 1) if step % 2 else (0, 1, 2, 3)
        selected = [streams[i] for i in order]
        keys = [(i, streams[i].match_id) for i in order]
        expected = caches[0].encode(keys, selected)[1]
        actual = caches[1].encode(keys, selected)[1]
        check_live(expected, actual, selected)
        assert caches[0].encoded_tokens == caches[1].encoded_tokens
        assert caches[0].rebuilds == caches[1].rebuilds
        for cache in caches:
            assert not any(kv[:cache.pool.page_tokens].any() for kv in cache.pool.kv)
        for stream, count in zip(streams, (1, 3, 27, 65)):
            append(stream, count, offset=step)
        if step == 2:
            streams[2].reset(2)
            append(streams[2], 177)
        if step == 4:
            for cache in caches:
                cache.prune({(0, 0), (1, 1)})
        if step == 6:
            with torch.no_grad():
                actor.public.weight.add_(1e-3)
    assert caches[1].attention_kernel_calls > 0
    assert caches[0].attention_kernel_calls == 0
    snapshot = actual.clone()
    for cache in caches:
        cache.clear()
        assert cache.pool.used_pages == 0
        cache.pool.reset()
    # The previously returned output remains independent of released pool pages.
    assert torch.equal(actual, snapshot)
    expected = caches[0].encode(keys, selected)[1]
    actual = caches[1].encode(keys, selected)[1]
    check_live(expected, actual, selected)


def test_cuda_merged_snapshot_slot_gap_and_shared_pool_isolation(cuda_precision):
    config = HistoryPolicyConfig(width=64, layers=2, heads=4)
    actors = [fresh_player(config, seed)[0].cuda().train() for seed in (3, 7, 11)]
    heads = SnapshotHeads(encoder=True)
    slots = [heads.slot(identity, actor) for identity, actor in zip((5, 9, 11), actors)]
    heads.retain({5, 11})
    actors, slots = [actors[0], actors[2]], [slots[0], slots[2]]
    arms = []
    for backend in ("torch", "triton"):
        pool = KVPagePool.for_actor(actors[0], pages=2)
        arms.append([PagedHistoryCache(actor, pool, chunk_size=27, page_span=True,
                                        attention_backend=backend) for actor in actors])
    streams = [PublicStream(i) for i in range(6)]
    for stream, count in zip(streams, (0, 1, 64, 65, 133, 257)):
        append(stream, count)
    for step in range(4):
        outputs = []
        size = (max(s.prefix + 1 for s in streams) + 63) // 64 * 64
        for caches in arms:
            parts = [(caches[0], [(i, i) for i in (0, 2, 3, 5)],
                      [streams[i] for i in (0, 2, 3, 5)], slots[0]),
                     (caches[1], [(i, i) for i in (1, 4)],
                      [streams[i] for i in (1, 4)], slots[1])]
            with torch.no_grad():
                outputs.append(merged_encode(parts, heads, size))
        check_live(outputs[0], outputs[1], [streams[i] for i in (0, 2, 3, 5, 1, 4)])
        for stream in streams:
            append(stream, 17, offset=step)
    assert sum(c.attention_kernel_calls for c in arms[1]) > 0
    assert sum(c.attention_kernel_calls for c in arms[0]) == 0
    # Fail before a mixed arm can refresh or overwrite any cache.
    mixed = [(arms[0][0], [(0, 0)], [streams[0]], slots[0]),
             (arms[1][1], [(1, 1)], [streams[1]], slots[1])]
    with torch.no_grad(), pytest.raises(ValueError, match="one attention backend"):
        merged_encode(mixed, heads, 512)
    with pytest.raises(RuntimeError, match="no_grad"):
        merged_encode([(arms[1][0], [(0, 0)], [streams[0]], slots[0])], heads, 512)


@pytest.mark.parametrize("merged", [False, True])
def test_cuda_real_self_play_choices_rng_logp_and_full_history(cuda_precision, merged):
    config = HistoryPolicyConfig(width=128, layers=4, heads=8)
    actor = fresh_player(config, 4)[0].cuda()
    opponents = {identity: fresh_player(config, 10 + identity)[0].cuda() for identity in (1, 2)}
    collectors = []
    for backend in ("torch", "triton"):
        with experimental_paged_attention(backend):
            collector = HistoryCollector(
                make_env(4, 21), actor, MatchEventStore(), SequenceRolloutBuffer(),
                torch.Generator(device="cuda").manual_seed(5), device="cuda",
                record_choices=True, seat_policy=lambda env, match: [0, 1, 0, 2],
                resolve_policy=opponents.__getitem__, kv_cache=True, paged_cache=True,
                page_span=True, batch_snapshot_policies=merged, batch_snapshot_encoder=merged)
            collector.collect(160)
        assert all(c.attention_backend == backend for c in collector.caches.values())
        collectors.append(collector)
    original, candidate = collectors
    assert len(original.choice_log) == len(candidate.choice_log)
    for a, b in zip(original.choice_log, candidate.choice_log):
        np.testing.assert_array_equal(a, b)
    assert torch.equal(original.generator.get_state(), candidate.generator.get_state())
    expected, actual = original.buffer.compact(), candidate.buffer.compact()
    assert expected.keys() == actual.keys()
    for name in expected:
        if name in ("logp", "behaviour_logp"):
            np.testing.assert_allclose(expected[name], actual[name], rtol=0, atol=1e-4)
        else:
            np.testing.assert_array_equal(expected[name], actual[name])
    assert sum(c.attention_kernel_calls for c in candidate.caches.values()) > 0
    assert set(original.store.streams) == set(candidate.store.streams)
    for key in original.store.streams:
        for a, b in zip(original.store.streams[key].arrays(), candidate.store.streams[key].arrays()):
            np.testing.assert_array_equal(a, b)
    # Both arms' stored behavior probabilities must recompute from full history.
    rows = np.arange(min(len(candidate.buffer), 256))
    assert len(rows) > 0
    for collector in collectors:
        inputs, chosen = collector.buffer.decision_inputs(rows, collector.store, "cuda")
        with torch.no_grad():
            dense_memory = actor.encode_batch(inputs.streams)
            recomputed = actor.candidate_log_probs(inputs, encoded=dense_memory)
        assert torch.isfinite(recomputed).all()
        stored = collector.buffer.compact()["behaviour_logp"][rows]
        np.testing.assert_allclose(recomputed[chosen].cpu().numpy(), stored, rtol=0, atol=1e-4)
    # Check learner backward independently: the rollout-only backend scope must
    # neither intercept nor detach actor recomputation.
    baseline_actor, scoped_actor = copy.deepcopy(actor), copy.deepcopy(actor)
    inputs, _ = candidate.buffer.decision_inputs(rows[:8], candidate.store, "cuda")
    with experimental_paged_attention("torch"):
        baseline_actor.candidate_log_probs(inputs).sum().backward()
    with experimental_paged_attention("triton"):
        scoped_actor.candidate_log_probs(inputs).sum().backward()
    for (name, p), q in zip(baseline_actor.named_parameters(), scoped_actor.parameters()):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            assert torch.isfinite(p.grad).all() and torch.isfinite(q.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-5, msg=name)
