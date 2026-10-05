"""Cache/SDPA equivalence, privacy, invalidation and real population rollout."""
import copy

import numpy as np
import pytest
import torch

from train.history_inference import BatchedHistoryCache, ForkedHistoryCache
from train.history_model import HistoryPolicyConfig, PublicStream, StreamBatch, fresh_player
from train.history_ppo import HistoryPPOConfig, HistoryTrainer
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer
from test_history_model import valid_token
from test_history_rollout import make_env


def append(stream, count, offset=0):
    for i in range(count):
        stream.append_token(valid_token((i+offset) % 4, 7+(i+offset) % 90, 27-i % 28),
                            (i+offset) // 17, (i+offset) % 4)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_sdpa_valid_prefix_values_and_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA backend acceptance")
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=2), 13)
    actor.to(device).train()
    optimized = copy.deepcopy(actor)
    optimized.causal_sdpa = True
    streams = [PublicStream(i) for i in range(3)]
    for s, n in zip(streams, [0, 11, 71]):
        append(s, n)
    batch = StreamBatch.from_streams(streams, device)
    first, second = actor.encode_batch(batch), optimized.encode_batch(batch)
    losses = []
    for output in (first, second):
        # Prefix query losses ignore trailing padding, just like real decisions.
        losses.append(sum(output[i, :s.prefix+1, :7].square().sum() for i, s in enumerate(streams)))
    for i, s in enumerate(streams):
        torch.testing.assert_close(first[i, :s.prefix+1], second[i, :s.prefix+1], rtol=3e-5, atol=3e-6)
    for loss in losses:
        loss.backward()
    for (name, p), other in zip(actor.named_parameters(), optimized.parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, other.grad, rtol=3e-4, atol=3e-5, msg=name)
    # Future/public padding cannot change a shorter valid prefix.
    before = optimized.encode_batch(batch).detach()
    batch.tokens[1, 11:] = 1
    after = optimized.encode_batch(batch).detach()
    torch.testing.assert_close(before[1, :12], after[1, :12], rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_batched_ragged_cache_prefill_append_reorder_and_weight_rebuild(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA cache acceptance")
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=2), 7)
    actor.to(device).train()
    cache = BatchedHistoryCache(actor, chunk_size=16)
    streams = [PublicStream(i) for i in range(3)]
    keys = [(i, i) for i in range(3)]
    for s, n in zip(streams, [0, 13, 69]):
        append(s, n)

    def compare(order):
        selected = [streams[i] for i in order]
        _, cached = cache.encode([keys[i] for i in order], selected)
        with torch.no_grad():
            direct = actor.encode_batch(StreamBatch.from_streams(selected, device))
        for row, stream in enumerate(selected):
            torch.testing.assert_close(cached[row, :stream.prefix+1], direct[row, :stream.prefix+1],
                                       rtol=3e-5, atol=3e-6)
        assert not cached.requires_grad

    compare([0, 1, 2])
    assert cache.encoded_tokens == 85  # one BOS per match, every public event once
    count = cache.encoded_tokens
    compare([2, 0])
    assert cache.encoded_tokens == count
    for s, n in zip(streams, [1, 3, 67]):
        append(s, n, s.prefix)
    compare([2, 1, 0])
    assert cache.encoded_tokens == count + 71
    with torch.no_grad():
        actor.public.weight.add_(0.05)
    compare([0, 2])
    assert set(cache.entries) == {keys[0], keys[2]}
    cache.prune({keys[2]})
    assert set(cache.entries) == {keys[2]}
    # Reset to the same ID and a longer prefix must not reuse the old encoding.
    streams[2].reset(2)
    append(streams[2], 151, 50)
    compare([2])
    assert cache.bytes > 0
    cache.clear()
    assert cache.bytes == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_population_rollout_choices_probabilities_and_private_isolation(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA sampled-action and population equivalence")
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=2), 4)
    actor.to(device)
    opponents = {i: fresh_player(actor.config, 10+i)[0] for i in (1, 2)}
    for opponent in opponents.values():
        opponent.to(device)
    collectors = []
    for cached in (False, True):
        collector = HistoryCollector(make_env(4, 21), actor, MatchEventStore(), SequenceRolloutBuffer(),
                                     torch.Generator(device=device).manual_seed(5), device=device,
                                     record_choices=True,
                                     seat_policy=lambda env, match: [0, 1, 0, 2],
                                     resolve_policy=opponents.__getitem__, kv_cache=cached)
        collector.collect(160)
        collectors.append(collector)
    dense, cached = collectors
    for a, b in zip(dense.choice_log, cached.choice_log):
        np.testing.assert_array_equal(a, b)
    a, b = dense.buffer.compact(), cached.buffer.compact()
    for key in a:
        if key in ("logp", "behaviour_logp"):
            np.testing.assert_allclose(a[key], b[key], rtol=0, atol=2e-5)
        else:
            np.testing.assert_array_equal(a[key], b[key])
    assert set(cached.caches) == {0, 1, 2}
    assert all(e.stream is cached.store.streams[key] for c in cached.caches.values()
               for key, e in c.entries.items())
    # Private queries may change, but the cache is exclusively raw public history.
    rows = np.arange(min(len(cached.buffer), 30))
    inputs, _ = cached.buffer.decision_inputs(rows, cached.store, device)
    public = actor.encode_batch(inputs.streams).detach()
    with torch.no_grad():
        original = actor.candidate_log_probs(inputs, public)
        inputs.obs = 1-inputs.obs
        changed = actor.candidate_log_probs(inputs, public)
    assert not torch.equal(original, changed)
    assert not public.requires_grad
    # A changed match assignment evicts the preceding match's cache entries.
    cached.assignment(0, 1)
    cached._prune_caches()
    assert all((0, 0) not in c.entries for c in cached.caches.values())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_trainer_update_resume_and_probability_recomputation(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA PPO integration acceptance")
    cfg = HistoryPPOConfig(width=32, layers=1, num_envs=2, steps_per_update=90,
                           epochs=1, minibatch_matches=1, snapshot_updates=1, seed=3,
                           causal_sdpa=True, rollout_kv_cache=True, profile_collection=True)
    trainer = HistoryTrainer(cfg, tmp_path / "start", device=device)
    line = trainer.update()
    assert line["encoder_grad_norm"] > 0
    assert line["collection_phase_seconds"] and line["collection_profile_synchronized"]
    assert not trainer.collector.caches[0].entries  # released before learning
    trainer.collect()
    rows = np.flatnonzero(trainer.buffer.compact()["version"] == 1)
    with torch.no_grad():
        logp, _ = trainer.recompute_log_probs(rows)
    np.testing.assert_allclose(logp.cpu().numpy(), trainer.buffer.compact()["logp"][rows],
                               rtol=0, atol=3e-5)
    checkpoint = trainer.save()
    restored = HistoryTrainer(cfg, tmp_path / "resume", device=device, resume=checkpoint)
    assert restored.collector.caches == {}
    assert restored.actor.causal_sdpa and restored.config.rollout_kv_cache
    assert torch.equal(restored.generator.get_state(), trainer.generator.get_state())
    restored.collect()
    assert 1 in restored.collector.caches
    assert restored.learn()["encoder_grad_norm"] > 0


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_forked_branch_cache_matches_full_encode_through_growth_and_chunks(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS acceptance")
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=2), 11)
    actor.to(device).eval()
    root_cache = BatchedHistoryCache(actor, chunk_size=16)
    root = PublicStream(0)
    append(root, 23)
    root_cache.prefill([(0, 0)], [root])
    branches = [copy.deepcopy(root) for _ in range(5)]
    forked = ForkedHistoryCache(root_cache, (0, 0), branches)
    assert forked.capacity == 32
    # Ragged steps: some rows idle, one grows past the capacity in chunked appends.
    for step, extra in enumerate([(1, 0, 2, 1, 0), (0, 3, 1, 0, 1), (40, 1, 0, 2, 0), (2, 0, 5, 1, 3)]):
        for stream, n in zip(branches, extra):
            append(stream, n, offset=stream.prefix)
        rows = [i for i, n in enumerate(extra) if n] + [4]
        metadata, memory = forked.encode(rows)
        expected = actor.encode_batch(StreamBatch.from_streams([branches[i] for i in rows], device))
        assert metadata.lengths.tolist() == [branches[i].prefix for i in rows]
        assert memory.shape[1] >= max(branches[i].prefix + 1 for i in rows)
        for j, i in enumerate(rows):
            n = branches[i].prefix + 1
            torch.testing.assert_close(memory[j, :n], expected[j, :n], rtol=3e-5, atol=3e-5)
            assert not memory[j, n:].any()
    assert forked.capacity == 128
    # The root entry is untouched by the branches.
    assert root_cache.entries[(0, 0)].length == root.prefix + 1


def test_cache_rejects_window_control():
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=1, window=16), 1)
    with pytest.raises(ValueError, match="full history"):
        BatchedHistoryCache(actor)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_long_match_cache_keeps_all_2403_tokens(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA long-prefix acceptance")
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=1), 42)
    actor.to(device)
    stream = PublicStream(6)
    append(stream, 2403)
    cache = BatchedHistoryCache(actor)
    _, encoded = cache.encode([(0, 6)], [stream])
    with torch.no_grad():
        dense = actor.encode_batch(StreamBatch.from_streams([stream], device))
    torch.testing.assert_close(encoded[0, :2404], dense[0], rtol=4e-5, atol=4e-6)
    assert cache.encoded_tokens == 2404
    append(stream, 2, 2403)
    cache.encode([(0, 6)], [stream])
    assert cache.encoded_tokens == 2406


def test_benchmark_cpu_training_guard_and_weighted_summary(tmp_path):
    from bench.history_stack import main, summarize
    with pytest.raises(SystemExit) as rejected:
        main(["--device", "cpu", "--output", str(tmp_path)])
    assert rejected.value.code == 2
    base = dict(learn_seconds=1, update_samples=100, mean_prefix=800, max_prefix=1000,
                cuda_peak_allocated_bytes=10, cuda_peak_reserved_bytes=20,
                collection_cache={"bytes": 5}, collection_phase_seconds={})
    result = summarize([dict(base, collect_seconds=1, step_decisions=100),
                        dict(base, collect_seconds=3, step_decisions=300)])
    assert result["collect_dps"] == 100
    assert result["unique_rows_per_collect_learn_second"] == 200/6
