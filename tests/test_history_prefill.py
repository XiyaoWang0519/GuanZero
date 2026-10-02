"""Batched learner cache prefill, the learner's own page pool and the learn-phase profile."""
import json

import numpy as np
import pytest
import torch

from train.history_inference import BatchedHistoryCache
from train.history_model import HistoryPolicyConfig, PublicStream, fresh_player
from train.history_paged_cache import KVPagePool, PagedHistoryCache
from train.history_ppo import (TORCH_PROFILE_ENV, TORCH_PROFILE_UPDATES_ENV, HistoryPPOConfig,
                               HistoryTrainer, parse_resume_overrides)
from test_history_inference import append
from test_history_snapshot_batch import players, run_collector

LENGTHS = [0, 3, 40, 150, 7, 90]


def devices():
    return ["cpu", "cuda"]


def skip_without(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA prefill acceptance")


def make_streams():
    streams = [PublicStream(i) for i in range(len(LENGTHS))]
    for stream, n in zip(streams, LENGTHS):
        append(stream, n)
    return streams


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("paged", [False, True])
def test_prefill_holds_the_same_entries_in_fewer_passes(device, paged):
    skip_without(device)
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=2, heads=4), 7)
    actor.to(device).train()
    build = (lambda: PagedHistoryCache(actor, chunk_size=16)) if paged else (
        lambda: BatchedHistoryCache(actor, chunk_size=16))
    lazy, eager = build(), build()
    streams = make_streams()
    keys = [(i, i) for i in range(len(streams))]
    passes = eager.prefill(keys, streams)
    # One pass per 16-token chunk of the longest stream (150 tokens and BOS).
    assert passes == eager.appends == -(-151 // 16)
    assert eager.prefill(keys, streams) == 0            # already up to date
    # Lazily, each subset rebuilds whatever it is the first to touch.
    for picked in ([3], [0, 2], [5, 1], [4]):
        subset = [keys[i] for i in picked], [streams[i] for i in picked]
        _, expected = lazy.encode(*subset)
        before = eager.appends
        _, actual = eager.encode(*subset)
        assert eager.appends == before                  # nothing left to encode
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    assert lazy.appends > eager.appends
    assert eager.encoded_tokens == lazy.encoded_tokens == sum(n + 1 for n in LENGTHS)
    assert eager.rebuilds == lazy.rebuilds == len(streams)
    assert ({k: e.length for k, e in eager.entries.items()}
            == {k: e.length for k, e in lazy.entries.items()})
    # New events after the prefill are appended as usual.
    append(streams[2], 5, offset=3)
    _, expected = lazy.encode([keys[2]], [streams[2]])
    _, actual = eager.encode([keys[2]], [streams[2]])
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_prefill_rejects_duplicate_keys_and_accepts_nothing():
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=2, heads=4), 7)
    cache = BatchedHistoryCache(actor.train())
    assert cache.prefill([], []) == 0
    stream = PublicStream(0)
    with pytest.raises(ValueError):
        cache.prefill([(0, 0), (0, 0)], [stream, stream])


def test_pool_reset_returns_storage_and_restores_its_size_in_one_step():
    pool = KVPagePool(layers=2, heads=4, width=32, device="cpu", page_tokens=8, pages=2)
    first = pool.allocate(5)
    assert pool.grows == 2 and 0 not in first
    with pytest.raises(RuntimeError):
        pool.reset()                                    # pages still in use
    pool.release(first)
    pool.reset()
    assert pool.pages == 0 and pool.reserved_bytes == 0 and pool.used_pages == 0
    assert pool.memory.numel() == 0 and not pool.kv
    grows = pool.grows
    pages = pool.allocate(1)                            # restores the earlier peak at once
    assert pool.grows == grows + 1 and pool.pages >= 6 and 0 not in pages
    assert len(set(pages + pool.allocate(4))) == 5 and pool.grows == grows + 1
    assert not pool.memory[:8].any() and not any(kv[:8].any() for kv in pool.kv)
    pool.reset.__self__.release(pool.allocate(0))       # allocating nothing is a no-op
    assert pool.grows == grows + 1


@pytest.mark.parametrize("device", devices())
def test_collector_gives_the_learner_its_own_pool_and_returns_it(device):
    skip_without(device)

    def invalidate(chunk, policies, collector):
        used = collector.learner_pool.used_pages
        assert used > 0 and collector.learner_pool is not collector.kv_pool
        snapshots = collector.kv_pool.used_pages
        collector.invalidate_learner_cache()
        assert collector.learner_pool.reserved_bytes == 0
        assert collector.kv_pool.used_pages == snapshots > 0    # snapshot pages untouched

    from test_history_snapshot_batch import assert_same_collection
    plain = run_collector(device, False, kv_cache=True,
                          between=lambda c, p, col: col.invalidate_learner_cache())[0]
    paged = run_collector(device, True, kv_cache=True, paged_cache=True,
                          batch_snapshot_encoder=True, between=invalidate)[0]
    assert_same_collection(plain, paged)
    metrics = paged.cache_metrics()
    assert metrics["pool_pages"] == paged.kv_pool.pages     # the learner pool is empty now


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("paged", [False, True])
def test_collector_prefill_rebuilds_once_per_collection(device, paged):
    """After every invalidation the prefilled collector rebuilds all learner
    streams in one batched pass per chunk. Tier 2 on learner seats: stored
    log-probabilities carry FP32 reduction-order noise; on these seeds it flips
    no choice, so everything else in the rollout is identical."""
    skip_without(device)
    invalidate = lambda chunk, policies, collector: collector.invalidate_learner_cache()
    extra = dict(paged_cache=True) if paged else {}
    lazy, lazy_stats = run_collector(device, True, kv_cache=True, steps=540, between=invalidate,
                                     profile=True, **extra)
    fast, fast_stats = run_collector(device, True, kv_cache=True, steps=540, between=invalidate,
                                     profile=True, prefill_learner_cache=True, **extra)
    assert len(lazy.choice_log) == len(fast.choice_log)
    for before, after in zip(lazy.choice_log, fast.choice_log):
        assert before.tobytes() == after.tobytes()
    assert torch.equal(lazy.generator.get_state(), fast.generator.get_state())
    assert lazy.policy_decisions == fast.policy_decisions
    for name, before in lazy.buffer.compact().items():
        after = fast.buffer.compact()[name]
        if name in ("logp", "behaviour_logp"):
            np.testing.assert_allclose(after, before, rtol=2e-5, atol=2e-5)
        else:
            assert before.dtype == after.dtype and before.tobytes() == after.tobytes()
    assert fast.caches[0].encoded_tokens == lazy.caches[0].encoded_tokens
    assert fast.caches[0].appends < lazy.caches[0].appends
    assert all("learner_prefill" in s.phase_seconds for s in fast_stats)
    assert not any("learner_prefill" in s.phase_seconds for s in lazy_stats)


def test_prefill_needs_the_match_keyed_kv_cache():
    with pytest.raises(ValueError):
        run_collector("cpu", False, kv_cache=False, prefill_learner_cache=True)
    with pytest.raises(ValueError):
        HistoryPPOConfig(rollout_prefill_learner_cache=True)
    assert parse_resume_overrides(["rollout_prefill_learner_cache=true", "profile_learn=true"]) == {
        "rollout_prefill_learner_cache": True, "profile_learn": True}


def trainer_config(**extra):
    return HistoryPPOConfig(width=32, layers=2, heads=4, num_envs=4, steps_per_update=24,
                            snapshot_updates=1, rollout_kv_cache=True, causal_sdpa=True,
                            batch_snapshot_policies=True, response_mode="auxiliary",
                            updates=4, **extra)


def test_trainer_prefill_and_learn_profile(tmp_path, monkeypatch):
    """The flag reaches the collector and survives a resume override; the learn
    profile and the kernel profile only add timings and files."""
    plain = HistoryTrainer(trainer_config(), tmp_path / "plain")
    plain.run()
    monkeypatch.setenv(TORCH_PROFILE_ENV, str(tmp_path / "profile"))
    monkeypatch.setenv(TORCH_PROFILE_UPDATES_ENV, "2")
    profiled = HistoryTrainer(trainer_config(profile_learn=True), tmp_path / "profiled")
    profiled.run()
    read = lambda path: [json.loads(line) for line in (path / "metrics.jsonl").open()]
    before, after = read(tmp_path / "plain"), read(tmp_path / "profiled")
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "update_samples",
                 "actor_grad_norm"):
        assert [line[name] for line in before] == [line[name] for line in after]
    for a, b in zip(plain.actor.parameters(), profiled.actor.parameters()):
        assert torch.equal(a, b)
    assert all(line["learn_phase_seconds"] == {} for line in before)
    phases = after[-1]["learn_phase_seconds"]
    assert {"values_and_gae", "batch_and_upload", "forward", "backward", "gradient_reduce",
            "norms_clip_and_stats", "optimizer"} <= set(phases)
    assert sum(phases.values()) <= after[-1]["learn_seconds"] * 1.05
    files = sorted(p.name for p in (tmp_path / "profile").iterdir())
    assert files == ["rank-0-update-2-collect.json", "rank-0-update-2-learn.json"]
    learn = json.loads((tmp_path / "profile" / "rank-0-update-2-learn.json").read_text())
    assert learn["phase"] == "learn" and learn["process_update"] == 2 and learn["events"]
    assert any(event["name"] == "aten::linear" and event["count"] > 0 for event in learn["events"])

    monkeypatch.delenv(TORCH_PROFILE_ENV)
    resumed = HistoryTrainer(HistoryPPOConfig(updates=6), tmp_path / "resumed",
                             resume=tmp_path / "plain" / "latest.pt",
                             resume_overrides={"rollout_prefill_learner_cache": True})
    assert resumed.collector.prefill_learner_cache
    resumed.run()
    lines = read(tmp_path / "resumed")
    assert [line["update"] for line in lines] == [5, 6]
    manifest = json.loads((tmp_path / "resumed" / "manifest.json").read_text())
    assert manifest["inference"]["rollout_prefill_learner_cache"] is True
    assert resumed.config_changes[-1]["changes"] == {"rollout_prefill_learner_cache": [False, True]}
