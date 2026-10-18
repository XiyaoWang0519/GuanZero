"""Opt-in Triton paged attention in the rollout KV cache
(``rollout_paged_attention_kernel``; acceptance tier 2, default off).

Where the kernel cannot run (no Triton, off CUDA, or a head depth / page size
it does not tile) the switch must be a no-op, bit for bit. Where it runs, the
cache's appends (learner, chunked prefill, merged snapshot appends) and, with
private K/V pages, the learner's private attention must match the gather +
SDPA path to FP32 reduction-order noise, and a real rollout must make the same
choices with the same generator.

The kernel runs on CPU only under Triton's interpreter, which Triton picks
when it is first imported; a normal test run imports it compiled. The
``interpret`` cases therefore run in a child pytest started with
``TRITON_INTERPRET=1`` (``test_interpreter_cases_in_subprocess``), as in
tests/test_history_paged_attention.py; on a CUDA host the ``cuda`` cases run
the compiled kernel in-process.
"""
import inspect
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pytest
import torch

import train.history_paged_attention as kernel
import train.history_paged_cache as paged
from train.history_model import DecisionInputs, HistoryPolicyConfig, fresh_player
from train.history_paged_cache import KVPagePool, PagedHistoryCache
from test_history_inference import append
from test_history_paged_cache import PoisonedPool
from test_history_rollout_memory import live, make_streams, record_log_probs
from test_history_snapshot_batch import run_collector

ROOT = Path(__file__).resolve().parents[1]
LOGP_ATOL = 1e-5        # rollout gates are 1e-4 pairwise and 3e-5 on recompute
# Head depth 16 (the production depth): the smallest the kernel tiles.
KERNEL_CONFIG = HistoryPolicyConfig(width=32, layers=2, heads=2, action_width=16,
                                    fusion_width=24, critic_width=16, response_mode="auxiliary")
# Head depth 8: the kernel cannot tile it, so the switch is a no-op even where Triton runs.
SHALLOW_CONFIG = HistoryPolicyConfig(width=32, layers=2, heads=4, action_width=16,
                                     fusion_width=24, critic_width=16, response_mode="auxiliary")


@pytest.fixture(params=["interpret", "cuda"])
def device(request, monkeypatch):
    """Where the kernel runs: the CPU interpreter (child process) or compiled CUDA."""
    if kernel.triton is None:
        pytest.skip("Triton is not installed")
    if request.param == "cuda":
        if kernel.INTERPRETED or not torch.cuda.is_available():
            pytest.skip("compiled CUDA kernel acceptance")
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
        return "cuda"
    if not kernel.INTERPRETED:
        pytest.skip("interpreter cases run in test_interpreter_cases_in_subprocess")
    return "cpu"


@pytest.fixture
def eager_groups(monkeypatch):
    """Group even the tiny test batches (production keeps the launch thresholds)."""
    monkeypatch.setattr(paged, "SPAN_GROUP_MIN_SHARE", 0.0)
    monkeypatch.setattr(paged, "SPAN_GROUP_MIN_ROWS", 0)


@pytest.fixture
def small_pages(monkeypatch):
    """Collector pools of 16-token pages, so short test streams span several pages."""
    original = KVPagePool.for_actor.__func__
    monkeypatch.setattr(KVPagePool, "for_actor", classmethod(
        lambda cls, actor, **kwargs: original(cls, actor, **{"page_tokens": 16, **kwargs})))


def force_splits(monkeypatch, splits):
    """Caches built after this launch the kernel with ``splits`` key splits (flash-decoding
    combine), which the interpreter's split heuristic (no SM count) never picks."""
    run = kernel.paged_append_attention_triton
    monkeypatch.setattr(kernel, "paged_append_attention_triton",
                        lambda *args, **kwargs: run(*args, splits=splits, **kwargs))


def players(config, device):
    policies = {i: fresh_player(config, 40 + i)[0].to(device) for i in range(4)}
    policies[0].batched_private_attention = True
    policies[0].wide_private_projection = True
    return policies


def rollout_pair(device, config, options, steps):
    """The same rollout without and with the switch: (collector, learner log-probs) each."""
    runs = []
    for flag in (False, True):
        policies = players(config, device)
        log = record_log_probs(policies[0])
        collector, _ = run_collector(
            device, True, kv_cache=True, paged_cache=True, page_span=True,
            batch_snapshot_encoder=True, prefill_learner_cache=True, steps=steps,
            policies=policies, paged_attention_kernel=flag,
            between=lambda chunk, policies, collector: collector.invalidate_learner_cache(),
            **options)
        runs.append((collector, log))
    return runs


def assert_same_rollout(plain, fast, plain_log, fast_log, exact):
    """Identical choices, sampler state and stored integers; returns the largest
    learner log-probability difference (every candidate of every step)."""
    assert len(plain.choice_log) == len(fast.choice_log)
    for before, after in zip(plain.choice_log, fast.choice_log):
        assert before.tobytes() == after.tobytes()
    assert torch.equal(plain.generator.get_state(), fast.generator.get_state())
    assert plain.policy_decisions == fast.policy_decisions
    worst = 0.0
    for before, after in zip(plain_log, fast_log, strict=True):
        assert before.shape == after.shape
        if exact:
            assert torch.equal(before, after)
        worst = max(worst, float((before - after).abs().max()))
    data, fast_data = plain.buffer.compact(), fast.buffer.compact()
    for name, before in data.items():
        after = fast_data[name]
        if name in ("logp", "behaviour_logp") and not exact:
            worst = max(worst, float(np.abs(after - before).max(initial=0.0)))
        else:
            assert before.dtype == after.dtype and before.tobytes() == after.tobytes(), name
    return worst


# ---- the switch is a no-op where the kernel cannot run ---------------------------

@pytest.mark.parametrize("mode", ["compiled", "interpret"])
def test_switch_is_bitwise_where_the_kernel_cannot_run(mode, small_pages, eager_groups):
    """Off CUDA with compiled Triton (any geometry), or under the interpreter with a head
    depth the kernel does not tile: every log-probability, choice, stored field and the
    generator are bitwise those of the default path, with every option it touches on."""
    if (mode == "interpret") != kernel.INTERPRETED:
        pytest.skip("runs in the matching Triton mode (the child process interprets)")
    config = SHALLOW_CONFIG if kernel.INTERPRETED else KERNEL_CONFIG
    for options in (dict(private_kv_pages=True), dict(span_groups=2)):
        (plain, plain_log), (fast, fast_log) = rollout_pair("cpu", config, options, steps=24)
        assert fast.paged_attention_kernel and not fast.paged_attention_kernel_active()
        assert all(c.attention_kernel and not c.kernel_active for c in fast.caches.values())
        assert all(not c.span_rows["kernel_appends"] for c in fast.caches.values())
        assert assert_same_rollout(plain, fast, plain_log, fast_log, exact=True) == 0.0
        assert ([c.span_rows for c in plain.caches.values()]
                == [c.span_rows for c in fast.caches.values()])


# ---- the kernel against the gather path -------------------------------------------

def decision_inputs(streams, device, seed):
    n = len(streams)
    obs = torch.as_tensor(np.random.default_rng(seed).integers(0, 2, (n, KERNEL_CONFIG.obs_dim)),
                          dtype=torch.uint8, device=device)
    return DecisionInputs(streams=None, match_index=torch.arange(n, device=device),
                          prefix=torch.as_tensor([s.prefix for s in streams], device=device),
                          obs=obs, seat=torch.arange(n, device=device) % 4, cand=None,
                          offsets=None, one_decision_per_stream=True)


@pytest.mark.parametrize("splits", [None, 3])
def test_cache_kernel_matches_gather_path(device, splits, eager_groups, monkeypatch):
    """A randomized workload (pool growth, page reuse, a stream reset, a prune, a weight
    change) through a cache with the kernel and one without (span groups and private
    K/V pages on both): the same entries and counters; memory, live pool rows and the
    learner's private attention over the stored K/V within FP32 order. Pools grow
    NaN-poisoned, so a kernel read of any unwritten row would show."""
    if splits is not None:
        force_splits(monkeypatch, splits)
    actor = fresh_player(KERNEL_CONFIG, 7)[0].to(device).train()
    actor.batched_private_attention = True
    caches = [PagedHistoryCache(actor, PoisonedPool.for_actor(actor, page_tokens=16, pages=2,
                                                              private_width=64),
                                chunk_size=32, page_span=True, span_groups=2, private_kv=True,
                                attention_kernel=flag) for flag in (False, True)]
    assert [cache.kernel_active for cache in caches] == [False, True]
    rng = np.random.default_rng(5)
    streams = make_streams([0, 3, 40, 150, 7, 90, 200, 12])
    worst = {"memory": 0.0, "pool": 0.0, "private attention": 0.0}
    for step in range(12):
        picked = [int(i) for i in rng.permutation(8)[:rng.integers(1, 9)]]
        keys = [(i, streams[i].match_id) for i in picked]
        batch = [streams[i] for i in picked]
        memory = [cache.encode(keys, batch)[1] for cache in caches]
        assert not memory[1].isnan().any()
        torch.testing.assert_close(memory[1], memory[0], rtol=1e-5, atol=1e-5)
        worst["memory"] = max(worst["memory"], float((memory[1] - memory[0]).abs().max()))
        for name in ("encoded_tokens", "rebuilds"):
            assert getattr(caches[0], name) == getattr(caches[1], name)
        assert ({k: (e.length, len(e.pages)) for k, e in caches[0].entries.items()}
                == {k: (e.length, len(e.pages)) for k, e in caches[1].entries.items()})
        for i in picked:
            append(streams[i], int(rng.integers(0, 6)), offset=step)
        rollout = [cache.encode_rollout(keys, batch, projected=True)[1] for cache in caches]
        assert rollout[0].attend is None and rollout[1].attend is not None
        assert rollout[1].groups == [] and rollout[1].projected
        inputs = decision_inputs(batch, device, step)
        with torch.no_grad():
            states = [actor.decision_states(m, inputs) for m in rollout]
        assert not states[1].isnan().any()
        torch.testing.assert_close(states[1], states[0], rtol=1e-5, atol=1e-5)
        worst["private attention"] = max(worst["private attention"],
                                         float((states[1] - states[0]).abs().max()))
        if step == 5:
            streams[3].reset(match_id=3)
            append(streams[3], 70)
        if step == 7:
            active = {(i, streams[i].match_id) for i in (0, 1, 3, 5, 6)}
            for cache in caches:
                cache.prune(active)
        if step == 9:
            with torch.no_grad():
                actor.public.weight.add_(1e-3)
    rows = [live(cache).to(device) for cache in caches]
    for name in ("memory", "private"):
        a, b = getattr(caches[0].pool, name)[rows[0]], getattr(caches[1].pool, name)[rows[1]]
        torch.testing.assert_close(b, a, rtol=1e-5, atol=1e-5)
        worst["pool"] = max(worst["pool"], float((b - a).abs().max()))
    for a, b in zip(caches[0].pool.kv, caches[1].pool.kv):
        torch.testing.assert_close(b[rows[1]], a[rows[0]], rtol=1e-5, atol=1e-5)
        worst["pool"] = max(worst["pool"], float((b[rows[1]] - a[rows[0]]).abs().max()))
    kernel_rows, plain_rows = caches[1].span_rows, caches[0].span_rows
    assert kernel_rows["kernel_appends"] == caches[1].appends and kernel_rows["append"] == 0
    assert kernel_rows["kernel_rollouts"] == 12 and plain_rows["kernel_appends"] == 0
    assert kernel_rows["append_valid"] == plain_rows["append_valid"]
    print(f"cache kernel (splits {splits or 'auto'}): max |diff| "
          + ", ".join(f"{name} {value:.2e}" for name, value in worst.items()))


@pytest.mark.parametrize("options", ["private_kv", "span_groups"])
def test_collector_kernel_matches_gather_path(device, options, small_pages, eager_groups,
                                              monkeypatch):
    """Real VecEnv rollouts with snapshot seats, page span, merged snapshot encodes and
    learner prefill (rebuilt from BOS after every chunk), kernel on against off: the
    same choices, generator state and stored integers; learner candidate
    log-probabilities within FP32 order. ``private_kv``: the learner's private
    attention reads its stored K/V in place. ``span_groups``: appends in the kernel,
    the learner's memory still gathered in span groups; forced key splits."""
    if options == "private_kv":
        extra = dict(private_kv_pages=True)
    else:
        extra = dict(span_groups=2)
        force_splits(monkeypatch, 2)
    (plain, plain_log), (fast, fast_log) = rollout_pair(device, KERNEL_CONFIG, extra, steps=24)
    worst = assert_same_rollout(plain, fast, plain_log, fast_log, exact=False)
    assert worst <= LOGP_ATOL, worst
    assert fast.paged_attention_kernel_active()
    assert all(c.kernel_active for c in fast.caches.values())
    learner = fast.caches[0].span_rows
    assert learner["kernel_appends"] == fast.caches[0].appends > 0 and learner["append"] == 0
    assert (learner["kernel_rollouts"] > 0) == (options == "private_kv")
    assert sum(c.span_rows["kernel_appends"] for k, c in fast.caches.items() if k) > 0
    assert all(c.span_rows["append_valid"] == p.span_rows["append_valid"]
               for c, p in zip(fast.caches.values(), plain.caches.values()))
    print(f"collector kernel ({options}): learner log-prob max |diff| {worst:.2e}")


def test_interpreter_cases_in_subprocess():
    if kernel.triton is None:
        pytest.skip("Triton is not installed")
    if kernel.INTERPRETED:
        pytest.skip("this process runs the interpreter cases itself")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYTEST_XDIST", "PYTEST_CURRENT_TEST"))}
    env["TRITON_INTERPRET"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider",
         str(Path(__file__)), "-k", "interpret and not subprocess"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    lines = result.stdout.strip().splitlines()
    summary = lines[-1] if lines else ""
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-3000:]
    assert " passed" in summary and "skipped" not in summary, summary
    for line in lines:          # the measured differences, for the report (pytest -s)
        if re.search(r"max \|diff\|", line):
            print(line.strip())


# ---- configuration -----------------------------------------------------------------

def test_paged_attention_kernel_default_off_and_configuration():
    from bench.history_arms import SWITCHABLE
    from train.history_ppo import (RESUME_OVERRIDES, HistoryPPOConfig, build_parser,
                                   config_from_args, parse_resume_overrides)
    from train.history_rollout import HistoryCollector
    assert HistoryPPOConfig().rollout_paged_attention_kernel is False
    parser = build_parser()
    assert not config_from_args(parser.parse_args(["--output", "x"])).rollout_paged_attention_kernel
    assert parser.parse_args(["--output", "x", "--rollout-paged-attention-kernel"]
                             ).rollout_paged_attention_kernel
    base = dict(rollout_kv_cache=True, rollout_paged_cache=True)
    HistoryPPOConfig(rollout_paged_attention_kernel=True, **base)
    for bad in (dict(base, rollout_paged_cache=False),
                dict(base, rollout_private_graphs=True)):
        with pytest.raises(ValueError, match="paged_attention_kernel"):
            HistoryPPOConfig(rollout_paged_attention_kernel=True, **bad)
    assert "rollout_paged_attention_kernel" in RESUME_OVERRIDES & SWITCHABLE
    assert parse_resume_overrides(["rollout_paged_attention_kernel=true"]) == {
        "rollout_paged_attention_kernel": True}
    assert inspect.signature(HistoryCollector).parameters["paged_attention_kernel"].default is False
    assert inspect.signature(PagedHistoryCache).parameters["attention_kernel"].default is False
    actor = fresh_player(KERNEL_CONFIG, 1)[0]
    store, buffer = object(), object()
    for bad in (dict(kv_cache=True), dict(kv_cache=False, paged_cache=False)):
        with pytest.raises(ValueError, match="paged attention kernel"):
            HistoryCollector(None, actor, store, buffer, paged_attention_kernel=True, **bad)
    # Geometry: powers of two, depth 16-64, pages 16-128, depth x page <= 4096.
    assert not kernel.kernel_supported("cpu", torch.float32, 8, 64)
    assert not kernel.kernel_supported("cpu", torch.float32, 16, 8)
    assert not kernel.kernel_supported("cpu", torch.float64, 16, 64)
    assert kernel.kernel_supported("cpu", torch.float32, 16, 64) == kernel.INTERPRETED
    assert not kernel.kernel_supported("cpu", torch.float32, 128, 64)     # shared memory
    assert not kernel.kernel_supported("cpu", torch.float32, 16, 256)
    assert not kernel.kernel_supported("cpu", torch.float32, 64, 128)
    if not kernel.INTERPRETED:
        assert not PagedHistoryCache(actor, attention_kernel=True).kernel_active


def test_trainer_records_and_switches_the_kernel(tmp_path):
    """The manifest records the request and whether the kernel runs (here on CPU: a
    no-op); ``bench.history_arms.switch`` sets it on a live trainer's collector."""
    import json
    from bench.history_arms import switch
    from train.history_ppo import HistoryPPOConfig, HistoryTrainer
    config = HistoryPPOConfig(width=32, layers=1, heads=2, num_envs=2, steps_per_update=12,
                              updates=1, epochs=1, minibatch_matches=2, snapshot_updates=1,
                              seed=3, causal_sdpa=True, rollout_kv_cache=True,
                              rollout_paged_cache=True, rollout_page_span=True,
                              rollout_batched_attention=True, rollout_private_kv_pages=True,
                              rollout_paged_attention_kernel=True)
    trainer = HistoryTrainer(config, tmp_path / "run")
    trainer.update()
    inference = json.loads((tmp_path / "run" / "manifest.json").read_text())["inference"]
    assert inference["rollout_paged_attention_kernel"] is True
    active = kernel.INTERPRETED                     # CPU: only the interpreter runs it
    assert inference["rollout_paged_attention_kernel_active"] is active
    assert all(c.kernel_active is active for c in trainer.collector.caches.values())
    switch(trainer, config, {"rollout_paged_attention_kernel": False})
    assert not trainer.collector.paged_attention_kernel and not trainer.collector.caches
    trainer.collect()
    assert all(not c.attention_kernel for c in trainer.collector.caches.values())
    switch(trainer, config, {})
    assert trainer.collector.paged_attention_kernel
