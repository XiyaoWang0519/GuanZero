"""Opt-in rollout public-memory layouts (acceptance tier 2, all default off):
span groups (``rollout_span_groups``), the learner's private K/V pages
(``rollout_private_kv_pages``) and the causal learner prefill
(``rollout_causal_prefill``).

Each is checked at the cache level against the default layout and at the
collector level on real VecEnv rollouts with snapshot seats: the same choices
and generator state, candidate log-probabilities within FP32 reduction-order
noise (the largest difference is printed for the report).
"""
import itertools

import numpy as np
import pytest
import torch

import train.history_paged_cache as paged
from train.history_model import HistoryPolicyConfig, PublicStream, RolloutMemory, fresh_player
from train.history_paged_cache import KVPagePool, PagedHistoryCache, span_plan
from test_history_inference import append
from test_history_snapshot_batch import run_collector

# Learner candidate log-probabilities against the default layout (CPU observed
# ~1e-7; the repository's rollout gates are 1e-4 pairwise, 3e-5 on recompute).
LOGP_ATOL = 1e-5
CONFIG = HistoryPolicyConfig(width=32, layers=2, heads=4, action_width=16, fusion_width=24,
                             critic_width=16, response_mode="auxiliary")


def devices():
    return ["cpu", "cuda"]


def skip_without(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA rollout memory acceptance")


@pytest.fixture
def eager_groups(monkeypatch):
    """Group even the tiny test batches (production keeps the launch thresholds)."""
    monkeypatch.setattr(paged, "SPAN_GROUP_MIN_SHARE", 0.0)
    monkeypatch.setattr(paged, "SPAN_GROUP_MIN_ROWS", 0)


def make_streams(lengths, offset=0):
    streams = [PublicStream(i) for i in range(len(lengths))]
    for stream, n in zip(streams, lengths):
        append(stream, n, offset=offset)
    return streams


# ---- span plan -------------------------------------------------------------------

def brute_force(spans, groups):
    n, best = len(spans), {}
    for g in range(1, groups + 1):
        for cuts in itertools.combinations(range(1, n), g - 1):
            bounds = (0, *cuts, n)
            rows = sum((b - a) * spans[a] for a, b in zip(bounds, bounds[1:]))
            best[g] = min(best.get(g, rows), rows)
    return best


def test_span_plan_minimises_padded_rows():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 9))
        spans = np.sort(rng.choice([64, 128, 192, 256, 512, 1024], n))[::-1].astype(np.int64)
        for groups in (1, 2, 3):
            plan = span_plan(spans, groups)
            assert plan[0][0] == 0 and plan[-1][1] == n and len(plan) <= groups
            assert all(a < b == c for (a, b, _), (c, _, _) in zip(plan, plan[1:]))
            assert all(span == spans[a] for a, _, span in plan)
            rows = sum((b - a) * span for a, b, span in plan)
            assert rows == min(brute_force(spans, groups).values())
    spans = np.asarray([1024, 1024, 64, 64, 64, 64], np.int64)
    assert span_plan(spans, 2) == [(0, 2, 1024), (2, 6, 64)]
    assert span_plan(spans, 2, min_rows=6145) == [(0, 6, 1024)]        # saves 3840 rows
    assert span_plan(spans, 2, min_share=0.7) == [(0, 6, 1024)]        # saves 62.5%
    assert span_plan(spans[:1], 3) == [(0, 1, 1024)]


# ---- cache level -----------------------------------------------------------------

@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("groups", [2, 3])
def test_span_groups_match_one_group(device, groups, eager_groups):
    """Grouped appends and grouped rollout memory against one group over a randomized
    workload (growth, resets, prunes, a weight change): the same entries, pages and
    counters; memory within FP32 order; one-group rollout memory is ``encode``'s."""
    skip_without(device)
    actor = fresh_player(CONFIG, 7)[0].to(device).train()
    actor.batched_private_attention = True
    caches = [PagedHistoryCache(actor, KVPagePool.for_actor(actor, page_tokens=16),
                                chunk_size=32, page_span=True, span_groups=g)
              for g in (1, groups)]
    rng = np.random.default_rng(groups)
    streams = make_streams([0, 3, 40, 150, 7, 90, 260, 12])
    worst, bitwise, total = 0.0, 0, 0
    for step in range(20):
        picked = [int(i) for i in rng.permutation(8)[:rng.integers(1, 9)]]
        keys = [(i, streams[i].match_id) for i in picked]
        batch = [streams[i] for i in picked]
        _, expected = caches[0].encode(keys, batch)
        _, whole = caches[0].encode_rollout(keys, batch)
        assert len(whole.groups) == 1 and whole.groups[0][0] is None
        assert torch.equal(whole.groups[0][1], expected)
        _, grouped = caches[1].encode_rollout(keys, batch)
        assert isinstance(grouped, RolloutMemory) and not grouped.projected
        lengths = [s.prefix + 1 for s in batch]
        seen = []
        for rows, part in grouped.groups:
            index = range(len(picked)) if rows is None else rows.tolist()
            seen.extend(index)
            span = part.shape[1]
            assert span == caches[1]._span(max(lengths[i] for i in index))
            for j, i in enumerate(index):
                n = lengths[i]
                torch.testing.assert_close(part[j, :n], expected[i, :n], rtol=1e-5, atol=1e-5)
                assert not part[j, n:].any()
                worst = max(worst, float((part[j, :n] - expected[i, :n]).abs().max()))
                bitwise += int(torch.equal(part[j, :n], expected[i, :n]))
                total += 1
        assert sorted(seen) == list(range(len(picked)))
        for name in ("encoded_tokens", "rebuilds"):
            assert getattr(caches[0], name) == getattr(caches[1], name)
        assert ({k: (e.length, len(e.pages)) for k, e in caches[0].entries.items()}
                == {k: (e.length, len(e.pages)) for k, e in caches[1].entries.items()})
        for i in picked:
            append(streams[i], int(rng.integers(0, 6)), offset=step)
        if step == 8:
            streams[3].reset(match_id=3)
            append(streams[3], 70)
        if step == 12:
            active = {(i, streams[i].match_id) for i in (0, 1, 3, 5, 6)}
            for cache in caches:
                cache.prune(active)
        if step == 15:
            with torch.no_grad():
                actor.public.weight.add_(1e-3)
    assert caches[1].span_rows["grouped_appends"] > 0
    assert caches[1].span_rows["append"] < caches[0].span_rows["append"]
    assert caches[1].span_rows["append_valid"] == caches[0].span_rows["append_valid"]
    print(f"span groups {groups}: memory max |diff| {worst:.2e}, {bitwise}/{total} streams bitwise")


@pytest.mark.parametrize("device", devices())
def test_grouped_private_attention_matches_padded(device, eager_groups):
    """``_attend_rollout`` over grouped memory against ``_attend_independent`` over the
    padded memory: one group is bitwise; groups differ by FP32 reduction order only."""
    skip_without(device)
    from train.history_model import DecisionInputs
    actor = fresh_player(CONFIG, 3)[0].to(device).train()
    actor.batched_private_attention = True
    streams = make_streams([5, 300, 64, 0, 129, 33, 260])
    keys = [(i, i) for i in range(len(streams))]
    cache = PagedHistoryCache(actor, KVPagePool.for_actor(actor), page_span=True, span_groups=3)
    _, padded = cache.encode(keys, streams)
    _, grouped = cache.encode_rollout(keys, streams)
    assert len(grouped.groups) > 1
    whole = RolloutMemory([(None, padded)])
    n = len(streams)
    rng = np.random.default_rng(1)
    obs = torch.as_tensor(rng.integers(0, 2, (n, CONFIG.obs_dim)), dtype=torch.uint8, device=device)
    inputs = DecisionInputs(streams=None, match_index=torch.arange(n, device=device),
                            prefix=torch.as_tensor([s.prefix for s in streams], device=device),
                            obs=obs, seat=torch.arange(n, device=device) % 4, cand=None,
                            offsets=None, one_decision_per_stream=True)
    for wide in (False, True):
        actor.wide_private_projection = wide
        with torch.no_grad():
            expected = actor.decision_states(padded, inputs)
            assert torch.equal(actor.decision_states(whole, inputs), expected)
            actual = actor.decision_states(grouped, inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        print(f"wide={wide}: grouped private attention max |diff| "
              f"{float((actual - expected).abs().max()):.2e}")
    with pytest.raises(ValueError, match="one decision per stream"):
        actor._attend_encoded(actor.q_proj(padded[:, 0]), grouped, inputs.match_index,
                              inputs.prefix, False)
    actor.batched_private_attention = False
    with pytest.raises(ValueError, match="batched private attention"):
        with torch.no_grad():
            actor.decision_states(grouped, inputs)
    with torch.no_grad():               # one ungrouped part: the per-row loop as before
        assert torch.equal(actor.decision_states(whole, inputs),
                           actor.decision_states(padded, inputs))


@pytest.mark.parametrize("device", devices())
def test_merged_encode_with_span_groups(device, eager_groups):
    """The merged snapshot append sorts by end and attends per group; memory equals
    the ungrouped merged pass to FP32 order and pages/counters stay per identity."""
    skip_without(device)
    from train.history_snapshot_batch import SnapshotHeads
    actors = [fresh_player(CONFIG, seed)[0].to(device).train() for seed in (1, 2)]
    heads = SnapshotHeads(encoder=True)
    slots = [heads.slot(identity, actor) for identity, actor in zip((4, 6), actors)]
    caches = {g: [PagedHistoryCache(actor, pool, chunk_size=32, page_span=True, span_groups=g)
                  for actor in actors]
              for g, pool in ((1, KVPagePool.for_actor(actors[0], page_tokens=16)),
                              (3, KVPagePool.for_actor(actors[0], page_tokens=16)))}
    streams = make_streams([0, 5, 200, 33, 1, 100, 12, 70])
    picks = [[0, 2, 3, 5], [1, 4, 6, 7]]
    for step in range(6):
        size = max(streams[i].prefix for pick in picks for i in pick) + 1
        memory = {g: paged.merged_encode(
            [(cache, [(i, i) for i in pick], [streams[i] for i in pick], slot)
             for cache, pick, slot in zip(caches[g], picks, slots)], heads, size)
            for g in caches}
        torch.testing.assert_close(memory[3], memory[1], rtol=1e-5, atol=1e-5)
        for one, grouped in zip(caches[1], caches[3]):
            assert one.encoded_tokens == grouped.encoded_tokens
            assert ({k: e.length for k, e in one.entries.items()}
                    == {k: e.length for k, e in grouped.entries.items()})
        for pick in picks:
            for i in pick:
                append(streams[i], (step * 7 + i) % 9, offset=step)
    assert caches[3][0].span_rows["grouped_appends"] > 0


# ---- collector level ---------------------------------------------------------------

def learner_players(device, wide=True):
    policies = {i: fresh_player(CONFIG, 40 + i)[0].to(device) for i in range(4)}
    policies[0].batched_private_attention = True
    policies[0].wide_private_projection = wide
    return policies


def record_log_probs(actor):
    """Keep every candidate log-probability vector the learner computes."""
    log, original = [], actor.candidate_log_probs

    def recording(inputs, encoded=None):
        result = original(inputs, encoded)
        log.append(result.detach().cpu().clone())
        return result

    actor.candidate_log_probs = recording
    return log


def compare_rollouts(device, options, steps=150, **common):
    """A default-layout rollout against one with ``options``: identical choices,
    sampler state and stored integers; the largest learner log-probability
    difference (every candidate of every step)."""
    runs = []
    for extra in ({}, options):
        policies = learner_players(device)
        log = record_log_probs(policies[0])
        collector, _ = run_collector(device, True, kv_cache=True, paged_cache=True,
                                     page_span=True, batch_snapshot_encoder=True, steps=steps,
                                     policies=policies, **common, **extra)
        runs.append((collector, log))
    (plain, plain_log), (fast, fast_log) = runs
    assert len(plain.choice_log) == len(fast.choice_log)
    for before, after in zip(plain.choice_log, fast.choice_log):
        assert before.tobytes() == after.tobytes()
    assert torch.equal(plain.generator.get_state(), fast.generator.get_state())
    assert plain.policy_decisions == fast.policy_decisions
    worst = 0.0
    for before, after in zip(plain_log, fast_log, strict=True):
        assert before.shape == after.shape
        worst = max(worst, float((before - after).abs().max()))
    data, fast_data = plain.buffer.compact(), fast.buffer.compact()
    for name, before in data.items():
        after = fast_data[name]
        if name in ("logp", "behaviour_logp"):
            worst = max(worst, float(np.abs(after - before).max(initial=0.0)))
        else:
            assert before.dtype == after.dtype and before.tobytes() == after.tobytes()
    assert worst < LOGP_ATOL
    return plain, fast, worst


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("groups", [2, 3])
def test_collector_span_groups(device, groups, eager_groups):
    skip_without(device)
    plain, fast, worst = compare_rollouts(device, dict(span_groups=groups))
    learner = fast.caches[0].span_rows
    assert learner["grouped_appends"] > 0 and learner["memory"] < plain.caches[0].span_rows["memory"]
    assert sum(c.span_rows["grouped_appends"] for k, c in fast.caches.items() if k) > 0
    print(f"collector span groups {groups}: learner log-prob max |diff| {worst:.2e}")


# ---- private K/V pages -----------------------------------------------------------

def live(cache):
    P = cache.pool.page_tokens
    return torch.as_tensor([entry.pages[t // P] * P + t % P for entry in cache.entries.values()
                            for t in range(entry.length)], dtype=torch.long)


@pytest.mark.parametrize("device", devices())
def test_private_kv_pages_hold_the_projected_memory(device, eager_groups):
    """Every live pool row's private K/V is ``kv_proj`` of its memory (to the GEMM
    row count); projected rollout memory equals ``kv_proj`` of ``encode``'s memory,
    grouped or not; a ``kv_proj`` update rebuilds them; a pool reset frees them."""
    skip_without(device)
    actor = fresh_player(CONFIG, 9)[0].to(device).train()
    actor.batched_private_attention = True
    cache = PagedHistoryCache(actor, chunk_size=32, page_span=True, span_groups=2,
                              private_kv=True)
    pool = cache.pool
    assert pool.private_width == 2 * CONFIG.width
    assert pool.page_bytes == pool.page_tokens * (2 * 2 * 32 + 32 + 64) * 4
    streams = make_streams([0, 3, 40, 150, 7, 90])
    keys = [(i, i) for i in range(len(streams))]
    worst = 0.0
    for step in range(10):
        _, memory = cache.encode(keys, streams)
        _, projected = cache.encode_rollout(keys, streams, projected=True)
        assert projected.projected
        rows = live(cache).to(device)
        with torch.no_grad():
            expected_rows = actor.kv_proj(pool.memory[rows])
            expected = actor.kv_proj(memory)
        torch.testing.assert_close(pool.private[rows], expected_rows, rtol=2e-6, atol=2e-6)
        worst = max(worst, float((pool.private[rows] - expected_rows).abs().max()))
        lengths = [s.prefix + 1 for s in streams]
        for group_rows, part in projected.groups:
            index = range(len(streams)) if group_rows is None else group_rows.tolist()
            for j, i in enumerate(index):
                torch.testing.assert_close(part[j, :lengths[i]], expected[i, :lengths[i]],
                                           rtol=2e-6, atol=2e-6)
                assert not part[j, lengths[i]:].any()
        for i, stream in enumerate(streams):
            append(stream, (3 * step + i) % 7, offset=step)
        if step == 5:
            with torch.no_grad():
                actor.kv_proj.weight.mul_(1.5)
    print(f"stored private K/V against kv_proj(memory): max |diff| {worst:.2e}")
    with pytest.raises(ValueError, match="private_kv"):
        PagedHistoryCache(actor).encode_rollout(keys, streams, projected=True)
    with pytest.raises(ValueError, match="private_width"):
        PagedHistoryCache(actor, KVPagePool.for_actor(actor), private_kv=True)
    cache.clear()
    pool.reset()
    assert pool.private.numel() == 0 and pool.reserved_bytes == 0
    _, projected = cache.encode_rollout(keys, streams, projected=True)    # regrows
    assert pool.private.shape == (pool.pages * pool.page_tokens, 64)
    assert not pool.private[:pool.page_tokens].any()


@pytest.mark.parametrize("device", devices())
def test_projected_rollout_memory_attention_is_the_padded_attention(device):
    """With the same keys/values (one ``kv_proj`` over the padded memory), attention
    over projected rollout memory is the padded path bit for bit."""
    skip_without(device)
    from train.history_model import DecisionInputs
    actor = fresh_player(CONFIG, 4)[0].to(device).train()
    actor.batched_private_attention = True
    streams = make_streams([5, 300, 64, 0, 129])
    _, padded = PagedHistoryCache(actor, page_span=True).encode(
        [(i, i) for i in range(len(streams))], streams)
    n = len(streams)
    obs = torch.as_tensor(np.random.default_rng(2).integers(0, 2, (n, CONFIG.obs_dim)),
                          dtype=torch.uint8, device=device)
    inputs = DecisionInputs(streams=None, match_index=torch.arange(n, device=device),
                            prefix=torch.as_tensor([s.prefix for s in streams], device=device),
                            obs=obs, seat=torch.arange(n, device=device) % 4, cand=None,
                            offsets=None, one_decision_per_stream=True)
    with torch.no_grad():
        projected = RolloutMemory([(None, actor.kv_proj(padded))], projected=True)
        for wide in (False, True):
            actor.wide_private_projection = wide
            assert torch.equal(actor.decision_states(projected, inputs),
                               actor.decision_states(padded, inputs))
        single = DecisionInputs(streams=None, match_index=inputs.match_index[:1],
                                prefix=inputs.prefix[:1], obs=obs[:1], seat=inputs.seat[:1],
                                cand=None, offsets=None, one_decision_per_stream=True)
        one = RolloutMemory([(None, actor.kv_proj(padded[:1]))], projected=True)
        assert torch.equal(actor.decision_states(one, single),
                           actor.decision_states(padded[:1], single))


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("groups", [1, 3])
def test_collector_private_kv_pages(device, groups, eager_groups):
    skip_without(device)
    plain, fast, worst = compare_rollouts(device, dict(private_kv_pages=True, span_groups=groups))
    assert fast.learner_pool.private is not None and fast.kv_pool.private is None
    assert plain.learner_pool.private is None
    print(f"collector private K/V pages (span groups {groups}): learner log-prob "
          f"max |diff| {worst:.2e}")


def test_private_kv_pages_default_off_and_configuration():
    from train.history_ppo import HistoryPPOConfig, build_parser, config_from_args
    from train.history_rollout import HistoryCollector
    import inspect
    assert HistoryPPOConfig().rollout_private_kv_pages is False
    assert not config_from_args(build_parser().parse_args(["--output", "x"])).rollout_private_kv_pages
    assert build_parser().parse_args(["--output", "x", "--rollout-private-kv-pages"]
                                     ).rollout_private_kv_pages
    base = dict(rollout_kv_cache=True, rollout_paged_cache=True, rollout_batched_attention=True)
    HistoryPPOConfig(rollout_private_kv_pages=True, **base)
    for drop in ("rollout_paged_cache", "rollout_batched_attention"):
        with pytest.raises(ValueError, match="private_kv_pages"):
            HistoryPPOConfig(rollout_private_kv_pages=True, **{**base, drop: False})
    assert inspect.signature(HistoryCollector).parameters["private_kv_pages"].default is False
    assert inspect.signature(PagedHistoryCache).parameters["private_kv"].default is False


# ---- causal prefill -------------------------------------------------------------

def position_rows(cache, key):
    entry, P = cache.entries[key], cache.pool.page_tokens
    return torch.as_tensor([entry.pages[t // P] * P + t % P for t in range(entry.length)])


def assert_close_pools(chunked, causal, keys, worst):
    for key in keys:
        a, b = position_rows(chunked, key), position_rows(causal, key)
        pairs = [(chunked.pool.memory, causal.pool.memory)]
        pairs += list(zip(chunked.pool.kv, causal.pool.kv))
        if chunked.private_kv:
            pairs.append((chunked.pool.private, causal.pool.private))
        for x, y in pairs:
            torch.testing.assert_close(y[b.to(y.device)], x[a.to(x.device)], rtol=1e-5, atol=1e-5)
            worst[0] = max(worst[0], float((y[b.to(y.device)] - x[a.to(x.device)]).abs().max()))


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("private", [False, True])
def test_causal_prefill_matches_chunked_prefill(device, private, monkeypatch):
    """Causal length-group passes hold the same entries, pages, counters and pool
    rows (keys/values, memory, private K/V) as chunked prefill, to FP32 order;
    partly encoded entries take the chunked path; later appends agree."""
    skip_without(device)
    monkeypatch.setattr(paged, "CAUSAL_PREFILL_TOKENS", 640)   # several passes per group
    actor = fresh_player(CONFIG, 13)[0].to(device).train()
    caches = [PagedHistoryCache(actor, KVPagePool.for_actor(
        actor, page_tokens=16, private_width=64 if private else 0), chunk_size=32,
        page_span=True, private_kv=private, causal_prefill=causal) for causal in (False, True)]
    streams = make_streams([0, 3, 40, 150, 7, 90, 260, 12, 1, 31])
    keys = [(i, i) for i in range(len(streams))]
    for cache in caches:            # two entries already partly encoded
        cache.encode(keys[:2], streams[:2])
    for stream in streams[:2]:
        append(stream, 50, offset=3)
    worst = [0.0]
    for round_ in range(2):
        passes = [cache.prefill(keys, streams) for cache in caches]
        assert passes[1] < passes[0]
        for name in ("encoded_tokens", "rebuilds"):
            assert getattr(caches[0], name) == getattr(caches[1], name)
        assert ({k: (e.length, len(e.pages)) for k, e in caches[0].entries.items()}
                == {k: (e.length, len(e.pages)) for k, e in caches[1].entries.items()})
        assert_close_pools(*caches, keys, worst)
        assert caches[1].prefill(keys, streams) == 0
        for step in range(3):
            for i, stream in enumerate(streams):
                append(stream, (i + step) % 5, offset=step)
            memory = [cache.encode(keys, streams)[1] for cache in caches]
            torch.testing.assert_close(memory[1], memory[0], rtol=1e-5, atol=1e-5)
            worst[0] = max(worst[0], float((memory[1] - memory[0]).abs().max()))
        assert_close_pools(*caches, keys, worst)
        with torch.no_grad():       # a weight update: everything is rebuilt from BOS
            actor.public.weight.add_(1e-3)
        for cache in caches:
            cache.refresh_weights()
            cache.pool.reset()
    print(f"causal prefill (private={private}): pool rows and memory max |diff| {worst[0]:.2e}")


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("options", ["causal", "all"])
def test_collector_causal_prefill(device, options, eager_groups, monkeypatch):
    """Learner cache prefill, causal against chunked (and with every option on)."""
    skip_without(device)
    spans = []
    causal_pass = PagedHistoryCache._causal_pass

    def spy(self, entries, targets, span):
        spans.append(span)
        return causal_pass(self, entries, targets, span)

    monkeypatch.setattr(PagedHistoryCache, "_causal_pass", spy)
    invalidate = lambda chunk, policies, collector: collector.invalidate_learner_cache()
    extra = dict(causal_prefill=True)
    if options == "all":
        extra.update(span_groups=3, private_kv_pages=True)
    plain, fast, worst = compare_rollouts(device, extra, steps=420, prefill_learner_cache=True,
                                          between=invalidate)
    assert len(set(spans)) > 1 and max(spans) > 128     # several length groups, long streams
    assert fast.caches[0].appends <= plain.caches[0].appends
    assert fast.caches[0].encoded_tokens == plain.caches[0].encoded_tokens
    print(f"collector causal prefill ({options}): learner log-prob max |diff| {worst:.2e}")


def test_causal_prefill_default_off_and_configuration():
    from train.history_ppo import HistoryPPOConfig, build_parser, config_from_args
    from train.history_rollout import HistoryCollector
    import inspect
    assert HistoryPPOConfig().rollout_causal_prefill is False
    assert not config_from_args(build_parser().parse_args(["--output", "x"])).rollout_causal_prefill
    assert build_parser().parse_args(["--output", "x", "--rollout-causal-prefill"]
                                     ).rollout_causal_prefill
    base = dict(rollout_kv_cache=True, rollout_paged_cache=True,
                rollout_prefill_learner_cache=True)
    HistoryPPOConfig(rollout_causal_prefill=True, **base)
    for drop in ("rollout_paged_cache", "rollout_prefill_learner_cache"):
        with pytest.raises(ValueError, match="causal_prefill"):
            HistoryPPOConfig(rollout_causal_prefill=True, **{**base, drop: False})
    assert inspect.signature(HistoryCollector).parameters["causal_prefill"].default is False
    assert inspect.signature(PagedHistoryCache).parameters["causal_prefill"].default is False


def test_span_groups_default_off_and_configuration():
    from train.history_ppo import HistoryPPOConfig, build_parser, config_from_args
    from train.history_rollout import HistoryCollector
    assert HistoryPPOConfig().rollout_span_groups == 1
    assert config_from_args(build_parser().parse_args(["--output", "x"])).rollout_span_groups == 1
    args = build_parser().parse_args(["--output", "x", "--rollout-span-groups", "3"])
    assert args.rollout_span_groups == 3
    for bad in (dict(rollout_span_groups=0), dict(rollout_span_groups=5),
                dict(rollout_span_groups=2, rollout_kv_cache=True)):
        with pytest.raises(ValueError, match="span_groups"):
            HistoryPPOConfig(**bad)
    HistoryPPOConfig(rollout_span_groups=2, rollout_kv_cache=True, rollout_paged_cache=True)
    import inspect
    assert inspect.signature(HistoryCollector).parameters["span_groups"].default == 1
    assert inspect.signature(PagedHistoryCache).parameters["span_groups"].default == 1


# ---- trainer -----------------------------------------------------------------------

def test_trainer_with_every_rollout_memory_option(tmp_path, eager_groups):
    """A tiny PPO run with all three options (and the production rollout set) gives
    finite metrics; stored log-probabilities match the learner's full-history
    recomputation within the repository's recompute gate; the manifest records the
    options and a resume may switch them."""
    import json
    from train.history_ppo import HistoryPPOConfig, HistoryTrainer, parse_resume_overrides
    options = dict(rollout_span_groups=3, rollout_private_kv_pages=True,
                   rollout_causal_prefill=True)
    config = HistoryPPOConfig(width=32, layers=2, heads=4, num_envs=4, steps_per_update=60,
                              updates=3, epochs=1, minibatch_matches=2, snapshot_updates=1,
                              seed=4, causal_sdpa=True, response_mode="auxiliary",
                              rollout_kv_cache=True, rollout_paged_cache=True,
                              rollout_page_span=True, rollout_batched_attention=True,
                              rollout_wide_projection=True, batch_snapshot_policies=True,
                              batch_snapshot_encoder=True, rollout_prefill_learner_cache=True,
                              learner_batched_attention=True, **options)
    trainer = HistoryTrainer(config, tmp_path / "run")
    lines = [trainer.update() for _ in range(3)]
    learned = [line for line in lines if line["update_samples"] > 0]
    assert len(learned) >= 2
    for line in learned:
        for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "actor_grad_norm"):
            assert np.isfinite(line[name]), (name, line[name])
    collector = trainer.collector
    assert collector.span_groups == 3 and collector.private_kv_pages and collector.causal_prefill
    trainer.collect()
    assert collector.learner_pool.private is not None
    assert collector.caches[0].span_rows["grouped_appends"] > 0
    data = trainer.buffer.compact()
    rows = np.flatnonzero(data["version"] == trainer.progress["updates"])
    assert rows.size
    with torch.no_grad():
        logp, _ = trainer.recompute_log_probs(rows)
    gap = float(np.abs(logp.cpu().numpy() - data["logp"][rows]).max())
    assert gap < 3e-5
    print(f"recompute against stored log-probabilities: max |diff| {gap:.2e}")
    manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
    assert {key: manifest["inference"][key] for key in options} == options
    assert parse_resume_overrides([f"{key}={value}" for key, value in options.items()]) == options
