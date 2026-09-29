"""Opt-in merged snapshot public encode (``batch_snapshot_encode``).

Acceptance tier 2 for frozen snapshot seats: one merged encode over stacked
encoder weights writes the same K/V and memory into each identity's own cache
entries, and returns the same public memory, as each identity's
``cache.encode`` up to FP32 reduction order. It draws no random numbers, so
with the flag on or off the learner rows and sampler state are bitwise equal
on these seeds (no snapshot choice flips).
"""
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from train import history_rollout as rollout
from train import history_snapshot_batch
from train.history_inference import BatchedHistoryCache
from train.history_model import HistoryPolicyConfig, PublicStream, fresh_player
from train.history_ppo import (HistoryPPOConfig, HistoryTrainer, build_parser, config_from_args,
                               parse_resume_overrides)
from train.history_snapshot_batch import (LAYOUT_FIELDS, SnapshotHeads, merged_layout,
                                          merged_log_probs)
from train.history_snapshot_encode import (ENCODE_FIELDS, SnapshotEncoders, _encoder_values,
                                           merged_encode, plan_encode, validate_encoder)
from train.history_transfers import upload_arrays
from test_history_rollout import make_env
from test_history_snapshot_batch import (Step, assert_same_collection, reference,
                                         strict_fp32, token)  # noqa: F401 (fixture)

# Merged vs per-identity encode (memory, K/V), FP32.
ATOL = 1e-5
TIMING_KEYS = {"collection_phase_seconds", "collection_profile_synchronized",
               "collection_group_phase_seconds", "collection_policy_call_rows",
               "collection_policy_batches", "decisions_per_sec", "collect_seconds",
               "learn_seconds", "learn_decisions_per_sec", "learn_exposures_per_sec",
               "learner_collect_decisions_per_sec", "elapsed_seconds",
               "rollout_batch_snapshot_encode", "snapshot_encoders",
               "rollout_trim_cuda_cache", "cuda_trim", "cuda_trim_seconds",
               # the merged encode keeps its positional table with the stacked
               # encoders, not in the per-identity caches (compared separately)
               "cache_bytes", "cache", "collection_cache"}
OBSERVED = {}


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA merged snapshot encode")
    return request.param


def config(width=32, layers=2, heads=4):
    return HistoryPolicyConfig(width=width, layers=layers, heads=heads, action_width=16,
                               fusion_width=24, critic_width=16, response_mode="auxiliary")


def players(identities, device, seed=40, stream_norm=False, **kwargs):
    result = {}
    for i in identities:
        actor = fresh_player(config(**kwargs), seed + i)[0].to(device)
        if stream_norm:
            actor.stream.norm = nn.LayerNorm(actor.config.width).to(device)
            with torch.no_grad():
                actor.stream.norm.weight.uniform_(0.5, 1.5)
                actor.stream.norm.bias.uniform_(-0.2, 0.2)
        result[i] = actor
    return result


# ---- unit level: plan + merged encode vs per-identity cache.encode ---------------

class Pair:
    """The same identities' caches twice: per-identity encode and merged encode."""

    def __init__(self, policies, device, chunk_size=16, cache_class=BatchedHistoryCache,
                 triton=False, **encoder_options):
        self.policies, self.device = policies, device
        self.make = lambda actor: cache_class(actor, chunk_size)
        self.plain, self.fast = {}, {}
        self.encoders = SnapshotEncoders(capacity=1, **encoder_options)
        self.triton = triton

    def cache(self, caches, identity):
        actor = self.policies[identity]
        if identity not in caches or caches[identity].actor is not actor:
            caches[identity] = self.make(actor)
        return caches[identity]

    def encode(self, groups):
        length = max(s.prefix for g in groups for s in g.streams) + 1
        width = next(iter(self.policies.values())).config.width
        actors = [self.policies[g.identity] for g in groups]
        # Reference: the collector's per-identity path.
        expected, first = torch.zeros(sum(len(g.keys) for g in groups), length, width,
                                      device=self.device), 0
        with torch.no_grad():
            for group in groups:
                _, encoded = self.cache(self.plain, group.identity).encode(group.keys, group.streams)
                span = min(length, encoded.shape[1])
                expected[first:first + len(group.keys), :span] = encoded[:, :span]
                first += len(group.keys)
        caches = [self.cache(self.fast, g.identity) for g in groups]
        plan = plan_encode(self.encoders, groups, actors, caches, length, triton=self.triton)
        fields = dict(zip(ENCODE_FIELDS, upload_arrays(plan.arrays, self.device)))
        memory = merged_encode(self.encoders, plan, fields, self.device)
        return plan, memory, expected

    def prune(self, identities):
        for caches in (self.plain, self.fast):
            for identity in list(caches):
                if identity not in identities:
                    del caches[identity]
        self.encoders.retain(identities)

    def compare_entries(self):
        worst = 0.0
        assert set(self.plain) == set(self.fast)
        for identity, plain in self.plain.items():
            fast = self.fast[identity]
            assert set(plain.entries) == set(fast.entries)
            assert (plain.rebuilds, plain.encoded_tokens) == (fast.rebuilds, fast.encoded_tokens)
            for key, a in plain.entries.items():
                b = fast.entries[key]
                assert (a.length, a.capacity, a.generation) == (b.length, b.capacity, b.generation)
                assert b.stream is a.stream
                for x, y in zip((*a.keys, *a.values, a.memory), (*b.keys, *b.values, b.memory)):
                    assert x.shape == y.shape
                    worst = max(worst, float((x - y).abs().max()))
                    # The zero tail past each entry's length is an invariant.
                    tail = y[:, a.length:] if y.ndim == 3 else y[a.length:]
                    assert not bool(tail.any())
        return worst


def grow(stream, rng, n):
    for _ in range(n):
        stream.append_token(token(rng), int(rng.integers(0, 3)), int(rng.integers(0, 4)))


def groups_for(streams, members):
    return [SimpleNamespace(identity=i, keys=list(keys), streams=[streams[k] for k in keys])
            for i, keys in sorted(members.items())]


def unit_scenario(device, stream_norm=False, triton=False, max_pack_bytes=128 << 20):
    """Several identities with different weights over 14 steps: mixed stream
    lengths and growth (including none), entries crossing the 32 bucket, an
    entry created mid-run, an identity pruned (its slot reused), an in-place
    weight update (cache cleared, rebuilt through the fallback), entries
    needing more than one chunk (fallback) and exactly one chunk (merged)."""
    rng = np.random.default_rng(3)
    policies = players((1, 2, 3, 4, 5), device, stream_norm=stream_norm)
    pair = Pair(policies, device, chunk_size=16, triton=triton, max_pack_bytes=max_pack_bytes)
    streams, members = {}, {1: [], 2: [], 3: [], 4: []}
    initial = {1: [0, 2, 29], 2: [5], 3: [1, 3, 30, 12, 0], 4: [14, 15]}   # 29/30: > chunk
    env = 0
    for identity, lengths in initial.items():
        for n in lengths:
            key = (env, 7)
            streams[key] = PublicStream(7)
            grow(streams[key], rng, n)
            members[identity].append(key)
            env += 1
    fallbacks, worst = [], 0.0
    for step in range(14):
        if step == 4:                       # an entry created during collection
            streams[(env, 7)] = PublicStream(7)
            grow(streams[(env, 7)], rng, 2)
            members[2].append((env, 7))
            env += 1
        if step == 6:                       # identity 3 leaves; identity 5 takes its slot
            freed = pair.encoders.slots[3].index
            del members[3]
            pair.prune({1, 2, 4})
            members[5] = [(env, 7), (env + 1, 7)]
            for key in members[5]:
                streams[key] = PublicStream(7)
                grow(streams[key], rng, 3)
            env += 2
        if step == 9:                       # an in-place weight update clears identity 1
            with torch.no_grad():
                policies[1].stream.layers[0].linear1.bias.add_(0.1)
        if step == 11:                      # a stream that exactly fills one chunk
            key = members[4][0]
            grow(streams[key], rng, 16 - (streams[key].prefix + 1 - pair.fast[4].entries[key].length))
        plan, memory, expected = pair.encode(groups_for(streams, members))
        fallbacks.append(len(plan.fallback))
        assert memory.shape == expected.shape and memory.dtype == torch.float32
        worst = max(worst, float((memory - expected).abs().max()))
        worst = max(worst, pair.compare_entries())
        for key in streams:                 # mixed growth, including none
            grow(streams[key], rng, int(rng.integers(0, 3)))
    print(f"max |merged - per-identity| memory/KV {device} stream_norm={stream_norm} "
          f"triton={triton}: {worst:.3e}")
    assert worst < ATOL
    # Step 0 (29/30 new tokens) and step 9 (rebuild after clear) fall back;
    # growth across the 32 bucket (29 -> 33) happened on the merged path.
    assert fallbacks[0] == 2 and fallbacks[9] == 1 and sum(fallbacks[1:9]) == 0
    assert sum(fallbacks[10:]) == 0
    metrics = pair.encoders.metrics()
    assert metrics["fallbacks"] == sum(fallbacks) and metrics["merged_calls"] == 14
    assert set(pair.encoders.slots) == {1, 2, 4, 5}
    assert pair.encoders.slots[5].index == freed     # identity 3's freed slot
    return pair, worst


@pytest.mark.parametrize("stream_norm,max_pack_bytes", [(False, 128 << 20), (True, 1)])
def test_merged_encode_matches_per_identity_caches(device, stream_norm, max_pack_bytes):
    # max_pack_bytes=1: one entry per attention chunk.
    pair, worst = unit_scenario(device, stream_norm, max_pack_bytes=max_pack_bytes)
    OBSERVED[("unit", device, stream_norm)] = worst
    metrics = pair.encoders.metrics()
    assert metrics["triton_launches"] == 0
    layers = next(iter(pair.policies.values())).config.layers
    one_chunk = metrics["merged_calls"] * layers
    assert (metrics["pack_chunks"] == one_chunk) == (max_pack_bytes > 1)
    assert metrics["pack_chunks"] >= one_chunk


class EmulatedUpdateAndPack:
    """``_update_and_pack`` in PyTorch on CPU, following the kernel's raw-pointer
    arithmetic: table rows, element strides, ROW_BASE, own-capacity offsets and
    load-before-store. Entry tensors are found by address in ``registry()``."""

    def __init__(self, registry):
        self.registry = registry
        self.launches = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(grid, *args, **kwargs)

    def run(self, grid, NEW_K, NEW_V, PACKED_K, PACKED_V, POINTERS, RANGES, BATCH, ROW_BASE,
            CAPACITY, LAYER, STACK_MODE, K0, K1, K2, K3, V0, V1, V2, V3, *, HEADS, DEPTH,
            BLOCK, num_warps):
        self.launches.append((grid, BATCH, ROW_BASE, CAPACITY, LAYER, HEADS, DEPTH))
        assert grid[0] == -(-HEADS * CAPACITY * DEPTH // BLOCK)
        pointers, ranges = POINTERS.reshape(-1), RANGES.reshape(-1)
        assert POINTERS.is_contiguous() and RANGES.is_contiguous()
        by_address = self.registry()
        new_k = torch.as_strided(NEW_K, (grid[1], HEADS, NEW_K.shape[2], DEPTH), (K0, K1, K2, K3),
                                 NEW_K.storage_offset())
        new_v = torch.as_strided(NEW_V, (grid[1], HEADS, NEW_V.shape[2], DEPTH), (V0, V1, V2, V3),
                                 NEW_V.storage_offset())
        index = torch.arange(HEADS * CAPACITY * DEPTH)
        head, position, depth = index // (CAPACITY * DEPTH), index // DEPTH % CAPACITY, index % DEPTH
        loaded = []
        for row in range(grid[1]):
            table_row = ROW_BASE + row
            own_capacity = int(pointers[table_row])
            start, count = int(ranges[table_row]), int(ranges[BATCH + table_row])
            entry_k = by_address[int(pointers[(1 + 2 * LAYER) * BATCH + table_row])].view(-1)
            entry_v = by_address[int(pointers[(2 + 2 * LAYER) * BATCH + table_row])].view(-1)
            fresh = (position >= start) & (position < start + count)
            limit = own_capacity if STACK_MODE else start + count
            old = (position < limit) & ~fresh
            own = (head * own_capacity + position) * DEPTH + depth
            source = (head, (position - start).clamp(0, new_k.shape[2] - 1), depth)  # masked
            k_new = torch.where(fresh, new_k[row][source], 0.0)
            v_new = torch.where(fresh, new_v[row][source], 0.0)
            k_old = torch.where(old, entry_k[own.clamp(max=entry_k.numel() - 1)], 0.0)
            v_old = torch.where(old, entry_v[own.clamp(max=entry_v.numel() - 1)], 0.0)
            loaded.append((entry_k, entry_v, own[fresh], k_new[fresh], v_new[fresh],
                           torch.where(fresh, k_new, k_old), torch.where(fresh, v_new, v_old)))
        # Every program loads before any program stores: fresh and old
        # positions are disjoint, so this matches the kernel's order.
        for row, (entry_k, entry_v, own, k_new, v_new, k_bits, v_bits) in enumerate(loaded):
            entry_k[own] = k_new
            entry_v[own] = v_new
            PACKED_K[row].view(-1)[:] = k_bits
            PACKED_V[row].view(-1)[:] = v_bits


def test_triton_arguments_with_an_emulated_kernel(monkeypatch):
    # CPU check of what the Triton path hands the kernel: the pointer table
    # (capacity, K/V per layer, memory as layer L), ranges, strides and
    # chunked launches (ROW_BASE), against the per-identity eager caches.
    from train import history_triton_cache as kernels
    holder = {}

    def registry():
        return {t.data_ptr(): t for cache in holder["pair"].fast.values()
                for entry in cache.entries.values()
                for t in (*entry.keys, *entry.values, entry.memory)}
    kernel = EmulatedUpdateAndPack(registry)
    monkeypatch.setattr(kernels, "triton", SimpleNamespace(cdiv=lambda a, b: -(-a // b)))
    monkeypatch.setattr(kernels, "_update_and_pack", kernel, raising=False)
    rows_per_launch = kernels.rows_per_launch
    monkeypatch.setattr(kernels, "rows_per_launch", lambda *args: min(3, rows_per_launch(*args)))
    original = Pair.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        holder["pair"] = self
    monkeypatch.setattr(Pair, "__init__", init)
    pair, worst = unit_scenario("cpu", triton=True, max_pack_bytes=100_000)
    metrics = pair.encoders.metrics()
    assert metrics["triton_launches"] == len(kernel.launches) > 0
    assert metrics["eager_packs"] == 0
    layers = next(iter(pair.policies.values())).config.layers
    assert {launch[4] for launch in kernel.launches} == set(range(layers + 1))
    assert any(launch[2] > 0 for launch in kernel.launches)       # chunked launches
    assert metrics["pack_chunks"] > metrics["merged_calls"] * layers   # chunked attention
    assert all(launch[5:] == (1, 32) for launch in kernel.launches if launch[4] == layers)


def test_encoder_slots_refresh_and_refuse(device):
    policies = players((1, 2, 3), device)
    encoders = SnapshotEncoders(capacity=1)

    def assert_slot(identity, actor):
        index = encoders.slot(identity, actor)
        for name, value in _encoder_values(actor).items():
            assert torch.equal(encoders.tensors[name][index], value), name
        return index

    assert (assert_slot(1, policies[1]), assert_slot(2, policies[2])) == (0, 1)
    writes = encoders.writes
    assert encoders.slot(1, policies[1]) == 0 and encoders.writes == writes
    with torch.no_grad():
        policies[1].stream.layers[1].self_attn.in_proj_weight.add_(1.0)
    assert assert_slot(1, policies[1]) == 0 and encoders.writes == writes + 1
    with torch.no_grad():   # rebound storage, same version counter
        policies[1].bos.data = policies[1].bos.data.clone() + 1
    assert assert_slot(1, policies[1]) == 0
    encoders.retain({2})
    assert assert_slot(3, policies[3]) == 0
    with pytest.raises(ValueError, match="one architecture"):
        encoders.slot(4, fresh_player(config(width=64), 1)[0].to(device))
    hooked = fresh_player(config(), 2)[0].to(device)
    hooked.stream.layers[0].linear1.register_forward_hook(lambda *args: None)
    with pytest.raises(ValueError, match="hooks"):
        validate_encoder(hooked)
    with pytest.raises(ValueError, match="KV cache"):
        rollout.HistoryCollector(make_env(2, 1), policies[1], rollout.MatchEventStore(),
                                 rollout.SequenceRolloutBuffer(), batch_snapshot_encode=True)


def production_size_check(device: str, longest: int, triton: bool) -> tuple[float, ...]:
    """Width 128, 4 layers, 8 heads; ~10 identities x ~8 streams. The first
    encode prefills prefixes up to ``longest`` (per-identity fallback where a
    stream needs more than one chunk), then two merged steps append 1-2 tokens
    per stream. Returns the worst memory/KV, merged-head log-prob (merged vs
    per-identity encode) and log-prob vs each identity's own actor differences."""
    rng = np.random.default_rng(12)
    spec = [(i, int(n), int(n)) for i, n in zip(range(1, 11), [8, 3, 12, 8, 1, 9, 8, 6, 10, 7])]
    step = Step(spec, seed=12, max_candidates=200, longest=longest)
    production = HistoryPolicyConfig(width=128, layers=4, heads=8, response_mode="auxiliary")
    policies = {i: fresh_player(production, 60 + i)[0].to(device) for i, _, _ in spec}
    for actor in policies.values():
        actor.causal_sdpa = True    # the dense reference encode, without a [T, T] mask
    cache_class = BatchedHistoryCache
    if triton:
        from train.history_triton_cache import TritonHistoryCache
        cache_class = TritonHistoryCache
    pair = Pair(policies, device, chunk_size=128, cache_class=cache_class, triton=triton)
    groups = step.groups()
    heads = SnapshotHeads()
    slots = [heads.slot(g.identity, policies[g.identity]) for g in groups]
    worst_memory = worst_logp = worst_reference = 0.0
    for round_ in range(3):
        if round_:
            for stream in step.streams.values():
                grow(stream, rng, int(rng.integers(1, 3)))
            step.prefix = np.asarray([step.streams[(e, m)].prefix
                                      for e, m in zip(step.env_id, step.match_id)], np.int64)
        plan, memory, expected = pair.encode(groups)
        assert bool(plan.fallback) == (round_ == 0)
        assert plan.triton == (triton and bool(plan.entries))
        layout = merged_layout(groups, slots, step.prefix, step.obs, step.seat, step.cand,
                               step.offsets, step.counts, step.env_id, step.match_id)
        fields = {name: torch.as_tensor(array, device=device)
                  for name, array in zip(LAYOUT_FIELDS, layout.arrays)}
        fast = merged_log_probs(heads, layout, fields, memory)
        plain = merged_log_probs(heads, layout, fields, expected)
        worst_memory = max(worst_memory, float((memory - expected).abs().max()),
                           pair.compare_entries())
        worst_logp = max(worst_logp, float((fast - plain).abs().max()))
        worst_reference = max(worst_reference, float(
            (fast - reference(step, policies, groups, device)).abs().max()))
    metrics = pair.encoders.metrics()
    assert metrics["merged_calls"] >= 2 and metrics["merged_streams"] >= 2 * len(step.streams)
    assert (metrics["triton_launches"] > 0) == triton
    print(f"production-size {device} (prefix <= {longest}, triton={triton}): memory/KV "
          f"{worst_memory:.3e}, log-prob vs per-identity encode {worst_logp:.3e}, "
          f"vs per-identity actor {worst_reference:.3e}")
    return worst_memory, worst_logp, worst_reference


def test_production_size_log_probs():
    # CPU at prefixes up to 300; tests/test_history_snapshot_encode_cuda.py runs
    # the ~2,000-token CUDA gate (Triton and eager).
    worst_memory, worst_logp, worst_reference = production_size_check("cpu", 300, False)
    OBSERVED[("production", "cpu")] = (worst_memory, worst_logp, worst_reference)
    assert worst_memory < ATOL and worst_logp < ATOL and worst_reference < 2e-5


# ---- collector ---------------------------------------------------------------------

def run_collector(device, encode, *, steps=36, identities=(0, 1, 2, 3), seat_policy=None,
                  between=None, chunk_size=None, actor_flags=None, **kwargs):
    policies = players(identities, device)
    for actor in policies.values():
        for name, value in (actor_flags or {}).items():
            setattr(actor, name, value)
    collector = rollout.HistoryCollector(
        make_env(6, 23), policies[0], rollout.MatchEventStore(), rollout.SequenceRolloutBuffer(),
        torch.Generator(device=device).manual_seed(31), device=device,
        seat_policy=seat_policy or (lambda env, match: [0, 1 + env % 3, 2, 1 + (env + match) % 3]),
        resolve_policy=policies.__getitem__, record_choices=True, kv_cache=True,
        batch_snapshot_policies=True, batch_snapshot_encode=encode, **kwargs)
    if chunk_size is not None:
        original = collector._cache

        def cache(identity, actor):
            result = original(identity, actor)
            result.chunk_size = chunk_size
            return result
        collector._cache = cache
    stats = []
    for chunk in range(3):
        stats.append(collector.collect(steps // 3))
        if between is not None:
            between(chunk, policies, collector)
    return collector, stats


def recording(monkeypatch):
    tables = []

    def record(*args):
        result = merged_log_probs(*args)
        tables.append(result.cpu())
        return result
    monkeypatch.setattr(history_snapshot_batch, "merged_log_probs", record)
    return tables


def compare_caches(plain, fast):
    worst = 0.0
    assert set(plain.caches) == set(fast.caches)
    for identity, a_cache in plain.caches.items():
        b_cache = fast.caches[identity]
        assert set(a_cache.entries) == set(b_cache.entries)
        assert a_cache.encoded_tokens == b_cache.encoded_tokens
        for key, a in a_cache.entries.items():
            b = b_cache.entries[key]
            assert (a.length, a.capacity) == (b.length, b.capacity)
            for x, y in zip((*a.keys, *a.values, a.memory), (*b.keys, *b.values, b.memory)):
                worst = max(worst, float((x - y).abs().max()))
    return worst


def run_pair(monkeypatch, device, **kwargs):
    runs, tables = [], []
    for encode in (False, True):
        tables.append(recording(monkeypatch))
        runs.append(run_collector(device, encode, **kwargs))
    return runs, tables


def test_collector_merged_encode_matches_per_identity(monkeypatch, device):
    ((plain, plain_stats), (fast, fast_stats)), (plain_tables, fast_tables) = run_pair(
        monkeypatch, device, profile=True)
    assert_same_collection(plain, fast)
    assert len(plain_tables) == len(fast_tables) > 0
    worst = max(float((a - b).abs().max()) for a, b in zip(plain_tables, fast_tables))
    worst_cache = compare_caches(plain, fast)
    OBSERVED[("collector", device)] = (worst, worst_cache)
    print(f"collector {device}: snapshot log-prob {worst:.3e}, cache K/V/memory {worst_cache:.3e}")
    assert worst < ATOL and worst_cache < ATOL
    for before, after in zip(plain_stats, fast_stats):
        same = lambda s: {k: v for k, v in asdict(s).items() if k not in rollout.PROFILE_STATS}
        assert same(before) == same(after)
    metrics = fast.snapshot_encoder_metrics()
    assert metrics["merged_calls"] == len(fast_tables) and metrics["fallbacks"] == 0
    assert metrics["slots"] and metrics["bytes"] > 0
    assert plain.snapshot_encoders is None and plain.snapshot_encoder_metrics() == {}
    # The learner's own cache never goes through the merged encode.
    assert fast.caches[0].encoded_tokens == plain.caches[0].encoded_tokens


def test_collector_fallback_and_snapshot_changes(monkeypatch, device):
    # Tiny prefill chunks, identities joining and leaving, one snapshot's
    # encoder updated in place (its cache clears and rebuilds through the
    # fallback) and one replaced by a new actor object between collects.
    def seats(env, match):
        base = 1 + (env + 2 * match) % 5
        return [0, base, 1 + (base % 5), 1 + ((base + 2) % 5)]

    def between(chunk, policies, collector):
        with torch.no_grad():
            if chunk == 0:
                policies[1].stream.layers[0].self_attn.in_proj_bias.add_(0.05)
                policies[3].stream_norm.weight.mul_(1.1)
            if chunk == 1:
                policies[2] = fresh_player(config(), 77)[0].to(device)

    ((plain, _), (fast, _)), (plain_tables, fast_tables) = run_pair(
        monkeypatch, device, steps=60, seat_policy=seats, between=between, chunk_size=3,
        identities=range(6))
    assert_same_collection(plain, fast)
    worst = max(float((a - b).abs().max()) for a, b in zip(plain_tables, fast_tables))
    assert worst < ATOL and compare_caches(plain, fast) < ATOL
    metrics = fast.snapshot_encoder_metrics()
    assert metrics["fallbacks"] > 0 and metrics["merged_calls"] > 0
    active = {int(i) for seats_ in fast.assignments.values() for i in seats_} - {0}
    assert set(fast.snapshot_encoders.slots) <= active


def test_collector_switches_between_collects(device):
    def between(chunk, policies, collector):
        collector.batch_snapshot_encode = chunk == 0      # on, off, on
    reference_run = run_collector(device, False)[0]
    switched = run_collector(device, True, between=between)[0]
    assert_same_collection(reference_run, switched)


def _top_level_aten(event, excluded, found):
    """Count the outermost aten ops below ``event``; ``excluded`` labels are
    counted separately in ``found`` instead of being descended into."""
    total = 0
    for child in event.cpu_children:
        if child.name in excluded:
            found[child.name] = found.get(child.name, 0) + 1
        elif child.name.startswith("aten::"):
            total += 1
        else:
            total += _top_level_aten(child, excluded, found)
    return total


def test_merged_encode_cuts_op_count(monkeypatch, capsys):
    # CPU op-count proxy for kernel launches: one collector step with ten
    # snapshot identities of about eight streams each, production architecture.
    # "encode" counts the outermost aten ops of the snapshot public encode; the
    # K/V write-and-pack and memory write (per-entry copies on CPU, one Triton
    # launch per layer on CUDA) are counted as regions instead, in "packs".
    from torch.autograd.profiler import record_function
    from torch.profiler import ProfilerActivity, profile
    from train import history_inference, history_snapshot_encode

    def labelled(owner, name, label):
        original = getattr(owner, name)

        def wrapper(*args, **kwargs):
            with record_function(label):
                return original(*args, **kwargs)
        monkeypatch.setattr(owner, name, wrapper)

    labelled(rollout.HistoryCollector, "_snapshot_memory", "snapshot_encode")
    labelled(history_snapshot_encode, "merged_encode", "snapshot_encode")
    for owner, name in ((history_inference.BatchedHistoryCache, "_pack_keys_values"),
                        (history_snapshot_encode, "_eager_pack"),
                        (history_snapshot_encode, "_eager_memory")):
        labelled(owner, name, "pack")
    production = HistoryPolicyConfig(width=128, layers=4, heads=8, response_mode="auxiliary")
    policies = {i: fresh_player(production, 90 + i)[0] for i in range(11)}
    counts = {}
    for encode in (False, True):
        collector = rollout.HistoryCollector(
            make_env(112, 5), policies[0], rollout.MatchEventStore(),
            rollout.SequenceRolloutBuffer(), torch.Generator().manual_seed(3),
            seat_policy=lambda env, match: [0, 1 + env % 10, 1 + (env + 3) % 10,
                                            1 + (env + 7) % 10],
            resolve_policy=policies.__getitem__, kv_cache=True, batch_snapshot_policies=True,
            batch_snapshot_encode=encode)
        collector.collect(6)
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            collector.collect(1)
        events = prof.events()
        step = sum(1 for e in events if e.name.startswith("aten::") and (
            e.cpu_parent is None or not e.cpu_parent.name.startswith("aten::")))
        regions = [e for e in events if e.name == "snapshot_encode"]
        found = {}
        inside = sum(_top_level_aten(e, {"pack"}, found) for e in regions)
        packs = sum(_top_level_aten(e, set(), {}) for e in events if e.name == "pack")
        counts[encode] = dict(step=step, encode=inside, pack_regions=found.get("pack", 0),
                              pack_ops=packs)
    OBSERVED["op_count"] = counts
    with capsys.disabled():
        print(f"\nouter aten ops, one step, ~10 identities x ~8 streams: "
              f"encode off {counts[False]}, on {counts[True]}")
    # Each identity's encode ran L pack regions; the merged encode runs L + 1.
    assert counts[True]["pack_regions"] == production.layers + 1
    assert counts[True]["encode"] < 0.2 * counts[False]["encode"]
    assert counts[True]["step"] < counts[False]["step"]


# ---- trainer -------------------------------------------------------------------------

def test_cli_config_and_resume(tmp_path):
    args = build_parser().parse_args(["--output", str(tmp_path), "--batch-snapshot-policies",
                                      "--rollout-kv-cache", "--batch-snapshot-encode"])
    assert config_from_args(args).batch_snapshot_encode
    assert not config_from_args(build_parser().parse_args(["--output", "x"])).batch_snapshot_encode
    with pytest.raises(ValueError, match="batch_snapshot_policies"):
        HistoryPPOConfig(batch_snapshot_encode=True, rollout_kv_cache=True)
    with pytest.raises(ValueError, match="rollout_kv_cache"):
        HistoryPPOConfig(batch_snapshot_encode=True, batch_snapshot_policies=True)
    HistoryPPOConfig(batch_snapshot_encode=True, rollout_kv_cache=True,
                     batch_snapshot_policies_schedule="1:off,1:on")
    base = dict(width=16, layers=1, heads=4, num_envs=2, steps_per_update=40, epochs=1,
                minibatch_matches=1, snapshot_updates=1, seed=5, causal_sdpa=True,
                rollout_kv_cache=True, batch_snapshot_policies=True)
    trainer = HistoryTrainer(HistoryPPOConfig(**base), tmp_path / "start")
    trainer.update()
    checkpoint = trainer.save()
    resumed = HistoryTrainer(HistoryPPOConfig(updates=3), tmp_path / "resumed", resume=checkpoint,
                             resume_overrides=parse_resume_overrides(
                                 ["batch_snapshot_encode=true"]))
    assert resumed.config_changes[-1] == dict(
        at_update=1, changes={"batch_snapshot_encode": [False, True]})
    line = resumed.update()
    assert line["rollout_batch_snapshot_encode"] is True
    manifest = (tmp_path / "resumed" / "manifest.json").read_text()
    assert '"batch_snapshot_encode": true' in manifest


def test_trainer_with_merged_encode_trains_identically(tmp_path):
    base = dict(width=16, layers=2, heads=4, num_envs=4, steps_per_update=30, epochs=1,
                minibatch_matches=1, snapshot_updates=1, seed=5, causal_sdpa=True,
                rollout_kv_cache=True, snapshot_probability=1.0, batch_snapshot_policies=True)
    start = HistoryTrainer(HistoryPPOConfig(**base), tmp_path / "start")
    for _ in range(3):
        start.update()
    checkpoint = start.save()
    trainers, lines = {}, {}
    for name, encode in (("plain", "false"), ("merged", "true")):
        trainer = HistoryTrainer(HistoryPPOConfig(updates=6), tmp_path / name, resume=checkpoint,
                                 resume_overrides=parse_resume_overrides(
                                     [f"batch_snapshot_encode={encode}"]))
        lines[name] = [trainer.update() for _ in range(3)]
        trainers[name] = trainer
    assert all(line["snapshot_encoders"].get("merged_calls") for line in lines["merged"])
    assert not any(line["snapshot_encoders"] for line in lines["plain"])
    for before, after in zip(lines["plain"], lines["merged"]):
        assert ({k: v for k, v in before.items() if k not in TIMING_KEYS}
                == {k: v for k, v in after.items() if k not in TIMING_KEYS})
        for name in ("cache", "collection_cache"):
            same = lambda line: {k: v for k, v in line[name].items() if k != "bytes"}
            assert same(before) == same(after)
    for a, b in zip(trainers["plain"].actor.parameters(), trainers["merged"].actor.parameters()):
        assert torch.equal(a, b)
    assert torch.equal(trainers["plain"].generator.get_state(),
                       trainers["merged"].generator.get_state())


@pytest.mark.parametrize("encode", [False, True])
def test_merged_arm_still_releases_snapshot_graphs(encode):
    # The merged arm frees snapshot private graphs at collect, encode on or off.
    released = []
    collector = run_collector("cpu", encode, steps=0)[0]
    collector.decision_graphs[2] = SimpleNamespace(clear=lambda: released.append(2))
    collector.collect(1)
    assert released == [2] and 2 not in collector.decision_graphs
    assert (collector.snapshot_encoders is not None) == encode
