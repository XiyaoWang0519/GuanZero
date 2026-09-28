"""Production graph lifecycle tests with CPU metadata and fake graph handles.

These exercise invalidation and collector ownership without claiming CUDA kernel
equivalence. Real capture, optimizer and resume acceptance live in the CUDA suite.
"""
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from train import history_cuda_graphs as graphs
from train.history_model import HistoryPolicyConfig, fresh_player
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer


def entry(size=16):
    return SimpleNamespace(graph=SimpleNamespace(reset=Mock()), bytes=size,
                           input_bytes=4, pool_bytes=size - 4)


@pytest.fixture
def cache(monkeypatch):
    # Bypass only CUDA allocation in __init__; all lifecycle methods are real.
    value = object.__new__(graphs.PrivateDecisionGraphs)
    value.actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=1), 7)
    value.device = torch.device("cpu")
    value.stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: value.stream)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    value.signature = graphs.storage_signature(value.actor)
    value._versions = graphs.version_signature(value.actor)
    value._interval_active, value._interval_backend = False, None
    value.max_entries, value.max_bytes, value.warmup = 2, 4096, 1
    value.admit_after, value.max_observed_shapes = 2, 3
    value.integer_alignment_agnostic = True
    value.entries, value.observed = OrderedDict(), OrderedDict()
    value.stats = dict.fromkeys(("hits", "misses", "captures", "evictions", "invalidations",
                                "budget_fallbacks", "eager_fallbacks", "admission_fallbacks",
                                "observed_evictions", "inplace_refreshes", "intervals",
                                "interval_ends", "interval_forwards", "full_refreshes"), 0)
    return value


def test_interval_validates_model_once_and_restores_per_call_validation(cache, monkeypatch):
    assert graphs.interval_backend_signature() == graphs.storage_signature(cache.actor)[-1]
    full = Mock(wraps=cache._parameter_signature)
    monkeypatch.setattr(cache, "_parameter_signature", full)
    with cache.collection_interval():
        for _ in range(4):
            cache._refresh()
        assert full.call_count == 1
        assert cache.stats["interval_forwards"] == 4
    assert not cache._interval_active and cache._interval_backend is None
    cache._refresh()
    assert full.call_count == 2
    assert cache.stats["intervals"] == cache.stats["interval_ends"] == 1


def test_nested_interval_and_explicit_refresh_reject_without_ending_outer(cache):
    with cache.collection_interval():
        with pytest.raises(RuntimeError, match="nested"):
            cache.begin_interval()
        with pytest.raises(RuntimeError, match="before a full refresh"):
            cache.refresh()
        assert cache._interval_active
    assert cache.stats["interval_ends"] == 1


def test_exception_and_clear_end_interval_idempotently(cache):
    old = entry()
    cache.entries["shape"] = old
    cache.observed["other"] = 1
    with pytest.raises(LookupError):
        with cache.collection_interval():
            cache.clear()
            raise LookupError("collection failed after pruning")
    assert not cache._interval_active and not cache.entries and not cache.observed
    old.graph.reset.assert_called_once_with()
    cache.end_interval()
    assert cache.stats["interval_ends"] == 1
    cache.actor.output_norm.eps *= 2
    cache._refresh()
    assert cache.stats["invalidations"] == 1


@pytest.mark.parametrize("change", ["add", "load_state_dict"])
def test_same_storage_updates_between_intervals_keep_entries(cache, change):
    old = entry()
    cache.entries["shape"] = old
    with cache.collection_interval():
        cache._refresh()
    with torch.no_grad():
        if change == "add":
            cache.actor.q_proj.weight.add_(0.001)
        else:
            state = {key: value.detach().clone() for key, value in cache.actor.state_dict().items()}
            state["q_proj.weight"].add_(0.001)
            cache.actor.load_state_dict(state)
    with cache.collection_interval():
        assert cache.entries["shape"] is old
        assert cache.stats["invalidations"] == 0
        assert cache.stats["inplace_refreshes"] == 1
    old.graph.reset.assert_not_called()


@pytest.mark.parametrize("change", ["storage", "parameter", "activation", "epsilon", "backend"])
def test_changes_between_intervals_invalidate_entries(cache, change):
    old = entry()
    cache.entries["shape"], cache.observed["other"] = old, 2
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    try:
        with cache.collection_interval():
            cache._refresh()
        if change == "storage":
            cache.actor.q_proj.weight.data = cache.actor.q_proj.weight.detach().clone()
        elif change == "parameter":
            cache.actor.q_proj.weight = torch.nn.Parameter(cache.actor.q_proj.weight.detach())
        elif change == "activation":
            cache.actor.private[1] = torch.nn.Tanh()
        elif change == "epsilon":
            cache.actor.output_norm.eps *= 2
        else:
            torch.use_deterministic_algorithms(not previous[0])
        with cache.collection_interval():
            assert not cache.entries and not cache.observed
            assert cache.stats["invalidations"] == 1
        old.graph.reset.assert_called_once_with()
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize("change", ["stream", "autocast", "backend"])
def test_runtime_guards_survive_fast_interval_path(cache, monkeypatch, active, change):
    if active:
        cache.begin_interval()
    try:
        if change == "stream":
            monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())
            with pytest.raises(ValueError, match="original CUDA stream"):
                cache._refresh()
        elif change == "autocast":
            monkeypatch.setattr(torch, "is_autocast_enabled", lambda: True)
            with pytest.raises(ValueError, match="autocast"):
                cache._refresh()
        elif active:
            monkeypatch.setattr(graphs, "interval_backend_signature", lambda: ("changed",))
            with pytest.raises(RuntimeError, match="backend changed"):
                cache._refresh()
        else:
            # Outside an interval backend changes invalidate instead of rejecting.
            original = cache._parameter_signature()
            monkeypatch.setattr(cache, "_parameter_signature", lambda: (*original[:-1], ("changed",)))
            cache._refresh()
            assert cache.stats["invalidations"] == 1
    finally:
        cache.end_interval()
    assert not cache._interval_active


def test_shrinking_memory_budget_evicts_oldest_entries(cache):
    first, second = entry(100), entry(80)
    cache.entries.update(first=first, second=second)
    cache.observed.update(rejected=-1, promising=2)
    cache.set_memory_budget(90)
    assert list(cache.entries) == ["second"] and cache.bytes == 80
    first.graph.reset.assert_called_once_with()
    second.graph.reset.assert_not_called()
    assert cache.observed == {"rejected": -1, "promising": 2}
    cache.set_memory_budget(70)
    assert cache.bytes == 0 and cache.stats["evictions"] == 2
    second.graph.reset.assert_called_once_with()
    with pytest.raises(ValueError, match="positive"):
        cache.set_memory_budget(0)
    assert cache.max_bytes == 70


def test_growing_memory_budget_retries_only_previously_rejected_shapes(cache):
    cache.observed.update(first=1, rejected=-1, frequent=2, rejected_too=-1)
    cache.set_memory_budget(cache.max_bytes)
    assert len(cache.observed) == 4
    cache.set_memory_budget(cache.max_bytes + 1)
    assert list(cache.observed.items()) == [("first", 1), ("frequent", 2)]


def test_admission_lru_is_bounded_and_retains_oversize_marker(cache):
    assert cache._observe("a") == 1
    assert cache._observe("b") == 1
    assert cache._observe("a") == 2
    cache.observed["b"] = -1
    assert cache._observe("b") == -1
    cache._observe("c")
    cache._observe("d")
    assert list(cache.observed.items()) == [("b", -1), ("c", 1), ("d", 1)]
    assert cache.stats["observed_evictions"] == 1


def test_forward_admission_budget_retry_and_owned_outputs(cache, monkeypatch):
    encoded = torch.zeros(2, 4, 32, requires_grad=True)
    inputs = SimpleNamespace(decisions=2, one_decision_per_stream=True,
                             obs=torch.zeros(2, 8, dtype=torch.uint8),
                             seat=torch.zeros(2, dtype=torch.int64),
                             prefix=torch.ones(2, dtype=torch.int64))
    gradient_modes = []
    def eager(inputs, encoded):
        gradient_modes.append(torch.is_grad_enabled())
        return encoded[:, 0] + 1
    monkeypatch.setattr(cache, "_eager", eager)
    handles = []
    def capture(values):
        buffers = tuple(value.detach().clone() for value in values)
        output = buffers[0][:, 0] + 1
        handle = SimpleNamespace(reset=Mock(), replay=Mock(side_effect=lambda: output.copy_(buffers[0][:, 0] + 1)))
        handles.append(handle)
        return graphs.PrivateEntry(handle, buffers, inputs, output, (), 5000, 1000, 4000)
    monkeypatch.setattr(cache, "_capture", capture)
    first = cache.forward(inputs, encoded)
    assert cache.stats["admission_fallbacks"] == 1
    cache.forward(inputs, encoded)
    cache.forward(inputs, encoded)
    assert cache.stats["captures"] == 1 and not cache.entries
    handles[0].reset.assert_called_once_with()
    assert cache.stats["budget_fallbacks"] == 2
    cache.set_memory_budget(6000)
    cache.forward(inputs, encoded)  # Starts admission again after budget growth.
    owned = cache.forward(inputs, encoded)
    assert cache.stats["captures"] == 2 and len(cache.entries) == 1
    with torch.no_grad():
        encoded.add_(2)
    borrowed = cache.forward(inputs, encoded, copy_outputs=False)
    assert cache.stats["hits"] == 1
    assert torch.equal(owned, first) and torch.equal(borrowed, first + 2)
    assert not first.requires_grad and not borrowed.requires_grad
    assert gradient_modes and not any(gradient_modes)  # Graph API is inference-only.


def test_shapes_that_cannot_fit_skip_before_admission_and_capture(cache, monkeypatch):
    # inputs 2*4*32*4 + 3*16 = 1072 bytes; kv_proj(encoded) = 2*4*64*4 = 2048 bytes.
    encoded = torch.zeros(2, 4, 32)
    inputs = SimpleNamespace(decisions=2, one_decision_per_stream=True,
                             obs=torch.zeros(2, 8, dtype=torch.uint8),
                             seat=torch.zeros(2, dtype=torch.int64),
                             prefix=torch.ones(2, dtype=torch.int64))
    values = cache._values(inputs, encoded)
    assert cache.minimum_entry_bytes(values) == 1072 + 2048
    monkeypatch.setattr(cache, "_eager", lambda inputs, encoded: encoded[:, 0])
    capture = Mock(side_effect=AssertionError("must not capture"))
    monkeypatch.setattr(cache, "_capture", capture)
    cache.max_bytes = 3000         # inputs fit (the old check), inputs + K/V do not
    for _ in range(4):
        cache.forward(inputs, encoded)
    capture.assert_not_called()
    assert cache.stats["size_skips"] == 4 and cache.stats["budget_fallbacks"] == 4
    assert cache.stats["admission_fallbacks"] == 0 and not cache.observed


class FakeGraphs:
    """No CUDA execution; collector must honor the real graph lifecycle API."""
    def __init__(self, actor, **options):
        self.actor, self.options = actor, options
        self.bytes, self.max_bytes = 0, options.get("max_bytes", 128 << 20)
        self.active = False
        self.begins = self.ends = self.clears = self.calls = 0
        self.fail_begin = False
        self.budgets = []

    def begin_interval(self):
        assert not self.active
        if self.fail_begin:
            raise LookupError("begin failed")
        self.active = True
        self.begins += 1

    def end_interval(self):
        if self.active:
            self.ends += 1
        self.active = False

    def clear(self):
        self.end_interval()
        self.bytes = 0
        self.clears += 1

    def set_memory_budget(self, value):
        self.max_bytes = value
        self.budgets.append(value)

    def log_probs(self, inputs, encoded):
        assert not torch.is_grad_enabled()
        self.calls += 1
        return torch.tensor([-0.5])


@pytest.fixture
def collector(monkeypatch):
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=1), 9)
    monkeypatch.setattr(graphs, "PrivateDecisionGraphs", FakeGraphs)
    # Constructor stores the CUDA device but allocates nothing; step is replaced.
    return HistoryCollector(None, actor, MatchEventStore(), SequenceRolloutBuffer(),
                            device="cuda", kv_cache=True, private_graphs=True)


def test_collector_begins_existing_and_lazy_graphs_and_ends_both(collector, monkeypatch):
    existing = FakeGraphs(collector.actor)
    collector.decision_graphs[0] = existing
    original_actor_fields = set(vars(collector.actor))
    def step(stats):
        assert existing.active and collector._graph_collecting
        collector._graph_log_probs(1, collector.actor, None, None)
        assert collector.decision_graphs[1].active
        stats.decisions += 1
    monkeypatch.setattr(collector, "step", step)
    stats = collector.collect(2, version=4)
    assert stats.decisions == 2 and collector.version == 4
    assert not collector._graph_collecting
    for graph in collector.decision_graphs.values():
        assert graph.begins == graph.ends == 1 and not graph.active
    assert collector.decision_graphs[1].calls == 2
    assert set(vars(collector.actor)) == original_actor_fields


@pytest.mark.parametrize("failure", ["step", "begin_existing", "begin_lazy"])
def test_collector_finally_ends_graphs_after_failure(collector, monkeypatch, failure):
    existing = FakeGraphs(collector.actor)
    collector.decision_graphs[0] = existing
    def step(stats):
        collector._graph_log_probs(2, collector.actor, None, None)
        raise LookupError("step failed")
    monkeypatch.setattr(collector, "step", step)
    if failure == "begin_existing":
        other = FakeGraphs(collector.actor)
        other.fail_begin = True
        collector.decision_graphs[1] = other
    elif failure == "begin_lazy":
        def factory(*args, **kwargs):
            other = FakeGraphs(*args, **kwargs)
            other.fail_begin = True
            return other
        monkeypatch.setattr(graphs, "PrivateDecisionGraphs", factory)
    with pytest.raises(LookupError):
        collector.collect(1)
    assert not collector._graph_collecting
    assert all(not graph.active for graph in collector.decision_graphs.values())
    assert existing.begins == existing.ends == 1


def test_collector_prune_and_actor_replacement_end_removed_intervals(collector, monkeypatch):
    old = FakeGraphs(collector.actor)
    stale = FakeGraphs(collector.actor)
    collector.decision_graphs.update({0: old, 1: stale})
    replacement, _ = fresh_player(collector.actor.config, 10)
    collector.assignments[(0, 0)] = np.zeros(4, np.int64)
    def step(stats):
        collector._prune_caches()
        assert not stale.active and stale.clears == 1
        collector._graph_log_probs(0, replacement, None, None)
        assert not old.active and old.clears == 1
        assert collector.decision_graphs[0].active
    monkeypatch.setattr(collector, "step", step)
    collector.collect(1)
    new = collector.decision_graphs[0]
    assert new.actor is replacement
    assert old.begins == old.ends == stale.begins == stale.ends == 1
    assert new.begins == new.ends == 1


def test_collector_nested_collect_rejects_and_outer_finally_ends(collector, monkeypatch):
    graph = FakeGraphs(collector.actor)
    collector.decision_graphs[0] = graph
    def step(stats):
        with pytest.raises(RuntimeError, match="nested"):
            collector.collect(1)
        assert graph.active and collector._graph_collecting
    monkeypatch.setattr(collector, "step", step)
    collector.collect(1)
    assert graph.begins == graph.ends == 1


def test_collector_shared_budget_fallback_and_per_policy_cap(collector, monkeypatch):
    collector._graph_log_probs(0, collector.actor, None, None)
    first = collector.decision_graphs[0]
    assert first.max_bytes == 128 << 20 and first.begins == 0
    collector.private_graph_budget_bytes = 200
    first.bytes = 130
    collector._graph_log_probs(1, collector.actor, None, None)
    second = collector.decision_graphs[1]
    assert second.max_bytes == 70
    second.bytes = 70
    eager = Mock(return_value=torch.tensor([-1.0]))
    monkeypatch.setattr(collector.actor, "candidate_log_probs", eager)
    result = collector._graph_log_probs(2, collector.actor, None, None)
    assert torch.equal(result, torch.tensor([-1.0])) and 2 not in collector.decision_graphs
    eager.assert_called_once_with(None, encoded=None)
    collector._graph_log_probs(0, collector.actor, None, None)
    assert first.budgets[-1] == 130  # Exclude its own reservation from availability.


def test_per_policy_cap_follows_configuration(collector):
    collector.private_graph_policy_budget_bytes = 300 << 20
    collector.private_graph_budget_bytes = 1000 << 20
    collector._graph_log_probs(0, collector.actor, None, None)
    assert collector.decision_graphs[0].max_bytes == 300 << 20
    collector.decision_graphs[0].bytes = 900 << 20
    collector._graph_log_probs(1, collector.actor, None, None)
    assert collector.decision_graphs[1].max_bytes == 100 << 20   # total minus others


def test_trainer_config_passes_graph_budgets():
    from train.history_ppo import HistoryPPOConfig, build_parser, config_from_args
    assert HistoryPPOConfig().rollout_graph_policy_budget_mb == 128
    args = build_parser().parse_args(["--output", "x", "--rollout-graph-budget-mb", "2048",
                                      "--rollout-graph-policy-budget-mb", "1024"])
    config = config_from_args(args)
    assert (config.rollout_graph_budget_mb, config.rollout_graph_policy_budget_mb) == (2048, 1024)
    with pytest.raises(ValueError):
        HistoryPPOConfig(rollout_graph_policy_budget_mb=0)


def test_learner_invalidation_keeps_private_graphs_for_next_interval(collector):
    private = FakeGraphs(collector.actor)
    public, snapshot = SimpleNamespace(clear=Mock()), SimpleNamespace(clear=Mock())
    collector.decision_graphs[0] = private
    collector.caches.update({0: public, 1: snapshot})
    collector.invalidate_learner_cache()
    public.clear.assert_called_once_with()
    snapshot.clear.assert_not_called()
    assert collector.decision_graphs[0] is private and private.clears == 0
