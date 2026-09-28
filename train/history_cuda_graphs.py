"""Bounded FP32 private-decision CUDA graphs for full-history rollout.

Opt in through HistoryCollector. The collector owns caches separately from
actors and checkpoints. Graphs retain parameter storage across in-place weight
updates; public KV histories still require normal learner invalidation.
Sampling and the complete candidate head remain ordinary eager operations.

Calls are serialized on one CUDA stream. Autocast and forward hooks are rejected.
Memory limits cover retained input storage and native graph pools. Capture may
transiently exceed that limit; an oversized graph is discarded immediately.
"""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from train.history_model import DecisionInputs, HistoryActor, segment_log_softmax


def tensor_layout(value: Tensor) -> tuple:
    # Preserve layout and pointer alignment, including packed upload views.
    return (tuple(value.shape), tuple(value.stride()), value.dtype,
            value.device, value.data_ptr() % 256)


def static_copy(value: Tensor) -> Tensor:
    if any(stride < 0 for stride in value.stride()):
        raise ValueError("negative input strides are unsupported")
    if any(size > 1 and stride == 0 for size, stride in zip(value.shape, value.stride())):
        raise ValueError("overlapping expanded inputs are unsupported")
    span = 1 + sum((size - 1) * stride for size, stride in zip(value.shape, value.stride()))
    if value.numel() == 0:
        span = 0
    backing = torch.empty(span + 256 // value.element_size(), dtype=value.dtype, device=value.device)
    offset = ((value.data_ptr() - backing.data_ptr()) % 256) // value.element_size()
    result = backing.as_strided(value.shape, value.stride(), offset)
    result.copy_(value)
    if tensor_layout(result) != tensor_layout(value):
        raise RuntimeError("static input layout/alignment does not match original")
    return result


def graph_pool_bytes(graph: torch.cuda.CUDAGraph, device: torch.device) -> int:
    """Count reserved bytes in this graph's own allocator pool, not a delta."""
    pool = tuple(graph.pool())
    snapshot = torch.cuda.memory_snapshot()
    if not all("segment_pool_id" in segment for segment in snapshot):
        raise RuntimeError("allocator snapshot cannot identify graph pools")
    segments = [segment for segment in snapshot
                if tuple(segment["segment_pool_id"]) == pool
                and segment["device"] == device.index]
    if not segments:
        raise RuntimeError("graph pool accounting unavailable; refuse unbounded caching")
    return sum(segment["total_size"] for segment in segments)


def _callable_identity(value):
    return id(getattr(value, "__func__", value))


def _constant(value):
    return value is None or isinstance(value, (bool, int, float, str)) or (
        isinstance(value, tuple) and all(_constant(item) for item in value))


def _storage_identity(name, value):
    storage = value.untyped_storage()
    return (name, id(value), value.data_ptr(), storage.data_ptr(), storage.nbytes(),
            value.storage_offset(), tuple(value.shape), tuple(value.stride()),
            value.dtype, value.device)


def storage_signature(actor):
    """Identity of captured storage, operations and backend, excluding values.

    Parameter versions intentionally do not participate. Scalar module settings
    include LayerNorm epsilon and activation options, which can change operations
    without replacing parameters. Module objects/functions remain identified too.
    """
    from torch.nn.modules import module as module_registry
    if module_registry._global_forward_hooks or module_registry._global_forward_pre_hooks:
        raise ValueError("private graphs require actors without global forward hooks")
    modules = []
    for name, module in actor.named_modules():
        if module._forward_hooks or module._forward_pre_hooks:
            raise ValueError("private graphs require actors without forward hooks")
        constants = tuple(sorted((key, value) for key, value in vars(module).items()
                                 if _constant(value)))
        operations = tuple(sorted((key, _callable_identity(value))
                                  for key, value in vars(module).items()
                                  if name and callable(value)))
        modules.append((name, id(module), type(module), module.training,
                        _callable_identity(module.forward), constants, operations))
    parameters = tuple(_storage_identity(name, value) for name, value in actor.named_parameters())
    buffers = tuple(_storage_identity(name, value) for name, value in actor.named_buffers())
    private_methods = tuple((name, _callable_identity(getattr(actor, name))) for name in
                            ("decision_states", "_attend", "_attend_independent", "_project_independent"))
    backend = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
               torch.backends.cudnn.allow_tf32, torch.backends.cuda.flash_sdp_enabled(),
               torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled(),
               torch.backends.cuda.cudnn_sdp_enabled(), torch.backends.mha.get_fastpath_enabled(),
               torch.are_deterministic_algorithms_enabled(),
               torch.is_deterministic_algorithms_warn_only_enabled(),
               torch.is_inference_mode_enabled())
    # The root module id also works when an external registry passes a weak proxy.
    return (modules[0][1], actor.config, tuple(modules), parameters, buffers, private_methods, backend)


def storage_references(actor):
    # Detached aliases retain the actual captured storages independently of the
    # Parameter objects. Holding the Parameter alone is insufficient if a caller
    # later replaces parameter.data or uses set_ to rebind its storage.
    return tuple(value.detach() for value in (*actor.parameters(), *actor.buffers()))


def version_signature(actor):
    return tuple((id(value), value._version) for value in (*actor.parameters(), *actor.buffers()))


@dataclass
class PrivateEntry:
    graph: Any
    buffers: tuple[Tensor, ...]
    inputs: DecisionInputs
    output: Tensor
    parameter_refs: tuple[Tensor, ...]
    bytes: int
    input_bytes: int
    pool_bytes: int


def interval_backend_signature():
    # Must cover the V4 backend tuple without traversing model metadata.
    return (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32, torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(), torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled(), torch.backends.mha.get_fastpath_enabled(),
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.is_inference_mode_enabled())


class PrivateDecisionGraphs:
    """Exact-shape LRU; current weights are read through retained storage."""
    retains_inplace_updates = True
    supports_collection_intervals = True

    def __init__(self, actor: HistoryActor, *, max_entries: int = 32,
                 max_bytes: int = 128 << 20, warmup: int = 3,
                 admit_after: int = 3, max_observed_shapes: int = 256,
                 integer_alignment_agnostic: bool = True):
        if min(max_entries, max_bytes, warmup, admit_after, max_observed_shapes) < 1:
            raise ValueError("graph cache limits and warmup must be positive")
        self._interval_active = False
        self._interval_backend = None
        self.actor = actor
        self.device = next(actor.parameters()).device
        if self.device.type != "cuda" or actor.config.window:
            raise ValueError("private graphs require a CUDA full-history actor")
        if torch.is_autocast_enabled() or any(p.dtype != torch.float32 for p in actor.parameters()):
            raise ValueError("private graphs require FP32 with autocast disabled")
        self.max_entries, self.max_bytes, self.warmup = max_entries, max_bytes, warmup
        self.admit_after, self.max_observed_shapes = admit_after, max_observed_shapes
        self.integer_alignment_agnostic = bool(integer_alignment_agnostic)
        self.entries = OrderedDict()
        self.observed = OrderedDict()
        self.signature = self._parameter_signature()
        self._versions = version_signature(actor)
        self.stream = torch.cuda.current_stream(self.device)
        self.side = torch.cuda.Stream(device=self.device)
        self.stats = dict(hits=0, misses=0, captures=0, evictions=0, invalidations=0,
                          budget_fallbacks=0, size_skips=0, eager_fallbacks=0, admission_fallbacks=0,
                          observed_evictions=0, inplace_refreshes=0, intervals=0, interval_ends=0,
                          interval_forwards=0, full_refreshes=0)

    @property
    def bytes(self):
        return sum(entry.bytes for entry in self.entries.values())

    def set_memory_budget(self, max_bytes: int) -> None:
        if max_bytes < 1:
            raise ValueError("graph memory budget must be positive")
        if max_bytes > self.max_bytes:
            # A shape rejected while another policy occupied the shared budget
            # may fit after that policy is pruned.
            for key in list(self.observed):
                if self.observed[key] < 0:
                    del self.observed[key]
        self.max_bytes = max_bytes
        while self.entries and self.bytes > max_bytes:
            self._evict()

    def metrics(self):
        return dict(self.stats, entries=len(self.entries), bytes=self.bytes,
                    max_entries=self.max_entries, max_bytes=self.max_bytes,
                    input_bytes=sum(e.input_bytes for e in self.entries.values()),
                    pool_bytes=sum(e.pool_bytes for e in self.entries.values()),
                    observed_shapes=len(self.observed), capture_scope="private_only",
                    parameter_update_policy="retain_same_storage", interval_active=self._interval_active)

    def _parameter_signature(self):
        return storage_signature(self.actor)

    def _evict(self) -> None:
        torch.cuda.synchronize(self.device)
        _, entry = self.entries.popitem(last=False)
        entry.graph.reset()
        self.stats["evictions"] += 1

    def clear(self) -> None:
        self.end_interval()
        if self.entries:
            torch.cuda.synchronize(self.device)
            for entry in self.entries.values():
                entry.graph.reset()
            self.entries.clear()
        self.observed.clear()

    def _full_refresh(self):
        if torch.cuda.current_stream(self.device) != self.stream:
            raise ValueError("graph replay must stay on its original CUDA stream")
        if torch.is_autocast_enabled():
            raise ValueError("private graphs require CUDA autocast disabled")
        signature = self._parameter_signature()
        versions = version_signature(self.actor)
        if signature != self.signature:
            self.clear()  # Synchronizes before releasing graphs and old storage aliases.
            self.signature = signature
            self.stats["invalidations"] += 1
        elif versions != self._versions:
            self.stats["inplace_refreshes"] += 1
        self._versions = versions
        self.stats["full_refreshes"] += 1

    def _values(self, inputs: DecisionInputs, encoded: Tensor) -> tuple[Tensor, ...]:
        values = (encoded, inputs.obs, inputs.seat, inputs.prefix)
        if any(value.device != self.device for value in values):
            raise ValueError("all private graph inputs must be on the actor's CUDA device")
        if encoded.dtype != torch.float32:
            raise ValueError("encoded public memory must remain FP32")
        return values

    def _eager(self, inputs: DecisionInputs, encoded: Tensor) -> Tensor:
        return self.actor.decision_states(encoded, inputs)

    def _capture(self, values: tuple[Tensor, ...]) -> PrivateEntry:
        buffers = tuple(static_copy(value) for value in values)
        encoded, obs, seat, prefix = buffers
        # The selected full-history layout never reads stream metadata,
        # match_index or candidate fields. Do not allocate or capture them.
        inputs = DecisionInputs(streams=None, match_index=None, prefix=prefix,
                                obs=obs, seat=seat, cand=None, offsets=None,
                                candidate_rows=None, one_decision_per_stream=True)
        self.side.wait_stream(self.stream)
        with torch.cuda.stream(self.side):
            for _ in range(self.warmup):
                self._eager(inputs, encoded)
        self.stream.wait_stream(self.side)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=self.side):
                output = self._eager(inputs, encoded)
            graph.replay()
            torch.cuda.synchronize(self.device)
            pool_bytes = graph_pool_bytes(graph, self.device)
            input_bytes = sum(value.untyped_storage().nbytes() for value in buffers)
            parameter_refs = storage_references(self.actor)
            return PrivateEntry(graph, buffers, inputs, output, parameter_refs,
                                input_bytes + pool_bytes, input_bytes, pool_bytes)
        except Exception:
            torch.cuda.synchronize(self.device)
            graph.reset()
            raise

    def minimum_entry_bytes(self, values: tuple[Tensor, ...]) -> int:
        """Lower bound of a captured entry: its static input copies plus the
        ``kv_proj(encoded)`` activation ([N, S, 2 * width] FP32), which must be
        allocated inside the graph's private pool during capture."""
        encoded = values[0]
        inputs = sum(value.numel() * value.element_size() for value in values)
        keys_values = (encoded.shape[0] * encoded.shape[1] * 2 * self.actor.config.width
                       * encoded.element_size())
        return inputs + keys_values

    def _key(self, values: tuple[Tensor, ...]) -> tuple:
        layouts = []
        for index, value in enumerate(values):
            layout = tensor_layout(value)
            eligible = ((index == 1 and value.dtype == torch.uint8)
                        or (index in (2, 3) and value.dtype == torch.int64))
            if self.integer_alignment_agnostic and eligible:
                layout = (*layout[:-1], None)
            layouts.append(layout)
        return tuple(layouts)

    def _observe(self, key: tuple) -> int:
        previous = self.observed.pop(key, 0)
        count = previous + 1 if previous >= 0 else previous
        self.observed[key] = count
        while len(self.observed) > self.max_observed_shapes:
            self.observed.popitem(last=False)
            self.stats["observed_evictions"] += 1
        return count

    @torch.no_grad()
    def forward(self, inputs: DecisionInputs, encoded: Tensor, *,
                copy_outputs: bool = True) -> Tensor:
        self._refresh()
        if not inputs.one_decision_per_stream:
            self.stats["eager_fallbacks"] += 1
            return self._eager(inputs, encoded)
        values = self._values(inputs, encoded)
        if encoded.shape[0] != inputs.decisions:
            raise ValueError("one-decision-per-stream layout does not match memory")
        key = self._key(values)
        entry = self.entries.get(key)
        if entry is None:
            self.stats["misses"] += 1
            if self.minimum_entry_bytes(values) > self.max_bytes:
                # Cannot fit: skip before admission, warm-up and capture.
                self.stats["budget_fallbacks"] += 1
                self.stats["size_skips"] = self.stats.get("size_skips", 0) + 1
                return self._eager(inputs, encoded)
            count = self._observe(key)
            if count < 0:
                self.stats["budget_fallbacks"] += 1
                return self._eager(inputs, encoded)
            if count < self.admit_after:
                self.stats["admission_fallbacks"] += 1
                return self._eager(inputs, encoded)
            while len(self.entries) >= self.max_entries:
                self._evict()
            entry = self._capture(values)
            self.stats["captures"] += 1
            if entry.bytes > self.max_bytes:
                entry.graph.reset()
                # Avoid recapturing a known oversized shape on every call.
                # The negative marker lives in the same bounded LRU.
                self.observed[key] = -1
                self.stats["budget_fallbacks"] += 1
                return self._eager(inputs, encoded)
            while self.entries and self.bytes + entry.bytes > self.max_bytes:
                self._evict()
            self.entries[key] = entry
            self.observed.pop(key, None)
        else:
            self.stats["hits"] += 1
            self.entries.move_to_end(key)
            for destination, source in zip(entry.buffers, values):
                destination.copy_(source)
            entry.graph.replay()
        return entry.output.clone() if copy_outputs else entry.output

    @torch.no_grad()
    def log_probs(self, inputs: DecisionInputs, encoded: Tensor) -> Tensor:
        state = self.forward(inputs, encoded, copy_outputs=False)
        rows = inputs.rows
        logits = self.actor.candidate_logits(state, inputs.cand, inputs.offsets, rows=rows)
        return segment_log_softmax(logits, rows, inputs.decisions)

    def _refresh(self):
        if not self._interval_active:
            self._full_refresh()
            return
        if torch.cuda.current_stream(self.device) != self.stream:
            raise ValueError("graph replay must stay on its original CUDA stream")
        if torch.is_autocast_enabled():
            raise ValueError("private graph intervals require CUDA autocast disabled")
        if interval_backend_signature() != self._interval_backend:
            raise RuntimeError("numerical backend changed during a frozen collection interval")
        self.stats["interval_forwards"] += 1

    @torch.no_grad()
    def refresh(self):
        if self._interval_active:
            raise RuntimeError("end the collection interval before a full refresh")
        self._full_refresh()

    @torch.no_grad()
    def begin_interval(self):
        """Validate once, then require frozen actor state until end_interval()."""
        if self._interval_active:
            raise RuntimeError("nested collection intervals are unsupported")
        self._full_refresh()
        self._interval_backend = interval_backend_signature()
        self._interval_active = True
        self.stats["intervals"] += 1

    @torch.no_grad()
    def end_interval(self):
        """Idempotent cleanup, including failed collection and pruned policies."""
        if self._interval_active:
            self.stats["interval_ends"] += 1
        self._interval_active = False
        self._interval_backend = None

    @contextmanager
    def collection_interval(self):
        self.begin_interval()
        try:
            yield self
        finally:
            self.end_interval()
