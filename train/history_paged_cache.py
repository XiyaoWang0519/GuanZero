"""Paged public KV cache for full-history rollout (opt-in ``rollout_paged_cache``).

``BatchedHistoryCache`` gives every cached match its own K/V and memory
tensors, so each encode call costs a handful of tensor operations per match:
key/value writes, packing copies, memory write-back. On CUDA those are kernel
launches and Python dispatch, and a vector step's encode is host-bound.

``PagedHistoryCache`` keeps the same entries, the same append chunks and the
same attention inputs, but stores every match's public K/V and top-layer
memory in pages of one ``KVPagePool``. Per append it runs a fixed number of
operations whatever the number of matches: one indexed write of the new
tokens' keys and values per layer, one row gather that builds the zero-padded
``[matches, heads, capacity, depth]`` attention operands, one indexed memory
write. The host work per match is list bookkeeping and one token slice.

Exactness. Every floating-point operation (embedding, projections, SDPA,
residuals, norms) is the one ``BatchedHistoryCache._append`` performs, on the
same values: gathered rows past a match's length read the pool's zero page,
exactly the zeros of the per-entry layout. The keys and values reach SDPA as
strided views of the token-major gather rather than contiguous stacks; on CPU
this is bitwise identical (tests/test_history_paged_cache.py). The target
backend must be checked with the same test before a GPU run relies on bitwise
identity; otherwise the difference is reduction-order noise (acceptance tier 2).

``page_span`` (opt-in) pads attention keys and the returned memory only to
the next whole page past the longest stream instead of the next power of two.
Position encodings are unchanged; the SDPA reduction shapes change, so this is
acceptance tier 2 (on CPU most calls stay bitwise, single-token calls do not).

Exactness of the unconditional trims (bitwise on CPU, checked against a frozen
copy of the previous implementation in tests/history_paged_reference.py):
the attention mask is built once per append as the additive float tensor SDPA
would otherwise derive from the boolean mask in every layer; row indices are
expanded page-major; the merged path gathers only queries; pool storage grows
uninitialized; and memory is gathered a whole page at a time. The last two
rest on the pool invariants below: no row is read before it is written,
except page 0 and the memory (and private) rows of an entry's pages past its
length, which are zeroed when the page is allocated. The gathered memory is
therefore the same tensor as before, zero past each stream's length.

Three further opt-in options (``PagedHistoryCache`` arguments, each acceptance
tier 2 and default off): ``span_groups`` attends per end-sorted span group
instead of padding every stream to the batch's longest; ``private_kv`` stores
the learner's private-attention keys/values per token, so a rollout step does
not project the whole history again; ``causal_prefill`` rebuilds streams from
BOS in one causal pass per length group. ``encode_rollout`` returns the
learner's memory for those layouts as a ``history_model.RolloutMemory``.

``attention_kernel`` (opt-in, tier 2, default off) runs every append's
attention as one Triton paged-attention call per layer
(``train.history_paged_attention``) that reads each stream's keys and values
in place from the pool pages up to its own end: no padded gather, no mask,
no span groups. With ``private_kv`` the learner's one-query private attention
reads its stored keys/values the same way. The kernel runs only where it can
(CUDA FP32, or Triton's CPU interpreter, with a head depth and page size it
tiles); anywhere else the switch is a no-op and every path is the gather +
SDPA one, bit for bit.

A pool is shared by the caches built on it: a cache releases its pages on
``clear``/``prune``. Page 0 is never allocated and stays zero. A pool grows by
reallocation and does not shrink on its own; its reserved size is the
high-water mark of live pages. The collector keeps the frozen snapshot
identities on one pool and the learner on its own: the learner's cache is
discarded before every PPO update, and ``KVPagePool.reset`` then returns that
pool's storage instead of holding it through learning. The next allocation
restores the size the pool had, in one step and without a copy.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from train.history_attention import finish
from train.history_inference import BatchedHistoryCache, bucket
from train.history_model import HistoryActor, PublicStream, RolloutMemory, StreamBatch
from train.history_transfers import upload_arrays
from train.logs import TOKEN_DIM

PAGE_TOKENS = 64


class KVPagePool:
    """Token-major page storage: per layer ``[rows, 2, heads, depth]`` keys and
    values, plus ``[rows, width]`` top-layer memory; ``rows = pages * page_tokens``.
    ``private_width`` > 0 adds ``[rows, private_width]`` per-token private
    attention keys/values (``PagedHistoryCache(private_kv=True)``).
    """

    def __init__(self, layers: int, heads: int, width: int, device, dtype=torch.float32, *,
                 page_tokens: int = PAGE_TOKENS, pages: int = 16, private_width: int = 0) -> None:
        if (min(layers, heads, width, page_tokens) < 1 or width % heads or pages < 2
                or private_width < 0):
            raise ValueError("positive pool dimensions and at least two pages required")
        self.layers, self.heads, self.width = layers, heads, width
        self.depth = width // heads
        self.page_tokens = page_tokens
        self.private_width = private_width
        self.device, self.dtype = torch.device(device), dtype
        self.kv: list[Tensor] = []
        self.memory = torch.zeros(0, width, device=self.device, dtype=dtype)
        self.private = (torch.zeros(0, private_width, device=self.device, dtype=dtype)
                        if private_width else None)
        # Within-page row offsets, for page-major row expansion (``PagedHistoryCache._rows``).
        self.offsets = torch.arange(page_tokens, device=self.device)
        self.pages = 0
        self.free: list[int] = []
        self.grows = 0
        self._restore = 0               # pages to restore after a reset
        self._peak_used = 0             # most pages in use since the last reset
        self._grow(pages)

    @classmethod
    def for_actor(cls, actor: HistoryActor, **kwargs) -> "KVPagePool":
        cfg, prototype = actor.config, actor.bos
        return cls(cfg.layers, cfg.heads, cfg.width, prototype.device, prototype.dtype, **kwargs)

    def check(self, actor: HistoryActor) -> None:
        cfg, prototype = actor.config, actor.bos
        if ((cfg.layers, cfg.heads, cfg.width) != (self.layers, self.heads, self.width)
                or prototype.device != self.device or prototype.dtype != self.dtype):
            raise ValueError("a shared KV page pool needs one architecture, device and dtype")

    @property
    def page_bytes(self) -> int:
        element = torch.empty((), dtype=self.dtype).element_size()
        return (self.page_tokens * (self.layers * 2 * self.width + self.width + self.private_width)
                * element)

    @property
    def reserved_bytes(self) -> int:
        return self.pages * self.page_bytes

    @property
    def used_pages(self) -> int:
        return max(self.pages - 1, 0) - len(self.free)

    def _storage(self, rows: int, *shape: int) -> Tensor:
        """New, uninitialized pool storage (see ``_grow``)."""
        return torch.empty(rows, *shape, device=self.device, dtype=self.dtype)

    def _grow(self, pages: int) -> None:
        # Uninitialized beyond the copied rows: no row is read before it is
        # written, except page 0 (zeroed here once) and the memory and private
        # rows of newly allocated pages (zeroed by ``zero_pages``).
        rows, old_rows = pages * self.page_tokens, self.pages * self.page_tokens
        kv = [self._storage(rows, 2, self.heads, self.depth) for _ in range(self.layers)]
        memory = self._storage(rows, self.width)
        private = self._storage(rows, self.private_width) if self.private_width else None
        new = (*kv, memory, *(() if private is None else (private,)))
        if old_rows:
            old = (*self.kv, self.memory, *(() if private is None else (self.private,)))
            for storage, previous in zip(new, old, strict=True):
                storage[:old_rows].copy_(previous)
        else:
            for storage in new:
                storage[:self.page_tokens].zero_()
        self.kv, self.memory, self.private = kv, memory, private
        # Page 0 stays unallocated: the zero rows every padded gather reads.
        self.free.extend(range(pages - 1, max(self.pages, 1) - 1, -1))
        self.pages = pages
        self.grows += 1

    def reserve(self, count: int) -> None:
        """Make at least ``count`` pages free, growing once if they are not."""
        missing = count - len(self.free)
        if missing > 0:
            # Page 0 is not allocatable, so an empty pool needs one page more.
            pages = max(self.pages, 1) + max(missing, self.pages // 4)
            self._grow(max(pages, self._restore))
            self._restore = 0

    def allocate(self, count: int) -> list[int]:
        if count <= 0:
            return []
        self.reserve(count)
        pages = self.free[-count:]
        del self.free[-count:]
        self._peak_used = max(self._peak_used, self.used_pages)
        return pages

    def reset(self) -> None:
        """Drop the storage. Only valid when no page is in use; the next
        allocation grows straight to the most pages used since the last reset
        (plus an eighth), so a steady workload allocates once per cycle."""
        if self.used_pages > 0:
            raise RuntimeError("resetting a KV page pool with live pages")
        if self._peak_used:
            self._restore = self._peak_used + self._peak_used // 8 + 1
        self._peak_used = 0
        self.kv = []
        self.memory = torch.zeros(0, self.width, device=self.device, dtype=self.dtype)
        if self.private is not None:
            self.private = torch.zeros(0, self.private_width, device=self.device, dtype=self.dtype)
        self.pages = 0
        self.free = []

    def release(self, pages: list[int]) -> None:
        self.free.extend(pages)

    def zero_pages(self, pages: Tensor) -> None:
        """Zero the memory (and private) rows of ``pages`` (int64, on the pool's
        device): a newly allocated page holds no position yet, so whole-page
        gathers read zeros past each stream's length, as the zero page gives."""
        self.memory.view(self.pages, -1).index_fill_(0, pages, 0.0)
        if self.private is not None:
            self.private.view(self.pages, -1).index_fill_(0, pages, 0.0)


@dataclass
class PagedEntry:
    stream: PublicStream
    generation: int
    length: int = 0                 # encoded positions, BOS included
    pages: list[int] = field(default_factory=list)


def span_plan(spans: np.ndarray, groups: int, min_share: float = 0.0, min_rows: int = 0
              ) -> list[tuple[int, int, int]]:
    """Contiguous groups ``(begin, end, span)`` of a batch sorted by length.

    ``spans[i]`` (non-increasing) is the padded span row ``i`` alone would
    need; a group pads every member to its first member's span. Chooses at
    most ``groups`` groups minimising the padded rows ``sum n_g * span_g``,
    by dynamic programming over the points where the span drops. A group is
    added only if it saves at least ``min_share`` of the one-group rows and
    ``min_rows`` rows over the best plan with one group fewer.
    """
    n = len(spans)
    starts = np.flatnonzero(np.r_[True, spans[1:] != spans[:-1]]) if n else np.zeros(1, np.int64)
    if groups < 2 or len(starts) < 2:
        return [(0, n, int(spans[0]))]
    bounds = np.r_[starts, n].astype(np.int64)
    m = len(starts)
    # cost[i, j]: rows of one group over [bounds[i], bounds[j]), i < j.
    cost = ((bounds[None, :] - bounds[:, None]) * np.r_[spans[starts], 0][:, None]).astype(np.float64)
    cost[np.tril_indices(m + 1)] = np.inf
    best = np.full(m + 1, np.inf)
    best[0] = 0.0
    totals, choices = [], []
    for _ in range(min(groups, m)):
        candidates = best[:, None] + cost
        choice = candidates.argmin(0)
        best = candidates[choice, np.arange(m + 1)]
        totals.append(best[m])
        choices.append(choice)
    chosen = 1
    for g in range(2, len(totals) + 1):
        saving = totals[chosen - 1] - totals[g - 1]
        if saving > 0 and saving >= min_share * totals[0] and saving >= min_rows:
            chosen = g
    plan, j = [], m
    for g in range(chosen, 0, -1):
        i = int(choices[g - 1][j])
        plan.append((int(bounds[i]), int(bounds[j]), int(spans[bounds[i]])))
        j = i
    return plan[::-1]


@dataclass(frozen=True)
class KernelAttention:
    """One append's (or rollout step's) inputs to the Triton paged attention
    kernel: ``table`` page ids and ``[B]`` query ``starts`` and key ``ends`` on
    the device; ``max_end`` (host) bounds the kernel's key splits."""
    run: Callable[..., Tensor]
    table: Tensor
    starts: Tensor | None
    ends: Tensor
    page_tokens: int
    max_end: int

    def __call__(self, q: Tensor, kv: Tensor, starts: Tensor | None = None) -> Tensor:
        """``q [B, H, W, D]`` attended over ``kv [rows, 2, H, D]`` pool rows."""
        return self.run(q, kv, self.table, self.starts if starts is None else starts, self.ends,
                        self.page_tokens, max_end=self.max_end)


@dataclass(frozen=True)
class PagedPrivateKV:
    """Private attention over the learner's stored keys/values in place
    (``PagedHistoryCache(private_kv=True, attention_kernel=True)``): decision
    ``i`` attends to stream ``i``'s positions ``<= prefix[i]``. Fills
    ``RolloutMemory.attend``."""
    attention: KernelAttention
    kv: Tensor                      # pool.private viewed as [rows, 2, heads, depth]

    def __call__(self, q: Tensor, prefix: Tensor) -> Tensor:
        """``q [n, heads, depth]`` -> ``[n, heads, depth]``."""
        return self.attention(q[:, :, None], self.kv, prefix)[:, :, 0]


# Span groups (``PagedHistoryCache(span_groups=G)``): an extra group must save
# this share of the one-group padded rows, and this many rows, to be worth
# its extra gather and SDPA launches per layer.
SPAN_GROUP_MIN_SHARE = 0.1
SPAN_GROUP_MIN_ROWS = 2048
# Causal prefill (``PagedHistoryCache(causal_prefill=True)``): length groups
# (``history_model.length_groups``) and the most padded tokens in one pass,
# which bounds its transient activations.
CAUSAL_PREFILL_GROUPS = 4
CAUSAL_PREFILL_TOKENS = 1 << 15


class PagedHistoryCache(BatchedHistoryCache):
    """``BatchedHistoryCache`` over a shared ``KVPagePool`` (see the module notes).

    ``span_groups`` (opt-in, > 1): every append sorts its streams by end and
    runs the key/value gather and SDPA per span group (``span_plan``), each
    padded only to its own longest stream; token-wise layers still run once
    over the batch, whose order is invisible to the pool. ``encode_rollout``
    groups the returned memory the same way. FP32 reduction shapes change
    (acceptance tier 2); with one group everything is as without it.

    ``private_kv`` (opt-in; the pool needs ``private_width = 2 * width``)
    also stores every new token's ``actor.kv_proj(memory)`` when it is
    appended, so ``encode_rollout(projected=True)`` returns the private
    attention keys/values instead of memory for the actor to project again
    at every step. Same dot products; only the projection's GEMM row count
    differs (acceptance tier 2). The parameter signature covers ``kv_proj``,
    so a weight change rebuilds them with the rest of the cache.

    ``causal_prefill`` (opt-in): ``prefill`` encodes every entry that starts
    from BOS in one causal pass per length group (``is_causal`` SDPA over its
    own fresh keys/values; no pool gather, no mask) instead of one masked
    append per ``chunk_size`` tokens, and writes the same pool rows. Other
    entries take the chunked path. Same function of the same tokens; FP32
    reduction shapes differ (acceptance tier 2).

    ``attention_kernel`` (opt-in): where ``kernel_active`` (see the module
    notes), every append (chunked prefill and merged appends included) runs
    one Triton paged-attention call per layer over its whole batch in input
    order instead of the per-span-group gather + SDPA, and with
    ``private_kv`` ``encode_rollout(projected=True)`` returns the stored
    keys/values in place (``RolloutMemory.attend``) instead of gathering them.
    Same masks and keys; the kernel's online softmax changes FP32 reduction
    order (acceptance tier 2). Inactive, nothing changes.
    """

    def __init__(self, actor: HistoryActor, pool: KVPagePool | None = None,
                 chunk_size: int = 128, *, page_span: bool = False, span_groups: int = 1,
                 private_kv: bool = False, causal_prefill: bool = False,
                 attention_kernel: bool = False) -> None:
        super().__init__(actor, chunk_size)
        self.pool = pool if pool is not None else KVPagePool.for_actor(
            actor, private_width=2 * actor.config.width if private_kv else 0)
        self.pool.check(actor)
        self.page_span = page_span
        if span_groups < 1:
            raise ValueError("span groups must be positive")
        self.span_groups = int(span_groups)
        if private_kv and self.pool.private_width != 2 * actor.config.width:
            raise ValueError("private K/V pages need a pool with private_width = 2 * width")
        self.private_kv = bool(private_kv)
        self.causal_prefill = bool(causal_prefill)
        self.attention_kernel = bool(attention_kernel)
        # The kernel entry point where it can run here; None keeps the gather + SDPA path.
        self._paged_kernel = None
        if attention_kernel:
            from train import history_paged_attention as kernel
            if kernel.kernel_supported(self.pool.device, self.pool.dtype, self.pool.depth,
                                       self.pool.page_tokens):
                self._paged_kernel = kernel.paged_append_attention_triton
        # Host counters of attention key rows: gathered (padded) and valid, for
        # appends and returned memory; how many appends ran in groups; padded
        # positions of causal prefill passes; appends and rollout steps whose
        # attention ran in the paged kernel (which gathers nothing).
        self.span_rows = dict(append=0, append_valid=0, memory=0, memory_valid=0,
                              grouped_appends=0, causal_padded=0, kernel_appends=0,
                              kernel_rollouts=0)

    @property
    def kernel_active(self) -> bool:
        """Whether appends attend with the Triton paged kernel (``attention_kernel``
        requested and runnable on this pool's device, dtype and geometry)."""
        return self._paged_kernel is not None

    def _span(self, longest: int) -> int:
        """Attention keys and memory positions to materialize for ``longest``."""
        if not self.page_span:
            return bucket(longest)
        P = self.pool.page_tokens
        return min(bucket(longest), -(-longest // P) * P)

    def _memory_size(self, longest: int) -> int:
        return self._span(longest)

    def _spans(self, ends: np.ndarray) -> np.ndarray:
        """``_span`` of every element of ``ends`` (positive int64), vectorized."""
        # bucket(n) = max(32, 2 ** bit_length(n - 1)); frexp's exponent of a
        # non-negative integer below 2 ** 53 is exactly its bit length.
        spans = np.maximum(32, np.left_shift(np.int64(1), np.frexp((ends - 1).astype(np.float64))[1]))
        if self.page_span:
            P = self.pool.page_tokens
            spans = np.minimum(spans, -(-ends // P) * P)
        return spans.astype(np.int64)

    def _groups(self, ends: np.ndarray) -> tuple[np.ndarray | None, list[tuple[int, int, int]]]:
        """``(order, plan)``: the batch order (None: unchanged) and its span groups
        ``(begin, end, span)`` in that order; one group without ``span_groups``."""
        if self.span_groups < 2 or len(ends) < 2:
            return None, [(0, len(ends), self._span(int(ends.max())))]
        order = np.argsort(-ends, kind="stable")
        plan = span_plan(self._spans(ends[order]), self.span_groups, SPAN_GROUP_MIN_SHARE,
                         SPAN_GROUP_MIN_ROWS)
        if len(plan) == 1:          # keep the batch exactly as it came
            return None, [(0, len(ends), plan[0][2])]
        return order, plan

    # -- storage ------------------------------------------------------------

    def clear(self) -> None:
        for entry in self.entries.values():
            self.pool.release(entry.pages)
        super().clear()

    def prune(self, active: set[tuple[int, int]]) -> None:
        for key in list(self.entries):
            if key not in active:
                self.pool.release(self.entries.pop(key).pages)

    @property
    def bytes(self) -> int:
        pages = sum(len(entry.pages) for entry in self.entries.values())
        position_bytes = (sum(t.numel() * t.element_size() for t in self._position_cache[1:])
                          if self._position_cache is not None else 0)
        return pages * self.pool.page_bytes + position_bytes

    def _entry(self, key: tuple[int, int], stream: PublicStream) -> PagedEntry:
        size = stream.prefix + 1
        old = self.entries.get(key)
        # A replaced/reset stream starts from BOS, even if a caller reuses a key.
        if old is not None and (old.stream is not stream or old.generation != stream.generation
                                or old.length > size):
            self.pool.release(old.pages)
            old = None
        if old is not None:
            return old
        entry = PagedEntry(stream, stream.generation)
        self.entries[key] = entry
        self.rebuilds += 1
        return entry

    def _page_table(self, entries: list[PagedEntry], capacity: int) -> np.ndarray:
        """``[len(entries), blocks]`` page ids covering ``capacity`` positions; 0 pads."""
        P = self.pool.page_tokens
        table = np.zeros((len(entries), -(-capacity // P)), np.int64)
        for i, entry in enumerate(entries):
            pages = entry.pages[:table.shape[1]]
            table[i, :len(pages)] = pages
        return table

    def _rows(self, table: Tensor, lengths: Tensor, positions: Tensor) -> Tensor:
        """Pool row of every position in ``positions`` (an ``arange`` from 0) for each
        table row; row 0 (the zero page) at or past that row's length. Expanded
        page-major: ``table[b, j] * P + o`` for in-page offset ``o``."""
        P = self.pool.page_tokens
        span = len(positions)
        blocks = -(-span // P)
        rows = (table[:, :blocks, None] * P + self.pool.offsets).view(len(table), blocks * P)
        if blocks * P != span:
            rows = rows[:, :span]
        return torch.where(positions < lengths[:, None], rows, 0)

    def _allocate(self, entries: list[PagedEntry], ends: np.ndarray) -> np.ndarray:
        """Extend every entry's pages to cover ``ends``; returns the new page ids,
        whose memory rows the caller zeroes (``KVPagePool.zero_pages``)."""
        P, new = self.pool.page_tokens, []
        for entry, end in zip(entries, ends.tolist()):
            missing = -(-end // P) - len(entry.pages)
            if missing > 0:
                pages = self.pool.allocate(missing)
                entry.pages.extend(pages)
                new.extend(pages)
        return np.asarray(new, np.int64)

    # -- encoding -----------------------------------------------------------

    def _gather_table(self, source: Tensor, table: Tensor, size: int) -> Tensor:
        """``[len(table), size, features]`` rows of ``source`` (a ``[rows, features]``
        pool tensor) at the positions of each uploaded page-table row, gathered
        one whole page at a time. Rows past an entry's length read zeros: page 0
        or the zeroed tail of its last page."""
        P = self.pool.page_tokens
        features = source.shape[1]
        gathered = torch.index_select(source.view(-1, P * features), 0, table.view(-1))
        gathered = gathered.view(len(table), -1, features)
        return gathered if gathered.shape[1] == size else gathered[:, :size].contiguous()

    def _gather_memory(self, entries: list[PagedEntry], size: int) -> Tensor:
        table, = upload_arrays((self._page_table(entries, size),), self.pool.device)
        self.span_rows["memory"] += len(entries) * size
        self.span_rows["memory_valid"] += sum(entry.length for entry in entries)
        return self._gather_table(self.pool.memory, table, size)

    @torch.no_grad()
    def encode_rollout(self, keys: list[tuple[int, int]], streams: list[PublicStream], *,
                       projected: bool = False, preuploaded_lengths: Tensor | None = None,
                       host_lengths: tuple[int, ...] | None = None
                       ) -> tuple[StreamBatch, RolloutMemory]:
        """``encode`` for one decision per stream (decision ``i`` reads stream
        ``i``), returned as a ``RolloutMemory``: the streams in up to
        ``span_groups`` end-sorted span groups, each padded only to its own
        longest stream, of top-layer memory or, ``projected``, the stored
        private attention keys/values (``private_kv``). With one group and no
        projection the memory is exactly ``encode``'s. Projected with the
        paged kernel active, nothing is gathered: ``RolloutMemory.attend``
        reads the stored keys/values in place (valid until the next append)."""
        if projected and not self.private_kv:
            raise ValueError("projected rollout memory needs a private_kv cache")
        targets = self._encode_targets(keys, streams, preuploaded_lengths, host_lengths)
        entries = self._extend(keys, streams, targets)
        source = self.pool.private if projected else self.pool.memory
        lengths = np.asarray(targets, np.int64)
        if projected and self._paged_kernel is not None:
            longest = int(lengths.max())
            table, ends = upload_arrays((self._page_table(entries, longest), lengths),
                                        self.pool.device)
            pool = self.pool
            attend = PagedPrivateKV(self._kernel_attention(table, None, ends, longest),
                                    source.view(-1, 2, pool.heads, pool.depth))
            self.span_rows["memory_valid"] += int(lengths.sum())
            self.span_rows["kernel_rollouts"] += 1
            return (self._stream_metadata(targets, preuploaded_lengths, source.device),
                    RolloutMemory([], True, attend))
        order, plan = self._groups(lengths)
        ordered = entries if order is None else [entries[i] for i in order]
        tables = [self._page_table(ordered[a:b], span) for a, b, span in plan]
        rows = [] if order is None else [order[a:b] for a, b, _ in plan]
        uploaded = upload_arrays((*tables, *rows), self.pool.device)
        groups = []
        for index, (a, b, span) in enumerate(plan):
            groups.append((None if order is None else uploaded[len(plan) + index],
                           self._gather_table(source, uploaded[index], span)))
            self.span_rows["memory"] += (b - a) * span
        self.span_rows["memory_valid"] += int(lengths.sum())
        return (self._stream_metadata(targets, preuploaded_lengths, source.device),
                RolloutMemory(groups, projected))

    @staticmethod
    def _mask(allowed: Tensor, dtype: torch.dtype) -> Tensor:
        """The additive ``[B, 1, W, span]`` mask SDPA derives from boolean ``allowed``
        (``where(allowed, 0, -inf)``), built once per append, not once per layer."""
        mask = torch.full(allowed.shape, float("-inf"), dtype=dtype, device=allowed.device)
        return mask.masked_fill_(allowed, 0.0)[:, None]

    def _append_groups(self, ends: np.ndarray
                       ) -> tuple[np.ndarray | None, list[tuple[int, int, int]]]:
        """``_groups`` for an append's attention; one group in input order with the
        paged kernel, which stops at every stream's own end."""
        if self._paged_kernel is not None:
            return None, [(0, len(ends), self._span(int(ends.max())))]
        return self._groups(ends)

    def _kernel_attention(self, table: Tensor, starts: Tensor | None, ends: Tensor,
                          max_end: int) -> KernelAttention:
        """The paged kernel's inputs for one append (built once, used by every layer)."""
        return KernelAttention(self._paged_kernel, table, starts, ends, self.pool.page_tokens,
                               max_end)

    def _attention_plan(self, plan: list[tuple[int, int, int]], positions: Tensor,
                        end_positions: Tensor, table: Tensor, key_positions: Tensor,
                        dtype: torch.dtype) -> list[tuple[int, int, int, Tensor, Tensor]]:
        """Per span group ``(begin, end, span, rows, mask)``: the pool rows to
        gather and the additive mask, built once per append for every layer."""
        groups = []
        for a, b, span in plan:
            keys = key_positions[:span]
            mask = self._mask((keys[None, None] <= positions[a:b, :, None])
                              & (keys[None, None] < end_positions[a:b, None, None]), dtype)
            rows = self._rows(table[a:b], end_positions[a:b], keys).view(-1)
            groups.append((a, b, span, rows, mask))
            self.span_rows["append"] += (b - a) * span
        if len(groups) > 1:
            self.span_rows["grouped_appends"] += 1
        return groups

    @staticmethod
    def _attend(q: Tensor, kv: Tensor,
                groups: list[tuple[int, int, int, Tensor, Tensor]] | KernelAttention) -> Tensor:
        """SDPA of queries ``q`` ``[batch, heads, width, depth]`` over the keys and
        values gathered from pool rows ``kv`` per span group, in batch order; or
        the paged kernel over the pool rows in place."""
        if isinstance(groups, KernelAttention):
            return groups(q, kv)
        heads, depth = kv.shape[2], kv.shape[3]
        parts = []
        for a, b, span, rows, mask in groups:
            # [n * span, 2, heads, depth] -> keys/values [n, heads, span, depth]
            packed = torch.index_select(kv, 0, rows).view(b - a, span, 2, heads, depth).permute(2, 0, 3, 1, 4)
            parts.append(F.scaled_dot_product_attention(q[a:b], packed[0], packed[1],
                                                        attn_mask=mask, dropout_p=0.0))
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def _append(self, entries: list[PagedEntry], counts: list[int]) -> None:
        actor, cfg, pool = self.actor, self.actor.config, self.pool
        device, dtype = actor.bos.device, actor.bos.dtype
        batch, width = len(entries), max(counts)
        starts = np.asarray([e.length for e in entries], np.int64)
        count_array = np.asarray(counts, np.int64)
        ends = starts + count_array
        order, plan = self._append_groups(ends)
        if order is not None:       # end-sorted; outputs go to the pool by row
            entries = [entries[i] for i in order]
            counts = [counts[i] for i in order]
            starts, count_array, ends = starts[order], count_array[order], ends[order]
        self.span_rows["append_valid"] += int(ends.sum())
        capacity = bucket(int(ends.max()))
        span = plan[0][2]
        P = pool.page_tokens
        new_pages = self._allocate(entries, ends)
        # Upload only new raw events, not the complete historical prefix.
        tokens = np.zeros((batch, width, TOKEN_DIM), np.uint8)
        rounds, phases = np.zeros((batch, width), np.int64), np.zeros((batch, width), np.int64)
        for i, (entry, count) in enumerate(zip(entries, counts)):
            begin = max(entry.length - 1, 0)
            offset = int(entry.length == 0)  # first new item is BOS
            count -= offset
            if count:
                end = begin + count
                tokens[i, offset:offset + count] = entry.stream.tokens[begin:end]
                rounds[i, offset:offset + count] = entry.stream.rounds[begin:end]
                phases[i, offset:offset + count] = entry.stream.phases[begin:end]
        table = self._page_table(entries, span)
        # New positions (row b, slot t < count_b) and the pool rows they fill.
        write_batch = np.repeat(np.arange(batch, dtype=np.int64), count_array)
        write_slot = np.arange(len(write_batch), dtype=np.int64) - np.repeat(
            np.cumsum(count_array) - count_array, count_array)
        position = starts[write_batch] + write_slot
        write_rows = table[write_batch, position // P] * P + position % P
        fresh = np.flatnonzero(starts == 0)
        (t, r, p, start_positions, end_positions, table_d, write_batch_d, write_slot_d,
         write_rows_d, fresh_d, new_pages_d) = upload_arrays(
            (tokens, rounds, phases, starts, ends, table, write_batch, write_slot, write_rows,
             fresh, new_pages), device)
        if len(new_pages):
            pool.zero_pages(new_pages_d)
        state = (actor.public(t.to(dtype)) + actor.round_embedding(r.clamp(0, cfg.max_rounds - 1))
                 + actor.phase_embedding(p.clamp(0, 3)))
        if len(fresh):
            state[fresh_d, 0] = actor.bos[0, 0]
        position_table, key_positions = self._position_buffers(capacity)
        positions = start_positions[:, None] + key_positions[:width]
        state = state + position_table[positions.clamp(max=capacity - 1)]
        if self._paged_kernel is not None:
            groups = self._kernel_attention(table_d, start_positions, end_positions,
                                            int(ends.max()))
            self.span_rows["kernel_appends"] += 1
        else:
            groups = self._attention_plan(plan, positions, end_positions, table_d,
                                          key_positions, dtype)
        heads, depth = cfg.heads, cfg.width // cfg.heads
        for layer_index, layer in enumerate(actor.stream.layers):
            attention = layer.self_attn
            # train.history_attention.project, keeping the fused q/k/v output.
            qkv = F.linear(layer.norm1(state), attention.in_proj_weight, attention.in_proj_bias)
            qkv = qkv.view(batch, width, 3, heads, depth)
            q = qkv[:, :, 0].transpose(1, 2)
            kv = pool.kv[layer_index]
            kv[write_rows_d] = qkv[write_batch_d, write_slot_d, 1:]
            state = finish(layer, state, self._attend(q, kv, groups))
        if actor.stream.norm is not None:
            state = actor.stream.norm(state)
        state = actor.stream_norm(state)
        self._write_memory(state[write_batch_d, write_slot_d], write_rows_d)
        for entry, end in zip(entries, ends.tolist()):
            entry.length = end
        self.encoded_tokens += int(count_array.sum())

    def _write_memory(self, memory: Tensor, rows: Tensor) -> None:
        """Store new tokens' top-layer memory at pool ``rows``, and with
        ``private_kv`` their private attention keys/values."""
        self.pool.memory[rows] = memory
        if self.private_kv:
            projection = self.actor.kv_proj
            self.pool.private[rows] = F.linear(memory, projection.weight, projection.bias)

    # -- causal prefill -----------------------------------------------------

    @torch.no_grad()
    def prefill(self, keys: list[tuple[int, int]], streams: list[PublicStream]) -> int:
        """``BatchedHistoryCache.prefill``; with ``causal_prefill`` the entries that
        start from BOS are encoded in causal length-group passes first."""
        if not self.causal_prefill or not keys:
            return super().prefill(keys, streams)
        if len(keys) != len(streams) or len(set(keys)) != len(keys):
            raise ValueError("one distinct match key per public stream required")
        before = self.appends
        self.refresh_weights()
        entries = [self._entry(key, stream) for key, stream in zip(keys, streams)]
        fresh = [i for i, entry in enumerate(entries) if entry.length == 0]
        if fresh:
            self._causal_prefill([entries[i] for i in fresh],
                                 np.asarray([streams[i].prefix + 1 for i in fresh], np.int64))
        self._extend(keys, streams, [s.prefix + 1 for s in streams])
        return self.appends - before

    def _causal_prefill(self, entries: list[PagedEntry], targets: np.ndarray) -> None:
        """Encode fresh ``entries`` from BOS up to ``targets`` positions: per
        length group, under a token budget, one causal pass over the whole
        prefix that writes keys/values and memory into the entries' pages."""
        from train.history_model import length_groups
        P = self.pool.page_tokens
        # Never more length groups than the chunked path would take passes.
        groups = min(CAUSAL_PREFILL_GROUPS, -(-int(targets.max()) // self.chunk_size))
        for part in length_groups(targets - 1, groups, self.actor.config.width):
            part = part[np.argsort(-targets[part], kind="stable")]
            span = -(-int(targets[part[0]]) // P) * P
            count = max(1, CAUSAL_PREFILL_TOKENS // span)
            for begin in range(0, len(part), count):
                members = part[begin:begin + count]
                self._causal_pass([entries[i] for i in members], targets[members], span)
                self.appends += 1

    def _causal_pass(self, entries: list[PagedEntry], targets: np.ndarray, span: int) -> None:
        actor, cfg, pool = self.actor, self.actor.config, self.pool
        device, dtype = actor.bos.device, actor.bos.dtype
        batch, P = len(entries), pool.page_tokens
        self.span_rows["causal_padded"] += batch * span
        tokens = np.zeros((batch, span, TOKEN_DIM), np.uint8)
        rounds, phases = np.zeros((batch, span), np.int64), np.zeros((batch, span), np.int64)
        for i, (entry, target) in enumerate(zip(entries, targets.tolist())):
            events = target - 1                 # position 0 is BOS
            tokens[i, 1:target] = entry.stream.tokens[:events]
            rounds[i, 1:target] = entry.stream.rounds[:events]
            phases[i, 1:target] = entry.stream.phases[:events]
        new_pages = self._allocate(entries, targets)
        table = self._page_table(entries, span)
        write_batch = np.repeat(np.arange(batch, dtype=np.int64), targets)
        write_slot = np.arange(len(write_batch), dtype=np.int64) - np.repeat(
            np.cumsum(targets) - targets, targets)
        write_rows = table[write_batch, write_slot // P] * P + write_slot % P
        t, r, p, write_batch_d, write_slot_d, write_rows_d, new_pages_d = upload_arrays(
            (tokens, rounds, phases, write_batch, write_slot, write_rows, new_pages), device)
        if len(new_pages):
            pool.zero_pages(new_pages_d)
        state = (actor.public(t.to(dtype)) + actor.round_embedding(r.clamp(0, cfg.max_rounds - 1))
                 + actor.phase_embedding(p.clamp(0, 3)))
        state[:, 0] = actor.bos[0, 0]
        position_table, _ = self._position_buffers(bucket(span))
        state = state + position_table[:span]
        heads, depth = cfg.heads, cfg.width // cfg.heads
        for layer_index, layer in enumerate(actor.stream.layers):
            attention = layer.self_attn
            qkv = F.linear(layer.norm1(state), attention.in_proj_weight, attention.in_proj_bias)
            qkv = qkv.view(batch, span, 3, heads, depth)
            pool.kv[layer_index][write_rows_d] = qkv[write_batch_d, write_slot_d, 1:]
            # Trailing padding cannot reach a valid position under causal attention.
            q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
            state = finish(layer, state, F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=True))
        if actor.stream.norm is not None:
            state = actor.stream.norm(state)
        state = actor.stream_norm(state)
        self._write_memory(state[write_batch_d, write_slot_d], write_rows_d)
        for entry, target in zip(entries, targets.tolist()):
            entry.length = target
        self.encoded_tokens += int(targets.sum())


# ---- merged snapshot encoding ----------------------------------------------------

def merged_encode(parts: list[tuple[PagedHistoryCache, list[tuple[int, int]], list[PublicStream], int]],
                  heads, size: int) -> Tensor:
    """Encode several frozen identities' streams in ONE append pass per chunk.

    ``parts`` are ``(cache, keys, streams, slot)`` per identity, in identity
    order; ``slot`` indexes ``heads`` (a ``SnapshotHeads`` built with
    ``encoder=True``), whose stacked weights stand in for each identity's
    encoder. Entries, pages, chunking and counters stay per identity cache;
    only the arithmetic is shared: every linear layer runs as one batched
    matrix multiply over the identities' slots, attention as one SDPA over all
    rows. Returns ``[sum of streams, size, width]`` memory in part order, zero
    past each stream's length. Same function as ``cache.encode`` per part,
    with FP32 reduction-order differences (acceptance tier 2).
    """
    if not parts:
        raise ValueError("nothing to encode")
    pool, chunk = parts[0][0].pool, parts[0][0].chunk_size
    entries, targets, slots, owners = [], [], [], []
    for cache, keys, streams, slot in parts:
        if cache.pool is not pool or cache.chunk_size != chunk:
            raise ValueError("merged encoding needs one page pool and one prefill chunk")
        if not keys or len(keys) != len(streams) or len(set(keys)) != len(keys):
            raise ValueError("one distinct match key per public stream required")
        if cache.private_kv:
            raise ValueError("merged encoding does not store private K/V pages")
        cache.refresh_weights()
        for key, stream in zip(keys, streams):
            entries.append(cache._entry(key, stream))
            targets.append(stream.prefix + 1)
            slots.append(int(slot))
            owners.append(cache)
    while True:
        short = [i for i, (e, n) in enumerate(zip(entries, targets)) if e.length < n]
        if not short:
            break
        counts = [min(chunk, targets[i] - entries[i].length) for i in short]
        _merged_append(parts[0][0], [entries[i] for i in short], counts,
                       [slots[i] for i in short], heads)
        parts[0][0].appends += 1    # one shared pass, counted once
        for i, count in zip(short, counts):
            owners[i].encoded_tokens += count
    return parts[0][0]._gather_memory(entries, size)


def _merged_append(cache: PagedHistoryCache, entries: list[PagedEntry], counts: list[int],
                   slots: list[int], heads) -> None:
    """``PagedHistoryCache._append`` for entries of several identities.

    Linear layers, layer norms and embeddings run slot-padded: row ``b`` of
    slot ``s`` sits at ``s * M + rank``, ``M`` the most rows of one slot, and
    every layer is one ``baddbmm`` against the stacked ``[slot, in, out]``
    weights. Padding rows are zero tokens whose outputs are never written.
    Attention runs over the real rows only, per span group with
    ``cache.span_groups`` (the entries then sorted by end before the slot
    layout is laid out), or in the paged kernel when ``cache.kernel_active``.
    """
    from train.history_snapshot_batch import _linear, _norm
    pool, tensors, cfg, eps = cache.pool, heads.tensors, heads.config, heads.eps
    device = pool.device
    order, plan = cache._append_groups(np.asarray([e.length for e in entries], np.int64)
                                       + np.asarray(counts, np.int64))
    if order is not None:
        entries = [entries[i] for i in order]
        counts = [counts[i] for i in order]
        slots = [slots[i] for i in order]
    batch, width = len(entries), max(counts)
    S = max(slots) + 1
    rank = np.zeros(batch, np.int64)
    seen: dict[int, int] = {}
    for i, slot in enumerate(slots):
        rank[i] = seen.get(slot, 0)
        seen[slot] = rank[i] + 1
    M = max(seen.values())
    R = S * M
    pad = np.asarray(slots, np.int64) * M + rank
    starts = np.asarray([e.length for e in entries], np.int64)
    count_array = np.asarray(counts, np.int64)
    ends = starts + count_array
    cache.span_rows["append_valid"] += int(ends.sum())
    capacity = bucket(int(ends.max()))
    span = plan[0][2]
    P = pool.page_tokens
    new_pages = cache._allocate(entries, ends)
    tokens = np.zeros((R, width, TOKEN_DIM), np.uint8)
    rounds, phases = np.zeros((R, width), np.int64), np.zeros((R, width), np.int64)
    starts_pad = np.zeros(R, np.int64)
    starts_pad[pad] = starts
    for i, (entry, count) in enumerate(zip(entries, counts)):
        begin = max(entry.length - 1, 0)
        offset = int(entry.length == 0)  # first new item is BOS
        count -= offset
        if count:
            end = begin + count
            tokens[pad[i], offset:offset + count] = entry.stream.tokens[begin:end]
            rounds[pad[i], offset:offset + count] = entry.stream.rounds[begin:end]
            phases[pad[i], offset:offset + count] = entry.stream.phases[begin:end]
    table = cache._page_table(entries, span)
    write_batch = np.repeat(np.arange(batch, dtype=np.int64), count_array)
    write_slot = np.arange(len(write_batch), dtype=np.int64) - np.repeat(
        np.cumsum(count_array) - count_array, count_array)
    position = starts[write_batch] + write_slot
    write_rows = table[write_batch, position // P] * P + position % P
    fresh = np.flatnonzero(starts == 0)
    (t, r, p, starts_pad_d, start_positions, end_positions, table_d, pad_d, write_pad_d,
     write_slot_d, write_rows_d, fresh_pad_d, fresh_slot_d, new_pages_d) = upload_arrays(
        (tokens, rounds, phases, starts_pad, starts, ends, table, pad, pad[write_batch],
         write_slot, write_rows, pad[fresh], np.asarray(slots, np.int64)[fresh], new_pages),
        device)
    if len(new_pages):
        pool.zero_pages(new_pages_d)
    W, H = cfg.width, cfg.heads
    depth = W // H
    flat = lambda x: x.reshape(S, M * width, x.shape[-1])
    state = _linear(flat(t.float()), tensors, "e_pub", S).view(R, width, W)
    slot_of = (torch.arange(R, device=device) // M)[:, None]
    state = (state + tensors["e_round"][slot_of, r.clamp(0, cfg.max_rounds - 1)]
             + tensors["e_phase"][slot_of, p.clamp(0, 3)])
    if len(fresh):
        state[fresh_pad_d, 0] = tensors["e_bos"][fresh_slot_d]
    position_table, key_positions = cache._position_buffers(capacity)
    state = state + position_table[(starts_pad_d[:, None] + key_positions[:width])
                                   .clamp(max=capacity - 1)]
    if cache.kernel_active:
        groups = cache._kernel_attention(table_d, start_positions, end_positions,
                                         int(ends.max()))
        cache.span_rows["kernel_appends"] += 1
    else:
        positions = start_positions[:, None] + key_positions[:width]
        groups = cache._attention_plan(plan, positions, end_positions, table_d, key_positions,
                                       state.dtype)
    for index in range(cfg.layers):
        hidden = _norm(flat(state), tensors, f"e{index}_n1", S, eps)
        qkv = _linear(hidden, tensors, f"e{index}_in", S).view(R, width, 3, H, depth)
        kv = pool.kv[index]
        kv[write_rows_d] = qkv[write_pad_d, write_slot_d, 1:]
        q = qkv[:, :, 0][pad_d].transpose(1, 2)        # the real rows' queries only
        attended = cache._attend(q, kv, groups)
        attended = state.new_zeros(R, width, W).index_copy_(
            0, pad_d, attended.transpose(1, 2).reshape(batch, width, W))
        state = state + _linear(flat(attended), tensors, f"e{index}_o", S).view(R, width, W)
        hidden = _norm(flat(state), tensors, f"e{index}_n2", S, eps)
        hidden = _linear(torch.relu(_linear(hidden, tensors, f"e{index}_l1", S)), tensors,
                         f"e{index}_l2", S)
        state = state + hidden.view(R, width, W)
    state = _norm(flat(state), tensors, "e_out", S, eps).view(R, width, W)
    pool.memory[write_rows_d] = state[write_pad_d, write_slot_d]
    for entry, end in zip(entries, ends.tolist()):
        entry.length = end
