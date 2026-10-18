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
except page 0 and the memory rows of an entry's pages past its length, which
are zeroed when the page is allocated. The gathered memory is therefore the
same tensor as before, zero past each stream's length.

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

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from train.history_attention import finish
from train.history_inference import BatchedHistoryCache, bucket
from train.history_model import HistoryActor, PublicStream
from train.history_transfers import upload_arrays
from train.logs import TOKEN_DIM

PAGE_TOKENS = 64


class KVPagePool:
    """Token-major page storage: per layer ``[rows, 2, heads, depth]`` keys and
    values, plus ``[rows, width]`` top-layer memory; ``rows = pages * page_tokens``.
    """

    def __init__(self, layers: int, heads: int, width: int, device, dtype=torch.float32, *,
                 page_tokens: int = PAGE_TOKENS, pages: int = 16) -> None:
        if min(layers, heads, width, page_tokens) < 1 or width % heads or pages < 2:
            raise ValueError("positive pool dimensions and at least two pages required")
        self.layers, self.heads, self.width = layers, heads, width
        self.depth = width // heads
        self.page_tokens = page_tokens
        self.device, self.dtype = torch.device(device), dtype
        self.kv: list[Tensor] = []
        self.memory = torch.zeros(0, width, device=self.device, dtype=dtype)
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
        return self.page_tokens * (self.layers * 2 * self.width + self.width) * element

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
        # written, except page 0 (zeroed here once) and the memory rows of
        # newly allocated pages (zeroed by ``zero_memory``).
        rows, old_rows = pages * self.page_tokens, self.pages * self.page_tokens
        kv = [self._storage(rows, 2, self.heads, self.depth) for _ in range(self.layers)]
        memory = self._storage(rows, self.width)
        if old_rows:
            for new, old in zip(kv, self.kv):
                new[:old_rows].copy_(old)
            memory[:old_rows].copy_(self.memory)
        else:
            for storage in (*kv, memory):
                storage[:self.page_tokens].zero_()
        self.kv, self.memory = kv, memory
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
        self.pages = 0
        self.free = []

    def release(self, pages: list[int]) -> None:
        self.free.extend(pages)

    def zero_memory(self, pages: Tensor) -> None:
        """Zero the memory rows of ``pages`` (int64, on the pool's device): a newly
        allocated page holds no position yet, so whole-page memory gathers read
        zeros past each stream's length, as the zero page gives."""
        self.memory.view(self.pages, -1).index_fill_(0, pages, 0.0)


@dataclass
class PagedEntry:
    stream: PublicStream
    generation: int
    length: int = 0                 # encoded positions, BOS included
    pages: list[int] = field(default_factory=list)


class PagedHistoryCache(BatchedHistoryCache):
    """``BatchedHistoryCache`` over a shared ``KVPagePool`` (see the module notes)."""

    def __init__(self, actor: HistoryActor, pool: KVPagePool | None = None,
                 chunk_size: int = 128, *, page_span: bool = False) -> None:
        super().__init__(actor, chunk_size)
        self.pool = pool if pool is not None else KVPagePool.for_actor(actor)
        self.pool.check(actor)
        self.page_span = page_span

    def _span(self, longest: int) -> int:
        """Attention keys and memory positions to materialize for ``longest``."""
        if not self.page_span:
            return bucket(longest)
        P = self.pool.page_tokens
        return min(bucket(longest), -(-longest // P) * P)

    def _memory_size(self, longest: int) -> int:
        return self._span(longest)

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
        whose memory rows the caller zeroes (``KVPagePool.zero_memory``)."""
        P, new = self.pool.page_tokens, []
        for entry, end in zip(entries, ends.tolist()):
            missing = -(-end // P) - len(entry.pages)
            if missing > 0:
                pages = self.pool.allocate(missing)
                entry.pages.extend(pages)
                new.extend(pages)
        return np.asarray(new, np.int64)

    # -- encoding -----------------------------------------------------------

    def _gather_pages(self, source: Tensor, entries: list[PagedEntry], size: int) -> Tensor:
        """``[len(entries), size, features]`` rows of ``source`` (a ``[rows, features]``
        pool tensor) at each entry's positions, gathered one whole page at a time.
        Rows past an entry's length read zeros: page 0 or the zeroed tail of its
        last page."""
        P = self.pool.page_tokens
        table, = upload_arrays((self._page_table(entries, size),), self.pool.device)
        features = source.shape[1]
        gathered = torch.index_select(source.view(-1, P * features), 0, table.view(-1))
        gathered = gathered.view(len(entries), -1, features)
        return gathered if gathered.shape[1] == size else gathered[:, :size].contiguous()

    def _gather_memory(self, entries: list[PagedEntry], size: int) -> Tensor:
        return self._gather_pages(self.pool.memory, entries, size)

    @staticmethod
    def _mask(allowed: Tensor, dtype: torch.dtype) -> Tensor:
        """The additive ``[B, 1, W, span]`` mask SDPA derives from boolean ``allowed``
        (``where(allowed, 0, -inf)``), built once per append, not once per layer."""
        mask = torch.full(allowed.shape, float("-inf"), dtype=dtype, device=allowed.device)
        return mask.masked_fill_(allowed, 0.0)[:, None]

    def _append(self, entries: list[PagedEntry], counts: list[int]) -> None:
        actor, cfg, pool = self.actor, self.actor.config, self.pool
        device, dtype = actor.bos.device, actor.bos.dtype
        batch, width = len(entries), max(counts)
        starts = np.asarray([e.length for e in entries], np.int64)
        count_array = np.asarray(counts, np.int64)
        ends = starts + count_array
        capacity = bucket(int(ends.max()))
        span = self._span(int(ends.max()))
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
            pool.zero_memory(new_pages_d)
        state = (actor.public(t.to(dtype)) + actor.round_embedding(r.clamp(0, cfg.max_rounds - 1))
                 + actor.phase_embedding(p.clamp(0, 3)))
        if len(fresh):
            state[fresh_d, 0] = actor.bos[0, 0]
        position_table, key_positions = self._position_buffers(capacity)
        positions = start_positions[:, None] + key_positions[:width]
        state = state + position_table[positions.clamp(max=capacity - 1)]
        key_positions = key_positions[:span]
        mask = self._mask((key_positions[None, None] <= positions[:, :, None])
                          & (key_positions[None, None] < end_positions[:, None, None]), dtype)
        rows = self._rows(table_d, end_positions, key_positions).view(-1)
        heads, depth = cfg.heads, cfg.width // cfg.heads
        for layer_index, layer in enumerate(actor.stream.layers):
            attention = layer.self_attn
            # train.history_attention.project, keeping the fused q/k/v output.
            qkv = F.linear(layer.norm1(state), attention.in_proj_weight, attention.in_proj_bias)
            qkv = qkv.view(batch, width, 3, heads, depth)
            q = qkv[:, :, 0].transpose(1, 2)
            kv = pool.kv[layer_index]
            kv[write_rows_d] = qkv[write_batch_d, write_slot_d, 1:]
            # [batch * span, 2, heads, depth] -> keys/values [batch, heads, span, depth]
            packed = torch.index_select(kv, 0, rows).view(batch, span, 2, heads, depth).permute(2, 0, 3, 1, 4)
            attended = F.scaled_dot_product_attention(q, packed[0], packed[1],
                                                      attn_mask=mask, dropout_p=0.0)
            state = finish(layer, state, attended)
        if actor.stream.norm is not None:
            state = actor.stream.norm(state)
        state = actor.stream_norm(state)
        pool.memory[write_rows_d] = state[write_batch_d, write_slot_d]
        for entry, end in zip(entries, ends.tolist()):
            entry.length = end
        self.encoded_tokens += int(count_array.sum())


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
    Attention runs over the real rows only.
    """
    from train.history_snapshot_batch import _linear, _norm
    pool, tensors, cfg, eps = cache.pool, heads.tensors, heads.config, heads.eps
    device = pool.device
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
    capacity = bucket(int(ends.max()))
    span = cache._span(int(ends.max()))
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
        pool.zero_memory(new_pages_d)
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
    positions = start_positions[:, None] + key_positions[:width]
    key_positions = key_positions[:span]
    mask = cache._mask((key_positions[None, None] <= positions[:, :, None])
                       & (key_positions[None, None] < end_positions[:, None, None]), state.dtype)
    rows = cache._rows(table_d, end_positions, key_positions).view(-1)
    for index in range(cfg.layers):
        hidden = _norm(flat(state), tensors, f"e{index}_n1", S, eps)
        qkv = _linear(hidden, tensors, f"e{index}_in", S).view(R, width, 3, H, depth)
        kv = pool.kv[index]
        kv[write_rows_d] = qkv[write_pad_d, write_slot_d, 1:]
        q = qkv[:, :, 0][pad_d].transpose(1, 2)        # the real rows' queries only
        packed = torch.index_select(kv, 0, rows).view(batch, span, 2, H, depth).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, packed[0], packed[1],
                                                  attn_mask=mask, dropout_p=0.0)
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
