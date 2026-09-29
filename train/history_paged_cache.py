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

The pool is shared by every policy identity of one collector: an identity's
cache releases its pages on ``clear``/``prune``. Page 0 is never allocated and
stays zero. The pool grows by reallocation and never shrinks; its reserved
size is the high-water mark of live pages.
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
        self.pages = 0
        self.free: list[int] = []
        self.grows = 0
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
        return self.pages - 1 - len(self.free)

    def _grow(self, pages: int) -> None:
        rows, old_rows = pages * self.page_tokens, self.pages * self.page_tokens
        kv = [torch.zeros(rows, 2, self.heads, self.depth, device=self.device, dtype=self.dtype)
              for _ in range(self.layers)]
        memory = torch.zeros(rows, self.width, device=self.device, dtype=self.dtype)
        if old_rows:
            for new, old in zip(kv, self.kv):
                new[:old_rows].copy_(old)
            memory[:old_rows].copy_(self.memory)
        self.kv, self.memory = kv, memory
        # Page 0 stays unallocated: the zero rows every padded gather reads.
        self.free.extend(range(pages - 1, max(self.pages, 1) - 1, -1))
        self.pages = pages
        self.grows += 1

    def allocate(self, count: int) -> list[int]:
        if count <= 0:
            return []
        if len(self.free) < count:
            self._grow(max(self.pages + self.pages // 2, self.pages + count - len(self.free)))
        pages = self.free[-count:]
        del self.free[-count:]
        return pages

    def release(self, pages: list[int]) -> None:
        self.free.extend(pages)


@dataclass
class PagedEntry:
    stream: PublicStream
    generation: int
    length: int = 0                 # encoded positions, BOS included
    pages: list[int] = field(default_factory=list)


class PagedHistoryCache(BatchedHistoryCache):
    """``BatchedHistoryCache`` over a shared ``KVPagePool`` (see the module notes)."""

    def __init__(self, actor: HistoryActor, pool: KVPagePool | None = None,
                 chunk_size: int = 128) -> None:
        super().__init__(actor, chunk_size)
        self.pool = pool if pool is not None else KVPagePool.for_actor(actor)
        self.pool.check(actor)
        self._selectors: tuple | None = None

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
        """Pool row of every position in ``positions`` (an ``arange``) for each table
        row; row 0 (the zero page) at or past that row's length."""
        P = self.pool.page_tokens
        page = table[:, positions // P]
        rows = page * P + positions % P
        return torch.where(positions[None] < lengths[:, None], rows, torch.zeros_like(rows))

    # -- encoding -----------------------------------------------------------

    def _gather_memory(self, entries: list[PagedEntry], size: int) -> Tensor:
        table, lengths = upload_arrays(
            (self._page_table(entries, size), np.asarray([e.length for e in entries], np.int64)),
            self.pool.device)
        rows = self._rows(table, lengths, torch.arange(size, device=self.pool.device))
        return self.pool.memory[rows.view(-1)].view(len(entries), size, self.pool.width)

    def _append(self, entries: list[PagedEntry], counts: list[int]) -> None:
        actor, cfg, pool = self.actor, self.actor.config, self.pool
        device, dtype = actor.bos.device, actor.bos.dtype
        batch, width = len(entries), max(counts)
        starts = np.asarray([e.length for e in entries], np.int64)
        count_array = np.asarray(counts, np.int64)
        ends = starts + count_array
        capacity = bucket(int(ends.max()))
        P = pool.page_tokens
        for entry, end in zip(entries, ends.tolist()):
            missing = -(-end // P) - len(entry.pages)
            if missing > 0:
                entry.pages.extend(pool.allocate(missing))
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
        table = self._page_table(entries, capacity)
        # New positions (row b, slot t < count_b) and the pool rows they fill.
        write_batch = np.repeat(np.arange(batch, dtype=np.int64), count_array)
        write_slot = np.arange(len(write_batch), dtype=np.int64) - np.repeat(
            np.cumsum(count_array) - count_array, count_array)
        position = starts[write_batch] + write_slot
        write_rows = table[write_batch, position // P] * P + position % P
        fresh = np.flatnonzero(starts == 0)
        (t, r, p, start_positions, end_positions, table_d, write_batch_d, write_slot_d,
         write_rows_d, fresh_d) = upload_arrays(
            (tokens, rounds, phases, starts, ends, table, write_batch, write_slot, write_rows,
             fresh), device)
        state = (actor.public(t.to(dtype)) + actor.round_embedding(r.clamp(0, cfg.max_rounds - 1))
                 + actor.phase_embedding(p.clamp(0, 3)))
        if len(fresh):
            state[fresh_d, 0] = actor.bos[0, 0]
        position_table, key_positions = self._position_buffers(capacity)
        positions = start_positions[:, None] + key_positions[:width]
        state = state + position_table[positions.clamp(max=capacity - 1)]
        allowed = ((key_positions[None, None] <= positions[:, :, None])
                   & (key_positions[None, None] < end_positions[:, None, None]))
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
            # [batch * capacity, 2, heads, depth] -> keys/values [batch, heads, capacity, depth]
            packed = kv[rows].view(batch, capacity, 2, heads, depth).permute(2, 0, 3, 1, 4)
            attended = F.scaled_dot_product_attention(q, packed[0], packed[1],
                                                      attn_mask=allowed[:, None], dropout_p=0.0)
            state = finish(layer, state, attended)
        if actor.stream.norm is not None:
            state = actor.stream.norm(state)
        state = actor.stream_norm(state)
        pool.memory[write_rows_d] = state[write_batch_d, write_slot_d]
        for entry, end in zip(entries, ends.tolist()):
            entry.length = end
        self.encoded_tokens += int(count_array.sum())
