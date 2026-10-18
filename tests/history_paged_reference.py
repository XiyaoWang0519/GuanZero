"""Frozen reference: the paged cache's append and gather as of 2026-10-17 (commit 090f337).

tests/test_history_paged_cache.py runs this copy side by side with
``train.history_paged_cache`` to show that the unconditional trims since
(additive mask built once per append, page-major row expansion, query-only
gather in the merged append, uninitialized pool growth, whole-page memory
gathers) are bitwise on CPU. Do not edit except to follow API renames.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from train.history_attention import finish
from train.history_inference import bucket
from train.history_model import PublicStream
from train.history_paged_cache import KVPagePool, PagedEntry, PagedHistoryCache
from train.history_transfers import upload_arrays
from train.logs import TOKEN_DIM


class LegacyPool(KVPagePool):
    """Zero-filled growth, as before."""

    def _storage(self, rows, *shape):
        return torch.zeros(rows, *shape, device=self.device, dtype=self.dtype)


class LegacyPagedHistoryCache(PagedHistoryCache):
    """Per-position row gathers and a boolean attention mask, as before."""

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
        return torch.index_select(self.pool.memory, 0, rows.view(-1)).view(len(entries), size, self.pool.width)

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
        table = self._page_table(entries, span)
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
        key_positions = key_positions[:span]
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
            # [batch * span, 2, heads, depth] -> keys/values [batch, heads, span, depth]
            packed = torch.index_select(kv, 0, rows).view(batch, span, 2, heads, depth).permute(2, 0, 3, 1, 4)
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


def legacy_merged_encode(parts: list[tuple[PagedHistoryCache, list[tuple[int, int]], list[PublicStream], int]],
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
        legacy_merged_append(parts[0][0], [entries[i] for i in short], counts,
                             [slots[i] for i in short], heads)
        parts[0][0].appends += 1    # one shared pass, counted once
        for i, count in zip(short, counts):
            owners[i].encoded_tokens += count
    return parts[0][0]._gather_memory(entries, size)


def legacy_merged_append(cache: PagedHistoryCache, entries: list[PagedEntry], counts: list[int],
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
    for entry, end in zip(entries, ends.tolist()):
        missing = -(-end // P) - len(entry.pages)
        if missing > 0:
            entry.pages.extend(pool.allocate(missing))
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
     write_slot_d, write_rows_d, fresh_pad_d, fresh_slot_d) = upload_arrays(
        (tokens, rounds, phases, starts_pad, starts, ends, table, pad, pad[write_batch],
         write_slot, write_rows, pad[fresh], np.asarray(slots, np.int64)[fresh]), device)
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
    allowed = ((key_positions[None, None] <= positions[:, :, None])
               & (key_positions[None, None] < end_positions[:, None, None]))
    rows = cache._rows(table_d, end_positions, key_positions).view(-1)
    for index in range(cfg.layers):
        hidden = _norm(flat(state), tensors, f"e{index}_n1", S, eps)
        qkv = _linear(hidden, tensors, f"e{index}_in", S).view(R, width, 3, H, depth)
        kv = pool.kv[index]
        kv[write_rows_d] = qkv[write_pad_d, write_slot_d, 1:]
        q = qkv[pad_d][:, :, 0].transpose(1, 2)
        packed = torch.index_select(kv, 0, rows).view(batch, span, 2, H, depth).permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(q, packed[0], packed[1],
                                                  attn_mask=allowed[:, None], dropout_p=0.0)
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
