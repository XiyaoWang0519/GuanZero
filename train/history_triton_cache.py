"""Optional uint32-only CUDA KV update/pack for frozen rollout histories.

One kernel per public layer writes the new K/V into Entry-owned storage and
gathers the exact eager dense layout. No floating-point operation is introduced
or changed. The original _append still performs every projection, attention,
residual and normalization with its original shapes.

The cache is restricted to one CUDA stream at a time. A pointer table owns strong
Tensor references, and records that stream on their allocations before exposing
raw addresses to Triton. PyTorch may therefore release a table after launches
without recycling referenced allocations while those launches are pending.
clear() releases the stream binding; callers must finish using returned outputs
before using a different stream, as for the original cache.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import time

import numpy as np
import torch

from train.history_inference import BatchedHistoryCache

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU probe and original cache remain usable without Triton.
    triton = tl = None


if triton is not None:
    @triton.jit(do_not_specialize=[
        "BATCH", "CAPACITY", "LAYER", "STACK_MODE", "K0", "K1", "K2", "K3",
        "V0", "V1", "V2", "V3",
    ])
    def _update_and_pack(
        NEW_K, NEW_V, PACKED_K, PACKED_V, POINTERS, RANGES,
        BATCH, CAPACITY, LAYER, STACK_MODE,
        K0, K1, K2, K3, V0, V1, V2, V3,
        HEADS: tl.constexpr, DEPTH: tl.constexpr, BLOCK: tl.constexpr,
    ):
        row = tl.program_id(1)
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = index < HEADS * CAPACITY * DEPTH
        head = index // (CAPACITY * DEPTH)
        position = index // DEPTH % CAPACITY
        depth = index % DEPTH
        own_capacity = tl.load(POINTERS + row)
        start = tl.load(RANGES + row)
        count = tl.load(RANGES + BATCH + row)
        end = start + count
        entry_k = tl.load(POINTERS + (1 + 2 * LAYER) * BATCH + row)
        entry_v = tl.load(POINTERS + (2 + 2 * LAYER) * BATCH + row)
        entry_k = entry_k.to(tl.pointer_type(tl.uint32))
        entry_v = entry_v.to(tl.pointer_type(tl.uint32))
        # Strides are element strides of potentially non-contiguous QKV views.
        new_k = NEW_K.to(tl.pointer_type(tl.uint32))
        new_v = NEW_V.to(tl.pointer_type(tl.uint32))
        packed_k = PACKED_K.to(tl.pointer_type(tl.uint32))
        packed_v = PACKED_V.to(tl.pointer_type(tl.uint32))
        fresh = valid & (position >= start) & (position < end)
        source_k = row * K0 + head * K1 + (position - start) * K2 + depth * K3
        source_v = row * V0 + head * V1 + (position - start) * V2 + depth * V3
        k_new = tl.load(new_k + source_k, mask=fresh, other=0)
        v_new = tl.load(new_v + source_v, mask=fresh, other=0)
        own_offset = (head * own_capacity + position) * DEPTH + depth
        # The eager insufficient-capacity branch zeros everything after end.
        # The stack branch copies the actual cache tail, whose invariant is zero.
        limit = tl.where(STACK_MODE != 0, own_capacity, end)
        old = valid & (position < limit) & ~fresh
        k_old = tl.load(entry_k + own_offset, mask=old, other=0)
        v_old = tl.load(entry_v + own_offset, mask=old, other=0)
        k_bits = tl.where(fresh, k_new, k_old)
        v_bits = tl.where(fresh, v_new, v_old)
        tl.store(entry_k + own_offset, k_new, mask=fresh)
        tl.store(entry_v + own_offset, v_new, mask=fresh)
        packed_offset = row * HEADS * CAPACITY * DEPTH + index
        tl.store(packed_k + packed_offset, k_bits, mask=valid)
        tl.store(packed_v + packed_offset, v_bits, mask=valid)


def _storage_bytes(tensors):
    return sum({t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                for t in tensors if t is not None}.values())


@dataclass
class _PointerTable:
    key: tuple
    pointers: torch.Tensor  # [1 + 2 * layers, batch]: capacity, K0, V0, ...
    references: tuple      # Strong refs until every raw-pointer launch is queued.
    entries: tuple


@dataclass
class _CopyContext:
    table: _PointerTable
    ranges: torch.Tensor    # [2, batch]: start, count
    entries: tuple
    counts: tuple
    stream: object


class TritonHistoryCache(BatchedHistoryCache):
    """Public KV cache with optional Triton copies and exact eager fallbacks.

    At most one pointer table is retained. Its entries already belong to the
    normal cache; prune()/clear() release the table so retired histories do not
    remain resident. Dynamic start/count metadata is uploaded once per append,
    reused across every layer, then released. Pointer/capacity metadata is only
    uploaded when its exact ordered composition changes. Both uploads share one
    pinned slab on a table miss; the async transfer owns an immutable host slab.
    """

    def __init__(self, actor, chunk_size=128, *, enabled=True, min_batch=1, max_batch=128,
                 max_table_bytes=128 << 10, max_packed_bytes=256 << 20, block=256):
        super().__init__(actor, chunk_size)
        if enabled and actor.bos.device.type == "cuda" and triton is None:
            raise RuntimeError("Triton is required when the CUDA Triton KV cache is enabled")
        if min(min_batch, max_batch, max_table_bytes, max_packed_bytes) < 1 or min_batch > max_batch:
            raise ValueError("positive Triton cache limits required")
        if block not in (128, 256, 512, 1024):
            raise ValueError("block must be a supported power of two")
        self.enabled = enabled
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.max_table_bytes = max_table_bytes
        self.max_packed_bytes = max_packed_bytes
        self.block = block
        self._pointer_table = None
        self._copy_context = None
        self._owner_stream = None
        self.copy_stats = Counter()
        self.launch_host_seconds = 0.0

    @property
    def metadata_bytes(self):
        tensors = []
        if self._pointer_table is not None:
            tensors.append(self._pointer_table.pointers)
        if self._copy_context is not None:
            tensors.append(self._copy_context.ranges)
        return _storage_bytes(tensors)

    @property
    def bytes(self):
        return super().bytes + self.metadata_bytes

    def clear(self):
        self._copy_context = self._pointer_table = self._owner_stream = None
        super().clear()
        self.copy_stats["clears"] += 1

    def prune(self, active):
        super().prune(active)
        if self._pointer_table is not None:
            live = {id(entry) for entry in self.entries.values()}
            if any(id(entry) not in live for entry in self._pointer_table.entries):
                self._pointer_table = None
                self.copy_stats["table_prunes"] += 1

    @torch.no_grad()
    def encode(self, keys, streams, **kwargs):
        # Unchanged prefixes skip _append; growth copies old allocations inside
        # _entry. Reject a different stream before either path can touch them.
        owner = self._owner_stream
        if owner is not None and torch.cuda.current_stream(owner.device) != owner:
            raise RuntimeError("TritonHistoryCache must run on its owning CUDA stream")
        device = self.actor.bos.device
        if device.type == "cuda" and owner is None:
            self._owner_stream = torch.cuda.current_stream(device)
        try:
            return super().encode(keys, streams, **kwargs)
        finally:
            # Base invalidation can call clear() before an eager-only encode.
            # Bind partially completed/error paths as well as successful ones.
            if device.type == "cuda" and self._owner_stream is None:
                self._owner_stream = torch.cuda.current_stream(device)

    def _supported_entries(self, entries, counts):
        actor, cfg = self.actor, self.actor.config
        batch = len(entries)
        if (not self.enabled or triton is None or actor.bos.device.type != "cuda"
                or actor.bos.dtype != torch.float32 or not self.min_batch <= batch <= self.max_batch
                or len(counts) != batch or len({id(e) for e in entries}) != batch
                or (2 * cfg.layers + 5) * batch * 8 > self.max_table_bytes):
            return False
        return all(0 <= entry.length < entry.length + count <= entry.capacity
                   and len(entry.keys) == len(entry.values) == cfg.layers
                   for entry, count in zip(entries, counts))

    def _metadata(self, entries, counts):
        device, cfg = self.actor.bos.device, self.actor.config
        stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
        # Eager fallback writes the same Entry allocations, so it must obey the
        # owner restriction even when Triton eligibility changes between calls.
        if self._owner_stream is not None and self._owner_stream != stream:
            raise RuntimeError("TritonHistoryCache must run on its owning CUDA stream")
        if not self._supported_entries(entries, counts):
            self._pointer_table = None
            return None
        self._owner_stream = stream
        refs = tuple(tensor for entry in entries
                     for pair in zip(entry.keys, entry.values) for tensor in pair)
        if any(tensor.dtype != torch.float32 or tensor.device != device
               or not tensor.is_contiguous()
               or tensor.shape != (cfg.heads, entry.capacity, cfg.width // cfg.heads)
               for entry in entries for tensor in (*entry.keys, *entry.values)):
            self._pointer_table = None
            return None
        # Both view addresses and capacities are live-checked; replacement of a
        # Tensor on an existing Entry cannot silently keep an obsolete pointer.
        pointers = tuple(tensor.data_ptr() for tensor in refs)
        if len(set(pointers)) != len(pointers):
            self._pointer_table = None
            return None
        key = (tuple(id(entry) for entry in entries), tuple(e.capacity for e in entries),
               tuple(id(tensor) for tensor in refs), pointers, device)
        old = self._pointer_table
        changed = old is None or old.key != key
        if changed:
            # Overlapping Entry tensors would create racing raw-pointer writes.
            # Normal cache allocations are independent; unusual aliases use the
            # original ordered eager copies instead.
            bases = [tensor.untyped_storage().data_ptr() for tensor in refs]
            if len(set(bases)) != len(bases):
                self._pointer_table = None
                return None
        rows, batch = (2 * cfg.layers + 3 if changed else 2), len(entries)
        host = torch.empty((rows, batch), dtype=torch.int64, pin_memory=True)
        array = host.numpy()
        array[-2] = [entry.length for entry in entries]
        array[-1] = counts
        if changed:
            array[0] = [entry.capacity for entry in entries]
            for layer in range(cfg.layers):
                array[1 + 2 * layer] = [entry.keys[layer].data_ptr() for entry in entries]
                array[2 + 2 * layer] = [entry.values[layer].data_ptr() for entry in entries]
        uploaded = host.to(device, non_blocking=True)
        # PyTorch's pinned allocator records the H2D event before recycling host.
        # Recording Entry allocations handles their later release after rawptr
        # launches, including prune, growth, reset, and model invalidation.
        if changed:
            # A Tensor reference alone does not preserve its former allocation
            # when .data is rebound. Detached aliases keep that storage alive.
            storage_refs = tuple(tensor.detach() for tensor in refs)
            for tensor in storage_refs:
                tensor.record_stream(stream)
            old = self._pointer_table = _PointerTable(key, uploaded[:-2], storage_refs, tuple(entries))
            self.copy_stats["pointer_uploads"] += 1
        else:
            self.copy_stats["pointer_reuses"] += 1
        self.copy_stats["metadata_uploads"] += 1
        self.copy_stats["metadata_h2d_bytes"] += host.numel() * host.element_size()
        context = _CopyContext(old, uploaded[-2:], tuple(entries), tuple(counts), stream)
        retained = _storage_bytes([old.pointers, context.ranges])
        self.copy_stats["max_metadata_bytes"] = max(self.copy_stats["max_metadata_bytes"], retained)
        return context

    def _append(self, entries, counts):
        if self._copy_context is not None:
            raise RuntimeError("TritonHistoryCache does not support reentrant appends")
        # Upload before the eager embedding/project work, then reuse across L
        # calls to _pack_keys_values. No original floating-point op is changed.
        self._copy_context = self._metadata(entries, counts)
        try:
            super()._append(entries, counts)
        finally:
            self._copy_context = None

    def _pack_keys_values(self, entries, counts, layer_index, new_k, new_v, capacity):
        cfg, batch = self.actor.config, len(entries)
        shape = (batch, cfg.heads, capacity, cfg.width // cfg.heads)
        context = self._copy_context
        if context is None:
            context = self._metadata(entries, counts)
        supported = (context is not None and 0 <= layer_index < cfg.layers
                     and capacity >= max(e.length + n for e, n in zip(entries, counts))
                     and 2 * batch * cfg.width * capacity * 4 <= self.max_packed_bytes
                     and new_k.dtype == new_v.dtype == torch.float32
                     and new_k.device == new_v.device == self.actor.bos.device
                     and new_k.shape == new_v.shape
                     and new_k.ndim == 4 and new_k.shape[:2] == (batch, cfg.heads)
                     and new_k.shape[2] >= max(counts)
                     and new_k.shape[3] == cfg.width // cfg.heads
                     and min(*new_k.stride(), *new_v.stride()) >= 0)
        if not supported:
            self.copy_stats["eager_packs"] += 1
            return super()._pack_keys_values(entries, counts, layer_index, new_k, new_v, capacity)
        if (tuple(id(entry) for entry in entries) != tuple(id(entry) for entry in context.entries)
                or tuple(counts) != context.counts):
            raise RuntimeError("Entry ordering/counts changed within one append")
        if torch.cuda.current_stream(new_k.device) != context.stream:
            raise RuntimeError("Triton KV pack changed CUDA stream during append")
        packed_k, packed_v = new_k.new_empty(shape), new_v.new_empty(shape)
        stack_mode = int(all(entry.capacity >= capacity for entry in entries))
        grid = (triton.cdiv(cfg.width * capacity, self.block), batch)
        started = time.perf_counter()
        with torch.cuda.device(new_k.device):
            _update_and_pack[grid](
                new_k, new_v, packed_k, packed_v, context.table.pointers, context.ranges,
                batch, capacity, layer_index, stack_mode, *new_k.stride(), *new_v.stride(),
                HEADS=cfg.heads, DEPTH=cfg.width // cfg.heads, BLOCK=self.block,
                num_warps=4,
            )
        self.launch_host_seconds += time.perf_counter() - started
        self.copy_stats["triton_packs"] += 1
        self.copy_stats["removed_entry_copy_calls"] += 2 * batch
        self.copy_stats["insufficient_capacity_packs"] += 1 - stack_mode
        return packed_k, packed_v
