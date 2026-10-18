"""Optional uint32-only CUDA KV update/pack for frozen rollout histories.

One kernel per public layer writes the new K/V into Entry-owned storage and
gathers the exact eager dense layout. No floating-point operation is introduced
or changed. The original _append still performs every projection, attention,
residual and normalization with its original shapes.

The cache is restricted to one CUDA stream at a time. One cache-wide pointer
table has a column per registered Entry (an insertion-ordered registry with a
free list). Registering an Entry validates its tensors once, keeps detached
aliases of their storages and records that stream on their allocations before
exposing raw addresses to Triton; PyTorch may therefore release a column after
launches without recycling referenced allocations while they are pending. The
table is uploaded again only when the registry changes: an Entry is created,
replaced by growth or reset, pruned, or the cache is cleared. Each append then
uploads only a ``[3, batch]`` int64 array (table column, start, count) and the
kernel reads capacity and K/V addresses through the column. clear() releases
the stream binding; callers must finish using returned outputs before using a
different stream, as for the original cache.
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
        "BATCH", "ROW_BASE", "SLOTS", "CAPACITY", "LAYER", "STACK_MODE", "K0", "K1", "K2",
        "K3", "V0", "V1", "V2", "V3",
    ])
    def _update_and_pack(
        NEW_K, NEW_V, PACKED_K, PACKED_V, POINTERS, RANGES,
        BATCH, ROW_BASE, SLOTS, CAPACITY, LAYER, STACK_MODE,
        K0, K1, K2, K3, V0, V1, V2, V3,
        HEADS: tl.constexpr, DEPTH: tl.constexpr, BLOCK: tl.constexpr,
    ):
        # NEW_*/PACKED_* start at this launch's first row; the range table
        # covers the whole append, so it is read at ROW_BASE + row. Its first
        # row names the Entry's column in the cache-wide [1 + 2L, SLOTS]
        # pointer table (capacity, K0, V0, K1, V1, ...).
        row = tl.program_id(1)
        table_row = ROW_BASE + row
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = index < HEADS * CAPACITY * DEPTH
        head = index // (CAPACITY * DEPTH)
        position = index // DEPTH % CAPACITY
        depth = index % DEPTH
        column = tl.load(RANGES + table_row)
        start = tl.load(RANGES + BATCH + table_row)
        count = tl.load(RANGES + 2 * BATCH + table_row)
        end = start + count
        own_capacity = tl.load(POINTERS + column)
        entry_k = tl.load(POINTERS + (1 + 2 * LAYER) * SLOTS + column)
        entry_v = tl.load(POINTERS + (2 + 2 * LAYER) * SLOTS + column)
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


INT32_LIMIT = 2**31 - 1
MAX_GRID_ROWS = 65535    # CUDA grid dimension y


def rows_per_launch(batch: int, span: int, views) -> int:
    """Rows per kernel launch keeping every in-kernel element offset in int32.

    ``span`` is one packed row (``width * capacity`` elements); ``views`` are the
    new K/V views as ``(row_stride, in_row_extent)``. Every offset is at most
    ``row * row_stride + extent`` for a launch-local ``row``. Chunking only
    splits the grid; each launch copies the same bits.
    """
    extent = max([span, *(int(e) for _, e in views)])
    per_row = max([span, *(int(r) for r, _ in views)])
    if span <= 0 or extent > INT32_LIMIT:
        return 0
    rows = (INT32_LIMIT - extent) // per_row + 1
    return max(0, min(batch, rows, MAX_GRID_ROWS))


def _view(tensor):
    return tensor.stride(0), sum((n - 1) * st for n, st in zip(tensor.shape[1:], tensor.stride()[1:]))


def _storage_bytes(tensors):
    return sum({t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
                for t in tensors if t is not None}.values())


@dataclass
class _PointerTable:
    """One upload of the registry: column c describes ``entries[c]``."""
    pointers: torch.Tensor  # [1 + 2 * layers, columns]: capacity, K0, V0, ...
    references: tuple      # per column: detached K/V storage aliases (() when free)
    entries: tuple         # per column: the registered Entry, or None when free


@dataclass
class _CopyContext:
    table: _PointerTable
    ranges: torch.Tensor    # [3, batch]: table column, start, count
    entries: tuple
    counts: tuple
    stream: object


def _upload(array: np.ndarray, device: torch.device) -> torch.Tensor:
    """int64 host array to ``device``: one pinned non_blocking H2D on CUDA.

    PyTorch's pinned allocator records the H2D event before recycling the
    host slab, so the slab may be dropped right after the call."""
    if device.type != "cuda":
        return torch.from_numpy(np.array(array, dtype=np.int64))
    host = torch.empty(array.shape, dtype=torch.int64, pin_memory=True)
    host.numpy()[...] = array
    return host.to(device, non_blocking=True)


class TritonHistoryCache(BatchedHistoryCache):
    """Public KV cache with optional Triton copies and exact eager fallbacks.

    One cache-wide pointer table covers every registered Entry. An Entry is
    registered (validated, aliased, given a column) the first time an append
    uses it on the Triton path, and leaves the registry when the base cache
    replaces it (growth, reset or reused key), prunes it, or clears. A retired
    Entry's column is freed at once, so retired histories do not remain
    resident. The device table is uploaded again only after such a change
    (``table_rebuilds``, also counted as ``pointer_uploads``); otherwise it is
    reused (``table_reuses``/``pointer_reuses``). Column, start and count are
    uploaded once per append, reused across every layer, then released.

    Per append, each Entry costs one registry lookup and a check that its
    capacity and K/V addresses still match its column (a Tensor whose ``.data``
    was rebound re-registers). The tensor checks (dtype, device, contiguity,
    shape, distinct addresses, storage not shared with another registered
    Entry) run once at registration; in-place resizing of an Entry tensor is
    unsupported, as is ``.data`` mutation of parameters for the base cache.
    ``copy_stats`` counts every eager fallback by reason (``fallback_*`` and
    ``eager_packs``), Triton launches and chunked packs.
    """
    max_rows_per_launch = None   # tests force chunking; production uses the int32 bound

    def __init__(self, actor, chunk_size=128, *, enabled=True, min_batch=1, max_batch=None,
                 max_table_bytes=None, max_packed_bytes=None, block=256):
        # None = no limit. Batch size is unbounded: launches are chunked so every
        # in-kernel offset stays int32; per append only 24 bytes per stream are
        # uploaded, plus the cache-wide pointer table (8 * (2L + 1) bytes per
        # column) after a registry change. Packed K/V has exactly the eager
        # path's size, so capping it only forced eager copies.
        super().__init__(actor, chunk_size)
        if enabled and actor.bos.device.type == "cuda" and triton is None:
            raise RuntimeError("Triton is required when the CUDA Triton KV cache is enabled")
        limits = [value for value in (max_batch, max_table_bytes, max_packed_bytes)
                  if value is not None]
        if min([min_batch, *limits]) < 1 or (max_batch is not None and min_batch > max_batch):
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
        self._reset_registry()

    # -- registry: one pointer-table column per live Entry ------------------------

    def _reset_registry(self):
        self._pointer_table = None
        self._columns = {}          # id(Entry) -> column; the registry holds the Entry
        self._column_entries = []   # column -> Entry, or None when free
        self._column_keys = []      # column -> (capacity, K/V data_ptrs) at registration
        self._column_refs = []      # column -> detached storage aliases
        self._bases = {}            # storage base address -> column (alias check)
        self._free = []
        self._host = np.zeros((1 + 2 * self.actor.config.layers, 0), dtype=np.int64)

    @staticmethod
    def _entry_key(entry):
        return (entry.capacity, *(t.data_ptr() for t in entry.keys),
                *(t.data_ptr() for t in entry.values))

    def _unregister(self, column):
        entry = self._column_entries[column]
        del self._columns[id(entry)]
        for tensor in self._column_refs[column]:
            del self._bases[tensor.untyped_storage().data_ptr()]
        self._column_entries[column] = None
        self._column_keys[column] = None
        self._column_refs[column] = ()
        self._host[:, column] = 0
        self._free.append(column)
        self._pointer_table = None

    def _register(self, entry, key, stream):
        """Validate ``entry`` once and give it a column; False: eager fallback."""
        device, cfg = self.actor.bos.device, self.actor.config
        tensors = tuple(tensor for pair in zip(entry.keys, entry.values) for tensor in pair)
        if any(tensor.dtype != torch.float32 or tensor.device != device
               or not tensor.is_contiguous()
               or tensor.shape != (cfg.heads, entry.capacity, cfg.width // cfg.heads)
               for tensor in tensors) or len(set(key[1:])) != len(tensors):
            self.copy_stats["fallback_entry_tensors"] += 1
            return False
        # Overlapping Entry tensors would create racing raw-pointer writes.
        # Normal cache allocations are independent; unusual aliases use the
        # original ordered eager copies instead.
        bases = [tensor.untyped_storage().data_ptr() for tensor in tensors]
        if len(set(bases)) != len(bases) or any(base in self._bases for base in bases):
            self.copy_stats["fallback_aliased_storage"] += 1
            return False
        # A Tensor reference alone does not preserve its former allocation
        # when .data is rebound. Detached aliases keep that storage alive, and
        # recording the stream covers release after pending raw-pointer launches.
        refs = tuple(tensor.detach() for tensor in tensors)
        if stream is not None:
            for tensor in refs:
                tensor.record_stream(stream)
        if self._free:
            column = self._free.pop()
        else:
            column = len(self._column_entries)
            self._column_entries.append(None)
            self._column_keys.append(None)
            self._column_refs.append(())
            if column >= self._host.shape[1]:
                grown = np.zeros((self._host.shape[0], max(64, 2 * self._host.shape[1])),
                                 dtype=np.int64)
                grown[:, :self._host.shape[1]] = self._host
                self._host = grown
        self._columns[id(entry)] = column
        self._column_entries[column] = entry
        self._column_keys[column] = key
        self._column_refs[column] = refs
        for base in bases:
            self._bases[base] = column
        self._host[0, column] = entry.capacity
        self._host[1:, column] = [tensor.data_ptr() for tensor in tensors]
        self._pointer_table = None
        return True

    def _table_columns(self, entries, stream):
        """Registry columns of ``entries`` (registering new ones), or None."""
        columns = []
        lookup, keys = self._columns, self._column_keys
        for entry in entries:
            key = self._entry_key(entry)
            column = lookup.get(id(entry))
            if column is not None and keys[column] != key:
                self._unregister(column)   # capacity or a K/V address changed
                column = None
            if column is None:
                if not self._register(entry, key, stream):
                    return None
                column = lookup[id(entry)]
            columns.append(column)
        return columns

    def _table(self, device):
        """The uploaded registry, uploading it first if it changed."""
        table = self._pointer_table
        if table is None:
            used = len(self._column_entries)
            table = self._pointer_table = _PointerTable(
                _upload(self._host[:, :used], device), tuple(self._column_refs),
                tuple(self._column_entries))
            self.copy_stats["table_rebuilds"] += 1
            self.copy_stats["pointer_uploads"] += 1
            self.copy_stats["metadata_h2d_bytes"] += 8 * self._host.shape[0] * used
        else:
            self.copy_stats["table_reuses"] += 1
            self.copy_stats["pointer_reuses"] += 1
        return table

    def _entry(self, key, stream):
        old = self.entries.get(key)
        result = super()._entry(key, stream)
        if old is not None and result is not old:
            column = self._columns.get(id(old))
            if column is not None:
                self._unregister(column)
                self.copy_stats["table_replacements"] += 1
        return result

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
        self._copy_context = self._owner_stream = None
        self._reset_registry()
        super().clear()
        self.copy_stats["clears"] += 1

    def prune(self, active):
        super().prune(active)
        live = {id(entry) for entry in self.entries.values()}
        retired = [column for column, entry in enumerate(self._column_entries)
                   if entry is not None and id(entry) not in live]
        for column in retired:
            self._unregister(column)
        if retired:
            self.copy_stats["table_prunes"] += 1

    def _on_owner_stream(self, call, *args, **kwargs):
        # Unchanged prefixes skip _append; growth copies old allocations inside
        # _entry. Reject a different stream before either path can touch them.
        owner = self._owner_stream
        if owner is not None and torch.cuda.current_stream(owner.device) != owner:
            raise RuntimeError("TritonHistoryCache must run on its owning CUDA stream")
        device = self.actor.bos.device
        if device.type == "cuda" and owner is None:
            self._owner_stream = torch.cuda.current_stream(device)
        try:
            return call(*args, **kwargs)
        finally:
            # Base invalidation can call clear() before an eager-only encode.
            # Bind partially completed/error paths as well as successful ones.
            if device.type == "cuda" and self._owner_stream is None:
                self._owner_stream = torch.cuda.current_stream(device)

    @torch.no_grad()
    def encode(self, keys, streams, **kwargs):
        return self._on_owner_stream(super().encode, keys, streams, **kwargs)

    @torch.no_grad()
    def prefill(self, keys, streams):
        return self._on_owner_stream(super().prefill, keys, streams)

    def _supported_entries(self, entries, counts):
        actor, cfg = self.actor, self.actor.config
        batch = len(entries)
        reason = None
        if not self.enabled or triton is None or actor.bos.device.type != "cuda":
            reason = "fallback_disabled"
        elif actor.bos.dtype != torch.float32:
            reason = "fallback_dtype"
        elif batch < self.min_batch:
            reason = "fallback_below_min_batch"
        elif self.max_batch is not None and batch > self.max_batch:
            reason = "fallback_above_max_batch"
        elif len(counts) != batch or len({id(e) for e in entries}) != batch:
            reason = "fallback_entry_layout"
        elif (self.max_table_bytes is not None
              and (2 * cfg.layers + 5) * batch * 8 > self.max_table_bytes):
            reason = "fallback_table_bytes"
        elif not all(0 <= entry.length < entry.length + count <= entry.capacity
                     and len(entry.keys) == len(entry.values) == cfg.layers
                     for entry, count in zip(entries, counts)):
            reason = "fallback_entry_layout"
        if reason is not None:
            self.copy_stats[reason] += 1
            return False
        return True

    def _metadata(self, entries, counts):
        device, cfg = self.actor.bos.device, self.actor.config
        stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
        # Eager fallback writes the same Entry allocations, so it must obey the
        # owner restriction even when Triton eligibility changes between calls.
        if self._owner_stream is not None and self._owner_stream != stream:
            raise RuntimeError("TritonHistoryCache must run on its owning CUDA stream")
        if not self._supported_entries(entries, counts):
            return None
        self._owner_stream = stream
        # Registration (validation, aliases, record_stream) happens only for
        # Entries new to the registry; known ones are a lookup and an address check.
        columns = self._table_columns(entries, stream)
        if columns is None:
            return None
        table = self._table(device)
        ranges = _upload(np.array([columns, [entry.length for entry in entries], counts],
                                  dtype=np.int64), device)
        self.copy_stats["metadata_uploads"] += 1
        self.copy_stats["metadata_h2d_bytes"] += 3 * 8 * len(entries)
        context = _CopyContext(table, ranges, tuple(entries), tuple(counts), stream)
        retained = _storage_bytes([table.pointers, context.ranges])
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
        rows = (rows_per_launch(batch, cfg.width * capacity, (_view(new_k), _view(new_v)))
                if new_k.ndim == new_v.ndim == 4 else 0)
        supported = (context is not None and 0 <= layer_index < cfg.layers
                     and capacity >= max(e.length + n for e, n in zip(entries, counts))
                     and (self.max_packed_bytes is None
                          or 2 * batch * cfg.width * capacity * 4 <= self.max_packed_bytes)
                     and rows >= 1
                     and new_k.dtype == new_v.dtype == torch.float32
                     and new_k.device == new_v.device == self.actor.bos.device
                     and new_k.shape == new_v.shape
                     and new_k.ndim == 4 and new_k.shape[:2] == (batch, cfg.heads)
                     and new_k.shape[2] >= max(counts)
                     and new_k.shape[3] == cfg.width // cfg.heads
                     and min(*new_k.stride(), *new_v.stride()) >= 0)
        if not supported:
            self.copy_stats["eager_packs"] += 1
            if context is not None:
                self.copy_stats["fallback_pack_shape"] += 1
            return super()._pack_keys_values(entries, counts, layer_index, new_k, new_v, capacity)
        if (tuple(id(entry) for entry in entries) != tuple(id(entry) for entry in context.entries)
                or tuple(counts) != context.counts):
            raise RuntimeError("Entry ordering/counts changed within one append")
        if torch.cuda.current_stream(new_k.device) != context.stream:
            raise RuntimeError("Triton KV pack changed CUDA stream during append")
        packed_k, packed_v = new_k.new_empty(shape), new_v.new_empty(shape)
        stack_mode = int(all(entry.capacity >= capacity for entry in entries))
        rows = min(rows, self.max_rows_per_launch or rows)
        started = time.perf_counter()
        with torch.cuda.device(new_k.device):
            for base in range(0, batch, rows):
                end = min(batch, base + rows)
                grid = (triton.cdiv(cfg.width * capacity, self.block), end - base)
                _update_and_pack[grid](
                    new_k[base:end], new_v[base:end], packed_k[base:end], packed_v[base:end],
                    context.table.pointers, context.ranges,
                    batch, base, context.table.pointers.shape[1], capacity, layer_index,
                    stack_mode,
                    *new_k.stride(), *new_v.stride(),
                    HEADS=cfg.heads, DEPTH=cfg.width // cfg.heads, BLOCK=self.block,
                    num_warps=4,
                )
                self.copy_stats["triton_launches"] += 1
        self.launch_host_seconds += time.perf_counter() - started
        self.copy_stats["triton_packs"] += 1
        self.copy_stats["chunked_packs"] += int(rows < batch)
        self.copy_stats["removed_entry_copy_calls"] += 2 * batch
        self.copy_stats["insufficient_capacity_packs"] += 1 - stack_mode
        return packed_k, packed_v
