"""Opt-in merged public-stream encode for the frozen snapshot seats of one vector step.

``HistoryCollector(batch_snapshot_encode=True)`` (with ``batch_snapshot_policies``
and the rollout KV cache) replaces the per-identity ``cache.encode`` calls of a
merged snapshot step with ONE padded encode over every snapshot identity's new
public tokens. It is the encoder-side twin of ``train.history_snapshot_batch``:

* ``SnapshotEncoders`` stacks each identity's stream-encoder weights (``public``,
  ``round_embedding``, ``phase_embedding``, ``bos``, every layer's ``norm1``,
  ``in_proj``, ``out_proj``, ``norm2``, ``linear1``, ``linear2``, then
  ``stream.norm`` if present and ``stream_norm``) into ``[slots, ...]`` tensors,
  with the slot/registry/signature/retain discipline of ``SnapshotHeads``.
* Layout ``[S, M, w, width]``: S slots in use, M the most entries (streams) of
  one identity in this call, w the most new tokens of one entry. Every
  per-identity linear is one ``baddbmm`` over the stacked weights and every
  layer norm one ``layer_norm`` plus a per-slot affine. Padded cells are zero
  tokens whose outputs are never read back.
* Attention runs per entry against that entry's OWN keys/values: the new K/V
  of every real entry are written into its identity cache's ``Entry`` storage
  and packed (zero past each entry's end) into ``[E, heads, cap, depth]``,
  then one SDPA call covers all E entries. With the Triton cache this
  write-and-pack is one ``_update_and_pack`` launch per layer across all
  identities, through a per-step ``[1 + 2L + 2, E]`` pointer table (one column
  per entry) and ``[3, E]`` ranges (column, start, count) in the kernel's own
  layout; otherwise it is the eager per-entry copy. The pack is split into
  entry chunks of at most ``max_pack_bytes`` (one pack and one SDPA each), so
  the transient K/V stays near the old per-identity size. The same kernel (one
  head, depth = width, key and value pointers both naming ``Entry.memory``)
  writes the new memory rows and gathers the ``[streams, T, width]`` memory
  ``merged_log_probs`` reads, in one launch.

Storage never leaves the per-identity caches: ``Entry`` objects are created and
grown by the owning cache's ``_entry``, the cache's parameter signature still
clears it on a weight change, and ``prune``/``bytes``/``rebuilds``/
``encoded_tokens`` and the non-merged path are unchanged.

Fallback rule: an identity whose entries need more than ``max_new_tokens`` new
tokens in this step (default: the cache's ``chunk_size``, i.e. more than one
``_append`` chunk; rebuilds after ``clear()`` or long new entries) is encoded by
its own ``cache.encode`` instead, and counted in ``metrics()['fallbacks']``.

Acceptance tier 2: the same FP32 function as ``BatchedHistoryCache._append``
with a different floating-point reduction order (batched GEMMs, a shared
positional table and packed length). Frozen snapshot seats only; the learner's
encode is untouched. The encode draws no random numbers, so the sampler
generator is consumed exactly as before.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
import weakref

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from train.history_attention import validate as validate_stream
from train.history_inference import Entry, bucket
from train.history_model import HistoryActor, sinusoidal
from train.history_snapshot_batch import _Slot
from train.logs import TOKEN_DIM

# Order of the uploaded plan fields (``EncodePlan.arrays``).
ENCODE_FIELDS = ("tokens", "rounds", "phases", "bos", "positions", "grid", "starts", "ends",
                 "pointers", "ranges")


def encoder_parameters(actor: HistoryActor) -> tuple[Tensor, ...]:
    """Every parameter the merged encode reads, from the live module registries."""
    result = [actor.public.weight, actor.public.bias, actor.round_embedding.weight,
              actor.phase_embedding.weight, actor.bos]
    for layer in actor.stream.layers:
        attention = layer.self_attn
        result += [layer.norm1.weight, layer.norm1.bias, attention.in_proj_weight,
                   attention.in_proj_bias, attention.out_proj.weight, attention.out_proj.bias,
                   layer.norm2.weight, layer.norm2.bias, layer.linear1.weight, layer.linear1.bias,
                   layer.linear2.weight, layer.linear2.bias]
    if actor.stream.norm is not None:
        result += [actor.stream.norm.weight, actor.stream.norm.bias]
    result += [actor.stream_norm.weight, actor.stream_norm.bias]
    return tuple(result)


def encoder_signature(actor: HistoryActor) -> tuple:
    """Parameter identity, version and storage address of the encoder weights."""
    return tuple((id(t), t._version, t.data_ptr(), t.dtype, t.device)
                 for t in encoder_parameters(actor))


def _norms(actor: HistoryActor) -> list[nn.Module]:
    norms = [n for layer in actor.stream.layers for n in (layer.norm1, layer.norm2)]
    if actor.stream.norm is not None:
        norms.append(actor.stream.norm)
    return norms + [actor.stream_norm]


def validate_encoder(actor: HistoryActor) -> None:
    """Refuse encoder configurations the merged encode does not implement exactly."""
    config = actor.config
    if config.window:
        raise ValueError("merged snapshot encode requires the full-history actor")
    validate_stream(actor.stream)
    if len(actor.stream.layers) != config.layers:
        raise ValueError("merged snapshot encode requires config.layers stream layers")
    for layer in actor.stream.layers:
        if not (layer.activation is F.relu or isinstance(layer.activation, nn.ReLU)):
            raise ValueError("merged snapshot encode requires ReLU stream layers")
        if (layer.self_attn.in_proj_bias is None or layer.self_attn.out_proj.bias is None
                or layer.linear1.bias is None or layer.linear2.bias is None):
            raise ValueError("merged snapshot encode requires biased stream projections")
    norms = _norms(actor)
    if any(not isinstance(n, nn.LayerNorm) or not n.elementwise_affine or n.bias is None
           or n.normalized_shape != (config.width,) or n.eps != norms[0].eps for n in norms):
        raise ValueError("merged snapshot encode requires affine width layer norms, one eps")
    if any(t.dtype != torch.float32 for t in encoder_parameters(actor)):
        raise ValueError("merged snapshot encode is FP32 only")
    for module in actor.modules():
        if module._forward_hooks or module._forward_pre_hooks:
            raise ValueError("merged snapshot encode bypasses module calls; "
                             "actors with forward hooks are refused")


def _encoder_values(actor: HistoryActor) -> dict[str, Tensor]:
    """One identity's encoder in the stacked layout: Linear weights as [in, out],
    biases and norm affines as [1, out]."""
    width = actor.config.width
    linear = lambda weight, bias, name: {f"{name}_w": weight.t(), f"{name}_b": bias[None]}
    norm = lambda module, name: {f"{name}_w": module.weight[None], f"{name}_b": module.bias[None]}
    values = {**linear(actor.public.weight, actor.public.bias, "public"),
              "round": actor.round_embedding.weight, "phase": actor.phase_embedding.weight,
              "bos": actor.bos.reshape(1, width)}
    for index, layer in enumerate(actor.stream.layers):
        attention = layer.self_attn
        values.update(norm(layer.norm1, f"n1_{index}"))
        values.update(linear(attention.in_proj_weight, attention.in_proj_bias, f"in_{index}"))
        values.update(linear(attention.out_proj.weight, attention.out_proj.bias, f"out_{index}"))
        values.update(norm(layer.norm2, f"n2_{index}"))
        values.update(linear(layer.linear1.weight, layer.linear1.bias, f"l1_{index}"))
        values.update(linear(layer.linear2.weight, layer.linear2.bias, f"l2_{index}"))
    if actor.stream.norm is not None:
        values.update(norm(actor.stream.norm, "stream_norm"))
    values.update(norm(actor.stream_norm, "final_norm"))
    return values


class SnapshotEncoders:
    """Stacked public-stream encoder weights of the frozen snapshot identities.

    One slot per identity, exactly as ``SnapshotHeads``: ``slot(identity,
    actor)`` returns the identity's slot after checking the slot was written
    from this very actor object (weak reference) with the same encoder
    parameter signature, otherwise (re)writes it first; ``retain`` frees the
    slots of identities no longer assigned to any match. Collector-owned: never
    actor or checkpoint state. Also counts merged calls and fallbacks.
    """

    def __init__(self, capacity: int = 4, *, block: int = 256,
                 max_pack_bytes: int | None = 128 << 20) -> None:
        if capacity < 1:
            raise ValueError("slot capacity must be positive")
        if max_pack_bytes is not None and max_pack_bytes < 1:
            raise ValueError("max_pack_bytes must be positive (None: one pack)")
        if block not in (128, 256, 512, 1024):
            raise ValueError("block must be a supported power of two")
        self.capacity = capacity
        self.block = block
        self.max_pack_bytes = max_pack_bytes    # K + V of one attention chunk
        self.tensors: dict[str, Tensor] | None = None
        self.eps: float | None = None
        self.config = None
        self.has_stream_norm: bool | None = None
        self.slots: dict[int, _Slot] = {}
        self.writes = 0
        self.calls = 0              # merged encodes
        self.streams = 0            # entries covered by merged encodes
        self.tokens = 0             # new tokens appended by merged encodes
        self.fallbacks = 0          # identity encodes left to their own cache
        self.fallback_streams = 0
        self.triton_launches = 0
        self.eager_packs = 0        # eager K/V chunk packs (and eager memory writes)
        self.pack_chunks = 0        # attention chunks (one pack + one SDPA each)
        self._positions: tuple[tuple, Tensor] | None = None

    @property
    def bytes(self) -> int:
        stacked = sum(t.numel() * t.element_size() for t in (self.tensors or {}).values())
        positions = (self._positions[1].numel() * self._positions[1].element_size()
                     if self._positions is not None else 0)
        return stacked + positions

    def _allocate(self, values: dict[str, Tensor], capacity: int) -> None:
        old = self.tensors
        self.tensors = {name: value.new_zeros((capacity, *value.shape))
                        for name, value in values.items()}
        if old is not None:
            for name, tensor in old.items():
                self.tensors[name][:len(tensor)].copy_(tensor)
        self.capacity = capacity

    def _free_index(self) -> int:
        used = {slot.index for slot in self.slots.values()}
        return next(i for i in range(len(used) + 1) if i not in used)

    @torch.no_grad()
    def slot(self, identity: int, actor: HistoryActor) -> int:
        signature = encoder_signature(actor)
        record = self.slots.get(int(identity))
        if record is not None and record.actor() is actor and record.signature == signature:
            return record.index
        validate_encoder(actor)
        eps = actor.stream_norm.eps
        has_norm = actor.stream.norm is not None
        if self.config is not None and (actor.config != self.config or eps != self.eps
                                        or has_norm != self.has_stream_norm):
            raise ValueError("stacked snapshot encoders require one architecture")
        values = _encoder_values(actor)
        index = record.index if record is not None else self._free_index()
        if self.tensors is None or index >= self.capacity:
            self._allocate(values, max(self.capacity, 2 * index, index + 1))
        for name, value in values.items():
            target = self.tensors[name]
            if target.shape[1:] != value.shape or target.device != value.device:
                raise ValueError("stacked snapshot encoders require one architecture and device")
            target[index].copy_(value)
        self.config, self.eps, self.has_stream_norm = actor.config, eps, has_norm
        self.slots[int(identity)] = _Slot(index, weakref.ref(actor), signature)
        self.writes += 1
        return index

    def retain(self, identities) -> None:
        keep = {int(i) for i in identities}
        for identity in [i for i in self.slots if i not in keep]:
            del self.slots[identity]

    def position_table(self, length: int, device, dtype) -> Tensor:
        key = (length, self.config.width, device, dtype)
        if self._positions is None or self._positions[0] != key:
            self._positions = (key, sinusoidal(*key))
        return self._positions[1]

    def metrics(self) -> dict:
        return dict(bytes=self.bytes, capacity=self.capacity if self.tensors is not None else 0,
                    slots=len(self.slots), writes=self.writes, merged_calls=self.calls,
                    merged_streams=self.streams, merged_tokens=self.tokens,
                    fallbacks=self.fallbacks, fallback_streams=self.fallback_streams,
                    triton_launches=self.triton_launches, eager_packs=self.eager_packs,
                    pack_chunks=self.pack_chunks)


# ---- plan ------------------------------------------------------------------------

@dataclass
class EncodePlan:
    """Host-side plan of one merged encode (collector group order)."""
    arrays: tuple[np.ndarray, ...]      # ENCODE_FIELDS, uploaded with the step
    entries: list[Entry]                # merged entries, group then key order
    counts: list[int]                   # new tokens of each merged entry
    caches: list                        # owning cache of each merged entry
    stream_rows: np.ndarray             # output memory row of each merged entry
    fallback: list[tuple]               # (cache, keys, streams, first, last) per fallback group
    streams: int                        # rows of the output memory (all groups)
    length: int                         # T: columns of the output memory
    slots: int                          # S
    per_slot: int                       # M
    width: int                          # w
    capacity: int                       # packed K/V length, bucket of the longest end
    full: bool                          # the [S, M] grid holds exactly the entries, in order
    direct: bool                        # merged entries are every output row, in order
    triton: bool
    references: tuple                   # Entry tensors named by the pointer table


def _bind_stream(cache, device) -> None:
    """``TritonHistoryCache.encode``'s owning-stream rule, for merged appends."""
    if device.type != "cuda" or not hasattr(cache, "_owner_stream"):
        return
    current = torch.cuda.current_stream(device)
    if cache._owner_stream is not None and cache._owner_stream != current:
        raise RuntimeError("TritonHistoryCache must run on its owning CUDA stream")


def plan_encode(encoders: SnapshotEncoders, groups, actors, caches, length: int, *,
                triton: bool = False, max_new_tokens: int | None = None) -> EncodePlan:
    """Entries, counts and padded index arrays of one merged encode.

    ``groups`` carry ``identity``, ``keys`` and ``streams`` (the collector's
    snapshot groups, in identity order); ``caches`` are their per-identity
    caches. Output rows follow group then key order, ``length`` columns.
    Invalidates a cache whose actor weights changed (as ``encode`` does) and
    creates or grows entries through the cache's own ``_entry``.
    """
    merged, fallback, first = [], [], 0
    for group, actor, cache in zip(groups, actors, caches):
        if cache.actor is not actor:
            raise ValueError("each identity's cache must belong to its actor")
        device = actor.bos.device
        _bind_stream(cache, device)
        if cache._signature() != cache.signature:
            cache.clear()
        if device.type == "cuda" and hasattr(cache, "_owner_stream"):
            cache._owner_stream = torch.cuda.current_stream(device)
        last = first + len(group.keys)
        entries = [cache._entry(key, stream) for key, stream in zip(group.keys, group.streams)]
        counts = [stream.prefix + 1 - entry.length
                  for entry, stream in zip(entries, group.streams)]
        limit = cache.chunk_size if max_new_tokens is None else max_new_tokens
        if max(counts) > limit:
            fallback.append((cache, list(group.keys), list(group.streams), first, last))
        else:
            merged.append((group, actor, cache, entries, counts, first))
        first = last
    streams = first
    empty = np.zeros(0, np.int64)
    if not merged:
        return EncodePlan((np.zeros((0, TOKEN_DIM), np.uint8), empty, empty, np.zeros(0, bool),
                           empty, empty, empty, empty, empty, empty), [], [], [], empty, fallback,
                          streams, length, 0, 0, 0, 0, True, False, False, ())
    slots = [encoders.slot(group.identity, actor) for group, actor, *_ in merged]
    cfg = encoders.config
    S = max(slots) + 1
    M = max(len(item[3]) for item in merged)
    w = max(1, max(max(item[4]) for item in merged))
    all_entries = [e for item in merged for e in item[3]]
    all_counts = [c for item in merged for c in item[4]]
    capacity = bucket(max(e.length + c for e, c in zip(all_entries, all_counts)))
    rounds_n = cfg.max_rounds
    cells = S * M * w
    tokens = np.zeros((cells, TOKEN_DIM), np.uint8)
    rounds = np.zeros(cells, np.int64)
    phases = np.zeros(cells, np.int64)
    bos = np.zeros(cells, bool)
    positions = np.zeros(cells, np.int64)
    grid, stream_rows, starts, ends, owners = [], [], [], [], []
    columns = np.arange(w, dtype=np.int64)
    for (group, actor, cache, entries, counts, row), slot in zip(merged, slots):
        base_cell = slot * M
        rounds[base_cell * w:(base_cell + M) * w] = slot * rounds_n
        phases[base_cell * w:(base_cell + M) * w] = slot * 4
        for m, (entry, count, stream) in enumerate(zip(entries, counts, group.streams)):
            cell = base_cell + m
            base = cell * w
            start = entry.length
            begin = max(start - 1, 0)
            offset = int(start == 0)       # first new item is BOS
            n = count - offset
            if n > 0:
                end = begin + n
                tokens[base + offset:base + offset + n] = stream.tokens[begin:end]
                rounds[base + offset:base + offset + n] += np.clip(
                    stream.rounds[begin:end], 0, rounds_n - 1)
                phases[base + offset:base + offset + n] += np.clip(stream.phases[begin:end], 0, 3)
            bos[base] = start == 0 and count > 0
            positions[base:base + w] = np.minimum(start + columns, capacity - 1)
            grid.append(cell)
            stream_rows.append(row + m)
            starts.append(start)
            ends.append(start + count)
            owners.append(cache)
    grid = np.asarray(grid, np.int64)
    stream_rows = np.asarray(stream_rows, np.int64)
    full = len(grid) == S * M and bool(np.array_equal(grid, np.arange(S * M)))
    direct = not fallback and bool(np.array_equal(stream_rows, np.arange(streams)))
    starts = np.asarray(starts, np.int64)
    ends = np.asarray(ends, np.int64)
    pointers, ranges, references = empty, empty, ()
    if triton:
        pointers, references = _pointer_table(all_entries, cfg)
        triton = pointers.size > 0
        if triton:
            ranges = np.stack((np.arange(len(all_entries), dtype=np.int64), starts,
                               ends - starts))
    arrays = (tokens, rounds, phases, bos, positions, grid, starts, ends, pointers, ranges)
    return EncodePlan(arrays, all_entries, all_counts, owners, stream_rows, fallback, streams,
                      length, S, M, w, capacity, full, direct, triton, references)


def _pointer_table(entries: list[Entry], cfg) -> tuple[np.ndarray, tuple]:
    """``[1 + 2L + 2, E]`` int64, column e for entry e, in the layout of
    ``TritonHistoryCache``'s table (row 0 capacity, rows 1 + 2l / 2 + 2l the
    layer-l K/V addresses) plus two rows naming ``Entry.memory``, read as
    "layer L" by the memory write. Empty when an entry does not have the
    cache's own layout."""
    from train.history_triton_cache import triton
    if triton is None:
        return np.zeros(0, np.int64), ()
    depth = cfg.width // cfg.heads
    references = []
    for entry in entries:
        if (len(entry.keys) != cfg.layers or len(entry.values) != cfg.layers
                or entry.memory.shape != (entry.capacity, cfg.width)
                or not entry.memory.is_contiguous() or entry.memory.dtype != torch.float32):
            return np.zeros(0, np.int64), ()
        for tensor in (*entry.keys, *entry.values):
            if (tensor.shape != (cfg.heads, entry.capacity, depth) or not tensor.is_contiguous()
                    or tensor.dtype != torch.float32):
                return np.zeros(0, np.int64), ()
    layers, batch = cfg.layers, len(entries)
    table = np.empty((1 + 2 * layers + 2, batch), np.int64)
    table[0] = [entry.capacity for entry in entries]
    for layer in range(layers):
        table[1 + 2 * layer] = [entry.keys[layer].data_ptr() for entry in entries]
        table[2 + 2 * layer] = [entry.values[layer].data_ptr() for entry in entries]
        references += [entry.keys[layer] for entry in entries]
        references += [entry.values[layer] for entry in entries]
    table[1 + 2 * layers] = table[2 + 2 * layers] = [entry.memory.data_ptr() for entry in entries]
    references += [entry.memory for entry in entries]
    # Raw-pointer writes need independent allocations (the cache's own invariant).
    if len({int(p) for p in table[1:1 + 2 * layers + 1].ravel()}) != (2 * layers + 1) * batch:
        return np.zeros(0, np.int64), ()
    return table, tuple(references)


# ---- encode ----------------------------------------------------------------------

def _linear(x: Tensor, tensors: dict[str, Tensor], name: str, slots: int) -> Tensor:
    return torch.baddbmm(tensors[f"{name}_b"][:slots], x, tensors[f"{name}_w"][:slots])


def _norm(x: Tensor, tensors: dict[str, Tensor], name: str, slots: int, eps: float) -> Tensor:
    normalized = F.layer_norm(x, (x.shape[-1],), eps=eps)
    return torch.addcmul(tensors[f"{name}_b"][:slots], normalized, tensors[f"{name}_w"][:slots])


def _launch(encoders, new_k, new_v, packed_k, packed_v, pointers, ranges, capacity, layer,
            heads, depth, first: int = 0) -> bool:
    """``_update_and_pack`` for entries ``first:first + len(new_k)``: ``pointers``
    is the plan's ``[rows, E]`` table and ``ranges`` its ``[3, E]`` (column,
    start, count). Arguments go by the kernel's parameter names, so a signature
    change fails loudly. False if the int32 guard refuses."""
    from train import history_triton_cache as kernels
    batch, entries = new_k.shape[0], ranges.shape[1]
    if pointers.shape[1] != entries or ranges.shape[0] != 3:
        raise ValueError("pointer table and ranges must have one column per entry")
    rows = kernels.rows_per_launch(batch, heads * depth * capacity,
                                   (kernels._view(new_k), kernels._view(new_v)))
    if rows < 1 or min(*new_k.stride(), *new_v.stride()) < 0:
        return False
    # (Tests run an emulation of the kernel on CPU tensors.)
    device = torch.cuda.device(new_k.device) if new_k.is_cuda else contextlib.nullcontext()
    with device:
        for base in range(0, batch, rows):
            end = min(batch, base + rows)
            grid = (kernels.triton.cdiv(heads * depth * capacity, encoders.block), end - base)
            (K0, K1, K2, K3), (V0, V1, V2, V3) = new_k.stride(), new_v.stride()
            kernels._update_and_pack[grid](
                NEW_K=new_k[base:end], NEW_V=new_v[base:end],
                PACKED_K=packed_k[base:end], PACKED_V=packed_v[base:end],
                POINTERS=pointers, RANGES=ranges,
                BATCH=entries, ROW_BASE=first + base, SLOTS=entries,
                CAPACITY=capacity, LAYER=layer, STACK_MODE=0,
                K0=K0, K1=K1, K2=K2, K3=K3, V0=V0, V1=V1, V2=V2, V3=V3,
                HEADS=heads, DEPTH=depth, BLOCK=encoders.block, num_warps=4)
            encoders.triton_launches += 1
    return True


def _eager_pack(entries: list[Entry], counts: list[int], layer: int, new_k: Tensor,
                new_v: Tensor, capacity: int) -> tuple[Tensor, Tensor]:
    """``BatchedHistoryCache._pack_keys_values`` across identities."""
    for i, (entry, count) in enumerate(zip(entries, counts)):
        start = entry.length
        entry.keys[layer][:, start:start + count].copy_(new_k[i, :, :count])
        entry.values[layer][:, start:start + count].copy_(new_v[i, :, :count])
    if all(entry.capacity >= capacity for entry in entries):
        # Unwritten cache positions stay zero: stacking gives the zero-padded pack.
        return (torch.stack([entry.keys[layer][:, :capacity] for entry in entries]),
                torch.stack([entry.values[layer][:, :capacity] for entry in entries]))
    packed_k = new_k.new_zeros(len(entries), new_k.shape[1], capacity, new_k.shape[3])
    packed_v = torch.zeros_like(packed_k)
    for i, (entry, count) in enumerate(zip(entries, counts)):
        end = entry.length + count
        packed_k[i, :, :end].copy_(entry.keys[layer][:, :end])
        packed_v[i, :, :end].copy_(entry.values[layer][:, :end])
    return packed_k, packed_v


def _eager_memory(plan: EncodePlan, state: Tensor) -> Tensor:
    """New memory rows into each entry, then the ``[E, T, width]`` gather."""
    length = plan.length
    for i, (entry, count) in enumerate(zip(plan.entries, plan.counts)):
        entry.memory[entry.length:entry.length + count].copy_(state[i, :count])
    if all(entry.capacity >= length for entry in plan.entries):
        return torch.stack([entry.memory[:length] for entry in plan.entries])
    memory = state.new_zeros(len(plan.entries), length, state.shape[-1])
    for i, (entry, count) in enumerate(zip(plan.entries, plan.counts)):
        end = entry.length + count
        memory[i, :end].copy_(entry.memory[:end])
    return memory


@torch.no_grad()
def merged_encode(encoders: SnapshotEncoders, plan: EncodePlan, fields: dict[str, Tensor],
                  device) -> Tensor:
    """Encode every planned stream; returns ``[plan.streams, plan.length, width]``,
    each row the stream's public memory, zero past its length (what
    ``cache.encode`` returns, cut or padded to ``plan.length``)."""
    memory = None
    if plan.entries:
        merged = _forward(encoders, plan, fields)
        encoders.calls += 1
        encoders.streams += len(plan.entries)
        encoders.tokens += sum(plan.counts)
        if plan.direct:
            memory = merged
    width = (encoders.config.width if encoders.config is not None
             else plan.fallback[0][0].actor.config.width)
    if memory is None:
        memory = torch.zeros(plan.streams, plan.length, width, device=device)
        if plan.entries:
            memory.index_copy_(0, torch.as_tensor(plan.stream_rows, device=device), merged)
    for cache, keys, streams, first, last in plan.fallback:
        _, encoded = cache.encode(keys, streams)
        span = min(plan.length, encoded.shape[1])
        memory[first:last, :span].copy_(encoded[:, :span])
        encoders.fallbacks += 1
        encoders.fallback_streams += last - first
    return memory


def _attend_chunk(encoders, plan: EncodePlan, layer: int, first: int, last: int, q: Tensor,
                  new_k: Tensor, new_v: Tensor, allowed: Tensor,
                  table: tuple[Tensor, Tensor] | None) -> Tensor:
    """Write and pack entries ``first:last``' K/V, then their SDPA. The packed
    K/V are released on return, before the next chunk allocates its own."""
    cfg = encoders.config
    heads, depth = cfg.heads, cfg.width // cfg.heads
    packed = None
    encoders.pack_chunks += 1
    if table is not None:
        packed_k = new_k.new_empty(last - first, heads, plan.capacity, depth)
        packed_v = new_v.new_empty(last - first, heads, plan.capacity, depth)
        if _launch(encoders, new_k[first:last], new_v[first:last], packed_k, packed_v, *table,
                   plan.capacity, layer, heads, depth, first):
            packed = packed_k, packed_v
    if packed is None:
        encoders.eager_packs += 1
        packed = _eager_pack(plan.entries[first:last], plan.counts[first:last], layer,
                             new_k[first:last], new_v[first:last], plan.capacity)
    return F.scaled_dot_product_attention(q[first:last], *packed,
                                          attn_mask=allowed[first:last], dropout_p=0.0)


def _forward(encoders: SnapshotEncoders, plan: EncodePlan, fields: dict[str, Tensor]) -> Tensor:
    tensors, cfg, eps = encoders.tensors, encoders.config, encoders.eps
    S, M, w, E = plan.slots, plan.per_slot, plan.width, len(plan.entries)
    width, heads = cfg.width, cfg.heads
    depth = width // heads
    cells = M * w
    device = tensors["bos"].device
    dtype = tensors["bos"].dtype
    grid = fields["grid"]
    # Embeddings, [S, M * w, width]: (public + round) + phase, BOS, then position.
    state = _linear(fields["tokens"].view(S, cells, TOKEN_DIM).to(dtype), tensors, "public", S)
    state = state + tensors["round"].view(-1, width)[fields["rounds"]].view(S, cells, width)
    state = state + tensors["phase"].view(-1, width)[fields["phases"]].view(S, cells, width)
    state = torch.where(fields["bos"].view(S, cells, 1), tensors["bos"][:S], state)
    table = encoders.position_table(plan.capacity, device, dtype)
    state = state + table[fields["positions"]].view(S, cells, width)
    # Entry e's query j sits at start_e + j; it sees its own keys up to there.
    key_positions = torch.arange(plan.capacity, device=device)
    query_positions = fields["starts"][:, None] + key_positions[:w][None]
    allowed = ((key_positions[None, None] <= query_positions[:, :, None])
               & (key_positions[None, None] < fields["ends"][:, None, None]))[:, None]
    triton = plan.triton
    if triton:
        pointer_table = fields["pointers"]
        ranges = fields["ranges"]
    per_entry = 2 * heads * plan.capacity * depth * state.element_size()
    chunk = (E if encoders.max_pack_bytes is None
             else max(1, min(E, encoders.max_pack_bytes // per_entry)))
    for layer in range(cfg.layers):
        hidden = _norm(state, tensors, f"n1_{layer}", S, eps)
        qkv = _linear(hidden, tensors, f"in_{layer}", S).view(S * M, w, 3, heads, depth)
        if not plan.full:
            qkv = qkv.index_select(0, grid)
        q, new_k, new_v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
        parts = [_attend_chunk(encoders, plan, layer, first, min(E, first + chunk), q,
                               new_k, new_v, allowed,
                               (pointer_table, ranges) if triton else None)
                 for first in range(0, E, chunk)]
        attended = parts[0] if len(parts) == 1 else torch.cat(parts)
        del parts
        attended = attended.transpose(1, 2).reshape(E, w, width)
        if not plan.full:
            attended = attended.new_zeros(S * M, w, width).index_copy_(0, grid, attended)
        state = state + _linear(attended.view(S, cells, width), tensors, f"out_{layer}", S)
        hidden = _norm(state, tensors, f"n2_{layer}", S, eps)
        hidden = torch.relu(_linear(hidden, tensors, f"l1_{layer}", S))
        state = state + _linear(hidden, tensors, f"l2_{layer}", S)
    if encoders.has_stream_norm:
        state = _norm(state, tensors, "stream_norm", S, eps)
    state = _norm(state, tensors, "final_norm", S, eps)
    state = state.view(S * M, w, width)
    if not plan.full:
        state = state.index_select(0, grid)
    memory = None
    if triton:
        # One head of depth `width`: key and value pointers both name Entry.memory
        # (rows 1 + 2L and 2 + 2L of the table); the same thread stores the
        # same bits twice. Positions past each end gather as zeros.
        new = state[:, None]
        out = state.new_empty(E, 1, plan.length, width)
        if _launch(encoders, new, new, out, out, pointer_table, ranges, plan.length,
                   cfg.layers, 1, width):
            memory = out.view(E, plan.length, width)
    if memory is None:
        encoders.eager_packs += 1
        memory = _eager_memory(plan, state)
    for entry, count, cache in zip(plan.entries, plan.counts, plan.caches):
        entry.length += count
        cache.encoded_tokens += count
    return memory
