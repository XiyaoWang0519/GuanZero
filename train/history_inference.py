"""Batched, public-only, per-policy match KV cache for frozen rollout intervals.

Every entry belongs to one policy object and one (environment, match). Raw
histories remain authoritative. Ordinary optimizer/load_state_dict changes
invalidate the whole policy cache and the next query rebuilds from raw events.
Private observations never enter the cache. PPO recomputation never uses it.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from train.history_attention import finish, project, validate
from train.history_model import HistoryActor, PublicStream, StreamBatch, sinusoidal
from train.history_transfers import upload_arrays
from train.logs import TOKEN_DIM


def bucket(length: int) -> int:
    return max(32, 1 << (max(length, 1) - 1).bit_length())


@dataclass
class Entry:
    stream: PublicStream
    generation: int
    length: int                    # includes BOS
    capacity: int
    keys: list[Tensor]
    values: list[Tensor]
    memory: Tensor


class BatchedHistoryCache:
    def __init__(self, actor: HistoryActor, chunk_size: int = 128) -> None:
        if actor.config.window or chunk_size < 1:
            raise ValueError("KV cache requires full history and a positive prefill chunk")
        validate(actor.stream)
        self.actor = actor
        self.chunk_size = chunk_size
        self.entries: dict[tuple[int, int], Entry] = {}
        self.signature = self._signature()
        self._frozen_signature: tuple | None = None
        self.encoded_tokens = 0
        self.rebuilds = 0
        # Keep only the last exact shape. Rebuilding after a bucket change avoids
        # assuming that differently sized sin/cos kernels are bitwise identical.
        self._position_cache: tuple[tuple, Tensor, Tensor] | None = None

    def _signature(self) -> tuple:
        # .data mutation is unsupported, as with PyTorch autograd versioning.
        # Walk live registries in parameters() order without constructing module
        # names. Re-reading them still detects replaced parameters/submodules.
        modules, parameters, stack, result = set(), set(), [self.actor], []
        while stack:
            module = stack.pop()
            if module is None or id(module) in modules:
                continue
            modules.add(id(module))
            for parameter in module._parameters.values():
                if parameter is not None and id(parameter) not in parameters:
                    parameters.add(id(parameter))
                    result.append((id(parameter), parameter._version,
                                   parameter.device, parameter.dtype))
            stack.extend(reversed(tuple(module._modules.values())))
        return tuple(result)

    def clear(self) -> None:
        self.entries.clear()
        self._position_cache = None
        self.signature = self._signature()

    def refresh_weights(self) -> None:
        """Check live parameters unless the caller owns a frozen interval."""
        if self._frozen_signature is None and self._signature() != self.signature:
            self.clear()

    @contextmanager
    def frozen_weights(self):
        """Validate at both ends of an explicitly frozen collection interval.

        The trainer owns its actor weights while collecting. Standalone encode
        calls keep their per-call mutation checks. A mutation inside this scope
        raises and clears stale cache state on exit. The caller must discard
        that failed rollout; the trainer propagates the error before learning.
        """
        if self._frozen_signature is not None:
            raise RuntimeError("nested frozen-weight intervals are unsupported")
        self.refresh_weights()
        self._frozen_signature = self.signature
        try:
            yield self
        finally:
            expected = self._frozen_signature
            self._frozen_signature = None
            if self._signature() != expected:
                self.clear()
                raise RuntimeError("actor weights changed during frozen collection")

    def prune(self, active: set[tuple[int, int]]) -> None:
        for key in list(self.entries):
            if key not in active:
                del self.entries[key]

    @property
    def bytes(self) -> int:
        entry_bytes = sum(t.numel() * t.element_size() for e in self.entries.values()
                          for t in (*e.keys, *e.values, e.memory))
        position_bytes = (sum(t.numel() * t.element_size() for t in self._position_cache[1:])
                          if self._position_cache is not None else 0)
        return entry_bytes + position_bytes

    def _position_buffers(self, capacity: int) -> tuple[Tensor, Tensor]:
        prototype = self.actor.bos
        key = (capacity, self.actor.config.width, prototype.device, prototype.dtype)
        cached = self._position_cache
        if cached is None or cached[0] != key:
            cached = (key, sinusoidal(*key), torch.arange(capacity, device=prototype.device))
            self._position_cache = cached
        return cached[1], cached[2]

    def _entry(self, key: tuple[int, int], stream: PublicStream) -> Entry:
        size = stream.prefix + 1
        old = self.entries.get(key)
        # A replaced/reset stream starts from BOS, even if a caller reuses a key.
        if old is not None and (old.stream is not stream or old.generation != stream.generation
                                or old.length > size):
            old = None
        if old is not None and old.capacity >= size:
            return old
        capacity = bucket(size)
        cfg, prototype = self.actor.config, self.actor.bos
        shape = (cfg.heads, capacity, cfg.width // cfg.heads)
        result = Entry(stream, stream.generation, 0, capacity,
                       [prototype.new_zeros(shape) for _ in range(cfg.layers)],
                       [prototype.new_zeros(shape) for _ in range(cfg.layers)],
                       prototype.new_zeros(capacity, cfg.width))
        if old is not None:
            result.length = old.length
            for target, source in zip(result.keys + result.values, old.keys + old.values):
                target[:, :old.length].copy_(source[:, :old.length])
            result.memory[:old.length].copy_(old.memory[:old.length])
        else:
            self.rebuilds += 1
        self.entries[key] = result
        return result

    @torch.no_grad()
    def encode(self, keys: list[tuple[int, int]], streams: list[PublicStream], *,
               preuploaded_lengths: Tensor | None = None,
               host_lengths: tuple[int, ...] | None = None
               ) -> tuple[StreamBatch, Tensor]:
        """Encode streams, optionally reusing an already uploaded lengths tensor.

        The caller must have uploaded ``host_lengths`` unchanged, in stream
        order. Host values and tensor metadata are checked without a CUDA read;
        the returned metadata shares the supplied tensor's storage.
        """
        if not keys or len(keys) != len(streams) or len(set(keys)) != len(keys):
            raise ValueError("one distinct match key per public stream required")
        targets = [s.prefix + 1 for s in streams]
        if preuploaded_lengths is not None:
            if host_lengths != tuple(n - 1 for n in targets):
                raise ValueError("uploaded host lengths must match stream order and prefixes")
            if (preuploaded_lengths.shape != (len(streams),)
                    or preuploaded_lengths.dtype != torch.long
                    or preuploaded_lengths.device != self.actor.bos.device):
                raise ValueError("uploaded lengths must be int64 [streams] on the actor device")
        elif host_lengths is not None:
            raise ValueError("host lengths require an uploaded lengths tensor")
        self.refresh_weights()
        entries = [self._entry(key, stream) for key, stream in zip(keys, streams)]
        while any(e.length < n for e, n in zip(entries, targets)):
            indices = [i for i, (e, n) in enumerate(zip(entries, targets)) if e.length < n]
            self._append([entries[i] for i in indices],
                         [min(self.chunk_size, targets[i] - entries[i].length) for i in indices])
        # Power-of-two shapes reduce allocator churn without truncating history.
        memory = self._gather_memory(entries, self._memory_size(max(targets)))
        device = memory.device
        metadata = StreamBatch(torch.empty(len(entries), 0, TOKEN_DIM, dtype=torch.uint8, device=device),
                               torch.empty(len(entries), 0, dtype=torch.long, device=device),
                               torch.empty(len(entries), 0, dtype=torch.long, device=device),
                               preuploaded_lengths if preuploaded_lengths is not None else
                               torch.tensor([n - 1 for n in targets], device=device))
        return metadata, memory

    def _memory_size(self, longest: int) -> int:
        """Padded length of the memory ``encode`` returns for ``longest`` positions."""
        return bucket(longest)

    def _gather_memory(self, entries: list[Entry], size: int) -> Tensor:
        """``[len(entries), size, width]`` top-layer memory, zero past each length,
        in storage independent of the cache."""
        if all(entry.capacity >= size for entry in entries):
            # Padding in each cache stays zero. stack also keeps the returned
            # snapshot independent of later appends, including a single stream.
            return torch.stack([entry.memory[:size] for entry in entries])
        memory = self.actor.bos.new_zeros(len(entries), size, self.actor.config.width)
        for i, entry in enumerate(entries):
            memory[i, :entry.length].copy_(entry.memory[:entry.length])
        return memory

    def _pack_keys_values(self, entries: list[Entry], counts: list[int], layer_index: int,
                          new_k: Tensor, new_v: Tensor, capacity: int
                          ) -> tuple[Tensor, Tensor]:
        if all(entry.capacity >= capacity for entry in entries):
            for i, (entry, count) in enumerate(zip(entries, counts)):
                start, end = entry.length, entry.length + count
                entry.keys[layer_index][:, start:end].copy_(new_k[i, :, :count])
                entry.values[layer_index][:, start:end].copy_(new_v[i, :, :count])
            # Unwritten cache positions stay zero, so stacking these views gives
            # exactly the old zero-filled packing without per-entry copy calls.
            keys = [entry.keys[layer_index][:, :capacity] for entry in entries]
            values = [entry.values[layer_index][:, :capacity] for entry in entries]
            if len(entries) == 1 and entries[0].capacity == capacity:
                return keys[0].unsqueeze(0), values[0].unsqueeze(0)
            return torch.stack(keys), torch.stack(values)
        cfg = self.actor.config
        packed_k = new_k.new_zeros(len(entries), cfg.heads, capacity, cfg.width // cfg.heads)
        packed_v = torch.zeros_like(packed_k)
        for i, (entry, count) in enumerate(zip(entries, counts)):
            start, end = entry.length, entry.length + count
            entry.keys[layer_index][:, start:end].copy_(new_k[i, :, :count])
            entry.values[layer_index][:, start:end].copy_(new_v[i, :, :count])
            packed_k[i, :, :end].copy_(entry.keys[layer_index][:, :end])
            packed_v[i, :, :end].copy_(entry.values[layer_index][:, :end])
        return packed_k, packed_v

    def _append(self, entries: list[Entry], counts: list[int]) -> None:
        actor, cfg = self.actor, self.actor.config
        device, dtype = actor.bos.device, actor.bos.dtype
        batch, width = len(entries), max(counts)
        starts = [e.length for e in entries]
        ends = [s + n for s, n in zip(starts, counts)]
        capacity = bucket(max(ends))
        # Upload only new raw events, not the complete historical prefix.
        tokens = np.zeros((batch, width, TOKEN_DIM), np.uint8)
        rounds, phases = np.zeros((batch, width), np.int64), np.zeros((batch, width), np.int64)
        for i, (entry, count) in enumerate(zip(entries, counts)):
            begin = max(entry.length - 1, 0)
            offset = int(entry.length == 0)  # first new item is BOS
            count -= offset
            if count:
                end = begin + count
                tokens[i, offset:offset+count] = entry.stream.tokens[begin:end]
                rounds[i, offset:offset+count] = entry.stream.rounds[begin:end]
                phases[i, offset:offset+count] = entry.stream.phases[begin:end]
        t, r, p, start_positions, end_positions = upload_arrays(
            (tokens, rounds, phases, np.asarray(starts, dtype=np.int64),
             np.asarray(ends, dtype=np.int64)), device)
        state = (actor.public(t.to(dtype)) + actor.round_embedding(r.clamp(0, cfg.max_rounds-1))
                 + actor.phase_embedding(p.clamp(0, 3)))
        for i, start in enumerate(starts):
            if start == 0:
                state[i, 0] = actor.bos[0, 0]
        position_table, key_positions = self._position_buffers(capacity)
        positions = start_positions[:, None] + key_positions[:width]
        state = state + position_table[positions.clamp(max=capacity-1)]
        allowed = ((key_positions[None, None] <= positions[:, :, None])
                   & (key_positions[None, None] < end_positions[:, None, None]))
        for layer_index, layer in enumerate(actor.stream.layers):
            q, new_k, new_v = project(layer, state)
            packed_k, packed_v = self._pack_keys_values(entries, counts, layer_index,
                                                       new_k, new_v, capacity)
            attended = F.scaled_dot_product_attention(q, packed_k, packed_v,
                                                      attn_mask=allowed[:, None], dropout_p=0.0)
            state = finish(layer, state, attended)
        if actor.stream.norm is not None:
            state = actor.stream.norm(state)
        state = actor.stream_norm(state)
        for i, (entry, count) in enumerate(zip(entries, counts)):
            entry.memory[entry.length:entry.length+count].copy_(state[i, :count])
            entry.length += count
        self.encoded_tokens += sum(counts)
