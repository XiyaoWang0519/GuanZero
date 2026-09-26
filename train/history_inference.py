"""Batched, public-only, per-policy match KV cache for frozen rollout intervals.

Every entry belongs to one policy object and one (environment, match). Raw
histories remain authoritative. Ordinary optimizer/load_state_dict changes
invalidate the whole policy cache and the next query rebuilds from raw events.
Private observations never enter the cache. PPO recomputation never uses it.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from train.history_attention import finish, project, validate
from train.history_model import HistoryActor, PublicStream, StreamBatch, sinusoidal
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
        self.encoded_tokens = 0
        self.rebuilds = 0

    def _signature(self) -> tuple:
        # .data mutation is unsupported, as with PyTorch autograd versioning.
        return tuple((id(p), p._version, p.device, p.dtype) for p in self.actor.parameters())

    def clear(self) -> None:
        self.entries.clear()
        self.signature = self._signature()

    def prune(self, active: set[tuple[int, int]]) -> None:
        for key in list(self.entries):
            if key not in active:
                del self.entries[key]

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for e in self.entries.values()
                   for t in (*e.keys, *e.values, e.memory))

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
                       [prototype.new_empty(shape) for _ in range(cfg.layers)],
                       [prototype.new_empty(shape) for _ in range(cfg.layers)],
                       prototype.new_empty(capacity, cfg.width))
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
    def encode(self, keys: list[tuple[int, int]], streams: list[PublicStream]
               ) -> tuple[StreamBatch, Tensor]:
        if not keys or len(keys) != len(streams) or len(set(keys)) != len(keys):
            raise ValueError("one distinct match key per public stream required")
        if self._signature() != self.signature:
            self.clear()
        entries = [self._entry(key, stream) for key, stream in zip(keys, streams)]
        targets = [s.prefix + 1 for s in streams]
        while any(e.length < n for e, n in zip(entries, targets)):
            indices = [i for i, (e, n) in enumerate(zip(entries, targets)) if e.length < n]
            self._append([entries[i] for i in indices],
                         [min(self.chunk_size, targets[i] - entries[i].length) for i in indices])
        # Power-of-two shapes reduce allocator churn without truncating history.
        size = bucket(max(targets))
        memory = self.actor.bos.new_zeros(len(entries), size, self.actor.config.width)
        for i, entry in enumerate(entries):
            memory[i, :entry.length].copy_(entry.memory[:entry.length])
        device = memory.device
        metadata = StreamBatch(torch.empty(len(entries), 0, TOKEN_DIM, dtype=torch.uint8, device=device),
                               torch.empty(len(entries), 0, dtype=torch.long, device=device),
                               torch.empty(len(entries), 0, dtype=torch.long, device=device),
                               torch.tensor([n - 1 for n in targets], device=device))
        return metadata, memory

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
                tokens[i, offset:offset+count] = np.stack(entry.stream.tokens[begin:end])
                rounds[i, offset:offset+count] = entry.stream.rounds[begin:end]
                phases[i, offset:offset+count] = entry.stream.phases[begin:end]
        t = torch.as_tensor(tokens, device=device)
        r, p = torch.as_tensor(rounds, device=device), torch.as_tensor(phases, device=device)
        state = (actor.public(t.to(dtype)) + actor.round_embedding(r.clamp(0, cfg.max_rounds-1))
                 + actor.phase_embedding(p.clamp(0, 3)))
        for i, start in enumerate(starts):
            if start == 0:
                state[i, 0] = actor.bos[0, 0]
        positions = torch.tensor(starts, device=device)[:, None] + torch.arange(width, device=device)
        state = state + sinusoidal(capacity, cfg.width, device, dtype)[positions.clamp(max=capacity-1)]
        key_positions = torch.arange(capacity, device=device)
        allowed = ((key_positions[None, None] <= positions[:, :, None])
                   & (key_positions[None, None] < torch.tensor(ends, device=device)[:, None, None]))
        for layer_index, layer in enumerate(actor.stream.layers):
            q, new_k, new_v = project(layer, state)
            packed_k = state.new_zeros(batch, cfg.heads, capacity, cfg.width // cfg.heads)
            packed_v = torch.zeros_like(packed_k)
            for i, (entry, count) in enumerate(zip(entries, counts)):
                start, end = entry.length, entry.length + count
                entry.keys[layer_index][:, start:end].copy_(new_k[i, :, :count])
                entry.values[layer_index][:, start:end].copy_(new_v[i, :, :count])
                packed_k[i, :, :end].copy_(entry.keys[layer_index][:, :end])
                packed_v[i, :, :end].copy_(entry.values[layer_index][:, :end])
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
