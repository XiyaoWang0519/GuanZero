"""v3 match memory: per-seat, per-round summaries carried across a match.

Mechanism, following DESIGN.md 7.3. When a round of a match ends, its whole
public token stream is encoded by the same causal public-stream layers the v2
tower owns, and the encoding is mean-pooled at the tokens of each absolute
seat. That gives one summary vector per seat per finished round, built from
public information only. At a decision in round ``r`` the private query
additionally attends over the summaries of the rounds strictly before ``r`` of
the same match, each key tagged with an absolute-seat embedding and a
round-distance embedding.

Two properties are deliberate.

*Causality.* Only rounds with a strictly smaller match-relative index enter the
memory, so round ``r`` never sees round ``r + 1``, and the summaries are built
from public tokens alone, so nothing private crosses a round boundary. The
tower is the **no_history** tower selected by task 3, so the current round's own
public stream is not visible either: the only sequence information the model
has is the match memory.

*Missing labels.* DESIGN 7.3 would also pool each seat's revealed remaining
cards at the round end. The task 2/3 collection schema (`train/logs.py`
schema 2) stores hidden hands only as the per-decision supervision target and
records nothing at the round end, so no revealed-hand field exists to pool.
Those inputs are therefore omitted; the summaries are public-stream only. A
future collection could add them, and `summarise` is where they would enter.

Control. `memory_masked` is the identical network with the memory keys absent
from the cross-attention, so its private query attends to the BOS token alone
and the model is functionally the no_history tower. Parameter count is exactly
identical (the tag embeddings and the public-stream layers still exist, they
just receive no gradient), which is what makes the paired comparison a
memory-only contrast rather than a capacity contrast.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from train.belief_model import HistoryBelief
from train.belief_probe import collate, count_parameters
from train.logs import TOKEN_DIM


class MemoryBelief(HistoryBelief):
    """The no_history query tower plus cross-round, public-only match memory."""

    def __init__(self, obs_dim: int, width: int = 256, layers: int = 4, *,
                 memory_rounds: int = 8, memory_masked: bool = False) -> None:
        if memory_rounds <= 0:
            raise ValueError("memory_rounds must be positive")
        super().__init__(obs_dim, width, layers, no_history=True)
        self.memory_rounds = memory_rounds
        self.memory_masked = memory_masked
        # Absolute seat and round distance tags. These are the only parameters
        # the memory adds on top of the no_history tower.
        self.memory_seat = nn.Embedding(4, width)
        self.memory_distance = nn.Embedding(memory_rounds, width)
        nn.init.normal_(self.memory_seat.weight, std=0.02)
        nn.init.normal_(self.memory_distance.weight, std=0.02)

    def summarise(self, tokens: torch.Tensor, lengths: torch.Tensor
                  ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool a batch of finished rounds into one vector per absolute seat.

        `tokens` is ``(rounds, length, TOKEN_DIM)`` of public tokens and
        `lengths` the true stream length of each round. Returns the summaries
        ``(rounds, 4, width)`` and a ``(rounds, 4)`` flag that is false for a
        seat that acted in no token of that round.
        """
        stream, _ = self.encode_stream(tokens, lengths)
        span = torch.arange(tokens.shape[1], device=tokens.device)
        active = (span[None] < lengths[:, None]).to(stream.dtype)
        # Token j of a round occupies stream position j + 1; BOS is position 0.
        weights = tokens[:, :, :4] * active[:, :, None]
        counts = weights.sum(1)
        pooled = torch.einsum("ulw,uls->usw", stream[:, 1:], weights)
        return pooled / counts.clamp(min=1.0)[..., None], counts > 0

    def memory_keys(self, memory: dict[str, torch.Tensor]
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build tagged memory keys and their padding mask for one batch."""
        summary, present = self.summarise(memory["tokens"], memory["lengths"])
        index, distance = memory["index"], memory["distance"]
        safe = index.clamp(min=0)
        keys = summary[safe]
        keys = (keys + self.memory_seat.weight[None, None]
                + self.memory_distance(distance.clamp(0, self.memory_rounds - 1))[:, :, None])
        padding = (index < 0)[:, :, None] | ~present[safe]
        rows, slots = index.shape
        return keys.reshape(rows, slots * 4, self.width), padding.reshape(rows, slots * 4)

    def forward(self, obs, tokens, lengths, seat, memory=None):
        stream, padding = self.encode_public(tokens, lengths)
        if (memory is not None and not self.memory_masked
                and memory["tokens"].shape[0] and memory["index"].shape[1]):
            keys, key_padding = self.memory_keys(memory)
            stream = torch.cat((stream, keys), dim=1)
            padding = torch.cat((padding, key_padding), dim=1)
        query = (self.private(obs) + self.seat(seat))[:, None]
        attended, _ = self.query(query, stream, stream, key_padding_mask=padding,
                                 need_weights=False)
        state = self.attention_norm(query + attended)
        state = self.output_norm(state + self.feed_forward(state))
        return self.head(state[:, 0]).reshape(-1, 3, 54, 3)


def matched_memory_models(obs_dim: int, width: int = 256, layers: int = 4, *,
                          memory_rounds: int = 8, tolerance: float = 0.001
                          ) -> dict[str, nn.Module]:
    """Pair the memory model with its masked control at identical parameters.

    Both models are matched to the scaled no_history tower of task 3: the
    memory reuses that tower's public-stream layers and its existing
    cross-attention block, so the only added tensors are the absolute-seat and
    round-distance tag embeddings, which is checked exactly. `tolerance` bounds
    the resulting relative difference and is 0.1% by default, which the
    production shape (width 256, four layers) satisfies; a tiny test shape has
    to loosen it because the same fixed tags are a larger share of a small
    tower. The masked control is a deep copy, so it is parameter identical.
    """
    memory = MemoryBelief(obs_dim, width, layers, memory_rounds=memory_rounds)
    masked = copy.deepcopy(memory)
    masked.memory_masked = True
    reference = count_parameters(HistoryBelief(obs_dim, width, layers, no_history=True))
    total = count_parameters(memory)
    if total - reference != (4 + memory_rounds) * width:
        raise ValueError("the memory adds tensors beyond its seat and distance tags")
    if abs(total - reference) / reference > tolerance:
        raise ValueError("memory tower is not parameter-matched to the no_history tower")
    return {"memory": memory, "memory_masked": masked}


def attach_memory(rounds: list[dict], memory_rounds: int = 8) -> dict[str, list[dict]]:
    """Group rounds into matches and link each round to its earlier rounds.

    Matches are keyed by the collection's match group, which encodes the
    ``(env_id, match_id)`` pair, and ordered by `round_index`. Each round gets
    a `previous` list of up to `memory_rounds` earlier rounds of the same
    match, most recent first. Previous rounds keep their full, never
    subsampled public token stream, which is what the summaries read.
    """
    if memory_rounds <= 0:
        raise ValueError("memory_rounds must be positive")
    matches: dict[str, list[dict]] = {}
    for record in rounds:
        matches.setdefault(record["group"], []).append(record)
    for group, records in matches.items():
        records.sort(key=lambda r: (int(r.get("round_index", -1)), id(r)))
        seen: list[dict] = []
        for record in records:
            index = int(record.get("round_index", -1))
            earlier = [r for r in seen if 0 <= int(r["round_index"]) < index]
            record["previous"] = list(reversed(earlier))[:memory_rounds]
            seen.append(record)
    return matches


def memory_collate(items: list[tuple[dict, int]], device: str, memory_rounds: int = 8,
                   max_tokens: int = 0) -> dict[str, torch.Tensor]:
    """Collate decisions plus the earlier-round streams their memory needs.

    Earlier rounds shared by several decisions of the batch are encoded once:
    `index` points into the deduplicated `tokens`/`lengths` tables and is -1
    for an unused slot, and `distance` is the round-index gap, at least one.
    """
    batch = collate(items, device)
    unique: dict[int, int] = {}
    order: list[dict] = []
    slots: list[list[tuple[int, int]]] = []
    for record, _ in items:
        row = []
        for previous in record.get("previous", ())[:memory_rounds]:
            key = id(previous)
            if key not in unique:
                unique[key] = len(order)
                order.append(previous)
            gap = int(record["round_index"]) - int(previous["round_index"])
            row.append((unique[key], max(1, gap)))
        slots.append(row)
    width = max((len(row) for row in slots), default=0)
    index = np.full((len(items), width), -1, np.int64)
    distance = np.zeros((len(items), width), np.int64)
    for row, entries in enumerate(slots):
        for column, (target, gap) in enumerate(entries):
            index[row, column] = target
            distance[row, column] = min(gap, memory_rounds) - 1
    length = max((len(r["tokens"]) for r in order), default=0)
    if max_tokens:
        length = min(length, max_tokens)
    tokens = np.zeros((len(order), length, TOKEN_DIM), np.float32)
    lengths = np.zeros(len(order), np.int64)
    for row, record in enumerate(order):
        keep = min(len(record["tokens"]), length)
        tokens[row, :keep] = record["tokens"][:keep]
        lengths[row] = keep
    batch["memory"] = {
        "tokens": torch.tensor(tokens, device=device),
        "lengths": torch.tensor(lengths, device=device),
        "index": torch.tensor(index, device=device),
        "distance": torch.tensor(distance, device=device),
    }
    return batch


def memory_forward(model: nn.Module, batch: dict) -> torch.Tensor:
    """Call a memory tower with the extra memory tensors of `memory_collate`."""
    return model(batch["obs"], batch["tokens"], batch["lengths"], batch["seat"],
                 memory=batch.get("memory"))
