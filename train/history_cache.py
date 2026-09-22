"""Bounded, single-environment inference cache for the offline history tower.

This is a cache foundation, not batched rollout or RL integration. Public events
are appended once; any seat may query the resulting shared prefix. The model
must remain in evaluation mode with frozen weights and architecture. After a
weight change, clear/recreate the cache and replay the public prefix. Call
``clear()`` at every round boundary. Overflow raises; history is never truncated.
"""
from __future__ import annotations

import math
from numbers import Integral

import torch
from torch.nn import functional as F

from train.belief_model import HistoryBelief
from train.logs import TOKEN_DIM


class PublicHistoryCache:
    """One BOS-prefixed public KV cache, with a bound excluding BOS.

    ``append_public`` takes [events, TOKEN_DIM] (or one [TOKEN_DIM] event),
    with no private observation argument. ``query`` takes [queries, obs_dim]
    plus absolute seats, returning [queries, 3, 54, 3] hidden-count logits.
    Inputs are copied/coerced to the model's device and floating dtype; neither
    API stores private observations. Methods run under torch inference mode.
    """

    def __init__(self, model: HistoryBelief, *, max_tokens: int = 160) -> None:
        if not isinstance(max_tokens, Integral) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self.model = model
        self.max_tokens = int(max_tokens)
        self.clear()

    def _validate_model(self) -> None:
        if any(module.training for module in self.model.modules()):
            raise ValueError("history cache requires an evaluation-mode model")
        if self.model.no_history:
            raise ValueError("history cache requires a model with history enabled")
        for layer in self.model.stream.layers:
            attn = layer.self_attn
            if (layer.norm_first or not attn._qkv_same_embed_dim
                    or attn.bias_k is not None or attn.bias_v is not None or attn.add_zero_attn):
                raise ValueError("history cache supports post-norm layers with standard self-attention")
        parameters = list(self.model.parameters())
        if any(p.device != self.model.bos.device or p.dtype != self.model.bos.dtype for p in parameters):
            raise ValueError("history cache requires one model device and floating dtype")

    def _signature(self) -> tuple:
        # Version counters catch ordinary in-place optimizer/load_state_dict
        # changes without hashing/copying weights. Direct .data writes are not
        # supported; callers must keep model weights and architecture frozen.
        return tuple((name, id(p), p._version, p.device, p.dtype)
                     for name, p in self.model.named_parameters())

    def _assert_frozen(self) -> None:
        if any(module.training for module in self.model.modules()):
            raise ValueError("history cache requires an evaluation-mode model")
        if self.model.no_history or self._signature() != self._model_signature:
            raise ValueError("model changed; clear or recreate cache and replay the public prefix")

    @property
    def public_length(self) -> int:
        """Number of appended public events, excluding BOS."""
        return self._memory.shape[1] - 1

    @property
    def public_memory(self) -> torch.Tensor:
        """An owned snapshot of top-layer memory, including BOS."""
        self._assert_frozen()
        return self._memory.clone()

    @torch.inference_mode()
    def clear(self) -> None:
        """Drop the previous round, accept current frozen weights, and encode BOS."""
        self._validate_model()
        self.device, self.dtype = self.model.bos.device, self.model.bos.dtype
        self._model_signature = self._signature()
        self._keys: list[torch.Tensor | None] = [None] * len(self.model.stream.layers)
        self._values: list[torch.Tensor | None] = [None] * len(self.model.stream.layers)
        self._memory = self.model.bos.new_empty((1, 0, self.model.width))
        self._append_embedding(self.model.bos)

    def _position(self, index: int) -> torch.Tensor:
        frequency = torch.exp(torch.arange(0, self.model.width, 2, device=self.device,
                                           dtype=self.dtype)
                              * (-math.log(10000.0) / self.model.width))
        result = torch.zeros(1, 1, self.model.width, device=self.device, dtype=self.dtype)
        result[..., 0::2] = torch.sin(index * frequency)
        result[..., 1::2] = torch.cos(index * frequency)
        return result

    def _append_embedding(self, embedding: torch.Tensor) -> None:
        state = embedding + self._position(self._memory.shape[1])
        for index, layer in enumerate(self.model.stream.layers):
            attention = layer.self_attn
            q, k, v = F.linear(state, attention.in_proj_weight,
                               attention.in_proj_bias).chunk(3, dim=-1)
            shape = (1, 1, attention.num_heads, attention.head_dim)
            q, k, v = (value.reshape(shape).transpose(1, 2) for value in (q, k, v))
            if self._keys[index] is not None:
                k = torch.cat((self._keys[index], k), dim=2)
                v = torch.cat((self._values[index], v), dim=2)
            self._keys[index], self._values[index] = k, v
            # The new query is the LAST position and may see the whole cache.
            # is_causal=True with a one-token query would mask all but the first key.
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
            attended = attended.transpose(1, 2).reshape(1, 1, self.model.width)
            attended = F.linear(attended, attention.out_proj.weight, attention.out_proj.bias)
            state = layer.norm1(state + layer.dropout1(attended))
            feed_forward = layer.linear2(layer.dropout(layer.activation(layer.linear1(state))))
            state = layer.norm2(state + layer.dropout2(feed_forward))
        if self.model.stream.norm is not None:
            state = self.model.stream.norm(state)
        self._memory = torch.cat((self._memory, state), dim=1)

    @torch.inference_mode()
    def append_public(self, tokens: torch.Tensor) -> None:
        self._assert_frozen()
        events = torch.as_tensor(tokens, device=self.device)
        if events.ndim == 1:
            events = events[None]
        if events.ndim != 2 or events.shape[1] != TOKEN_DIM:
            raise ValueError(f"public events must have shape [events, {TOKEN_DIM}]")
        if self.public_length + len(events) > self.max_tokens:
            raise ValueError(f"public history exceeds max_tokens={self.max_tokens}; no truncation performed")
        if (not bool(torch.isfinite(events).all()) or bool(((events != 0) & (events != 1)).any())
                or not bool((events[:, :4].sum(1) == 1).all())):
            raise ValueError("public events must be finite binary features with one absolute seat")
        if bool(events[:, 150:158].any()):
            raise ValueError("private tribute structure cannot enter the public cache")
        events = events.to(dtype=self.dtype)
        # Process incrementally even for a multi-event append so storage and
        # attention remain bounded by the same explicit prefix limit.
        for event in events:
            self._append_embedding(self.model.public(event).reshape(1, 1, -1))

    @torch.inference_mode()
    def encode_private(self, obs: torch.Tensor, seat: torch.Tensor | int) -> torch.Tensor:
        """Return private state embeddings [queries, width], leaving public cache untouched."""
        self._assert_frozen()
        observations = torch.as_tensor(obs, device=self.device, dtype=self.dtype)
        if observations.ndim == 1:
            observations = observations[None]
        if (observations.ndim != 2 or not len(observations)
                or observations.shape[1] != self.model.private[0].in_features
                or not bool(torch.isfinite(observations).all())):
            raise ValueError("private observations must be finite [queries, obs_dim] features")
        seats = torch.as_tensor(seat, device=self.device)
        if seats.ndim == 0:
            seats = seats.expand(len(observations))
        if (seats.shape != (len(observations),) or seats.dtype == torch.bool
                or torch.is_floating_point(seats) or seats.is_complex()
                or bool(((seats < 0) | (seats > 3)).any())):
            raise ValueError("absolute seats must be integers in [0, 3], one per query")
        query = (self.model.private(observations) + self.model.seat(seats.long()))[:, None]
        memory = self._memory.expand(len(observations), -1, -1)
        attended, _ = self.model.query(query, memory, memory, need_weights=False)
        state = self.model.attention_norm(query + attended)
        state = self.model.output_norm(state + self.model.feed_forward(state))
        return state[:, 0]

    @torch.inference_mode()
    def query(self, obs: torch.Tensor, seat: torch.Tensor | int) -> torch.Tensor:
        """Return hidden-hand logits without changing the shared public cache."""
        return self.model.head(self.encode_private(obs, seat)).reshape(-1, 3, 54, 3)
