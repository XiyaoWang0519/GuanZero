"""Multi-layer relative-position private cross-attention decoder and obs masking.

Implements the sequence-decoder architecture components:
- ``PRIVATE_OBS_KEEP_SLICES`` and ``mask_public_obs`` for device-only public
  observation masking without CUDA host synchronizations.
- ``compute_stream_pos_arrays``, ``extract_stream_pos``, and
  ``infer_decision_rounds`` for 1-based stream event seat/round metadata
  (0 reserved for BOS and padding).
- ``PrivateDecoderLayer`` and ``PrivateCrossDecoder`` for Pre-LN multi-layer
  cross-attention with query-conditioned relative seat/round/step attention
  biases, optional multi-slot queries, and FP32-equivalent dispatch across
  per-match, batched-private, wide-projection, and batched-match layouts.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from train.history_attention import private_sdpa

PRIVATE_OBS_MODES = ("full", "private_only", "card_counter")
PRIVATE_OBS_KEEP_SLICES = ((0, 108), (748, 790), (793, 805), (1430, 1439))
CARD_COUNTER_OBS_KEEP_SLICES = ((0, 216), (648, 805), (1430, 1439))
REL_SEAT_BUCKETS = 5
REL_STEP_EXACT = 32
REL_STEP_BUCKETS = 39


def private_obs_keep_mask(
    obs_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    mode: str = "private_only",
) -> Tensor:
    """Return a ``[1, obs_dim]`` tensor with ``1.0`` on kept dims for ``mode`` and ``0.0`` elsewhere."""
    if obs_dim < 0:
        raise ValueError("obs_dim must be non-negative")
    if mode not in PRIVATE_OBS_MODES:
        raise ValueError(f"private_obs_mode must be one of {PRIVATE_OBS_MODES}, got {mode!r}")
    if mode == "full":
        return torch.ones((1, obs_dim), device=device, dtype=dtype)
    slices = PRIVATE_OBS_KEEP_SLICES if mode == "private_only" else CARD_COUNTER_OBS_KEEP_SLICES
    mask = torch.zeros((1, obs_dim), device=device, dtype=dtype)
    for start, end in slices:
        lo = max(0, min(start, obs_dim))
        hi = max(0, min(end, obs_dim))
        if hi > lo:
            mask[:, lo:hi] = 1.0
    return mask


def mask_public_obs(
    obs: Tensor,
    seat: Tensor,
    prefix: Tensor,
    prob: Tensor | float,
    mode: str = "private_only",
) -> Tensor:
    """Deterministically mask public observation dims per row with probability ``prob``.

    Uses pure device tensor operations (no ``.item()`` or ``bool(prob)`` host syncs).
    """
    x = obs.float()
    keep = private_obs_keep_mask(x.shape[1], x.device, x.dtype, mode=mode)
    if mode == "full":
        return x
    k = min(108, x.shape[1])
    weights = torch.arange(1, k + 1, device=x.device, dtype=torch.int64)[None, :]
    hand_hash = (obs[:, :k].to(device=x.device, dtype=torch.int64) * weights).sum(dim=1)
    h = (
        prefix.to(device=x.device, dtype=torch.int64) * 2654435761
        + seat.to(device=x.device, dtype=torch.int64) * 40503
        + hand_hash * 1013904223
    ) & 0xFFFF
    u = h.to(x.dtype).unsqueeze(1) * (1.0 / 65536.0)
    p = (
        prob.to(device=x.device, dtype=x.dtype)
        if isinstance(prob, Tensor)
        else x.new_tensor(float(prob))
    )
    if p.ndim == 1:
        p = p.unsqueeze(-1)
    drop = u < p
    return torch.where(drop, x * keep, x)


def compute_stream_pos_arrays(
    tokens: np.ndarray,
    rounds: np.ndarray,
    max_rounds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pos_seat_events, pos_round_events)`` as ``uint8 [K]`` arrays.

    Values are 1-based (``1..4`` for seat, ``1..max_rounds`` for round) so ``0``
    remains reserved for BOS and padding.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be positive")
    tokens_arr = np.asarray(tokens)
    rounds_arr = np.asarray(rounds)
    if len(tokens_arr) != len(rounds_arr):
        raise ValueError("tokens and rounds must have the same length")
    if len(tokens_arr) == 0:
        empty = np.zeros(0, dtype=np.uint8)
        return empty, empty.copy()
    pos_seat = (tokens_arr[:, :4].argmax(axis=1) + 1).astype(np.uint8)
    pos_round = (np.clip(rounds_arr, 0, max_rounds - 1) + 1).astype(np.uint8)
    return pos_seat, pos_round


def extract_stream_pos(
    streams: Any,
    encoded_len: int,
    max_rounds: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Return ``(pos_seat, pos_round)`` as ``int64 [B, encoded_len]`` on ``device``.

    Position ``0`` (BOS) and padding positions ``s > length`` are ``0``; valid
    event positions ``1 <= s <= length`` hold ``1..4`` and ``1..max_rounds``.
    """
    if encoded_len < 0 or max_rounds < 1:
        raise ValueError("encoded_len must be non-negative and max_rounds positive")
    cached_seat = getattr(streams, "pos_seat", None)
    cached_round = getattr(streams, "pos_round", None)
    if cached_seat is not None and cached_round is not None:
        pos_seat = cached_seat.to(device=device, dtype=torch.long)
        pos_round = cached_round.to(device=device, dtype=torch.long)
        if pos_seat.shape[1] >= encoded_len:
            return pos_seat[:, :encoded_len], pos_round[:, :encoded_len]
        pad = encoded_len - pos_seat.shape[1]
        return F.pad(pos_seat, (0, pad), value=0), F.pad(pos_round, (0, pad), value=0)

    tokens = streams.tokens.to(device=device)
    rounds = streams.rounds.to(device=device)
    lengths = streams.lengths.to(device=device, dtype=torch.long)
    batch_size, raw_len = tokens.shape[0], tokens.shape[1]
    pos_seat = torch.zeros((batch_size, encoded_len), device=device, dtype=torch.long)
    pos_round = torch.zeros((batch_size, encoded_len), device=device, dtype=torch.long)
    avail = min(raw_len, max(encoded_len - 1, 0))
    if batch_size > 0 and avail > 0:
        valid = (
            torch.arange(1, avail + 1, device=device, dtype=torch.long)[None, :]
            <= lengths[:, None]
        )
        ev_seat = tokens[:, :avail, :4].argmax(dim=-1).to(torch.long) + 1
        ev_round = rounds[:, :avail].to(torch.long).clamp(0, max_rounds - 1) + 1
        pos_seat[:, 1 : 1 + avail] = torch.where(valid, ev_seat, 0)
        pos_round[:, 1 : 1 + avail] = torch.where(valid, ev_round, 0)
    return pos_seat, pos_round


def infer_decision_rounds(
    round_index: Tensor | None,
    pos_round: Tensor,
    match_index: Tensor,
    prefix: Tensor,
) -> Tensor:
    """Return 0-based ``int64 [n]`` decision rounds, inferring from ``pos_round`` if None."""
    if round_index is not None:
        return round_index.to(device=prefix.device, dtype=torch.long)
    if len(match_index) == 0 or pos_round.shape[1] == 0:
        return torch.zeros(len(match_index), device=prefix.device, dtype=torch.long)
    max_pos = max(pos_round.shape[1] - 1, 0)
    m_idx = match_index.to(device=pos_round.device, dtype=torch.long)
    clamped_prefix = prefix.to(device=pos_round.device, dtype=torch.long).clamp(min=0, max=max_pos)
    r = pos_round[m_idx, clamped_prefix].to(torch.long)
    return torch.where(r > 0, r - 1, torch.zeros_like(r))


def _wide_project(layer: nn.Linear, value: Tensor) -> Tensor:
    return layer(value)


def _project_independent(layer: nn.Linear, value: Tensor) -> Tensor:
    """Batch independent per-decision linear ops without widening the GEMM row count."""
    if len(value) == 0:
        return layer(value)
    if value.device.type == "cuda":
        return torch.cat([layer(value[i : i + 1]) for i in range(len(value))], dim=0)
    weights = layer.weight.t()[None].expand(len(value), -1, -1)
    return torch.baddbmm(layer.bias[None, None], value, weights)


def _compute_rel_buckets(
    pos_seat: Tensor,
    pos_round: Tensor,
    prefix: Tensor,
    seat: Tensor,
    my_round: Tensor,
    max_rounds: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Compute ``(allowed, seat_bucket, round_bucket, step_bucket)`` via broadcasting.

    ``pos_seat`` / ``pos_round`` have trailing dim ``S``; ``prefix`` / ``seat`` /
    ``my_round`` have trailing dim ``1``.
    """
    seq_len = pos_seat.shape[-1]
    positions = torch.arange(seq_len, device=pos_seat.device, dtype=torch.long)
    # Reshape positions to match leading dims up to [..., S]
    pos_view = positions.view(*((1,) * (pos_seat.ndim - 1)), seq_len)
    allowed = pos_view <= prefix
    valid_event = (pos_view > 0) & allowed & (pos_seat > 0)

    seat_bucket = torch.where(valid_event, ((pos_seat - 1 - seat) % 4) + 1, 0)
    round_bucket = torch.where(
        valid_event,
        (my_round - (pos_round - 1)).clamp(0, max_rounds - 1) + 1,
        0,
    )
    delta = (prefix - pos_view).clamp_min(0)
    coarse = (
        REL_STEP_EXACT
        + (delta >= 32).long()
        + (delta >= 64).long()
        + (delta >= 128).long()
        + (delta >= 256).long()
        + (delta >= 512).long()
        + (delta >= 1024).long()
    )
    raw_step = torch.where(delta < REL_STEP_EXACT, delta + 1, coarse)
    step_bucket = torch.where(valid_event, raw_step, 0)
    return allowed, seat_bucket, round_bucket, step_bucket


def _gather_attn_bias(
    rel_seat: Tensor,
    rel_round: Tensor,
    rel_step: Tensor,
    allowed: Tensor,
    seat_bucket: Tensor,
    round_bucket: Tensor,
    step_bucket: Tensor,
) -> Tensor:
    """Gather query-conditioned relative biases into ``float32 [B, heads, Q_rows, S]``."""
    heads = rel_seat.shape[1]
    sb = seat_bucket[:, None, :, :].expand(-1, heads, -1, -1)
    rb = round_bucket[:, None, :, :].expand(-1, heads, -1, -1)
    tb = step_bucket[:, None, :, :].expand(-1, heads, -1, -1)
    bias = (
        rel_seat.gather(-1, sb)
        + rel_round.gather(-1, rb)
        + rel_step.gather(-1, tb)
    ).to(torch.float32)
    return bias.masked_fill(~allowed[:, None, :, :], float("-inf"))


def _scatter_slots(
    tensor: Tensor,
    batches: int,
    slots: int,
    match_index: Tensor,
    rank: Tensor,
) -> Tensor:
    """Scatter ``[n, K, heads, feat]`` into ``[batches, heads, slots * K, feat]``."""
    k_slots, heads, feat = tensor.shape[1], tensor.shape[2], tensor.shape[3]
    padded = tensor.new_zeros(batches, slots, k_slots, heads, feat).index_put(
        (match_index, rank), tensor
    )
    return padded.reshape(batches, slots * k_slots, heads, feat).transpose(1, 2)


class PrivateDecoderLayer(nn.Module):
    """Pre-LN cross-attention with relative seat/round/step bias + Pre-LN ReLU FFN."""

    def __init__(self, width: int, heads: int, max_rounds: int) -> None:
        super().__init__()
        self.width = width
        self.heads = heads
        self.depth = width // heads
        self.max_rounds = max_rounds
        self.q_ln = nn.LayerNorm(width)
        self.q_proj = nn.Linear(width, width)
        self.kv_proj = nn.Linear(width, 2 * width)
        self.q_norm = nn.LayerNorm(self.depth)
        self.k_norm = nn.LayerNorm(self.depth)
        self.rel_seat_proj = nn.Linear(width, heads * REL_SEAT_BUCKETS)
        self.rel_round_proj = nn.Linear(width, heads * (max_rounds + 1))
        self.rel_step_proj = nn.Linear(width, heads * REL_STEP_BUCKETS)
        self.out_proj = nn.Linear(width, width)
        self.ffn_ln = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.ReLU(),
            nn.Linear(width * 4, width),
        )

    def project_kv(self, encoded: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(k, v)`` of shape ``[B, heads, S, depth]`` from ``encoded``."""
        batches, seq_len, _ = encoded.shape
        keys, values = self.kv_proj(encoded).chunk(2, dim=-1)
        k = self.k_norm(keys.view(batches, seq_len, self.heads, self.depth)).transpose(1, 2)
        v = values.view(batches, seq_len, self.heads, self.depth).transpose(1, 2)
        return k, v

    def project_queries(
        self,
        z: Tensor,
        project: Callable[[nn.Linear, Tensor], Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Project ``z [n, K, width]`` into normalized queries and relative bias tables."""
        n, k_slots, _ = z.shape
        q_in = self.q_ln(z)
        q = self.q_norm(project(self.q_proj, q_in).view(n, k_slots, self.heads, self.depth))
        rel_seat = project(self.rel_seat_proj, q_in).view(
            n, k_slots, self.heads, REL_SEAT_BUCKETS
        )
        rel_round = project(self.rel_round_proj, q_in).view(
            n, k_slots, self.heads, self.max_rounds + 1
        )
        rel_step = project(self.rel_step_proj, q_in).view(
            n, k_slots, self.heads, REL_STEP_BUCKETS
        )
        return q, rel_seat, rel_round, rel_step

    def finish_layer(
        self,
        z: Tensor,
        attended: Tensor,
        project: Callable[[nn.Linear, Tensor], Tensor],
    ) -> Tensor:
        """Apply attention residual projection and Pre-LN ReLU FFN residual."""
        z = z + project(self.out_proj, attended)
        hidden = self.ffn_ln(z)
        ffn_hidden = F.relu(project(self.ffn[0], hidden))
        return z + project(self.ffn[2], ffn_hidden)


class PrivateCrossDecoder(nn.Module):
    """Multi-layer causal cross-attention decoder over the encoded public stream."""

    def __init__(
        self,
        width: int,
        heads: int,
        layers: int,
        max_rounds: int = 16,
        query_slots: int = 1,
    ) -> None:
        super().__init__()
        if width <= 0 or heads <= 0 or width % heads != 0:
            raise ValueError("width and heads must be positive with width divisible by heads")
        if layers < 1 or max_rounds < 1 or query_slots < 1:
            raise ValueError("layers, max_rounds, and query_slots must be at least 1")
        self.width = width
        self.heads = heads
        self.depth = width // heads
        self.max_rounds = max_rounds
        self.query_slots = query_slots

        self.reader_bias = nn.Parameter(torch.zeros(width))
        self.query_round = nn.Embedding(max_rounds, width)
        nn.init.zeros_(self.query_round.weight)
        if query_slots > 1:
            self.slot_embeddings = nn.Parameter(torch.randn(query_slots, width) * 0.02)
            self.slot_proj = nn.Linear(query_slots * width, width)
        else:
            self.slot_embeddings = None
            self.slot_proj = None
        self.layers = nn.ModuleList(
            [PrivateDecoderLayer(width, heads, max_rounds) for _ in range(layers)]
        )

    def reset_for_migration(
        self,
        reader_bias: Tensor,
        residual_scale: float = 0.02,
    ) -> None:
        """Initialize migration state near identity around ``attention_norm(query + r_bar)``."""
        if reader_bias.shape != self.reader_bias.shape:
            raise ValueError(
                f"reader_bias must have shape {tuple(self.reader_bias.shape)}, "
                f"got {tuple(reader_bias.shape)}"
            )
        with torch.no_grad():
            self.reader_bias.copy_(
                reader_bias.to(device=self.reader_bias.device, dtype=self.reader_bias.dtype)
            )
            self.query_round.weight.zero_()
            for layer in self.layers:
                layer.out_proj.weight.mul_(residual_scale)
                layer.out_proj.bias.zero_()
                layer.ffn[2].weight.mul_(residual_scale)
                layer.ffn[2].bias.zero_()
            if self.slot_proj is not None:
                nn.init.normal_(self.slot_proj.weight, std=residual_scale)
                self.slot_proj.bias.zero_()

    def forward(
        self,
        query: Tensor,
        attention_norm: nn.LayerNorm,
        encoded: Tensor,
        pos_seat: Tensor,
        pos_round: Tensor,
        match_index: Tensor,
        prefix: Tensor,
        seat: Tensor,
        round_index: Tensor,
        *,
        one_decision_per_stream: bool = False,
        batched_private_attention: bool = False,
        wide_private_projection: bool = False,
        batched_match_attention: bool = False,
        rank: Tensor | None = None,
        slots: int = 0,
    ) -> Tensor:
        """Decode ``[n, width]`` decision states from ``encoded [B, S, width]``."""
        if encoded.ndim != 3 or encoded.shape[2] != self.width:
            raise ValueError(f"encoded must have shape [B, S, {self.width}]")
        n = len(query)
        my_round = round_index.to(device=query.device, dtype=torch.long).clamp(
            0, self.max_rounds - 1
        )
        base = attention_norm(query + self.reader_bias) + self.query_round(my_round)
        if n == 0:
            if self.slot_proj is None:
                return base
            return base + self.slot_proj(query.new_zeros(0, self.query_slots * self.width))

        batches, seq_len, _ = encoded.shape
        if batches == 0 or seq_len == 0:
            raise ValueError("encoded must have non-zero batch and sequence length when n > 0")

        match_index = match_index.to(device=query.device, dtype=torch.long)
        prefix = prefix.to(device=query.device, dtype=torch.long)
        seat = seat.to(device=query.device, dtype=torch.long)
        pos_seat_l = pos_seat.to(device=encoded.device, dtype=torch.long)
        pos_round_l = pos_round.to(device=encoded.device, dtype=torch.long)

        if self.slot_embeddings is None:
            z0 = base[:, None, :]
        else:
            z0 = base[:, None, :] + self.slot_embeddings[None, :, :]
        z = z0

        if batched_private_attention and one_decision_per_stream and n > 1:
            z = self._forward_batched_private(
                z, encoded, pos_seat_l, pos_round_l, prefix, seat, my_round, wide_private_projection
            )
        elif batched_match_attention and not one_decision_per_stream and n > 1:
            z = self._forward_batched_match(
                z, encoded, pos_seat_l, pos_round_l, match_index, prefix, seat, my_round, rank, slots
            )
        else:
            z = self._forward_per_match(
                z,
                encoded,
                pos_seat_l,
                pos_round_l,
                match_index,
                prefix,
                seat,
                my_round,
                one_decision_per_stream,
            )

        if self.slot_proj is None:
            return z[:, 0]
        return base + self.slot_proj((z - z0).reshape(n, -1))

    def _forward_batched_private(
        self,
        z: Tensor,
        encoded: Tensor,
        pos_seat: Tensor,
        pos_round: Tensor,
        prefix: Tensor,
        seat: Tensor,
        my_round: Tensor,
        wide_private_projection: bool,
    ) -> Tensor:
        n, seq_len = encoded.shape[0], encoded.shape[1]
        if n != len(z):
            raise ValueError("one_decision_per_stream requires one stream per decision")
        project = _wide_project if wide_private_projection else _project_independent
        allowed, seat_b, round_b, step_b = _compute_rel_buckets(
            pos_seat, pos_round, prefix[:, None], seat[:, None], my_round[:, None], self.max_rounds
        )
        k_slots = self.query_slots
        allowed_q = allowed[:, None, :].expand(n, k_slots, seq_len)
        seat_q = seat_b[:, None, :].expand(n, k_slots, seq_len)
        round_q = round_b[:, None, :].expand(n, k_slots, seq_len)
        step_q = step_b[:, None, :].expand(n, k_slots, seq_len)

        for layer in self.layers:
            k, v = layer.project_kv(encoded)
            q, r_seat, r_round, r_step = layer.project_queries(z, project)
            q_sdpa = q.transpose(1, 2)
            attn_bias = _gather_attn_bias(
                r_seat.permute(0, 2, 1, 3),
                r_round.permute(0, 2, 1, 3),
                r_step.permute(0, 2, 1, 3),
                allowed_q,
                seat_q,
                round_q,
                step_q,
            )
            mask = attn_bias if q_sdpa.dtype == torch.float32 else attn_bias.to(q_sdpa.dtype)
            out = private_sdpa(q_sdpa, k, v, attn_mask=mask)
            attended = out.transpose(1, 2).reshape(n, k_slots, self.width)
            z = layer.finish_layer(z, attended, project)
        return z

    def _forward_batched_match(
        self,
        z: Tensor,
        encoded: Tensor,
        pos_seat: Tensor,
        pos_round: Tensor,
        match_index: Tensor,
        prefix: Tensor,
        seat: Tensor,
        my_round: Tensor,
        rank: Tensor | None,
        slots: int,
    ) -> Tensor:
        n = len(z)
        batches, seq_len, _ = encoded.shape
        if rank is None:
            per_match = torch.bincount(match_index, minlength=batches)
            slots = int(per_match.max())
            order = torch.argsort(match_index, stable=True)
            starts = torch.cumsum(per_match, 0) - per_match
            rank = torch.empty_like(match_index)
            rank[order] = torch.arange(n, device=z.device) - starts[match_index[order]]
        else:
            rank = rank.to(device=z.device, dtype=torch.long)
            if slots <= 0:
                slots = int(rank.max()) + 1

        vis_prefix = prefix.new_zeros(batches, slots).index_put((match_index, rank), prefix)
        vis_seat = seat.new_zeros(batches, slots).index_put((match_index, rank), seat)
        vis_round = my_round.new_zeros(batches, slots).index_put((match_index, rank), my_round)

        allowed, seat_b, round_b, step_b = _compute_rel_buckets(
            pos_seat[:, None, :],
            pos_round[:, None, :],
            vis_prefix[:, :, None],
            vis_seat[:, :, None],
            vis_round[:, :, None],
            self.max_rounds,
        )
        k_slots = self.query_slots
        q_rows = slots * k_slots
        allowed_q = allowed[:, :, None, :].expand(batches, slots, k_slots, seq_len).reshape(
            batches, q_rows, seq_len
        )
        seat_q = seat_b[:, :, None, :].expand(batches, slots, k_slots, seq_len).reshape(
            batches, q_rows, seq_len
        )
        round_q = round_b[:, :, None, :].expand(batches, slots, k_slots, seq_len).reshape(
            batches, q_rows, seq_len
        )
        step_q = step_b[:, :, None, :].expand(batches, slots, k_slots, seq_len).reshape(
            batches, q_rows, seq_len
        )

        for layer in self.layers:
            k, v = layer.project_kv(encoded)
            q, r_seat, r_round, r_step = layer.project_queries(z, _wide_project)
            q_sdpa = _scatter_slots(q, batches, slots, match_index, rank)
            rs = _scatter_slots(r_seat, batches, slots, match_index, rank)
            rr = _scatter_slots(r_round, batches, slots, match_index, rank)
            rt = _scatter_slots(r_step, batches, slots, match_index, rank)
            attn_bias = _gather_attn_bias(rs, rr, rt, allowed_q, seat_q, round_q, step_q)
            mask = attn_bias if q_sdpa.dtype == torch.float32 else attn_bias.to(q_sdpa.dtype)
            out = private_sdpa(q_sdpa, k, v, attn_mask=mask)
            attended = out.transpose(1, 2).reshape(batches, slots, k_slots, self.width)[
                match_index, rank
            ]
            z = layer.finish_layer(z, attended, _wide_project)
        return z

    def _forward_per_match(
        self,
        z: Tensor,
        encoded: Tensor,
        pos_seat: Tensor,
        pos_round: Tensor,
        match_index: Tensor,
        prefix: Tensor,
        seat: Tensor,
        my_round: Tensor,
        one_decision_per_stream: bool,
    ) -> Tensor:
        batches, seq_len, _ = encoded.shape
        if one_decision_per_stream and batches != len(z):
            raise ValueError("one_decision_per_stream requires one stream per decision")
        kv_layers = [layer.project_kv(encoded) for layer in self.layers]
        k_slots = self.query_slots
        parts: list[Tensor] = []
        row_indices: list[Tensor] = []

        for b in range(batches):
            if one_decision_per_stream:
                rows: slice | Tensor = slice(b, b + 1)
                r_count = 1
            else:
                rows = (match_index == b).nonzero(as_tuple=True)[0]
                r_count = len(rows)
                if r_count == 0:
                    continue
                row_indices.append(rows)

            zb = z[rows]
            allowed, seat_b, round_b, step_b = _compute_rel_buckets(
                pos_seat[b : b + 1],
                pos_round[b : b + 1],
                prefix[rows, None],
                seat[rows, None],
                my_round[rows, None],
                self.max_rounds,
            )
            q_rows = r_count * k_slots
            allowed_q = allowed[:, None, :].expand(r_count, k_slots, seq_len).reshape(
                1, q_rows, seq_len
            )
            seat_q = seat_b[:, None, :].expand(r_count, k_slots, seq_len).reshape(
                1, q_rows, seq_len
            )
            round_q = round_b[:, None, :].expand(r_count, k_slots, seq_len).reshape(
                1, q_rows, seq_len
            )
            step_q = step_b[:, None, :].expand(r_count, k_slots, seq_len).reshape(
                1, q_rows, seq_len
            )

            for layer, (k_all, v_all) in zip(self.layers, kv_layers):
                q, r_seat, r_round, r_step = layer.project_queries(zb, _wide_project)
                q_sdpa = q.reshape(1, q_rows, self.heads, self.depth).transpose(1, 2)
                rs = r_seat.reshape(1, q_rows, self.heads, REL_SEAT_BUCKETS).transpose(1, 2)
                rr = r_round.reshape(1, q_rows, self.heads, self.max_rounds + 1).transpose(1, 2)
                rt = r_step.reshape(1, q_rows, self.heads, REL_STEP_BUCKETS).transpose(1, 2)
                attn_bias = _gather_attn_bias(rs, rr, rt, allowed_q, seat_q, round_q, step_q)
                mask = attn_bias if q_sdpa.dtype == torch.float32 else attn_bias.to(q_sdpa.dtype)
                out = private_sdpa(
                    q_sdpa,
                    k_all[b : b + 1],
                    v_all[b : b + 1],
                    attn_mask=mask,
                )
                attended = out.transpose(1, 2).reshape(r_count, k_slots, self.width)
                zb = layer.finish_layer(zb, attended, _wide_project)
            parts.append(zb)

        if one_decision_per_stream:
            return torch.cat(parts, dim=0)
        return torch.zeros_like(z).index_copy(0, torch.cat(row_indices), torch.cat(parts, dim=0))
