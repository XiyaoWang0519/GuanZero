"""Opt-in merged actor forward for the frozen snapshot seats of one vector step.

``HistoryCollector(batch_snapshot_policies=True)`` evaluates the decision rows
of every non-learner identity of a vector step in ONE batched actor call
instead of one call per identity. All resident snapshots are copies of the
learner architecture, so their decision-head weights (everything after the
public stream encoder) are stacked per identity into ``SnapshotHeads`` slots
and every layer runs as one batched matrix multiply over the slots:

* Row-side layers (private MLP, seat embedding, query, output projection,
  feed-forward, layer norms) and candidate-side layers (action tower, fusion
  head) use a slot-padded layout ``[slots, rows or candidates, features]``
  with ``baddbmm`` against the stacked weights. Padded positions are zero
  inputs whose outputs are never read back.
* Attention reads each row's own public memory (the identity's
  ``cache.encode`` output, copied into one zero-padded ``[streams, T, width]``
  tensor). Key/value projections are folded into the query and the output:
  ``softmax(q . (W_k e + b_k)) = softmax((W_k^T q) . e)`` because
  ``q . b_k`` is constant over positions, and
  ``sum_t a_t (W_v e_t + b_v) = W_v (sum_t a_t e_t) + b_v`` because the
  weights sum to one. The same function as the per-identity path in FP32,
  with a different floating-point reduction order; it never materializes the
  ``[streams, T, 2 * width]`` key/value projection.

Acceptance tier 2 (same distribution, float-order noise), frozen snapshot
seats only. Snapshot rows are never stored for training; the learner's call
is not touched. The public KV cache stays per identity.

Random stream: the per-identity path drew one ``torch.rand([n_g, K_g])`` per
identity, in identity order, after the learner. The merged sampler draws the
same shapes, in the same order, from the same generator into consecutive
slices of one buffer (``uniform_`` per identity: the operation ``torch.rand``
itself performs), so the generator consumption and every uniform are
unchanged; only the table they are added to carries float-order noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import weakref

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from train.history_decoder import (
    PrivateCrossDecoder,
    PrivateDecoderLayer,
    REL_SEAT_BUCKETS,
    REL_STEP_BUCKETS,
    _compute_rel_buckets,
    _gather_attn_bias,
    infer_decision_rounds,
    mask_public_obs,
)
from train.history_model import HistoryActor, segment_log_softmax

# Per-slot tensor names and the order of the uploaded layout fields.
LAYOUT_FIELDS = ("obs", "seat", "prefix", "cand", "row_pad", "seat_pad", "stream",
                 "cand_pad", "cand_state_pad", "cand_row", "cand_table", "uniform_base",
                 "uniform_last")


def _mlp_linears(module: nn.Module, layers: int) -> tuple[nn.Linear, ...]:
    """The Linear layers of ``train.model.mlp`` (Linear, ReLU pairs), checked."""
    children = list(module.children()) if isinstance(module, nn.Sequential) else []
    if (len(children) != 2 * layers
            or not all(isinstance(c, nn.Linear) for c in children[0::2])
            or not all(type(c) is nn.ReLU for c in children[1::2])):
        raise ValueError("merged snapshot inference requires the standard history actor head")
    return tuple(children[0::2])


def head_parameters(actor: HistoryActor) -> tuple[Tensor, ...]:
    """Every parameter (and mask buffer) the merged head reads, from live registries."""
    cfg = actor.config
    p0, p1 = actor.private[0], actor.private[2]
    a0, a1 = actor.action_tower[0], actor.action_tower[2]
    u0, u1, u2 = actor.fusion[0][0], actor.fusion[0][2], actor.fusion[1]
    f0, f1 = actor.feed_forward[0], actor.feed_forward[2]
    if cfg.private_decoder_layers == 0 and cfg.private_obs_mode == "full":
        return (p0.weight, p0.bias, p1.weight, p1.bias, actor.seat.weight,
                actor.q_proj.weight, actor.q_proj.bias, actor.kv_proj.weight, actor.kv_proj.bias,
                actor.out_proj.weight, actor.out_proj.bias,
                actor.attention_norm.weight, actor.attention_norm.bias,
                f0.weight, f0.bias, f1.weight, f1.bias,
                actor.output_norm.weight, actor.output_norm.bias,
                a0.weight, a0.bias, a1.weight, a1.bias,
                u0.weight, u0.bias, u1.weight, u1.bias, u2.weight, u2.bias)
    params: list[Tensor] = [p0.weight, p0.bias, p1.weight, p1.bias, actor.seat.weight]
    if cfg.private_decoder_layers > 0:
        dec = actor.private_decoder
        params.extend((dec.reader_bias, dec.query_round.weight))
        if dec.query_slots > 1:
            params.extend((dec.slot_embeddings, dec.slot_proj.weight, dec.slot_proj.bias))
        for layer in dec.layers:
            params.extend((
                layer.q_ln.weight, layer.q_ln.bias,
                layer.q_proj.weight, layer.q_proj.bias,
                layer.kv_proj.weight, layer.kv_proj.bias,
                layer.q_norm.weight, layer.q_norm.bias,
                layer.k_norm.weight, layer.k_norm.bias,
                layer.rel_seat_proj.weight, layer.rel_seat_proj.bias,
                layer.rel_round_proj.weight, layer.rel_round_proj.bias,
                layer.rel_step_proj.weight, layer.rel_step_proj.bias,
                layer.out_proj.weight, layer.out_proj.bias,
                layer.ffn_ln.weight, layer.ffn_ln.bias,
                layer.ffn[0].weight, layer.ffn[0].bias,
                layer.ffn[2].weight, layer.ffn[2].bias,
            ))
    else:
        params.extend((
            actor.q_proj.weight, actor.q_proj.bias,
            actor.kv_proj.weight, actor.kv_proj.bias,
            actor.out_proj.weight, actor.out_proj.bias,
        ))
    params.extend((
        actor.attention_norm.weight, actor.attention_norm.bias,
        f0.weight, f0.bias, f1.weight, f1.bias,
        actor.output_norm.weight, actor.output_norm.bias,
        a0.weight, a0.bias, a1.weight, a1.bias,
        u0.weight, u0.bias, u1.weight, u1.bias, u2.weight, u2.bias,
    ))
    if cfg.private_obs_mode != "full":
        params.append(actor.obs_mask_prob)
    return tuple(params)


def head_signature(actor: HistoryActor) -> tuple:
    """The public KV cache's invalidation signal (parameter identity and version),
    restricted to the head, plus the storage address so a rebound ``.data`` is
    also caught. Any change rewrites the identity's slot before it is read."""
    return tuple((id(t), t._version, t.data_ptr(), t.dtype, t.device)
                 for t in head_parameters(actor))


def _validate_decoder(actor: HistoryActor, eps: float) -> None:
    config = actor.config
    dec = actor.private_decoder
    if not isinstance(dec, PrivateCrossDecoder):
        raise ValueError("merged snapshot inference requires PrivateCrossDecoder")
    if (dec.width != config.width or dec.heads != config.heads
            or len(dec.layers) != config.private_decoder_layers
            or dec.max_rounds != config.max_rounds
            or dec.query_slots != config.private_query_slots):
        raise ValueError("private_decoder dimensions do not match actor.config")
    if (not isinstance(dec.query_round, nn.Embedding)
            or dec.query_round.num_embeddings != config.max_rounds
            or dec.query_round.embedding_dim != config.width):
        raise ValueError("merged snapshot inference requires standard query_round embedding")
    if config.private_query_slots > 1:
        if (dec.slot_embeddings is None
                or dec.slot_embeddings.shape != (config.private_query_slots, config.width)
                or not isinstance(dec.slot_proj, nn.Linear)
                or dec.slot_proj.bias is None
                or dec.slot_proj.in_features != config.private_query_slots * config.width
                or dec.slot_proj.out_features != config.width):
            raise ValueError("merged snapshot inference requires standard multi-slot projections")
    depth = config.width // config.heads
    for layer in dec.layers:
        if not isinstance(layer, PrivateDecoderLayer):
            raise ValueError("merged snapshot inference requires PrivateDecoderLayer")
        for norm_layer, shape in (
            (layer.q_ln, (config.width,)),
            (layer.ffn_ln, (config.width,)),
            (layer.q_norm, (depth,)),
            (layer.k_norm, (depth,)),
        ):
            if (not isinstance(norm_layer, nn.LayerNorm)
                    or not norm_layer.elementwise_affine
                    or norm_layer.bias is None
                    or norm_layer.normalized_shape != shape
                    or norm_layer.eps != eps):
                raise ValueError("merged snapshot inference requires matching affine decoder norms")
        for lin in (layer.q_proj, layer.kv_proj, layer.rel_seat_proj, layer.rel_round_proj,
                    layer.rel_step_proj, layer.out_proj):
            if not isinstance(lin, nn.Linear) or lin.bias is None:
                raise ValueError("merged snapshot inference requires biased decoder linears")
        ffn_children = list(layer.ffn.children()) if isinstance(layer.ffn, nn.Sequential) else []
        if (len(ffn_children) != 3 or not isinstance(ffn_children[0], nn.Linear)
                or ffn_children[0].bias is None or type(ffn_children[1]) is not nn.ReLU
                or not isinstance(ffn_children[2], nn.Linear) or ffn_children[2].bias is None):
            raise ValueError("merged snapshot inference requires standard ReLU decoder FFN")


def validate_actor(actor: HistoryActor) -> None:
    """Refuse configurations the merged head does not implement exactly."""
    config = actor.config
    if config.window:
        raise ValueError("merged snapshot inference requires the full-history actor")
    if config.response_mode == "explicit":
        raise ValueError("merged snapshot inference does not implement the explicit "
                         "response bridge")
    if config.private_qk_norm != "none":
        raise ValueError("merged snapshot inference does not implement private Q/K norms")
    if config.private_fusion != "sum":
        raise ValueError("merged snapshot inference does not implement the split "
                         "private fusion")
    if config.style_input:
        raise ValueError("merged snapshot inference does not implement style_input")
    if config.private_obs_mode != "full":
        if (actor.obs_mask_prob is None or actor.obs_mask_prob.dtype != torch.float32
                or actor.obs_mask_prob.numel() != 1):
            raise ValueError("non-full obs mode requires scalar float32 obs_mask_prob buffer")
    _mlp_linears(actor.private, 2)
    _mlp_linears(actor.action_tower, 2)
    _mlp_linears(actor.fusion[0], 2)
    feed = list(actor.feed_forward.children())
    if (len(feed) != 3 or not isinstance(feed[0], nn.Linear) or type(feed[1]) is not nn.ReLU
            or not isinstance(feed[2], nn.Linear) or not isinstance(actor.fusion[1], nn.Linear)):
        raise ValueError("merged snapshot inference requires the standard history actor head")
    norms = (actor.attention_norm, actor.output_norm)
    if any(not n.elementwise_affine or n.bias is None or n.normalized_shape != (config.width,)
           for n in norms):
        raise ValueError("merged snapshot inference requires affine width layer norms")
    if config.private_decoder_layers > 0:
        _validate_decoder(actor, actor.attention_norm.eps)
    if any(t.dtype != torch.float32 for t in head_parameters(actor)):
        raise ValueError("merged snapshot inference is FP32 only")
    for module in actor.modules():
        if module._forward_hooks or module._forward_pre_hooks:
            raise ValueError("merged snapshot inference bypasses module calls; "
                             "actors with forward hooks are refused")


def _head_values(actor: HistoryActor) -> dict[str, Tensor]:
    """One identity's head in the stacked layout: Linear weights as [in, out],
    biases as [1, out], key/value weights split per head (or decoder layers)."""
    cfg = actor.config
    width, heads = cfg.width, cfg.heads
    depth = width // heads
    p0, p1 = _mlp_linears(actor.private, 2)
    a0, a1 = _mlp_linears(actor.action_tower, 2)
    u0, u1 = _mlp_linears(actor.fusion[0], 2)
    u2 = actor.fusion[1]
    f0, f1 = actor.feed_forward[0], actor.feed_forward[2]
    linear = lambda layer, name: {f"{name}_w": layer.weight.t(), f"{name}_b": layer.bias[None]}
    norm = lambda layer, name: {f"{name}_w": layer.weight[None], f"{name}_b": layer.bias[None]}
    values: dict[str, Tensor] = {
        **linear(p0, "p0"), **linear(p1, "p1"), "seat": actor.seat.weight,
        "n1_w": actor.attention_norm.weight[None], "n1_b": actor.attention_norm.bias[None],
        **linear(f0, "f0"), **linear(f1, "f1"),
        "n2_w": actor.output_norm.weight[None], "n2_b": actor.output_norm.bias[None],
        **linear(a0, "a0"), **linear(a1, "a1"),
        **linear(u0, "u0"), **linear(u1, "u1"), **linear(u2, "u2"),
    }
    if cfg.private_decoder_layers > 0:
        dec = actor.private_decoder
        values["d_rb"] = dec.reader_bias[None]
        values["d_qround"] = dec.query_round.weight
        if dec.query_slots > 1:
            values["d_slots"] = dec.slot_embeddings
            values.update(linear(dec.slot_proj, "d_sproj"))
        for idx, layer in enumerate(dec.layers):
            values.update({
                **norm(layer.q_ln, f"d{idx}_qln"),
                **linear(layer.q_proj, f"d{idx}_q"),
                **linear(layer.kv_proj, f"d{idx}_kv"),
                **norm(layer.q_norm, f"d{idx}_qn"),
                **norm(layer.k_norm, f"d{idx}_kn"),
                **linear(layer.rel_seat_proj, f"d{idx}_rs"),
                **linear(layer.rel_round_proj, f"d{idx}_rr"),
                **linear(layer.rel_step_proj, f"d{idx}_rt"),
                **linear(layer.out_proj, f"d{idx}_o"),
                **norm(layer.ffn_ln, f"d{idx}_fln"),
                **linear(layer.ffn[0], f"d{idx}_f0"),
                **linear(layer.ffn[2], f"d{idx}_f1"),
            })
    else:
        kv_w, kv_b = actor.kv_proj.weight, actor.kv_proj.bias
        values.update({
            **linear(actor.q_proj, "q"),
            # query-side key weights: (W_k^T q) per head, [heads, depth, width]
            "k_w": kv_w[:width].reshape(heads, depth, width),
            # value weights applied after attention, [heads, width, depth]
            "v_w": kv_w[width:].reshape(heads, depth, width).transpose(1, 2),
            "v_b": kv_b[width:][None],
            **linear(actor.out_proj, "o"),
        })
    if cfg.private_obs_mode != "full":
        values["obs_mask_prob"] = actor.obs_mask_prob.reshape(1)
    return values


def encoder_parameters(actor: HistoryActor) -> tuple[Tensor, ...]:
    """Every parameter of the public stream encoder, from the live registries."""
    result = [actor.public.weight, actor.public.bias, actor.round_embedding.weight,
              actor.phase_embedding.weight, actor.bos, actor.stream_norm.weight,
              actor.stream_norm.bias]
    for layer in actor.stream.layers:
        attention = layer.self_attn
        result += [layer.norm1.weight, layer.norm1.bias, attention.in_proj_weight,
                   attention.in_proj_bias, attention.out_proj.weight, attention.out_proj.bias,
                   layer.norm2.weight, layer.norm2.bias, layer.linear1.weight, layer.linear1.bias,
                   layer.linear2.weight, layer.linear2.bias]
    return tuple(result)


def validate_encoder(actor: HistoryActor) -> None:
    """Refuse encoders the merged snapshot encode does not implement exactly."""
    from train.history_attention import validate
    validate(actor.stream)
    width = actor.config.width
    norms = [actor.stream_norm] + [n for layer in actor.stream.layers
                                   for n in (layer.norm1, layer.norm2)]
    if (actor.stream.norm is not None
            or any(layer.activation is not F.relu for layer in actor.stream.layers)
            or any(not n.elementwise_affine or n.bias is None or n.normalized_shape != (width,)
                   or n.eps != actor.stream_norm.eps for n in norms)
            or any(t.dtype != torch.float32 for t in encoder_parameters(actor))):
        raise ValueError("merged snapshot encoding requires the standard FP32 history encoder")


def _encoder_values(actor: HistoryActor) -> dict[str, Tensor]:
    """One identity's stream encoder in the stacked layout (see ``_head_values``)."""
    linear = lambda layer, name: {f"{name}_w": layer.weight.t(), f"{name}_b": layer.bias[None]}
    norm = lambda layer, name: {f"{name}_w": layer.weight[None], f"{name}_b": layer.bias[None]}
    values = {**linear(actor.public, "e_pub"), "e_round": actor.round_embedding.weight,
              "e_phase": actor.phase_embedding.weight, "e_bos": actor.bos.reshape(-1),
              **norm(actor.stream_norm, "e_out")}
    for index, layer in enumerate(actor.stream.layers):
        attention = layer.self_attn
        values.update({**norm(layer.norm1, f"e{index}_n1"),
                       f"e{index}_in_w": attention.in_proj_weight.t(),
                       f"e{index}_in_b": attention.in_proj_bias[None],
                       **linear(attention.out_proj, f"e{index}_o"),
                       **norm(layer.norm2, f"e{index}_n2"),
                       **linear(layer.linear1, f"e{index}_l1"),
                       **linear(layer.linear2, f"e{index}_l2")})
    return values


@dataclass
class _Slot:
    index: int
    actor: weakref.ref
    signature: tuple


class SnapshotHeads:
    """Stacked decision-head weights of the frozen snapshot identities.

    One slot per identity. ``slot(identity, actor)`` returns the identity's slot
    after checking that the slot was written from this very actor object
    (weak reference) with the same head parameter signature; otherwise it
    (re)writes the slot first, so a stale stacked copy is never read.
    ``retain`` frees the slots of identities no longer assigned to any match,
    driven by the collector's cache-pruning signal. Only the head is stacked
    unless ``encoder`` is set, which also stacks the public stream encoder for
    ``train.history_paged_cache.merged_encode``; the extra memory is
    ``bytes``. Collector-owned: never actor or checkpoint state.
    """

    def __init__(self, capacity: int = 4, *, encoder: bool = False) -> None:
        if capacity < 1:
            raise ValueError("slot capacity must be positive")
        self.capacity = capacity
        self.encoder = encoder
        self.tensors: dict[str, Tensor] | None = None
        self.eps: float | None = None
        self.config = None
        self.slots: dict[int, _Slot] = {}
        self.writes = 0

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.tensors or {}).values())

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
        signature = head_signature(actor)
        if self.encoder:
            signature += tuple((id(t), t._version, t.data_ptr(), t.dtype, t.device)
                               for t in encoder_parameters(actor))
        record = self.slots.get(int(identity))
        if record is not None and record.actor() is actor and record.signature == signature:
            return record.index
        validate_actor(actor)
        eps = actor.attention_norm.eps
        if self.encoder:
            validate_encoder(actor)
            if actor.stream_norm.eps != eps:
                raise ValueError("stacked snapshot weights need one layer norm epsilon")
        if self.config is not None and (actor.config != self.config or eps != self.eps
                                        or actor.output_norm.eps != eps):
            raise ValueError("stacked snapshot heads require one architecture")
        values = _head_values(actor)
        if self.encoder:
            values.update(_encoder_values(actor))
        index = record.index if record is not None else self._free_index()
        if self.tensors is None or index >= self.capacity:
            self._allocate(values, max(self.capacity, 2 * index, index + 1))
        for name, value in values.items():
            target = self.tensors[name]
            if target.shape[1:] != value.shape or target.device != value.device:
                raise ValueError("stacked snapshot heads require one architecture and device")
            target[index].copy_(value)
        self.config, self.eps = actor.config, eps
        self.slots[int(identity)] = _Slot(index, weakref.ref(actor), signature)
        self.writes += 1
        return index

    def retain(self, identities) -> None:
        keep = {int(i) for i in identities}
        for identity in [i for i in self.slots if i not in keep]:
            del self.slots[identity]

    def metrics(self) -> dict:
        return dict(bytes=self.bytes, capacity=self.capacity if self.tensors is not None else 0,
                    slots=len(self.slots), writes=self.writes)


@dataclass
class MergedLayout:
    """Host-side layout of one merged snapshot call (collector order)."""
    rows: np.ndarray            # batch rows of the merged call, identity order
    group_rows: list[tuple[int, int]]      # [begin, end) of each identity's rows
    group_streams: list[tuple[int, int]]   # [begin, end) of each identity's streams
    arrays: tuple[np.ndarray, ...]         # LAYOUT_FIELDS, uploaded with the step
    draws: list[tuple[int, int]]           # uniform buffer slice of each identity
    slots: int                  # S: highest used slot + 1
    rows_per_slot: int          # M
    cands_per_slot: int         # Cm
    table_width: int            # Kmax
    length: int                 # T: longest public prefix + BOS
    streams: int
    identity_streams: bool      # every row reads its own stream, in row order


def merged_layout(groups, slots: list[int], prefix: np.ndarray, obs: np.ndarray,
                  seat: np.ndarray, cand: np.ndarray, offsets: np.ndarray,
                  counts: np.ndarray, env_id: np.ndarray, match_id: np.ndarray
                  ) -> MergedLayout:
    """Index arrays placing every snapshot group's rows and candidates into the
    slot-padded layout. ``groups`` are the collector's per-identity groups in
    identity order and ``slots`` their ``SnapshotHeads`` slots."""
    from train.history_rollout import ragged_index
    rows = np.concatenate([g.rows for g in groups])
    n = len(rows)
    row_counts = counts[rows]
    S = max(slots) + 1
    row_numbers = np.arange(n, dtype=np.int64)
    seat_rows = seat[rows]
    cand_row = np.repeat(row_numbers, row_counts)
    candidate_numbers = np.arange(len(cand_row), dtype=np.int64)
    if len(groups) == 1:
        # A single identity needs no group expansion or cumulative boundaries.
        # This is common while the population is first filling.
        slot = slots[0]
        M, Cm = n, len(cand_row)
        K = max(int(groups[0].max_candidates), 1)
        row_pad = slot * M + row_numbers
        seat_pad = np.zeros(S * M, np.int64)
        seat_pad[row_pad] = slot * 4 + seat_rows
        uniform_base = row_numbers * K
        uniform_width = np.full(n, K, np.int64)
        cand_pad = slot * Cm + candidate_numbers
        group_rows = [(0, n)]
        draws = [(0, n * K)]
    else:
        sizes = np.asarray([len(g.rows) for g in groups], np.int64)
        widths = np.asarray([max(int(g.max_candidates), 1) for g in groups], np.int64)
        slot_ids = np.asarray(slots, np.int64)
        row_ends = np.cumsum(sizes)
        row_starts = row_ends - sizes
        M = int(sizes.max())
        group_cands = np.add.reduceat(row_counts, row_starts)
        Cm = int(group_cands.max())
        K = int(widths.max())
        # Build the row/candidate scatter indices once across all identities.
        # The old per-identity repeat/arange calls cost more than this arithmetic
        # when many frozen policies contribute only a few rows each.
        row_local = row_numbers - np.repeat(row_starts, sizes)
        row_pad = np.repeat(slot_ids * M, sizes) + row_local
        seat_pad = np.zeros(S * M, np.int64)
        seat_pad[row_pad] = np.repeat(slot_ids * 4, sizes) + seat_rows
        draws_per_group = sizes * widths
        draw_ends = np.cumsum(draws_per_group)
        draw_starts = draw_ends - draws_per_group
        uniform_width = np.repeat(widths, sizes)
        uniform_base = np.repeat(draw_starts, sizes) + uniform_width * row_local
        cand_starts = np.cumsum(group_cands) - group_cands
        cand_pad = candidate_numbers + np.repeat(slot_ids * Cm - cand_starts, group_cands)
        group_rows = list(zip(row_starts.tolist(), row_ends.tolist()))
        draws = list(zip(draw_starts.tolist(), draw_ends.tolist()))
    cand_state_pad = np.zeros(S * Cm, np.int64)
    cand_state_pad[cand_pad] = row_pad[cand_row]
    group_streams = []
    stream = np.empty(n, np.int64)
    stream_base = 0
    for group, (begin, end) in zip(groups, group_rows):
        size = len(group.rows)
        if group.one_decision_per_stream:
            # Collector keys preserve first-row order, so unique keys already
            # have exactly this stream index; avoid rebuilding their lookup.
            stream[begin:end] = np.arange(stream_base, stream_base + size, dtype=np.int64)
        else:
            keys = zip(env_id[group.rows].tolist(), match_id[group.rows].tolist())
            local = {key: i for i, key in enumerate(group.keys)}
            stream[begin:end] = stream_base + np.fromiter(
                (local[k] for k in keys), dtype=np.int64, count=size)
        group_streams.append((stream_base, stream_base + len(group.keys)))
        stream_base += len(group.keys)
    local_cand = candidate_numbers - (np.cumsum(row_counts) - row_counts)[cand_row]
    src = ragged_index(offsets[rows], row_counts)
    length = max(int(p) for g in groups for p in (s.prefix for s in g.streams)) + 1
    arrays = (obs[rows].astype(np.uint8, copy=False), seat_rows, prefix[rows],
              cand[src].astype(np.uint8, copy=False),
              row_pad, seat_pad, stream, cand_pad, cand_state_pad, cand_row,
              cand_row * K + local_cand, uniform_base, uniform_width - 1)
    return MergedLayout(rows, group_rows, group_streams, arrays, draws, S, M, Cm, K, length,
                        stream_base, bool(np.array_equal(stream, row_numbers)))


def _linear(x: Tensor, tensors: dict[str, Tensor], name: str, slots: int) -> Tensor:
    return torch.baddbmm(tensors[f"{name}_b"][:slots], x, tensors[f"{name}_w"][:slots])


def _norm(x: Tensor, tensors: dict[str, Tensor], name: str, slots: int, eps: float) -> Tensor:
    normalized = F.layer_norm(x, (x.shape[-1],), eps=eps)
    return torch.addcmul(tensors[f"{name}_b"][:slots], normalized, tensors[f"{name}_w"][:slots])


def _pad(values: Tensor, index: Tensor, size: int) -> Tensor:
    out = values.new_zeros((size, *values.shape[1:]))
    return out.index_copy_(0, index, values)


def decoder_extra_arrays(
    layout: MergedLayout,
    slots: Sequence[int],
    round_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(round_rows, stream_pad)`` host arrays for merged decoder inference."""
    round_rows = np.ascontiguousarray(round_index[layout.rows], dtype=np.int64)
    stream_slot = np.empty(layout.streams, dtype=np.int64)
    local_idx = np.empty(layout.streams, dtype=np.int64)
    seen: dict[int, int] = {}
    for (first, last), slot in zip(layout.group_streams, slots):
        s = int(slot)
        count = last - first
        start = seen.get(s, 0)
        stream_slot[first:last] = s
        local_idx[first:last] = np.arange(start, start + count, dtype=np.int64)
        seen[s] = start + count
    ms = max(seen.values(), default=1)
    stream_pad = np.ascontiguousarray(stream_slot * ms + local_idx, dtype=np.int64)
    return round_rows, stream_pad


def _resolve_stream_pad(
    layout: MergedLayout,
    group_slots: Sequence[int] | None,
    fields: dict[str, Tensor],
    device: torch.device,
) -> tuple[Tensor, int]:
    """Return ``(stream_pad, max_streams_per_slot)`` on ``device`` without host syncs when ``group_slots`` is given."""
    slots_seq = group_slots
    if slots_seq is None:
        m = layout.rows_per_slot
        slots_seq = [int((fields["row_pad"][begin] // m).item()) for begin, _ in layout.group_rows]
    seen: dict[int, int] = {}
    if "stream_pad" in fields:
        for (first, last), slot in zip(layout.group_streams, slots_seq):
            s = int(slot)
            seen[s] = seen.get(s, 0) + (last - first)
        return fields["stream_pad"].to(device=device, dtype=torch.long), max(seen.values(), default=1)
    stream_slot = np.empty(layout.streams, dtype=np.int64)
    local_idx = np.empty(layout.streams, dtype=np.int64)
    for (first, last), slot in zip(layout.group_streams, slots_seq):
        s = int(slot)
        count = last - first
        start = seen.get(s, 0)
        stream_slot[first:last] = s
        local_idx[first:last] = np.arange(start, start + count, dtype=np.int64)
        seen[s] = start + count
    ms = max(seen.values(), default=1)
    stream_pad = torch.as_tensor(stream_slot * ms + local_idx, device=device, dtype=torch.long)
    return stream_pad, ms


def _merged_decode_states(
    heads: SnapshotHeads,
    layout: MergedLayout,
    fields: dict[str, Tensor],
    memory: Tensor,
    query: Tensor,
    pos_seat: Tensor | None,
    pos_round: Tensor | None,
    group_slots: Sequence[int] | None,
) -> Tensor:
    """Stacked ``PrivateCrossDecoder`` forward across all active slots -> ``[S, M, width]``."""
    tensors, cfg = heads.tensors, heads.config
    S, M = layout.slots, layout.rows_per_slot
    width, heads_n = cfg.width, cfg.heads
    depth = width // heads_n
    rows = len(layout.rows)
    seq_len = memory.shape[1]

    if pos_seat is None:
        ps_all = torch.zeros((layout.streams, seq_len), device=memory.device, dtype=torch.long)
    else:
        ps_all = pos_seat.to(device=memory.device, dtype=torch.long)
        if ps_all.shape[1] > seq_len:
            ps_all = ps_all[:, :seq_len]
        elif ps_all.shape[1] < seq_len:
            ps_all = F.pad(ps_all, (0, seq_len - ps_all.shape[1]), value=0)
    if pos_round is None:
        pr_all = torch.zeros((layout.streams, seq_len), device=memory.device, dtype=torch.long)
    else:
        pr_all = pos_round.to(device=memory.device, dtype=torch.long)
        if pr_all.shape[1] > seq_len:
            pr_all = pr_all[:, :seq_len]
        elif pr_all.shape[1] < seq_len:
            pr_all = F.pad(pr_all, (0, seq_len - pr_all.shape[1]), value=0)

    if layout.identity_streams:
        own_ps, own_pr = ps_all, pr_all
        stream_idx = torch.arange(rows, device=memory.device, dtype=torch.long)
    else:
        stream_idx = fields["stream"]
        own_ps = ps_all[stream_idx]
        own_pr = pr_all[stream_idx]

    round_idx = infer_decision_rounds(
        fields.get("round_index"), pr_all, stream_idx, fields["prefix"]
    )
    my_round = round_idx.to(device=memory.device, dtype=torch.long).clamp(0, cfg.max_rounds - 1)
    row_pad = fields["row_pad"]
    q_round_rows = tensors["d_qround"][row_pad // M, my_round]
    q_round_emb = _pad(q_round_rows, row_pad, S * M).view(S, M, width)

    base = _norm(query + tensors["d_rb"][:S], tensors, "n1", S, heads.eps) + q_round_emb
    k_slots = cfg.private_query_slots
    if k_slots == 1:
        z0 = base[:, :, None, :]
    else:
        z0 = base[:, :, None, :] + tensors["d_slots"][:S, None, :, :]
    z = z0.reshape(S, M * k_slots, width)

    allowed, seat_b, round_b, step_b = _compute_rel_buckets(
        own_ps,
        own_pr,
        fields["prefix"][:, None],
        fields["seat"][:, None],
        my_round[:, None],
        cfg.max_rounds,
    )
    allowed_q = allowed[:, None, :].expand(rows, k_slots, seq_len)
    seat_q = seat_b[:, None, :].expand(rows, k_slots, seq_len)
    round_q = round_b[:, None, :].expand(rows, k_slots, seq_len)
    step_q = step_b[:, None, :].expand(rows, k_slots, seq_len)
    has_any = allowed_q.any(dim=-1, keepdim=True)

    stream_pad, ms = _resolve_stream_pad(layout, group_slots, fields, memory.device)
    mem_pad = _pad(memory, stream_pad, S * ms).view(S, ms * seq_len, width)
    max_rounds_p1 = cfg.max_rounds + 1

    for idx in range(cfg.private_decoder_layers):
        kv = _linear(mem_pad, tensors, f"d{idx}_kv", S)
        k_raw, v_raw = kv.chunk(2, dim=-1)
        k_normed = _norm(k_raw.reshape(S, ms * seq_len * heads_n, depth), tensors,
                         f"d{idx}_kn", S, heads.eps)
        k_streams = k_normed.view(S * ms, seq_len, heads_n, depth)[stream_pad]
        v_streams = v_raw.view(S * ms, seq_len, heads_n, depth)[stream_pad]
        k_own = (k_streams if layout.identity_streams else k_streams[stream_idx]).transpose(1, 2)
        v_own = (v_streams if layout.identity_streams else v_streams[stream_idx]).transpose(1, 2)

        q_in = _norm(z, tensors, f"d{idx}_qln", S, heads.eps)
        q_proj = _linear(q_in, tensors, f"d{idx}_q", S)
        q_normed = _norm(q_proj.reshape(S, M * k_slots * heads_n, depth), tensors,
                         f"d{idx}_qn", S, heads.eps)
        q_r = (q_normed.view(S * M, k_slots, heads_n, depth)[row_pad].transpose(1, 2))
        rs = (_linear(q_in, tensors, f"d{idx}_rs", S)
              .view(S * M, k_slots, heads_n, REL_SEAT_BUCKETS)[row_pad].permute(0, 2, 1, 3))
        rr = (_linear(q_in, tensors, f"d{idx}_rr", S)
              .view(S * M, k_slots, heads_n, max_rounds_p1)[row_pad].permute(0, 2, 1, 3))
        rt = (_linear(q_in, tensors, f"d{idx}_rt", S)
              .view(S * M, k_slots, heads_n, REL_STEP_BUCKETS)[row_pad].permute(0, 2, 1, 3))

        attn_bias = _gather_attn_bias(rs, rr, rt, allowed_q, seat_q, round_q, step_q)
        mask = torch.where(has_any[:, None], attn_bias, 0.0)
        if q_r.dtype != mask.dtype:
            mask = mask.to(q_r.dtype)
        out = F.scaled_dot_product_attention(q_r, k_own, v_own, attn_mask=mask, dropout_p=0.0)
        out = out * has_any[:, None].to(out.dtype)
        out_rows = out.transpose(1, 2).reshape(rows, k_slots, width)
        attended = _pad(out_rows, row_pad, S * M).view(S, M * k_slots, width)

        z = z + _linear(attended, tensors, f"d{idx}_o", S)
        ffn_in = _norm(z, tensors, f"d{idx}_fln", S, heads.eps)
        ffn_hidden = torch.relu(_linear(ffn_in, tensors, f"d{idx}_f0", S))
        z = z + _linear(ffn_hidden, tensors, f"d{idx}_f1", S)

    if k_slots == 1:
        return z.view(S, M, width)
    delta = (z.view(S, M, k_slots, width) - z0).reshape(S, M, k_slots * width)
    return base + _linear(delta, tensors, "d_sproj", S)


@torch.no_grad()
def merged_log_probs(
    heads: SnapshotHeads,
    layout: MergedLayout,
    fields: dict[str, Tensor],
    memory: Tensor,
    *,
    pos_seat: Tensor | None = None,
    pos_round: Tensor | None = None,
    group_slots: Sequence[int] | None = None,
) -> Tensor:
    """Log-probability of every snapshot candidate, ``[sum_k]``, in merged order.

    ``memory`` is ``[layout.streams, T, width]``: each identity's encoded
    public streams (as ``cache.encode`` or ``encode_batch`` return them),
    zero beyond each stream's length.
    """
    tensors, cfg = heads.tensors, heads.config
    S, M, Cm = layout.slots, layout.rows_per_slot, layout.cands_per_slot
    width, heads_n = cfg.width, cfg.heads
    depth = width // heads_n
    rows = len(layout.rows)
    if cfg.private_obs_mode != "full":
        row_slots = fields["row_pad"] // M
        row_prob = tensors["obs_mask_prob"][row_slots, 0]
        obs_in = mask_public_obs(
            fields["obs"], fields["seat"], fields["prefix"], row_prob, mode=cfg.private_obs_mode
        )
    else:
        obs_in = fields["obs"].float()
    # Private query, [S, M, width].
    x = _pad(obs_in, fields["row_pad"], S * M).view(S, M, -1)
    x = torch.relu(_linear(x, tensors, "p0", S))
    x = torch.relu(_linear(x, tensors, "p1", S))
    query = x + tensors["seat"][:S].reshape(S * 4, width)[fields["seat_pad"]].view(S, M, width)
    if cfg.private_decoder_layers > 0:
        base_state = _merged_decode_states(
            heads, layout, fields, memory, query, pos_seat, pos_round, group_slots
        )
        hidden = _linear(torch.relu(_linear(base_state, tensors, "f0", S)), tensors, "f1", S)
        state = _norm(base_state + hidden, tensors, "n2", S, heads.eps)
    else:
        q = _linear(query, tensors, "q", S).view(S, M, heads_n, depth).transpose(1, 2)
        # Keys folded into the query: [S, heads, M, width], then one row per decision.
        folded = torch.matmul(q, tensors["k_w"][:S]) * depth ** -0.5
        folded = folded.transpose(1, 2).reshape(S * M, heads_n, width)[fields["row_pad"]]
        own = memory if layout.identity_streams else memory[fields["stream"]]
        scores = torch.bmm(folded, own.transpose(1, 2))                  # [rows, heads, T]
        allowed = torch.arange(own.shape[1], device=own.device)[None] <= fields["prefix"][:, None]
        scores = scores.masked_fill(~allowed[:, None], float("-inf"))
        context = torch.bmm(torch.softmax(scores, dim=-1), own)           # [rows, heads, width]
        context = _pad(context, fields["row_pad"], S * M).view(S, M, heads_n, width).transpose(1, 2)
        attended = torch.matmul(context, tensors["v_w"][:S])               # [S, heads, M, depth]
        attended = attended.transpose(1, 2).reshape(S, M, width) + tensors["v_b"][:S]
        attended = _linear(attended, tensors, "o", S)
        state = _norm(query + attended, tensors, "n1", S, heads.eps)
        hidden = _linear(torch.relu(_linear(state, tensors, "f0", S)), tensors, "f1", S)
        state = _norm(state + hidden, tensors, "n2", S, heads.eps)
    # Candidates, [S, Cm, features].
    actions = _pad(fields["cand"].float(), fields["cand_pad"], S * Cm).view(S, Cm, -1)
    actions = torch.relu(_linear(actions, tensors, "a0", S))
    actions = torch.relu(_linear(actions, tensors, "a1", S))
    states = state.reshape(S * M, width)[fields["cand_state_pad"]].view(S, Cm, width)
    fused = torch.cat((states, actions), dim=-1)
    fused = torch.relu(_linear(fused, tensors, "u0", S))
    fused = torch.relu(_linear(fused, tensors, "u1", S))
    logits = _linear(fused, tensors, "u2", S).reshape(S * Cm)[fields["cand_pad"]]
    return segment_log_softmax(logits, fields["cand_row"], rows)


@torch.no_grad()
def merged_sample(log_probs: Tensor, layout: MergedLayout, fields: dict[str, Tensor],
                  generator: torch.Generator | None) -> tuple[Tensor, Tensor]:
    """``HistoryActor.act`` for the merged rows: one Gumbel-max draw per row.

    The uniforms are drawn per identity, in identity order, with exactly the
    per-identity table shapes ``[n_g, K_g]``, into slices of one buffer; the
    generator therefore advances exactly as the per-identity calls did.
    """
    rows, K = len(layout.rows), layout.table_width
    table = torch.full((rows * K,), float("-inf"), device=log_probs.device,
                       dtype=log_probs.dtype)
    table = table.index_put_((fields["cand_table"],), log_probs).view(rows, K)
    uniform = torch.empty(layout.draws[-1][1], device=log_probs.device, dtype=log_probs.dtype)
    for begin, end in layout.draws:
        uniform[begin:end].uniform_(generator=generator)
    # Row r's cell c reads its identity's [n_g, K_g] draw at (r, min(c, K_g - 1));
    # cells past K_g hold -inf log-probabilities and can never be chosen.
    columns = torch.arange(K, device=log_probs.device)[None]
    index = fields["uniform_base"][:, None] + torch.minimum(columns, fields["uniform_last"][:, None])
    uniform = uniform[index].clamp_min(1e-12)
    choice = (table + -(-uniform.log()).log()).argmax(1)
    return choice, table.gather(1, choice[:, None])[:, 0]
