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
import weakref

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

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
    """Every parameter the merged head reads, from the live module registries."""
    p0, p1 = actor.private[0], actor.private[2]
    a0, a1 = actor.action_tower[0], actor.action_tower[2]
    u0, u1, u2 = actor.fusion[0][0], actor.fusion[0][2], actor.fusion[1]
    f0, f1 = actor.feed_forward[0], actor.feed_forward[2]
    return (p0.weight, p0.bias, p1.weight, p1.bias, actor.seat.weight,
            actor.q_proj.weight, actor.q_proj.bias, actor.kv_proj.weight, actor.kv_proj.bias,
            actor.out_proj.weight, actor.out_proj.bias,
            actor.attention_norm.weight, actor.attention_norm.bias,
            f0.weight, f0.bias, f1.weight, f1.bias,
            actor.output_norm.weight, actor.output_norm.bias,
            a0.weight, a0.bias, a1.weight, a1.bias,
            u0.weight, u0.bias, u1.weight, u1.bias, u2.weight, u2.bias)


def head_signature(actor: HistoryActor) -> tuple:
    """The public KV cache's invalidation signal (parameter identity and version),
    restricted to the head, plus the storage address so a rebound ``.data`` is
    also caught. Any change rewrites the identity's slot before it is read."""
    return tuple((id(t), t._version, t.data_ptr(), t.dtype, t.device)
                 for t in head_parameters(actor))


def validate_actor(actor: HistoryActor) -> None:
    """Refuse configurations the merged head does not implement exactly."""
    config = actor.config
    if config.window:
        raise ValueError("merged snapshot inference requires the full-history actor")
    if config.response_mode == "explicit":
        raise ValueError("merged snapshot inference does not implement the explicit "
                         "response bridge")
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
    if any(t.dtype != torch.float32 for t in head_parameters(actor)):
        raise ValueError("merged snapshot inference is FP32 only")
    for module in actor.modules():
        if module._forward_hooks or module._forward_pre_hooks:
            raise ValueError("merged snapshot inference bypasses module calls; "
                             "actors with forward hooks are refused")


def _head_values(actor: HistoryActor) -> dict[str, Tensor]:
    """One identity's head in the stacked layout: Linear weights as [in, out],
    biases as [1, out], key/value weights split per head."""
    cfg = actor.config
    width, heads = cfg.width, cfg.heads
    depth = width // heads
    p0, p1 = _mlp_linears(actor.private, 2)
    a0, a1 = _mlp_linears(actor.action_tower, 2)
    u0, u1 = _mlp_linears(actor.fusion[0], 2)
    u2 = actor.fusion[1]
    f0, f1 = actor.feed_forward[0], actor.feed_forward[2]
    kv_w, kv_b = actor.kv_proj.weight, actor.kv_proj.bias
    linear = lambda layer, name: {f"{name}_w": layer.weight.t(), f"{name}_b": layer.bias[None]}
    return {
        **linear(p0, "p0"), **linear(p1, "p1"), "seat": actor.seat.weight,
        **linear(actor.q_proj, "q"),
        # query-side key weights: (W_k^T q) per head, [heads, depth, width]
        "k_w": kv_w[:width].reshape(heads, depth, width),
        # value weights applied after attention, [heads, width, depth]
        "v_w": kv_w[width:].reshape(heads, depth, width).transpose(1, 2),
        "v_b": kv_b[width:][None],
        **linear(actor.out_proj, "o"),
        "n1_w": actor.attention_norm.weight[None], "n1_b": actor.attention_norm.bias[None],
        **linear(f0, "f0"), **linear(f1, "f1"),
        "n2_w": actor.output_norm.weight[None], "n2_b": actor.output_norm.bias[None],
        **linear(a0, "a0"), **linear(a1, "a1"),
        **linear(u0, "u0"), **linear(u1, "u1"), **linear(u2, "u2"),
    }


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
    sizes = np.asarray([len(g.rows) for g in groups], np.int64)
    widths = np.asarray([max(int(g.max_candidates), 1) for g in groups], np.int64)
    slot_ids = np.asarray(slots, np.int64)
    row_ends = np.cumsum(sizes)
    row_starts = row_ends - sizes
    S = max(slots) + 1
    M = int(sizes.max())
    group_cands = np.add.reduceat(row_counts, row_starts)
    Cm = int(group_cands.max())
    K = int(widths.max())
    # Build the row/candidate scatter indices once across all identities.
    # The old per-identity repeat/arange calls cost more than this arithmetic
    # when many frozen policies contribute only a few rows each.
    row_local = np.arange(n, dtype=np.int64) - np.repeat(row_starts, sizes)
    row_pad = np.repeat(slot_ids * M, sizes) + row_local
    seat_pad = np.zeros(S * M, np.int64)
    seat_rows = seat[rows]
    seat_pad[row_pad] = np.repeat(slot_ids * 4, sizes) + seat_rows
    draws_per_group = sizes * widths
    draw_ends = np.cumsum(draws_per_group)
    draw_starts = draw_ends - draws_per_group
    uniform_width = np.repeat(widths, sizes)
    uniform_base = np.repeat(draw_starts, sizes) + uniform_width * row_local
    cand_row = np.repeat(np.arange(n, dtype=np.int64), row_counts)
    cand_starts = np.cumsum(group_cands) - group_cands
    cand_pad = np.arange(len(cand_row), dtype=np.int64)
    cand_pad += np.repeat(slot_ids * Cm - cand_starts, group_cands)
    cand_state_pad = np.zeros(S * Cm, np.int64)
    cand_state_pad[cand_pad] = row_pad[cand_row]
    group_rows = list(zip(row_starts.tolist(), row_ends.tolist()))
    draws = list(zip(draw_starts.tolist(), draw_ends.tolist()))
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
    local_cand = np.arange(len(cand_row), dtype=np.int64) - (np.cumsum(row_counts) - row_counts)[cand_row]
    src = ragged_index(offsets[rows], row_counts)
    length = max(int(p) for g in groups for p in (s.prefix for s in g.streams)) + 1
    arrays = (obs[rows].astype(np.uint8, copy=False), seat_rows, prefix[rows],
              cand[src].astype(np.uint8, copy=False),
              row_pad, seat_pad, stream, cand_pad, cand_state_pad, cand_row,
              cand_row * K + local_cand, uniform_base, uniform_width - 1)
    return MergedLayout(rows, group_rows, group_streams, arrays, draws, S, M, Cm, K, length,
                        stream_base, bool(np.array_equal(stream, np.arange(n))))


def _linear(x: Tensor, tensors: dict[str, Tensor], name: str, slots: int) -> Tensor:
    return torch.baddbmm(tensors[f"{name}_b"][:slots], x, tensors[f"{name}_w"][:slots])


def _norm(x: Tensor, tensors: dict[str, Tensor], name: str, slots: int, eps: float) -> Tensor:
    normalized = F.layer_norm(x, (x.shape[-1],), eps=eps)
    return torch.addcmul(tensors[f"{name}_b"][:slots], normalized, tensors[f"{name}_w"][:slots])


def _pad(values: Tensor, index: Tensor, size: int) -> Tensor:
    out = values.new_zeros((size, *values.shape[1:]))
    return out.index_copy_(0, index, values)


@torch.no_grad()
def merged_log_probs(heads: SnapshotHeads, layout: MergedLayout, fields: dict[str, Tensor],
                     memory: Tensor) -> Tensor:
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
    # Private query, [S, M, width].
    x = _pad(fields["obs"].float(), fields["row_pad"], S * M).view(S, M, -1)
    x = torch.relu(_linear(x, tensors, "p0", S))
    x = torch.relu(_linear(x, tensors, "p1", S))
    query = x + tensors["seat"][:S].reshape(S * 4, width)[fields["seat_pad"]].view(S, M, width)
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
