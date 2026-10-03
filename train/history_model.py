"""History Transformer player for self-play RL (DESIGN.md v0.6, sections 7.2, 7.3, 8.4).

This module is the contract that the sequence trainer (T2) and the
history-aware evaluators (T3) build on. It contains

* ``PublicStream``: the raw public event store of one match. One token per
  play, pass, engine-resolved pass and public exchange, in order, tagged with
  the round index and phase. Nothing private and nothing engine-private
  enters it: no hidden hands, no candidate lists, no tribute flags and no
  ``forced`` bit (that flag says the seat had no legal reply, which an
  opponent cannot know; publicly identical voluntary and forced passes get
  identical tokens).
* ``HistoryActor``: a causal Transformer over the public stream plus a
  private query for the acting seat. ``encode_stream`` takes public tokens
  only, so private information cannot enter a shared encoding by
  construction. The head scores every concrete candidate of the full
  canonical set; there is no top-k, reference policy or pruning anywhere.
* ``HistoryCritic``: a separate, randomly initialised value network over the
  privileged training state (observation plus hidden-hand counts). It shares
  no parameters with the actor.
* Checkpoint helpers that mark the lineage as ``stage = "history_ppo"`` with
  a random start and no teacher, and refuse to load anything else.

Only ``train.model.mlp`` (a layer helper) and ``train.logs.public_token`` are
reused from the old player; no old weights, references or datasets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from pathlib import Path
from typing import Any

import gd
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from train.ckpt import load_checkpoint, save_checkpoint
from train.logs import TOKEN_DIM, public_token
from train.model import mlp
from train.history_response import RESPONSE_CLASSES
from train.history_aux import (HIDDEN_SEATS, NEXT_LOGITS, NUM_CARDS, canonical_aux_heads,
                               parse_aux_heads)

STAGE = "history_ppo"
TOKEN_SCHEMA_VERSION = 1      # 186 public dims + round index + phase; no forced bit
PLAY_PHASE = int(gd.Phase.Play)
HIDDEN_DIM = 3 * 54


@dataclass(frozen=True)
class HistoryPolicyConfig:
    """Architecture of the history player. Frozen into every checkpoint."""
    width: int = 128
    layers: int = 4
    heads: int = 4
    max_rounds: int = 16
    # 0 sees the whole match; k > 0 is the reduced-history control that sees
    # only the last k public tokens before a decision (DESIGN 7.4).
    window: int = 0
    action_width: int = 128
    fusion_width: int = 128
    critic_width: int = 256
    critic_layers: int = 3
    obs_dim: int = int(gd.OBS_DIM)
    act_dim: int = int(gd.ACT_DIM)
    response_mode: str = "none"  # none | auxiliary | explicit
    # Planted-habit ORACLE arm only (train/history_habit.py): the opponents'
    # true style z enters the private query through a zero-initialised
    # embedding. Recorded in checkpoints only when on (``config_record``).
    style_input: bool = False
    # Auxiliary prediction heads on the shared encoder (train/history_aux.py):
    # a comma-separated subset of "next", "belief", "outcome" in that order.
    # They read the encoder/decision state and never enter the scoring path.
    # Recorded in checkpoints only when non-empty (``config_record``).
    aux_heads: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "aux_heads", canonical_aux_heads(self.aux_heads))
        if "next" in self.aux_head_names and self.window:
            raise ValueError("the next-token head needs the full-history encoder (window 0)")
        if min(self.width, self.layers, self.heads, self.max_rounds, self.action_width,
               self.fusion_width, self.critic_width, self.critic_layers) <= 0:
            raise ValueError("every size must be positive")
        if self.width % self.heads or self.window < 0:
            raise ValueError("width must be divisible by heads; window must not be negative")
        if self.obs_dim != int(gd.OBS_DIM) or self.act_dim != int(gd.ACT_DIM):
            raise ValueError("observation and action widths must match the engine")
        if self.response_mode not in ("none", "auxiliary", "explicit"):
            raise ValueError("response_mode must be none, auxiliary or explicit")

    @property
    def aux_head_names(self) -> tuple[str, ...]:
        return parse_aux_heads(self.aux_heads)


def config_record(config: HistoryPolicyConfig) -> dict[str, Any]:
    """``asdict(config)`` without ``style_input`` when it is off and without
    ``aux_heads`` when empty, so every other checkpoint keeps its exact
    architecture record."""
    record = asdict(config)
    if not record["style_input"]:
        del record["style_input"]
    if not record["aux_heads"]:
        del record["aux_heads"]
    return record


# ---- public stream ----------------------------------------------------------

class PublicStream:
    """Raw public events of one match for one environment, in order.

    ``append`` takes a ``gd.PublicActionEvent`` (or anything with ``seat``,
    ``encoded_action``, ``cards_left``, ``round_index`` and ``phase``).
    ``reset`` starts a new match. ``prefix`` is the number of tokens a
    decision taken now may read. Streams never store observations,
    candidates, hidden counts or the forced flag.

    Events live in contiguous growable arrays, so ``arrays()`` and the
    ``tokens``/``rounds``/``phases`` properties are O(1) read-only views of
    the first ``prefix`` events. A view never changes: appends write past
    every earlier view's end, growth copies into a new buffer, and ``reset``
    starts new buffers instead of overwriting the old ones.
    """

    __slots__ = ("_tokens", "_rounds", "_phases", "_length", "match_id", "generation")
    _INITIAL_CAPACITY = 64

    def __init__(self, match_id: int = -1) -> None:
        self._new_buffers(0)
        self.match_id = int(match_id)
        self.generation = 0

    def _new_buffers(self, capacity: int) -> None:
        self._tokens = np.zeros((capacity, TOKEN_DIM), dtype=np.uint8)
        self._rounds = np.zeros(capacity, dtype=np.int64)
        self._phases = np.zeros(capacity, dtype=np.int64)
        self._length = 0

    def reset(self, match_id: int = -1) -> None:
        self.generation += 1
        self._new_buffers(0)
        self.match_id = int(match_id)

    def append(self, event: Any) -> None:
        self.append_token(public_token(event), int(event.round_index), int(event.phase))

    def append_token(self, token: np.ndarray, round_index: int, phase: int) -> None:
        token = np.asarray(token, dtype=np.uint8)
        if token.shape != (TOKEN_DIM,):
            raise ValueError(f"public token must have {TOKEN_DIM} entries")
        if token[:4].sum() != 1 or token[158:].sum() != 1:
            raise ValueError("public token needs exactly one seat and one cards-left bit")
        if token[4 + 146:4 + 154].any():
            raise ValueError("private tribute flags must not enter the public stream")
        if round_index < 0 or phase < 0:
            raise ValueError("round index and phase must not be negative")
        n = self._length
        if n == len(self._rounds):
            capacity = max(self._INITIAL_CAPACITY, 2 * n)
            tokens, rounds, phases = self._tokens, self._rounds, self._phases
            self._new_buffers(capacity)
            self._tokens[:n] = tokens[:n]
            self._rounds[:n] = rounds[:n]
            self._phases[:n] = phases[:n]
        self._tokens[n] = token
        self._rounds[n] = int(round_index)
        self._phases[n] = int(phase)
        self._length = n + 1

    @staticmethod
    def _view(array: np.ndarray, length: int) -> np.ndarray:
        view = array[:length]
        view.flags.writeable = False
        return view

    @property
    def tokens(self) -> np.ndarray:
        """uint8 ``[prefix, TOKEN_DIM]``, read-only."""
        return self._view(self._tokens, self._length)

    @property
    def rounds(self) -> np.ndarray:
        """int64 ``[prefix]``, read-only."""
        return self._view(self._rounds, self._length)

    @property
    def phases(self) -> np.ndarray:
        """int64 ``[prefix]``, read-only."""
        return self._view(self._phases, self._length)

    @property
    def prefix(self) -> int:
        return self._length

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.tokens, self.rounds, self.phases


@dataclass
class StreamBatch:
    """Padded public streams of several matches on one device."""
    tokens: Tensor    # uint8 [B, T, TOKEN_DIM]
    rounds: Tensor    # int64 [B, T]
    phases: Tensor    # int64 [B, T]
    lengths: Tensor   # int64 [B]

    @staticmethod
    def host_arrays(streams: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Zero-padded ``(tokens, rounds, phases, lengths)`` host arrays, in field order."""
        count = len(streams)
        longest = max((len(t) for t, _, _ in streams), default=0)
        tokens = np.zeros((count, longest, TOKEN_DIM), dtype=np.uint8)
        rounds = np.zeros((count, longest), dtype=np.int64)
        phases = np.full((count, longest), PLAY_PHASE, dtype=np.int64)
        lengths = np.zeros(count, dtype=np.int64)
        for b, (t, r, p) in enumerate(streams):
            n = len(t)
            lengths[b] = n
            if n:
                tokens[b, :n] = t
                rounds[b, :n] = r
                phases[b, :n] = p
        return tokens, rounds, phases, lengths

    @staticmethod
    def from_arrays(streams: list[tuple[np.ndarray, np.ndarray, np.ndarray]], device
                    ) -> "StreamBatch":
        from train.history_transfers import upload_arrays
        # One packed host-to-device copy on CUDA; zero-copy views on the CPU.
        return StreamBatch(*upload_arrays(StreamBatch.host_arrays(streams), device))

    @staticmethod
    def from_streams(streams: list[PublicStream], device) -> "StreamBatch":
        return StreamBatch.from_arrays([s.arrays() for s in streams], device)


@dataclass
class MatchGroup:
    """Matches of one learner length group (``length_groups``) and their decisions."""
    matches: Tensor       # int64 [m], stream indices in the StreamBatch
    rows: Tensor          # int64 [r], decision indices whose match is in this group
    local_match: Tensor   # int64 [r], each decision's position in ``matches``
    tokens: int           # public tokens to encode: the longest cited prefix
    match_rank: Tensor | None = None  # int64 [r], host-planned (see DecisionInputs)
    match_slots: int = 0


def length_groups(cited: np.ndarray, groups: int, width: int) -> list[np.ndarray]:
    """Partition matches into at most ``groups`` sets of similar encode length.

    ``cited[m]`` is the number of tokens match ``m``'s decisions read. Sorted by
    length, contiguous runs are chosen by dynamic programming to minimise the
    padded encoder work, ``sum_g n_g * T_g * (1 + T_g / (6 * width))`` (linear
    layers plus attention, in units of a width-sized token row). Returns match
    index arrays, longest group first.
    """
    order = np.argsort(-cited, kind="stable")
    ordered = cited[order].astype(np.float64) + 1.0          # + BOS
    n = len(order)
    groups = max(1, min(int(groups), n))
    unit = ordered * (1.0 + ordered / (6.0 * width))          # cost per match, group led by i
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    choice = np.zeros((groups + 1, n + 1), np.int64)
    layers = [best]
    ends = np.arange(1, n + 1)
    for g in range(1, groups + 1):
        previous, current = layers[-1], np.full(n + 1, np.inf)
        for start in range(n):
            if not np.isfinite(previous[start]):
                continue
            cost = previous[start] + (ends[start:] - start) * unit[start]
            better = cost < current[start + 1:]
            current[start + 1:][better] = cost[better]
            choice[g, start + 1:][better] = start
        layers.append(current)
    g = int(np.argmin([layer[n] for layer in layers[1:]])) + 1
    bounds, end = [], n
    while g > 0 and end > 0:
        start = int(choice[g, end])
        bounds.append((start, end))
        end, g = start, g - 1
    return [order[a:b] for a, b in reversed(bounds)]


@dataclass
class DecisionInputs:
    """Everything the actor needs for a batch of decisions.

    ``match_index[i]`` selects the stream of decision ``i`` inside the
    ``StreamBatch``; ``prefix[i]`` is how many of its tokens the decision may
    read. Candidates are ragged: ``cand[offsets[i]:offsets[i + 1]]`` are the
    full canonical candidates of decision ``i`` in engine order.
    ``match_groups`` (learner only, optional) encodes the streams in length
    groups, each only as far as its decisions read, instead of one batch
    padded to the longest stream.
    """
    streams: StreamBatch
    match_index: Tensor   # int64 [n]
    prefix: Tensor        # int64 [n]
    obs: Tensor           # uint8 or float [n, obs_dim]
    seat: Tensor          # int64 [n]
    cand: Tensor          # uint8 or float [sum_k, act_dim]
    offsets: Tensor       # int64 [n + 1]
    candidate_rows: Tensor | None = None  # optional precomputed ragged integer index
    one_decision_per_stream: bool = False  # rows match stream order; collector only
    match_groups: list[MatchGroup] | None = None
    # Optional host-planned layout of batched match attention: each decision's
    # rank among its match's decisions (in row order) and the largest per-match
    # count. The same integers the device would compute, without its sync.
    match_rank: Tensor | None = None
    match_slots: int = 0
    # int64 [n], the opponents' style z in {-1, +1}; read only by an actor
    # with ``config.style_input`` (planted-habit ORACLE arm).
    style: Tensor | None = None

    @property
    def decisions(self) -> int:
        return len(self.obs)

    @property
    def counts(self) -> Tensor:
        return self.offsets[1:] - self.offsets[:-1]

    @property
    def rows(self) -> Tensor:
        if self.candidate_rows is not None:
            return self.candidate_rows
        return torch.repeat_interleave(torch.arange(self.decisions, device=self.obs.device),
                                       self.counts, output_size=len(self.cand))


# ---- actor --------------------------------------------------------------------

def sinusoidal(length: int, width: int, device, dtype) -> Tensor:
    positions = torch.arange(length, device=device, dtype=dtype)[:, None]
    frequencies = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype)
                            * (-math.log(10000.0) / width))
    table = torch.zeros(length, width, device=device, dtype=dtype)
    table[:, 0::2] = torch.sin(positions * frequencies)
    table[:, 1::2] = torch.cos(positions * frequencies)
    return table


def segment_log_softmax(scores: Tensor, rows: Tensor, count: int) -> Tensor:
    """Log-softmax of ``scores`` within the segments named by ``rows``."""
    top = torch.full((count,), float("-inf"), device=scores.device, dtype=scores.dtype)
    top = top.scatter_reduce(0, rows, scores, reduce="amax")
    shifted = (scores - top[rows]).exp()
    total = torch.zeros(count, device=scores.device, dtype=scores.dtype).index_add_(0, rows, shifted)
    return scores - top[rows] - total[rows].log()


@dataclass
class ExplorationSample:
    """One sampled candidate per decision under the exploration floor.

    ``logp`` is the target (temperature 1, no epsilon) log-probability that
    PPO's ratio uses; ``behaviour_logp`` is the log-probability of the
    distribution the choice was actually drawn from.
    """
    choice: Tensor            # int64 [n], relative to each decision's candidates
    logp: Tensor              # [n] log softmax(logits)[choice]
    behaviour_logp: Tensor    # [n] log pi_b(choice)
    uniform_pick: Tensor      # bool [n], the epsilon branch chose the candidate
    behaviour_entropy: Tensor  # [n] entropy of pi_b over the legal candidates


class HistoryActor(nn.Module):
    """Causal public-stream encoder, private query, full-candidate head.

    Information boundary by construction: ``encode_stream`` has no private
    argument, and ``decision_states`` only reads the encoded stream. The
    actor never sees other seats' hands, legal lists or the forced flag.
    """

    def __init__(self, config: HistoryPolicyConfig = HistoryPolicyConfig()) -> None:
        super().__init__()
        self.config = config
        # Runtime backend only: checkpoint weights/architecture stay identical.
        self.causal_sdpa = False
        # Opt-in until the host-equivalence checks also pass on the target GPU.
        # Keep each projection's original row shape when batching attention.
        self.batched_private_attention = False
        # Opt-in with batched attention: one q/out projection over all rows.
        # Same FP32 math; only the GEMM reduction order may differ (not bitwise).
        self.wide_private_projection = False
        # Opt-in: several decisions per match (the learner's minibatches) in
        # one padded attention call instead of a per-match loop with a device
        # nonzero each. Same FP32 math; reduction order may differ (not bitwise).
        self.batched_match_attention = False
        width = config.width
        self.public = nn.Linear(TOKEN_DIM, width)
        self.round_embedding = nn.Embedding(config.max_rounds, width)
        self.phase_embedding = nn.Embedding(4, width)
        self.bos = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, config.heads, width * 4, dropout=0.0,
                                           batch_first=True, norm_first=True)
        self.stream = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.stream_norm = nn.LayerNorm(width)
        self.private = mlp(config.obs_dim, width, 2)
        self.seat = nn.Embedding(4, width)
        self.q_proj = nn.Linear(width, width)
        self.kv_proj = nn.Linear(width, 2 * width)
        self.out_proj = nn.Linear(width, width)
        self.attention_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(nn.Linear(width, width * 4), nn.ReLU(),
                                          nn.Linear(width * 4, width))
        self.output_norm = nn.LayerNorm(width)
        self.action_tower = mlp(config.act_dim, config.action_width, 2)
        self.fusion = nn.Sequential(mlp(width + config.action_width, config.fusion_width, 2),
                                    nn.Linear(config.fusion_width, 1))
        if config.response_mode != "none":
            # Preserve the base actor AND subsequent critic initialization.
            # B/C allocate identical parameters. A zero bridge keeps their
            # initial policy identical; only C receives nonzero bridge inputs.
            with torch.random.fork_rng(devices=[]):
                self.response_head = nn.Sequential(
                    mlp(width + config.action_width, width, 2),
                    nn.Linear(width, RESPONSE_CLASSES))
                self.response_bridge = nn.Linear(RESPONSE_CLASSES,
                                                  width + config.action_width, bias=False)
                nn.init.zeros_(self.response_bridge.weight)
        if config.style_input:
            # After the base actor, so every other parameter keeps its initialisation and order.
            self.style_embedding = nn.Embedding(3, width)
            nn.init.zeros_(self.style_embedding.weight)
        # Auxiliary heads come last (``extend_actor`` and the optimizer remap rely
        # on it) and draw from a forked RNG stream, so the base actor and the
        # critic created after it are initialised exactly as without heads.
        heads = config.aux_head_names
        if heads:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(0x4175)
                if "next" in heads:
                    self.next_head = nn.Sequential(mlp(width, width, 1), nn.Linear(width, NEXT_LOGITS))
                if "belief" in heads:
                    self.belief_head = nn.Sequential(mlp(width, width, 1),
                                                     nn.Linear(width, NUM_CARDS * HIDDEN_SEATS))
                if "outcome" in heads:
                    self.outcome_head = nn.Sequential(mlp(width, width, 1), nn.Linear(width, 1))

    # -- auxiliary heads --------------------------------------------------------

    @property
    def aux_head_names(self) -> tuple[str, ...]:
        return self.config.aux_head_names

    def auxiliary_parameter_names(self) -> set[str]:
        """State-dict keys that belong to the auxiliary heads."""
        prefixes = tuple(f"{name}_head." for name in self.aux_head_names)
        return {key for key in self.state_dict() if key.startswith(prefixes)} if prefixes else set()

    def next_logits(self, encoded: Tensor) -> Tensor:
        """``[B, S, NEXT_LOGITS]`` next-token logits for encoded positions ``[B, S, width]``."""
        return self.next_head(encoded)

    def belief_logits(self, state: Tensor) -> Tensor:
        """``[n, 54, 3]``: per card, the relative seat (+1, +2, +3) of each unseen copy."""
        return self.belief_head(state).view(len(state), NUM_CARDS, HIDDEN_SEATS)

    def outcome_value(self, state: Tensor) -> Tensor:
        """``[n]`` predicted round return of the acting team."""
        return self.outcome_head(state).squeeze(-1)

    # -- public side ----------------------------------------------------------

    def encode_stream(self, tokens: Tensor, rounds: Tensor, phases: Tensor, lengths: Tensor
                      ) -> Tensor:
        """Causal encoding of padded public streams.

        Returns ``[B, T + 1, width]``: position 0 is BOS (an empty prefix),
        position ``s`` has seen exactly the first ``s`` tokens. Only public
        tokens enter here; there is deliberately no observation argument.
        """
        embedded = (self.public(tokens.float())
                    + self.round_embedding(rounds.clamp(0, self.config.max_rounds - 1))
                    + self.phase_embedding(phases.clamp(0, 3)))
        stream = torch.cat((self.bos.expand(len(tokens), -1, -1), embedded), dim=1)
        length = stream.shape[1]
        stream = stream + sinusoidal(length, self.config.width, stream.device, stream.dtype)
        if self.causal_sdpa:
            from train.history_attention import causal_encode
            return self.stream_norm(causal_encode(self.stream, stream))
        padding = torch.arange(length, device=stream.device)[None] > lengths[:, None]
        causal = torch.ones(length, length, device=stream.device, dtype=torch.bool).triu(1)
        return self.stream_norm(self.stream(stream, mask=causal, src_key_padding_mask=padding))

    def encode_batch(self, batch: StreamBatch) -> Tensor:
        return self.encode_stream(batch.tokens, batch.rounds, batch.phases, batch.lengths)

    def _windowed(self, inputs: DecisionInputs) -> tuple[Tensor, Tensor]:
        """Per-decision streams of the last ``window`` tokens; (stream, lengths)."""
        k = self.config.window
        s = inputs.streams
        device = inputs.obs.device
        lengths = inputs.prefix.clamp(max=k)
        start = inputs.prefix - lengths
        positions = start[:, None] + torch.arange(k, device=device)[None]
        valid = positions < inputs.prefix[:, None]
        positions = positions.clamp(min=0, max=max(s.tokens.shape[1] - 1, 0))
        rows = inputs.match_index[:, None].expand(-1, k)

        def gather(source: Tensor, fill: Tensor) -> Tensor:
            picked = source[rows, positions]
            return torch.where(valid if picked.ndim == 2 else valid[..., None], picked, fill)

        if s.tokens.shape[1] == 0:
            tokens = torch.zeros((inputs.decisions, k, TOKEN_DIM), dtype=torch.uint8, device=device)
            rounds = torch.zeros((inputs.decisions, k), dtype=torch.long, device=device)
            phases = torch.full((inputs.decisions, k), PLAY_PHASE, dtype=torch.long, device=device)
        else:
            tokens = gather(s.tokens, torch.zeros((), dtype=torch.uint8, device=device))
            rounds = gather(s.rounds, torch.zeros((), dtype=torch.long, device=device))
            phases = gather(s.phases, torch.full((), PLAY_PHASE, dtype=torch.long, device=device))
        return self.encode_stream(tokens, rounds, phases, lengths), lengths

    # -- private side ---------------------------------------------------------

    def _attend(self, query: Tensor, keys: Tensor, values: Tensor, allowed: Tensor) -> Tensor:
        """query [n, w]; keys/values [1, S, w] shared or [n, S, w] per row; allowed [n, S]."""
        width, heads = self.config.width, self.config.heads
        depth = width // heads
        q = self.q_proj(query).view(-1, heads, depth)
        k = keys.view(keys.shape[0], keys.shape[1], heads, depth)
        v = values.view(values.shape[0], values.shape[1], heads, depth)
        if keys.shape[0] == 1:
            out = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k[0].transpose(0, 1)[None], v[0].transpose(0, 1)[None],
                attn_mask=allowed[None, None])[0].transpose(0, 1)
        else:
            out = F.scaled_dot_product_attention(
                q[:, :, None], k.transpose(1, 2), v.transpose(1, 2),
                attn_mask=allowed[:, None, None])[:, :, 0]
        return self.out_proj(out.reshape(-1, width))

    @staticmethod
    def _project_independent(layer: nn.Linear, value: Tensor) -> Tensor:
        """Batch independent one-row linear problems without widening GEMM M."""
        if value.device.type == "cuda":
            # CUDA baddbmm uses a different reduction from the original M=1
            # linear call, even with deterministic algorithms and TF32 off.
            # Keep those projections exact while batching the attention below.
            return torch.cat([layer(value[index:index + 1])
                              for index in range(len(value))])
        weights = layer.weight.t()[None].expand(len(value), -1, -1)
        return torch.baddbmm(layer.bias[None, None], value[:, None], weights)[:, 0]

    def _attend_independent(self, query: Tensor, keys: Tensor, values: Tensor,
                            allowed: Tensor) -> Tensor:
        """Batch one query per public stream, retaining per-row projections.

        Widening the linear layers' GEMM rows changes their floating-point
        reductions. CPU uses equivalent one-row batched matrix multiplies;
        CUDA retains the individual linear calls. Both batch the independent
        SDPA problems, with exact checks required on the target backend.
        ``wide_private_projection`` instead runs each projection as one linear
        call over all rows: O(1) launches, FP32 kept, reduction order may differ.
        """
        count = len(query)
        width, heads = self.config.width, self.config.heads
        depth = width // heads
        project = ((lambda layer, value: layer(value)) if self.wide_private_projection
                   else self._project_independent)
        q = project(self.q_proj, query)
        q = q.view(count, heads, depth)[:, :, None]
        k = keys.view(count, keys.shape[1], heads, depth).transpose(1, 2)
        v = values.view(count, values.shape[1], heads, depth).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None, None])
        out = out[:, :, 0].reshape(count, width)
        return project(self.out_proj, out)

    def _attend_by_match(self, query: Tensor, keys: Tensor, values: Tensor,
                         match_index: Tensor, prefix: Tensor,
                         rank: Tensor | None = None, slots: int = 0) -> Tensor:
        """All matches' decisions in one padded SDPA call, [n, w].

        Decisions are placed at (match, rank within match); padding slots see
        position 0 only, so no row is fully masked, and are never read back.
        ``rank``/``slots`` may come planned from the host (``training_batch``);
        otherwise they are derived here at the cost of one host synchronization.
        """
        count, matches = len(query), keys.shape[0]
        width, heads = self.config.width, self.config.heads
        depth = width // heads
        if rank is None:
            per_match = torch.bincount(match_index, minlength=matches)
            slots = int(per_match.max())        # the only host synchronization
            order = torch.argsort(match_index, stable=True)
            starts = torch.cumsum(per_match, 0) - per_match
            rank = torch.empty_like(match_index)
            rank[order] = torch.arange(count, device=query.device) - starts[match_index[order]]
        q = self.q_proj(query).view(count, heads, depth)
        padded = q.new_zeros(matches, slots, heads, depth).index_put((match_index, rank), q)
        visible = prefix.new_zeros(matches, slots).index_put((match_index, rank), prefix)
        allowed = torch.arange(keys.shape[1], device=keys.device)[None, None] <= visible[..., None]
        k = keys.view(matches, keys.shape[1], heads, depth).transpose(1, 2)
        v = values.view(matches, values.shape[1], heads, depth).transpose(1, 2)
        out = F.scaled_dot_product_attention(padded.transpose(1, 2), k, v,
                                             attn_mask=allowed[:, None])
        out = out.transpose(1, 2)[match_index, rank].reshape(count, width)
        return self.out_proj(out)

    def decision_states(self, encoded: Tensor | None, inputs: DecisionInputs,
                        collect: list | None = None) -> Tensor:
        """One state per decision from its private query and visible prefix.

        ``encoded`` is ``encode_batch(inputs.streams)`` for the full-history
        actor and is ignored by the windowed control, which encodes its own
        per-decision windows. A decision attends to positions ``<= prefix``.
        ``collect``, when a list, receives every encoded stream tensor this
        call produced as ``(encoded, group)`` (``group`` None for the whole
        batch) for the next-token head; the computation itself is unchanged.
        """
        query = self.private(inputs.obs.float()) + self.seat(inputs.seat)
        if self.config.style_input:
            if inputs.style is None:
                raise ValueError("a style-input actor needs the opponents' style per decision")
            query = query + self.style_embedding(inputs.style + 1)
        if self.config.window:
            stream, lengths = self._windowed(inputs)
            keys, values = self.kv_proj(stream).chunk(2, dim=-1)
            allowed = torch.arange(stream.shape[1], device=stream.device)[None] <= lengths[:, None]
            attended = self._attend(query, keys, values, allowed)
        elif encoded is None and inputs.match_groups is not None:
            attended = self._attend_length_groups(query, inputs, collect)
        else:
            if encoded is None:
                encoded = self.encode_batch(inputs.streams)
            if collect is not None:
                collect.append((encoded, None))
            attended = self._attend_encoded(query, encoded, inputs.match_index, inputs.prefix,
                                            inputs.one_decision_per_stream,
                                            inputs.match_rank, inputs.match_slots)
        state = self.attention_norm(query + attended)
        return self.output_norm(state + self.feed_forward(state))

    def _attend_encoded(self, query: Tensor, encoded: Tensor, match_index: Tensor,
                        prefix: Tensor, one_decision_per_stream: bool,
                        rank: Tensor | None = None, slots: int = 0) -> Tensor:
        """Each decision's attention over its encoded stream, positions ``<= prefix``."""
        keys, values = self.kv_proj(encoded).chunk(2, dim=-1)
        positions = torch.arange(encoded.shape[1], device=encoded.device)
        if self.batched_private_attention and one_decision_per_stream and len(query) > 1:
            allowed = positions[None] <= prefix[:, None]
            return self._attend_independent(query, keys, values, allowed)
        if self.batched_match_attention and not one_decision_per_stream and len(query) > 1:
            return self._attend_by_match(query, keys, values, match_index, prefix, rank, slots)
        attended = torch.empty_like(query)
        for b in range(encoded.shape[0]):
            if one_decision_per_stream:
                # The collector already knows this layout on the host.
                # Keep the same per-match attention shapes, avoiding a
                # device nonzero (and its CUDA synchronization) per row.
                rows = slice(b, b + 1)
            else:
                rows = (match_index == b).nonzero(as_tuple=True)[0]
                if not len(rows):
                    continue
            allowed = positions[None] <= prefix[rows][:, None]
            attended[rows] = self._attend(query[rows], keys[b:b + 1], values[b:b + 1], allowed)
        return attended

    def _attend_length_groups(self, query: Tensor, inputs: DecisionInputs,
                              collect: list | None = None) -> Tensor:
        """``_attend_encoded`` per length group: each group's streams are encoded
        only as far as its decisions read (causal, so every visible position is
        exactly what the full-length encode gives) and padded only to the
        group's own longest. Same function; FP32 reduction order may differ."""
        s = inputs.streams
        parts, rows = [], []
        for group in inputs.match_groups:
            n = group.tokens
            encoded = self.encode_stream(s.tokens[group.matches, :n], s.rounds[group.matches, :n],
                                         s.phases[group.matches, :n],
                                         s.lengths[group.matches].clamp(max=n))
            if collect is not None:
                collect.append((encoded, group))
            parts.append(self._attend_encoded(query[group.rows], encoded, group.local_match,
                                              inputs.prefix[group.rows], False,
                                              group.match_rank, group.match_slots))
            rows.append(group.rows)
        return torch.zeros_like(query).index_copy(0, torch.cat(rows), torch.cat(parts))

    def candidate_outputs(self, state: Tensor, cand: Tensor, offsets: Tensor,
                          *, predict: bool = False, rows: Tensor | None = None,
                          response_rows: Tensor | None = None
                          ) -> tuple[Tensor, Tensor | None]:
        """Policy logits and optional response logits for every legal candidate.

        Prediction uses the observer state plus that candidate, without a
        future event or a target-seat label. The explicit bridge reads detached
        probabilities: PPO learns to use predictions, while the response head
        itself is supervised by actual public outcomes in both B and C.
        ``response_rows`` (auxiliary only) predicts just those candidates: the
        head is row-wise, so this is the same values as indexing afterwards.
        """
        if rows is None:
            counts = offsets[1:] - offsets[:-1]
            rows = torch.repeat_interleave(torch.arange(len(state), device=state.device), counts,
                                           output_size=len(cand))
        fused = torch.cat((state[rows], self.action_tower(cand.float())), dim=-1)
        response = None
        mode = self.config.response_mode
        if response_rows is not None and mode != "auxiliary":
            raise ValueError("response_rows needs the auxiliary response mode")
        if mode != "none" and (predict or mode == "explicit"):
            response = self.response_head(fused if response_rows is None else fused[response_rows])
        if mode == "explicit":
            feature = response.softmax(-1).detach() - 1.0 / RESPONSE_CLASSES
            fused = fused + self.response_bridge(feature)
        elif mode == "auxiliary" and torch.is_grad_enabled():
            # Matched bridge shape/parameter count without response information.
            # Its output is exactly zero, so inference skips it; training keeps
            # it so the bridge weight still receives its (zero) gradient.
            fused = fused + self.response_bridge(fused.new_zeros(len(cand), RESPONSE_CLASSES))
        return self.fusion(fused).squeeze(-1), response

    def candidate_logits(self, state: Tensor, cand: Tensor, offsets: Tensor,
                         *, rows: Tensor | None = None) -> Tensor:
        """One logit per concrete candidate of the full canonical set, [sum_k]."""
        return self.candidate_outputs(state, cand, offsets, rows=rows)[0]

    def candidate_log_probs(self, inputs: DecisionInputs, encoded: Tensor | None = None
                            ) -> Tensor:
        """Log-probability of every candidate, ``[sum_k]``, softmax within each decision."""
        state = self.decision_states(encoded, inputs)
        rows = inputs.rows
        logits = self.candidate_logits(state, inputs.cand, inputs.offsets, rows=rows)
        return segment_log_softmax(logits, rows, inputs.decisions)

    def forward(self, inputs: DecisionInputs) -> Tensor:
        return self.candidate_log_probs(inputs)

    @torch.no_grad()
    def act(self, inputs: DecisionInputs, generator: torch.Generator | None = None,
            greedy: bool = False, *, encoded: Tensor | None = None,
            max_candidates: int | None = None,
            inference_log_probs: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Sample (or take the argmax of) one candidate per decision.

        Returns ``(choice, log_prob)``: ``choice[i]`` is relative to the
        decision's own candidate range, so the engine choice index is exactly
        ``choice[i]``; ``log_prob[i]`` is the behaviour log-probability under
        this actor, which PPO stores.
        """
        table = self._log_prob_table(inputs, encoded, max_candidates, inference_log_probs)
        choice = table.argmax(1) if greedy else _gumbel_argmax(table, generator)
        chosen = table.gather(1, choice[:, None])[:, 0]
        return choice, chosen

    def _log_prob_table(self, inputs: DecisionInputs, encoded: Tensor | None,
                        max_candidates: int | None,
                        inference_log_probs: Tensor | None = None) -> Tensor:
        """``[n, width]`` candidate log-probabilities, ``-inf`` past each count."""
        # A collector-owned inference cache may supply the same complete
        # candidate vector. Sampling and its generator order remain here.
        log_probs = (self.candidate_log_probs(inputs, encoded=encoded)
                     if inference_log_probs is None else inference_log_probs)
        if inference_log_probs is not None and (
                log_probs.shape != (len(inputs.cand),) or log_probs.device != inputs.obs.device
                or log_probs.dtype != torch.float32):
            raise ValueError("cached inference must supply FP32 log probabilities for every candidate")
        counts = inputs.counts
        offsets = inputs.offsets
        longest = max_candidates if max_candidates is not None else (int(counts.max()) if len(counts) else 0)
        table = torch.full((inputs.decisions, max(longest, 1)), float("-inf"),
                           device=log_probs.device, dtype=log_probs.dtype)
        rows = inputs.rows
        local = torch.arange(len(log_probs), device=log_probs.device) - offsets[:-1][rows]
        table[rows, local] = log_probs
        return table

    @torch.no_grad()
    def explore(self, inputs: DecisionInputs, generator: torch.Generator | None = None, *,
                temperature: float = 1.0, epsilon: float = 0.0,
                encoded: Tensor | None = None, max_candidates: int | None = None,
                inference_log_probs: Tensor | None = None
                ) -> ExplorationSample:
        """Sample from pi_b = (1 - epsilon) softmax(logits / T) + epsilon / n_legal.

        With ``temperature == 1`` and ``epsilon == 0`` this is ``act``
        exactly: the same table, the same single Gumbel draw from
        ``generator``, the same choice, and ``behaviour_logp`` is the very
        ``logp`` tensor. With ``epsilon > 0`` two extra ``[n]`` uniform draws
        follow the Gumbel draw (the epsilon coin, then the uniform index), so
        the generator stream advances further than ``act`` would.
        """
        if not temperature > 0 or not 0 <= epsilon <= 1:
            raise ValueError("temperature must be positive and epsilon in [0, 1]")
        table = self._log_prob_table(inputs, encoded, max_candidates, inference_log_probs)
        counts = inputs.counts
        valid = torch.arange(table.shape[1], device=table.device)[None] < counts[:, None]
        behaviour = table if temperature == 1 else torch.log_softmax(table / temperature, 1)
        choice = _gumbel_argmax(behaviour, generator)
        uniform_pick = torch.zeros(len(choice), dtype=torch.bool, device=choice.device)
        if epsilon > 0:
            coin = torch.rand(len(choice), generator=generator, device=table.device,
                              dtype=torch.float64)
            draw = torch.rand(len(choice), generator=generator, device=table.device,
                              dtype=torch.float64)
            index = torch.minimum((draw * counts).long(), counts - 1)
            uniform_pick = coin < epsilon
            choice = torch.where(uniform_pick, index, choice)
            mixed = torch.logaddexp(behaviour + math.log1p(-epsilon) if epsilon < 1
                                    else torch.full_like(behaviour, float("-inf")),
                                    (math.log(epsilon) - counts.to(table.dtype).log())[:, None])
            behaviour = torch.where(valid, mixed, table)
        logp = table.gather(1, choice[:, None])[:, 0]
        behaviour_logp = behaviour.gather(1, choice[:, None])[:, 0]
        terms = torch.where(valid, behaviour.exp() * behaviour, torch.zeros_like(behaviour))
        return ExplorationSample(choice, logp, behaviour_logp, uniform_pick, -terms.sum(1))


def _gumbel_argmax(table: Tensor, generator: torch.Generator | None) -> Tensor:
    """One categorical draw per row of a log-probability table (one uniform per cell)."""
    uniform = torch.rand(table.shape, generator=generator, device=table.device,
                         dtype=table.dtype).clamp_min(1e-12)
    gumbel = -(-uniform.log()).log()
    return (table + gumbel).argmax(1)


# ---- critic -------------------------------------------------------------------

class HistoryCritic(nn.Module):
    """Separate value network over the privileged training state.

    Input is the acting seat's observation concatenated with the three hidden
    hands' per-card counts (training-time information the actor never sees).
    Randomly initialised; shares nothing with the actor or any old critic.
    """

    def __init__(self, config: HistoryPolicyConfig = HistoryPolicyConfig()) -> None:
        super().__init__()
        self.config = config
        self.tower = mlp(config.obs_dim + HIDDEN_DIM, config.critic_width, config.critic_layers)
        self.head = nn.Linear(config.critic_width, 1)

    def forward(self, obs: Tensor, hidden: Tensor) -> Tensor:
        state = torch.cat((obs.float(), hidden.float().reshape(len(obs), -1)), dim=-1)
        return self.head(self.tower(state)).squeeze(-1)


# ---- checkpoints -------------------------------------------------------------

def fresh_player(config: HistoryPolicyConfig, seed: int) -> tuple[HistoryActor, HistoryCritic]:
    """Randomly initialised actor and critic; nothing is loaded from disk."""
    torch.manual_seed(seed)
    return HistoryActor(config), HistoryCritic(config)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def checkpoint_payload(actor: HistoryActor, critic: HistoryCritic, *, lineage: str, seed: int,
                       optimizer: dict[str, Any] | None = None, config: dict[str, Any] | None = None,
                       progress: dict[str, Any] | None = None, rng: dict[str, Any] | None = None,
                       ) -> dict[str, Any]:
    """The on-disk record of a history player.

    ``stage`` marks the lineage, ``init``/``teacher`` record the training
    boundary of DESIGN 1.1, and ``token_schema`` pins what the actor reads.
    The remaining keys follow ``train.ckpt`` so one atomic writer serves both
    the old and the new players; old evaluators reject this stage explicitly.
    """
    if actor.config != critic.config:
        raise ValueError("actor and critic must share one architecture config")
    return {
        "stage": STAGE, "lineage": str(lineage), "init": "random", "teacher": None,
        "token_schema": TOKEN_SCHEMA_VERSION, "seed": int(seed),
        "model_config": config_record(actor.config), "model": actor.state_dict(),
        "critic": critic.state_dict(), "optimizer": optimizer or {},
        "config": dict(config or {}, stage=STAGE, init="random", teacher=None),
        "progress": dict(progress or {}), "rng": dict(rng or {}),
    }


def save_history_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    if payload.get("stage") != STAGE:
        raise ValueError("only history_ppo payloads are written by this helper")
    save_checkpoint(path, payload)


def load_history_checkpoint(path: str | Path, device: str | torch.device = "cpu"
                            ) -> tuple[HistoryActor, HistoryCritic, dict[str, Any]]:
    """Load a history player. Anything else (old MLP stages, teachers) is refused."""
    payload = load_checkpoint(path, device)
    if payload.get("stage") != STAGE:
        raise ValueError(f"not a history_ppo checkpoint: stage={payload.get('stage')!r}")
    if payload.get("init") != "random" or payload.get("teacher") is not None:
        raise ValueError("history players must start randomly and have no teacher")
    if payload.get("token_schema") != TOKEN_SCHEMA_VERSION:
        raise ValueError("token schema mismatch")
    config = HistoryPolicyConfig(**payload["model_config"])
    actor, critic = HistoryActor(config), HistoryCritic(config)
    actor.load_state_dict(payload["model"])
    critic.load_state_dict(payload["critic"])
    return actor.to(device), critic.to(device), payload


def extend_actor(base: HistoryActor, config: HistoryPolicyConfig) -> HistoryActor:
    """The same actor with ``config``'s auxiliary heads added (freshly
    initialised); every other parameter is copied, so the policy is unchanged.
    Only ``aux_heads`` may differ between the two configurations, and heads
    are never removed."""
    if config_record(replace(base.config, aux_heads=config.aux_heads)) != config_record(config):
        raise ValueError("extend_actor changes only the auxiliary heads")
    if not set(base.aux_head_names) <= set(config.aux_head_names):
        raise ValueError("auxiliary heads cannot be removed from a trained actor")
    actor = HistoryActor(config)
    missing, unexpected = actor.load_state_dict(base.state_dict(), strict=False)
    added = actor.auxiliary_parameter_names() - base.auxiliary_parameter_names()
    if unexpected or set(missing) != added:
        raise ValueError(f"extended actor weights do not fit: missing {missing}, extra {unexpected}")
    return actor.to(next(base.parameters()).device)
