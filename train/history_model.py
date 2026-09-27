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

from dataclasses import asdict, dataclass, field
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

    def __post_init__(self) -> None:
        if min(self.width, self.layers, self.heads, self.max_rounds, self.action_width,
               self.fusion_width, self.critic_width, self.critic_layers) <= 0:
            raise ValueError("every size must be positive")
        if self.width % self.heads or self.window < 0:
            raise ValueError("width must be divisible by heads; window must not be negative")
        if self.obs_dim != int(gd.OBS_DIM) or self.act_dim != int(gd.ACT_DIM):
            raise ValueError("observation and action widths must match the engine")
        if self.response_mode not in ("none", "auxiliary", "explicit"):
            raise ValueError("response_mode must be none, auxiliary or explicit")


# ---- public stream ----------------------------------------------------------

class PublicStream:
    """Raw public events of one match for one environment, in order.

    ``append`` takes a ``gd.PublicActionEvent`` (or anything with ``seat``,
    ``encoded_action``, ``cards_left``, ``round_index`` and ``phase``).
    ``reset`` starts a new match. ``prefix`` is the number of tokens a
    decision taken now may read. Streams never store observations,
    candidates, hidden counts or the forced flag.
    """

    __slots__ = ("tokens", "rounds", "phases", "match_id", "generation")

    def __init__(self, match_id: int = -1) -> None:
        self.tokens: list[np.ndarray] = []
        self.rounds: list[int] = []
        self.phases: list[int] = []
        self.match_id = int(match_id)
        self.generation = 0

    def reset(self, match_id: int = -1) -> None:
        self.generation += 1
        self.tokens.clear()
        self.rounds.clear()
        self.phases.clear()
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
        self.tokens.append(token)
        self.rounds.append(int(round_index))
        self.phases.append(int(phase))

    @property
    def prefix(self) -> int:
        return len(self.tokens)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tokens = (np.stack(self.tokens) if self.tokens
                  else np.zeros((0, TOKEN_DIM), dtype=np.uint8))
        return (tokens, np.asarray(self.rounds, dtype=np.int64),
                np.asarray(self.phases, dtype=np.int64))


@dataclass
class StreamBatch:
    """Padded public streams of several matches on one device."""
    tokens: Tensor    # uint8 [B, T, TOKEN_DIM]
    rounds: Tensor    # int64 [B, T]
    phases: Tensor    # int64 [B, T]
    lengths: Tensor   # int64 [B]

    @staticmethod
    def from_arrays(streams: list[tuple[np.ndarray, np.ndarray, np.ndarray]], device
                    ) -> "StreamBatch":
        count = len(streams)
        longest = max((len(t) for t, _, _ in streams), default=0)
        tokens = torch.zeros((count, longest, TOKEN_DIM), dtype=torch.uint8)
        rounds = torch.zeros((count, longest), dtype=torch.long)
        phases = torch.full((count, longest), PLAY_PHASE, dtype=torch.long)
        lengths = torch.zeros(count, dtype=torch.long)
        for b, (t, r, p) in enumerate(streams):
            n = len(t)
            lengths[b] = n
            if n:
                tokens[b, :n] = torch.as_tensor(t)
                rounds[b, :n] = torch.as_tensor(r)
                phases[b, :n] = torch.as_tensor(p)
        return StreamBatch(tokens.to(device), rounds.to(device), phases.to(device),
                           lengths.to(device))

    @staticmethod
    def from_streams(streams: list[PublicStream], device) -> "StreamBatch":
        return StreamBatch.from_arrays([s.arrays() for s in streams], device)


@dataclass
class DecisionInputs:
    """Everything the actor needs for a batch of decisions.

    ``match_index[i]`` selects the stream of decision ``i`` inside the
    ``StreamBatch``; ``prefix[i]`` is how many of its tokens the decision may
    read. Candidates are ragged: ``cand[offsets[i]:offsets[i + 1]]`` are the
    full canonical candidates of decision ``i`` in engine order.
    """
    streams: StreamBatch
    match_index: Tensor   # int64 [n]
    prefix: Tensor        # int64 [n]
    obs: Tensor           # uint8 or float [n, obs_dim]
    seat: Tensor          # int64 [n]
    cand: Tensor          # uint8 or float [sum_k, act_dim]
    offsets: Tensor       # int64 [n + 1]

    @property
    def decisions(self) -> int:
        return len(self.obs)

    @property
    def counts(self) -> Tensor:
        return self.offsets[1:] - self.offsets[:-1]

    @property
    def rows(self) -> Tensor:
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

    def decision_states(self, encoded: Tensor | None, inputs: DecisionInputs) -> Tensor:
        """One state per decision from its private query and visible prefix.

        ``encoded`` is ``encode_batch(inputs.streams)`` for the full-history
        actor and is ignored by the windowed control, which encodes its own
        per-decision windows. A decision attends to positions ``<= prefix``.
        """
        query = self.private(inputs.obs.float()) + self.seat(inputs.seat)
        if self.config.window:
            stream, lengths = self._windowed(inputs)
            keys, values = self.kv_proj(stream).chunk(2, dim=-1)
            allowed = torch.arange(stream.shape[1], device=stream.device)[None] <= lengths[:, None]
            attended = self._attend(query, keys, values, allowed)
        else:
            if encoded is None:
                encoded = self.encode_batch(inputs.streams)
            keys, values = self.kv_proj(encoded).chunk(2, dim=-1)
            positions = torch.arange(encoded.shape[1], device=encoded.device)
            attended = torch.empty_like(query)
            for b in range(encoded.shape[0]):
                rows = (inputs.match_index == b).nonzero(as_tuple=True)[0]
                if not len(rows):
                    continue
                allowed = positions[None] <= inputs.prefix[rows][:, None]
                attended[rows] = self._attend(query[rows], keys[b:b + 1], values[b:b + 1], allowed)
        state = self.attention_norm(query + attended)
        return self.output_norm(state + self.feed_forward(state))

    def candidate_outputs(self, state: Tensor, cand: Tensor, offsets: Tensor,
                          *, predict: bool = False) -> tuple[Tensor, Tensor | None]:
        """Policy logits and optional response logits for every legal candidate.

        Prediction uses the observer state plus that candidate, without a
        future event or a target-seat label. The explicit bridge reads detached
        probabilities: PPO learns to use predictions, while the response head
        itself is supervised by actual public outcomes in both B and C.
        """
        counts = offsets[1:] - offsets[:-1]
        rows = torch.repeat_interleave(torch.arange(len(state), device=state.device), counts,
                                       output_size=len(cand))
        fused = torch.cat((state[rows], self.action_tower(cand.float())), dim=-1)
        response = None
        mode = self.config.response_mode
        if mode != "none" and (predict or mode == "explicit"):
            response = self.response_head(fused)
        if mode == "explicit":
            feature = response.softmax(-1).detach() - 1.0 / RESPONSE_CLASSES
            fused = fused + self.response_bridge(feature)
        elif mode == "auxiliary":
            # Matched bridge shape/parameter count without response information.
            fused = fused + self.response_bridge(fused.new_zeros(len(cand), RESPONSE_CLASSES))
        return self.fusion(fused).squeeze(-1), response

    def candidate_logits(self, state: Tensor, cand: Tensor, offsets: Tensor) -> Tensor:
        """One logit per concrete candidate of the full canonical set, [sum_k]."""
        return self.candidate_outputs(state, cand, offsets)[0]

    def candidate_log_probs(self, inputs: DecisionInputs, encoded: Tensor | None = None
                            ) -> Tensor:
        """Log-probability of every candidate, ``[sum_k]``, softmax within each decision."""
        state = self.decision_states(encoded, inputs)
        logits = self.candidate_logits(state, inputs.cand, inputs.offsets)
        return segment_log_softmax(logits, inputs.rows, inputs.decisions)

    def forward(self, inputs: DecisionInputs) -> Tensor:
        return self.candidate_log_probs(inputs)

    @torch.no_grad()
    def act(self, inputs: DecisionInputs, generator: torch.Generator | None = None,
            greedy: bool = False, *, encoded: Tensor | None = None,
            max_candidates: int | None = None) -> tuple[Tensor, Tensor]:
        """Sample (or take the argmax of) one candidate per decision.

        Returns ``(choice, log_prob)``: ``choice[i]`` is relative to the
        decision's own candidate range, so the engine choice index is exactly
        ``choice[i]``; ``log_prob[i]`` is the behaviour log-probability under
        this actor, which PPO stores.
        """
        log_probs = self.candidate_log_probs(inputs, encoded=encoded)
        counts = inputs.counts
        offsets = inputs.offsets
        longest = max_candidates if max_candidates is not None else (int(counts.max()) if len(counts) else 0)
        table = torch.full((inputs.decisions, max(longest, 1)), float("-inf"),
                           device=log_probs.device, dtype=log_probs.dtype)
        local = torch.arange(len(log_probs), device=log_probs.device) - offsets[:-1][inputs.rows]
        table[inputs.rows, local] = log_probs
        if greedy:
            choice = table.argmax(1)
        else:
            uniform = torch.rand(table.shape, generator=generator, device=table.device,
                                 dtype=table.dtype).clamp_min(1e-12)
            gumbel = -(-uniform.log()).log()
            choice = (table + gumbel).argmax(1)
        chosen = table.gather(1, choice[:, None])[:, 0]
        return choice, chosen


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
        "model_config": asdict(actor.config), "model": actor.state_dict(),
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
