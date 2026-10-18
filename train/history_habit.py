"""Planted-habit diagnostic: a fixed styled opponent pack and the learner's views.

Diagnostic only (docs/reports/history-habit-diagnostic-plan-2026-09-29.md and
history-habit-phase0-2026-09-30.md). Approved exceptions, confined to runs with
``habit_pack`` set: training opponents are a fixed read-only pack (one frozen
base checkpoint plus a match-fixed logit bias), the learner continues from a
trained checkpoint, and the ORACLE view reads the true style. Nothing here is
used by the main lineage; every switch defaults to off.

Pack. Both enemy seats of a match play the base actor with
``pi_z(a) ∝ pi(a) exp(z * b * f(a))``, z ∈ {-1, +1} drawn once per match; f is
the declared axis feature. Every legal candidate is still scored and
selectable. Identity 1 is z = -1, identity 2 is z = +1. The two learner seats
(identity 0) are one team, chosen per match.

Views of the learner (``habit_view``):
  full    the whole match history (unchanged model and inputs);
  round   public stream reset at every round start: each decision encodes only
          the current round's events, from position 0, with their true round
          index (``RoundEventStore``). Blocks earlier rounds at the encoder,
          not only at the final cross-attention;
  oracle  full history plus the true z as a private input
          (``HistoryPolicyConfig.style_input``).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from train.history_model import (DecisionInputs, HistoryActor, PublicStream,
                                 load_history_checkpoint, segment_log_softmax)
from train.history_population import weights_digest
from train.history_rollout import MatchEventStore

STYLE_AXES = ("pass", "bomb", "lead_single")
HABIT_VIEWS = ("full", "round", "oracle")
ACT_TYPE = 108                 # action encoding: type one-hot, Pass .. BackTribute
TYPE_COUNT = 13
PASS, SINGLE = 0, 1
BOMB_TYPES = (8, 9, 10)        # Bomb, StraightFlush, JokerBomb
STYLE_IDENTITY = {-1: 1, 1: 2}
IDENTITY_STYLE = {1: -1, 2: 1}
ROUND_KEY_STRIDE = 1 << 16     # learner cache key: (env, match * stride + round)


def style_feature(inputs: DecisionInputs, axis: str) -> Tensor:
    """f(a) ∈ {0, 1} per candidate row of ``inputs`` (float, candidate order)."""
    types = inputs.cand[:, ACT_TYPE:ACT_TYPE + TYPE_COUNT].float().argmax(1)
    if axis == "pass":
        return (types == PASS).float()
    if axis == "bomb":
        return torch.isin(types, torch.tensor(BOMB_TYPES, device=types.device)).float()
    if axis == "lead_single":
        # Leading: no Pass among the decision's candidates.
        rows = inputs.rows
        has_pass = torch.zeros(inputs.decisions, device=types.device).index_add_(
            0, rows, (types == PASS).float()) > 0
        return ((types == SINGLE) & ~has_pass[rows]).float()
    raise ValueError(f"unknown style axis {axis!r}")


class StyledActor(HistoryActor):
    """A frozen base actor sampling from its match-fixed tilted distribution."""

    def set_style(self, axis: str, strength: float, z: int) -> "StyledActor":
        if axis not in STYLE_AXES or z not in (-1, 1) or not strength >= 0:
            raise ValueError("style needs a known axis, z in {-1, +1} and strength >= 0")
        self.style_axis, self.style_strength, self.style_z = axis, float(strength), int(z)
        return self

    def styled_log_probs(self, inputs: DecisionInputs, base: Tensor) -> Tensor:
        bias = (self.style_z * self.style_strength) * style_feature(inputs, self.style_axis)
        return segment_log_softmax(base + bias.to(base.dtype), inputs.rows, inputs.decisions)

    @torch.no_grad()
    def act(self, inputs: DecisionInputs, generator: torch.Generator | None = None,
            greedy: bool = False, *, encoded: Tensor | None = None,
            max_candidates: int | None = None,
            inference_log_probs: Tensor | None = None) -> tuple[Tensor, Tensor]:
        base = (self.candidate_log_probs(inputs, encoded=encoded)
                if inference_log_probs is None else inference_log_probs)
        return super().act(inputs, generator, greedy, encoded=encoded,
                           max_candidates=max_candidates,
                           inference_log_probs=self.styled_log_probs(inputs, base))


def load_styled_pair(path: str | Path, axis: str, strength: float, device
                     ) -> tuple[dict[int, StyledActor], dict[str, Any]]:
    """Identities 1 (z = -1) and 2 (z = +1) over one frozen copy each of ``path``."""
    base, _, payload = load_history_checkpoint(path, "cpu")
    digest = weights_digest(base.state_dict())
    models = {}
    for z, identity in STYLE_IDENTITY.items():
        model = StyledActor(base.config)
        model.load_state_dict(base.state_dict())
        models[identity] = model.set_style(axis, strength, z).requires_grad_(False).train().to(device)
    meta = dict(path=str(path), sha256=digest, lineage=payload.get("lineage"),
                update=payload.get("progress", {}).get("updates"), axis=axis,
                strength=float(strength), model_config=dict(payload["model_config"]))
    return models, meta


class HabitPack:
    """Seat assignment and frozen styled opponents; replaces the snapshot population.

    Per match: learner team t (seats t, t + 2) and z, both drawn from this
    pack's generator; both other seats are the styled identity of z.
    """

    def __init__(self, actor: HistoryActor, lineage: str, seed: int, path: str | Path,
                 axis: str, strength: float, device) -> None:
        self.actor, self.lineage = actor, lineage
        self.models, self.pack = load_styled_pair(path, axis, strength, device)
        base = self.pack["model_config"]
        own = {k: v for k, v in vars(actor.config).items() if k in base}
        if own != base:
            raise ValueError("the styled pack must share the learner's architecture")
        self.metadata = {i: dict(identity=i, lineage="habit-pack", z=IDENTITY_STYLE[i],
                                 sha256=self.pack["sha256"]) for i in self.models}
        self.rng = np.random.default_rng(seed)
        self.styles: dict[tuple[int, int], int] = {}
        self.seat_matches: Counter = Counter()
        self.decisions: Counter = Counter()

    def assignment(self, env: int, match: int) -> list[int]:
        team = int(self.rng.integers(2))
        z = 1 if self.rng.random() < 0.5 else -1
        seats = [STYLE_IDENTITY[z]] * 4
        seats[team] = seats[team + 2] = 0
        self.styles[(int(env), int(match))] = z
        self.seat_matches.update(seats)
        return seats

    def style(self, env: int, match: int) -> int:
        """The z of the learner's opponents in that match."""
        return self.styles[(int(env), int(match))]

    def resolve(self, identity: int) -> HistoryActor:
        if identity == 0:
            return self.actor
        if identity not in self.models:
            raise ValueError(f"unknown habit-pack identity {identity}")
        return self.models[identity]

    def snapshot(self, update: int) -> int:
        raise RuntimeError("the habit pack is fixed; snapshot_updates must be 0")

    def prune(self, assignments) -> None:
        # Called after learn: rows still carried belong to matches in progress.
        for key in [k for k in self.styles if k not in assignments]:
            del self.styles[key]

    def config(self) -> dict:
        return dict(pack=self.pack)

    def state_dict(self) -> dict:
        return dict(lineage=self.lineage, pack=self.pack, rng=self.rng.bit_generator.state,
                    seat_matches=dict(self.seat_matches), decisions=dict(self.decisions))

    def load_state_dict(self, state: dict) -> None:
        if state.get("lineage") != self.lineage or state.get("pack") != self.pack:
            raise ValueError("habit pack lineage/weights/style mismatch")
        self.rng.bit_generator.state = state["rng"]
        self.seat_matches = Counter(state["seat_matches"])
        self.decisions = Counter(state["decisions"])
        self.styles.clear()

    def metrics(self) -> dict:
        return dict(habit_pack=self.pack["sha256"][:16], axis=self.pack["axis"],
                    strength=self.pack["strength"], seat_matches=dict(self.seat_matches),
                    decisions=dict(self.decisions))


class RoundEventStore(MatchEventStore):
    """Full match streams (for the pack) plus one stream per (env, match, round).

    Round streams hold only that round's public events, tribute included,
    with their true round index; the ROUND learner reads them from position 0.
    """

    def __init__(self) -> None:
        super().__init__()
        self.rounds: dict[tuple[int, int, int], PublicStream] = {}

    def round_stream(self, env_id: int, match_id: int, round_index: int) -> PublicStream:
        key = (int(env_id), int(match_id), int(round_index))
        stream = self.rounds.get(key)
        if stream is None:
            self.stream(env_id, match_id)      # match-id monotonicity check
            stream = PublicStream(key[1])
            self.rounds[key] = stream
        return stream

    def ingest(self, events) -> int:
        for event in events:
            self.stream(event.env_id, event.match_id).append(event)
            self.round_stream(event.env_id, event.match_id, event.round_index).append(event)
        return len(events)

    def prune(self, cited: set[tuple[int, int]]) -> int:
        count = super().prune(cited)
        for key in [k for k in self.rounds if (k[0], k[1]) not in self.streams]:
            del self.rounds[key]
        return count
