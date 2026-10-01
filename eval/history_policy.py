"""History-aware evaluation policy over the full public match prefix.

Evaluators deliver events and maintain match boundaries through
``eval.history_events``. Its public names remain re-exported here for callers
using the original evaluator interface. The policy reads only public events
plus the acting seat's private observation and legal candidates.
"""
from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Sequence

import gd
import numpy as np

from train.public_history import PublicStream
from .history_events import (HistoryStreamStore, PublicEvent, apply_and_observe,
                             explicit_passes, history_listeners, needs_history,
                             resolve_forced_passes)

PLAY = int(gd.Phase.Play)
CARDS_LEFT_OFFSET = 4 + int(gd.ACT_DIM)   # train.public_history.public_token layout


class HistoryPolicy:
    """A ``history_ppo`` actor behind the scalar ``Policy`` protocol.

    Tribute and back-tribute choices go to the engine's tribute heuristic
    by default: the declared exchange-only exception of the fixed
    experiment boundary (``docs/STAGE_C_TODO.md``). Play decisions build a
    single-decision ``DecisionInputs`` whose prefix is the whole stream
    observed so far (full-prefix recomputation, the correctness reference of
    DESIGN.md 7.3). Greedy by default; ``sample=True`` draws from the
    actor's own softmax with a torch generator seeded from the loop's rng,
    so a fixed rng replays exactly.
    """

    needs_history = True

    def __init__(self, actor: Any, name: str = "history", device: str = "cpu",
                 sample: bool = False, heuristic_tribute: bool = True) -> None:
        import torch

        self.actor = actor.to(device).eval()
        self.name = name
        self.device = torch.device(device)
        self.sample = bool(sample)
        self.heuristic_tribute = bool(heuristic_tribute)
        self.action_mode = "canonical"
        self.stream = PublicStream()
        self.started = False
        self.matches_started = 0
        self.decisions = 0

    # -- stream bookkeeping ---------------------------------------------------

    def start_match(self, match_id: int = -1) -> None:
        """Forget the previous match; the next events belong to a new one."""
        self.stream.reset(match_id)
        self.started = True
        self.matches_started += 1

    def observe(self, event: Any) -> None:
        """Append one public action (any object shaped like ``PublicEvent``)."""
        if not self.started:
            raise RuntimeError(f"{self.name}: observe() before start_match(); the evaluator "
                               "did not open a match for this history policy")
        self.stream.append(event)

    @property
    def events_seen(self) -> int:
        """Public actions delivered since ``start_match()``; the next decision's prefix."""
        return self.stream.prefix

    def verify_stream(self, state: gd.MatchState) -> None:
        """Cheap public consistency check between the stream and the state.

        The engine does not expose its step counter, so a missing event is
        detected through what is public: every seat that has played this
        round must hold exactly the count its last play-phase token reported.
        This catches a dropped play by any seat; the exact proof of the event
        source is the VecEnv parity test, not this check.
        """
        rounds, phases, tokens = self.stream.rounds, self.stream.phases, self.stream.tokens
        current = int(state.round_index)
        if len(rounds) and rounds[-1] > current:
            raise RuntimeError(f"{self.name}: stream holds round {rounds[-1]} tokens but the "
                               f"state is in round {current}; start_match() is missing")
        seen: set[int] = set()
        for i in range(len(tokens) - 1, -1, -1):
            if rounds[i] != current:
                break
            if phases[i] != PLAY:
                continue
            seat = int(tokens[i][:4].argmax())
            if seat in seen:
                continue
            seen.add(seat)
            recorded = int(tokens[i][CARDS_LEFT_OFFSET:].argmax())
            actual = len(state.hand(seat))
            if recorded != actual:
                raise RuntimeError(f"{self.name}: seat {seat} holds {actual} cards but its last "
                                   f"observed play left {recorded}; a public event was not "
                                   "delivered, or this state is not the match start_match() opened")

    # -- decisions ------------------------------------------------------------

    def decision_inputs(self, engine: gd.Engine, state: gd.MatchState,
                        actions: Sequence[gd.Action]):
        """``DecisionInputs`` for one decision over the full prefix of this stream."""
        import torch
        from train.history_model import DecisionInputs, StreamBatch

        seat = int(state.to_move)
        obs = np.asarray(state.observation(seat), dtype=np.float32)[None, :]
        cand = np.stack([engine.encode_action(a, state, seat) for a in actions])
        streams = StreamBatch.from_streams([self.stream], self.device)
        return DecisionInputs(
            streams=streams,
            match_index=torch.zeros(1, dtype=torch.long, device=self.device),
            prefix=torch.tensor([self.stream.prefix], dtype=torch.long, device=self.device),
            obs=torch.as_tensor(obs, device=self.device),
            seat=torch.tensor([seat], dtype=torch.long, device=self.device),
            cand=torch.as_tensor(cand, device=self.device),
            offsets=torch.tensor([0, len(actions)], dtype=torch.long, device=self.device))

    def batch_inputs(self, streams: Sequence[Any], match_index: np.ndarray, seat: np.ndarray,
                     obs: np.ndarray, cand: np.ndarray, offsets: np.ndarray):
        """``DecisionInputs`` for several decisions over the given per-slot streams."""
        import torch
        from train.history_model import DecisionInputs, StreamBatch

        device = self.device
        prefix = np.asarray([streams[i].prefix for i in match_index], dtype=np.int64)
        return DecisionInputs(
            streams=StreamBatch.from_streams(list(streams), device),
            match_index=torch.as_tensor(np.asarray(match_index, dtype=np.int64), device=device),
            prefix=torch.as_tensor(prefix, device=device),
            obs=torch.as_tensor(np.ascontiguousarray(obs), device=device),
            seat=torch.as_tensor(np.asarray(seat, dtype=np.int64), device=device),
            cand=torch.as_tensor(np.ascontiguousarray(cand), device=device),
            offsets=torch.as_tensor(np.asarray(offsets, dtype=np.int64), device=device))

    def act(self, inputs: Any, generator: Any = None) -> np.ndarray:
        """Choice per decision (local candidate index), greedy or sampled."""
        import torch

        with torch.inference_mode():
            choice, log_prob = self.actor.act(inputs, generator, greedy=not self.sample)
        if not torch.isfinite(log_prob).all():
            raise ValueError(f"{self.name}: policy must produce finite log-probabilities")
        return choice.cpu().numpy()

    def select(self, engine: gd.Engine, state: gd.MatchState,
               actions: Sequence[gd.Action], rng: random.Random) -> int:
        if not self.started:
            raise RuntimeError(f"{self.name}: select() before start_match(); the evaluator "
                               "did not open a match for this history policy")
        self.verify_stream(state)
        self.decisions += 1
        if self.heuristic_tribute and int(state.phase) != PLAY:
            return engine.greedy(state)
        generator = None
        if self.sample:
            import torch

            generator = torch.Generator(device=self.device)
            generator.manual_seed(rng.getrandbits(63))
        choice = self.act(self.decision_inputs(engine, state, actions), generator)
        return int(choice[0])


def load_history_policy(path: str | Path, device: str = "cpu", margin: float = 0.0,
                        sample: bool = False, temperature: float | None = None) -> HistoryPolicy:
    """``load_policy`` branch for ``stage == "history_ppo"`` checkpoints."""
    from train.history_model import load_history_checkpoint

    from .policies import model_digest

    if margin:
        raise ValueError("a near-best margin is not supported for history policies")
    if temperature is not None:
        raise ValueError("history policies sample at their own temperature; use sample:<path>")
    path = Path(path)
    actor, _, payload = load_history_checkpoint(path, device)
    digest = model_digest(payload["model"])
    tribute_policy = payload.get("config", {}).get("tribute_policy", "heuristic")
    if tribute_policy not in ("heuristic", "learned"):
        raise ValueError("unsupported checkpoint tribute policy marker")
    name = f"{path.name}@{digest[:16]}/history"
    if tribute_policy == "learned":
        name += "/tribute=learned"
    if sample:
        name += "/sample"
    policy = HistoryPolicy(actor, name=name, device=device, sample=sample,
                           heuristic_tribute=tribute_policy == "heuristic")
    policy.checkpoint_id = digest
    policy.training_seed = payload.get("seed")
    policy.stage = payload["stage"]
    policy.base_checkpoint_id = None
    policy.base_training_seed = None
    policy.collection_seed = None
    policy.window = int(actor.config.window)
    return policy
