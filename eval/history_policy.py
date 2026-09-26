"""History-aware evaluation policy and the public event plumbing of the evaluators.

T3 of ``docs/STAGE_C_TODO.md``: every evaluator must deliver every public
action, in order, to a history policy and reset its stream at the right
boundary (DESIGN.md 7.3, 9.2). This module holds

* ``HistoryPolicy``: a ``history_ppo`` checkpoint (``train.history_model``)
  behind the scalar ``Policy`` protocol of ``eval.policies``. It keeps one
  ``PublicStream`` per match it is playing; the round loops call
  ``start_match()`` at a match boundary and ``observe()`` after every applied
  action of every seat. ``select()`` refuses to act without an open match.
* the scalar event source: ``apply_and_observe`` mirrors what ``VecEnv``
  records in ``cpp/src/env.cpp`` (seat, phase and round index before the
  action, the actor's public card count after it) from ``Engine.apply``, and
  ``resolve_forced_passes`` replays the engine's auto-pass rule explicitly so
  that engine-resolved passes reach the stream too. With these the stream
  built by a scalar loop equals, token for token, the stream a
  ``VecEnv(log_public_actions=True)`` produces (``tests/test_history_eval.py``).
* ``HistoryStreamStore``: per-slot streams for ``eval.batched`` fed from
  ``VecEnv.drain_public_actions()``, reset whenever a slot's ``match_id``
  changes.

Nothing here reads hidden hands, other seats' legal lists or the engine's
private ``forced`` flag into a token: ``PublicStream`` only takes the public
fields, and voluntary and forced passes get identical tokens.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

import gd
import numpy as np

PLAY = int(gd.Phase.Play)
CARDS_LEFT_OFFSET = 4 + int(gd.ACT_DIM)   # token layout of train.logs.public_token


# ---- events from Engine.apply -------------------------------------------------

@dataclass(frozen=True)
class PublicEvent:
    """One public action as ``PublicStream.append`` accepts it.

    The fields mirror ``gd.PublicActionEvent``: ``seat``, ``phase`` and
    ``round_index`` describe the state before the action, ``cards_left`` is
    the actor's public count after it. ``forced`` is diagnostic only; the
    stream never stores it.
    """
    seat: int
    phase: int
    round_index: int
    encoded_action: np.ndarray
    cards_left: int
    forced: bool = False
    action: Any = None


def needs_history(policy: object) -> bool:
    return bool(getattr(policy, "needs_history", False))


def history_listeners(policies: Iterable[object]) -> list:
    """The distinct history policies of a lineup, in seat order."""
    listeners: list = []
    for policy in policies:
        if needs_history(policy) and not any(policy is seen for seen in listeners):
            listeners.append(policy)
    return listeners


def apply_and_observe(engine: gd.Engine, state: gd.MatchState, action: gd.Action,
                      listeners: Sequence[Any] = (), forced: bool = False) -> PublicEvent:
    """``engine.apply`` plus the public event VecEnv would log for it.

    Seat, phase, round index and the encoded action are taken before the
    action, the actor's remaining count after it, exactly like
    ``VecEnv::Impl::apply``. Private tribute flags are zeroed. Every listener
    gets the same event object.
    """
    seat = int(state.to_move)
    phase = int(state.phase)
    round_index = int(state.round_index)
    encoded = np.array(engine.encode_action(action, state, seat), dtype=np.float32)
    encoded[int(gd.ACT_TRIBUTE_FLAGS):] = 0.0
    engine.apply(state, action)
    event = PublicEvent(seat, phase, round_index, encoded, len(state.hand(seat)), forced, action)
    for listener in listeners:
        listener.observe(event)
    return event


def resolve_forced_passes(engine: gd.Engine, state: gd.MatchState,
                          listeners: Sequence[Any] = ()) -> int:
    """Apply the lone-pass replies the engine would auto-pass; returns their count.

    This is ``Engine::skip_forced`` (``cpp/src/state.cpp``) done in Python so
    the passes become events. The engine must have ``auto_pass`` off while a
    history round runs (see ``explicit_passes``); with it on, ``apply`` would
    already have skipped them and nothing is left to resolve.
    """
    count = 0
    while int(state.phase) == PLAY:
        actions = engine.legal_actions(state)
        if len(actions) != 1 or not actions[0].is_pass:
            break
        apply_and_observe(engine, state, actions[0], listeners, forced=True)
        count += 1
    return count


@contextmanager
def explicit_passes(engine: gd.Engine, listeners: Sequence[Any]):
    """Turn the engine's auto-pass off for a history round; restore afterwards.

    Without listeners the engine is untouched, so the old scalar loops run
    exactly as before.
    """
    if not listeners:
        yield
        return
    saved = engine.auto_pass
    engine.auto_pass = False
    try:
        yield
    finally:
        engine.auto_pass = saved


# ---- the policy ---------------------------------------------------------------

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
        from train.history_model import PublicStream

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
        if rounds and rounds[-1] > current:
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


# ---- batched streams ----------------------------------------------------------

class HistoryStreamStore:
    """One ``PublicStream`` per VecEnv slot, fed from ``drain_public_actions()``.

    A slot restarts a new match inside ``pending()`` with a new ``match_id``;
    the store resets that slot's stream on the first event of the new match
    and, because the first decision of a match precedes its first event,
    also when ``sync`` sees a newer ``match_id`` on a pending row.
    """

    def __init__(self, num_envs: int) -> None:
        from train.history_model import PublicStream

        if num_envs < 1:
            raise ValueError("a stream store needs at least one slot")
        self.streams = [PublicStream() for _ in range(num_envs)]
        self.ingested = 0

    def __len__(self) -> int:
        return len(self.streams)

    def ingest(self, events: Iterable[Any]) -> int:
        count = 0
        for event in events:
            stream = self.streams[int(event.env_id)]
            if stream.match_id != int(event.match_id):
                stream.reset(int(event.match_id))
            stream.append(event)
            count += 1
        self.ingested += count
        return count

    def sync(self, env_ids: np.ndarray, match_ids: np.ndarray) -> None:
        for env_id, match_id in zip(env_ids, match_ids):
            stream = self.streams[int(env_id)]
            if stream.match_id != int(match_id):
                stream.reset(int(match_id))

    def select_streams(self, env_ids: np.ndarray) -> tuple[list, np.ndarray]:
        """The distinct streams of ``env_ids`` and each row's index into them."""
        unique, match_index = np.unique(np.asarray(env_ids, dtype=np.int64), return_inverse=True)
        return [self.streams[int(e)] for e in unique], match_index.reshape(-1)

    def prefix(self, env_id: int) -> int:
        return self.streams[int(env_id)].prefix
