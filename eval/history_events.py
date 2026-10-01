"""Capture public engine events and manage evaluator match streams.

Scalar loops deliver every action, including engine-resolved passes, in order.
Batched loops reset each slot's stream when its match generation changes.
This plumbing depends on the public schema, independently of policy models.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import gd
import numpy as np

from train.public_history import PublicStream

PLAY = int(gd.Phase.Play)


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


# ---- batched streams ----------------------------------------------------------

class HistoryStreamStore:
    """One ``PublicStream`` per VecEnv slot, fed from ``drain_public_actions()``.

    A slot restarts a new match inside ``pending()`` with a new ``match_id``;
    the store resets that slot's stream on the first event of the new match
    and, because the first decision of a match precedes its first event,
    also when ``sync`` sees a newer ``match_id`` on a pending row.
    """

    def __init__(self, num_envs: int) -> None:
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
