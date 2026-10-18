"""Public event tokens and per-match streams, independent of models and the engine.

Only public action encodings, seat, remaining card count, round index and phase
enter this schema. The forced flag and private tribute structure are excluded.
Arrays stay read-only and retain their contents across append, growth and reset.
"""
from __future__ import annotations

from typing import Any

import numpy as np

TOKEN_DIM = 4 + 154 + 28
TOKEN_SCHEMA_VERSION = 1      # 186 public dims + round index + phase; no forced bit


def public_token(event: object) -> np.ndarray:
    token = np.zeros(TOKEN_DIM, dtype=np.uint8)
    token[int(event.seat)] = 1
    token[4:158] = event.encoded_action
    # Tribute structure is private, even if an older engine emits those flags.
    token[4 + 146:4 + 154] = 0
    token[158 + int(event.cards_left)] = 1
    return token


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

    def _reserve(self, length: int) -> None:
        if length <= len(self._rounds):
            return
        n = self._length
        capacity = max(self._INITIAL_CAPACITY, 2 * len(self._rounds), length)
        tokens, rounds, phases = self._tokens, self._rounds, self._phases
        self._new_buffers(capacity)
        self._tokens[:n] = tokens[:n]
        self._rounds[:n] = rounds[:n]
        self._phases[:n] = phases[:n]
        self._length = n

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
        self._reserve(n + 1)
        self._tokens[n] = token
        self._rounds[n] = int(round_index)
        self._phases[n] = int(phase)
        self._length = n + 1

    @staticmethod
    def _validated_batch(tokens: np.ndarray, rounds: np.ndarray, phases: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate public arrays once before distributing them among streams."""
        tokens = np.asarray(tokens, dtype=np.uint8)
        rounds = np.asarray(rounds, dtype=np.int64)
        phases = np.asarray(phases, dtype=np.int64)
        if tokens.ndim != 2 or tokens.shape[1] != TOKEN_DIM:
            raise ValueError(f"public tokens must have shape [events, {TOKEN_DIM}]")
        if rounds.shape != (len(tokens),) or phases.shape != (len(tokens),):
            raise ValueError("rounds and phases need one entry per public token")
        if ((tokens[:, :4].sum(1) != 1).any()
                or (tokens[:, 158:].sum(1) != 1).any()):
            raise ValueError("public token needs exactly one seat and one cards-left bit")
        if tokens[:, 4 + 146:4 + 154].any():
            raise ValueError("private tribute flags must not enter the public stream")
        if (rounds < 0).any() or (phases < 0).any():
            raise ValueError("round index and phase must not be negative")
        return tokens, rounds, phases

    def _extend_validated(self, tokens: np.ndarray, rounds: np.ndarray,
                          phases: np.ndarray) -> None:
        """Append a slice of arrays already checked by ``_validated_batch``."""
        if not len(tokens):
            return
        start, end = self._length, self._length + len(tokens)
        self._reserve(end)
        self._tokens[start:end] = tokens
        self._rounds[start:end] = rounds
        self._phases[start:end] = phases
        self._length = end

    def extend_tokens(self, tokens: np.ndarray, rounds: np.ndarray, phases: np.ndarray) -> None:
        """Append public token arrays, preserving old views and owning the new data."""
        self._extend_validated(*self._validated_batch(tokens, rounds, phases))

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
