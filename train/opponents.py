"""Opponent interface between the Stage B learner loop and the opponent league.

The PPO loop (`train/ppo.py`, task B5) owns a `gd.VecEnv` in which the learner
controls both seats of one team per environment. Every pending row that belongs
to the other team is handed to an `OpponentSource`, which picks one local
candidate index per row. The league (`train/league.py`, task B7) is one
implementation; `GreedyOpponent` below is the simplest.

Contract, in the order the loop calls it:

1. `bind(env, learner_team)` once, before the first `pending()`.
   `learner_team[e]` is 0 or 1, the team the learner plays in environment `e`
   (team `t` owns seats `t` and `t + 2`). The source may call
   `env.set_styles` here and later; it must only change the style rows of
   non-learner seats.
2. `on_match_start(env_ids)` whenever an environment begins a new match,
   detected by the loop from a changed `match_id` (and for every environment
   after `bind`). The source fixes one opponent for that environment for the
   whole match. It is called before the first opponent row of that match.
3. `act(rows)` with every pending opponent row of the current step. Returns
   int32 local candidate indices, one per row, in row order.
4. `on_match_end(env_ids, learner_won)` when `RoundResult.match_winner >= 0`.

Only learner rows produce training samples; nothing an `OpponentSource` does
is trained on. Opponents see policy inputs only: `OpponentRows` carries no
hidden counts (CLAUDE.md, privileged information never reaches a policy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class OpponentRows:
    """The subset of a `DecisionBatch` that the opponent team must decide.

    Arrays are numpy views or copies; `cand` rows for row `i` are
    `cand[offsets[i]:offsets[i + 1]]`, and `offsets[0] == 0`.
    """
    obs: np.ndarray            # [rows, OBS_DIM] float32
    cand: np.ndarray           # [offsets[-1], ACT_DIM] float32
    offsets: np.ndarray        # [rows + 1] int32
    env_id: np.ndarray         # [rows] int32
    seat: np.ndarray           # [rows] int32
    phase: np.ndarray          # [rows] int32
    match_id: np.ndarray       # [rows] int64
    greedy_choice: np.ndarray  # [rows] int32, local index
    styled_choice: np.ndarray  # [rows] int32, local index under the env's styles

    @property
    def rows(self) -> int:
        return int(len(self.env_id))


class OpponentSource(Protocol):
    def bind(self, env, learner_team: np.ndarray) -> None: ...

    def on_match_start(self, env_ids: np.ndarray) -> None: ...

    def act(self, rows: OpponentRows) -> np.ndarray: ...

    def on_match_end(self, env_ids: np.ndarray, learner_won: np.ndarray) -> None: ...


class GreedyOpponent:
    """The in-engine greedy bot on every opponent seat."""

    def bind(self, env, learner_team: np.ndarray) -> None:
        del env, learner_team

    def on_match_start(self, env_ids: np.ndarray) -> None:
        del env_ids

    def act(self, rows: OpponentRows) -> np.ndarray:
        return np.asarray(rows.greedy_choice, np.int32)

    def on_match_end(self, env_ids: np.ndarray, learner_won: np.ndarray) -> None:
        del env_ids, learner_won
