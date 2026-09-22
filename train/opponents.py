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
   whole match. A match can end and the next one open inside a single
   `pending()`, so the loop must call `on_match_start` for every environment
   whose `match_id` changed in the current batch BEFORE it builds
   `OpponentRows` or reads `styled_choice`, and must read `styled_choice` from
   the batch after that call. `VecEnv.set_styles` recomputes the pending
   `styled_choice` in place (the batch's numpy view updates), so restyling in
   `on_match_start` reaches the first decision of the new match.
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
    # [rows] int32, local index under the env's styles; read after on_match_start
    styled_choice: np.ndarray

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


class FrozenModelOpponent:
    """A frozen Stage A network on every opponent seat (B5's default opponent).

    Play rows take the batched argmax of the network's play head, exactly as
    `eval.policies.ModelPolicy` plays a DMC checkpoint; tribute and
    back-tribute rows take the engine heuristic (`greedy_choice`), the
    deployed Stage A behaviour. Nothing here is trained. The network may be
    shared with the learner's frozen pruning reference.
    """

    def __init__(self, model, device: str = "cpu", chunk_size: int = 32768,
                 name: str = "frozen") -> None:
        import torch

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.model = model.to(device).eval().requires_grad_(False)
        self.device = torch.device(device)
        self.chunk_size = chunk_size
        self.name = name

    @classmethod
    def from_checkpoint(cls, path, device: str = "cpu",
                        chunk_size: int = 32768) -> "FrozenModelOpponent":
        from pathlib import Path

        from train.ckpt import load_checkpoint
        from train.model import GuandanModel, ModelConfig

        payload = load_checkpoint(path, device=device)
        if payload.get("stage", "dmc") != "dmc":
            raise ValueError("FrozenModelOpponent plays a Stage A (DMC) checkpoint")
        model = GuandanModel(ModelConfig(**payload["model_config"]))
        model.load_state_dict(payload["model"])
        return cls(model, device, chunk_size, name=f"frozen:{Path(path).name}")

    def bind(self, env, learner_team: np.ndarray) -> None:
        del env, learner_team

    def on_match_start(self, env_ids: np.ndarray) -> None:
        del env_ids

    def on_match_end(self, env_ids: np.ndarray, learner_won: np.ndarray) -> None:
        del env_ids, learner_won

    def act(self, rows: OpponentRows) -> np.ndarray:
        import torch

        from train.model import select_actions

        choices = np.array(rows.greedy_choice, dtype=np.int32, copy=True)
        play = np.asarray(rows.phase) == 3
        if not play.any():
            return choices
        offsets = np.asarray(rows.offsets, np.int64)
        index = np.flatnonzero(play)
        counts = offsets[index + 1] - offsets[index]
        if index.size == rows.rows:
            cand, local_offsets = rows.cand, offsets
        else:
            total = int(counts.sum())
            starts = np.repeat(offsets[index], counts)
            cand = rows.cand[starts + np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)]
            local_offsets = np.concatenate(([0], np.cumsum(counts)))
        with torch.inference_mode():
            device_offsets = torch.as_tensor(local_offsets, device=self.device)
            scores = self.model.score_candidates(
                torch.as_tensor(np.asarray(rows.obs)[index], dtype=torch.float32, device=self.device),
                torch.tensor(np.asarray(cand), dtype=torch.float32, device=self.device),
                device_offsets, torch.full((index.size,), 3, device=self.device),
                chunk_size=self.chunk_size, phase_code=3)
            if not bool(torch.isfinite(scores).all()):
                raise FloatingPointError("non-finite opponent scores")
            choices[index] = select_actions(scores, device_offsets).cpu().numpy()
        return choices
