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
from typing import Any, Callable, Protocol

import numpy as np


@dataclass(frozen=True)
class OpponentRows:
    """The subset of a `DecisionBatch` that the opponent team must decide.

    Arrays are numpy views or copies; `cand` rows for row `i` are
    `cand[offsets[i]:offsets[i + 1]]`, and `offsets[0] == 0`.

    In PPO rollouts `obs` and `cand` are `train.ppo.RowGather` views that
    gather from the engine's batch on use. Index them with 1-D integer
    arrays or take `np.asarray` of them, inside `act`: they are valid only
    until the environment advances, so an opponent that keeps features
    across steps must copy what it keeps.
    """
    obs: np.ndarray            # [rows, OBS_DIM] float32 (or a RowGather)
    cand: np.ndarray           # [offsets[-1], ACT_DIM] float32 (or a RowGather)
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


class CheckpointOpponent:
    """Any `eval.policies.load_policy` checkpoint on every opponent seat,
    playing exactly as it is evaluated: argmax Q for a Stage A checkpoint, the
    argmax logit over its own pruned set for a Stage B one. Tribute and
    back-tribute rows take the engine heuristic when the checkpoint's tribute
    policy is heuristic. `frozen:<path>` uses it for Stage B targets, which
    `FrozenModelOpponent` cannot play (B9, the exploiter against the league
    checkpoint). Nothing here is trained.
    """

    def __init__(self, path, device: str = "cpu") -> None:
        from pathlib import Path

        import gd

        from train.league import TorchModel

        # Argmax play never draws from the model's generator; the seed is moot.
        self.model = TorchModel(str(path), device, seed=0)
        self.name = f"frozen:{Path(path).name}"
        self.play_phase = int(gd.Phase.Play)

    def bind(self, env, learner_team: np.ndarray) -> None:
        del env, learner_team

    def on_match_start(self, env_ids: np.ndarray) -> None:
        del env_ids

    def on_match_end(self, env_ids: np.ndarray, learner_won: np.ndarray) -> None:
        del env_ids, learner_won

    def act(self, rows: OpponentRows) -> np.ndarray:
        return self.act_split(rows)[0]

    def act_split(self, rows: OpponentRows, defer: Callable[[Any], bool] | None = None
                  ) -> tuple[np.ndarray, list]:
        """`act`, deferring the model's rows to the caller when `defer(model)`
        holds (see `League.act_split`)."""
        from train.league import _gather

        choices = np.array(rows.greedy_choice, dtype=np.int32, copy=True)
        phase = np.asarray(rows.phase)
        index = np.arange(rows.rows)
        if self.model.heuristic_tribute:
            index = index[phase == self.play_phase]
        if not index.size:
            return choices, []
        if defer is not None and defer(self.model):
            return choices, [(self.model, index)]
        obs, cand, offsets = _gather(rows, index)
        local = np.asarray(self.model.choose(obs, cand, offsets, phase[index]), np.int32)
        sizes = offsets[1:] - offsets[:-1]
        if local.shape != index.shape or (local < 0).any() or (local >= sizes).any():
            raise ValueError(f"{self.name}: invalid candidate indices")
        choices[index] = local
        return choices, []


class FollowOpponent:
    """The newest checkpoint in a directory on every opponent seat (B11: an
    exploiter's opponent is the league learner's latest snapshot, written as
    `league/update-<n>.pt` by `PPOTrainer.league_snapshot`).

    An environment keeps the model it was given at match start for the whole
    match; a newer file is picked up at the next match start. The directory is
    rescanned at most every `rescan_seconds`. Until it holds a match for
    `pattern`, `fallback` plays (the learner's own starting checkpoint).
    Models play as `CheckpointOpponent` plays them; one is kept loaded per
    path in use.

    With `pinned`, the model is fixed until `retarget()` (an exploiter pins
    one snapshot per cycle, so it exploits a fixed target); the first target
    is the newest file at the first match start.
    """

    def __init__(self, directory, fallback, device: str = "cpu",
                 pattern: str = "update-*.pt", rescan_seconds: float = 30.0,
                 clock=None, pinned: bool = False) -> None:
        import time
        from pathlib import Path

        import gd

        self.directory = Path(directory)
        self.fallback = str(fallback)
        self.device = device
        self.pattern = pattern
        self.rescan_seconds = rescan_seconds
        self.clock = clock or time.monotonic
        self.play_phase = int(gd.Phase.Play)
        self.name = f"follow:{self.directory}"
        self.newest = self.fallback
        self.scanned_at = None
        self.pinned = pinned
        self.target: str | None = None
        self.assigned: list[str | None] = []
        self.models: dict = {}

    def current(self) -> str:
        """The model new matches get: the pinned target, or else the newest
        matching file (or the fallback), with rate-limited rescans."""
        if self.pinned and self.target is not None:
            return self.target
        path = self._scan()
        if self.pinned:
            self.target = path
        return path

    def retarget(self) -> str:
        """Rescan now; a pinned opponent moves to the newest file."""
        self.scanned_at = None
        self.target = None
        return self.current()

    def _scan(self) -> str:
        now = self.clock()
        if self.scanned_at is None or now - self.scanned_at >= self.rescan_seconds:
            self.scanned_at = now
            files = sorted(self.directory.glob(self.pattern)) if self.directory.is_dir() else []
            if files:
                self.newest = str(files[-1])
        return self.newest

    def _model(self, path: str):
        from train.league import TorchModel

        if path not in self.models:
            self.models[path] = TorchModel(path, self.device, seed=0)
        return self.models[path]

    def bind(self, env, learner_team: np.ndarray) -> None:
        del env
        self.assigned = [None] * len(learner_team)

    def on_match_start(self, env_ids: np.ndarray) -> None:
        path = self.current()
        for e in np.asarray(env_ids, np.int64).reshape(-1):
            self.assigned[int(e)] = path
        live = set(self.assigned) | {path}
        for old in [p for p in self.models if p not in live]:
            del self.models[old]

    def on_match_end(self, env_ids: np.ndarray, learner_won: np.ndarray) -> None:
        del env_ids, learner_won

    def act(self, rows: OpponentRows) -> np.ndarray:
        return self.act_split(rows)[0]

    def act_split(self, rows: OpponentRows, defer: Callable[[Any], bool] | None = None
                  ) -> tuple[np.ndarray, list]:
        """`act`, deferring the rows of each model with `defer(model)` to the
        caller (see `League.act_split`)."""
        from train.league import _gather

        choices = np.array(rows.greedy_choice, dtype=np.int32, copy=True)
        phase = np.asarray(rows.phase)
        groups: dict[str, list[int]] = {}
        for r, e in enumerate(np.asarray(rows.env_id, np.int64)):
            path = self.assigned[int(e)]
            if path is None:
                raise RuntimeError(f"environment {int(e)} has no opponent; call on_match_start")
            groups.setdefault(path, []).append(r)
        deferred = []
        for path, members in groups.items():
            model = self._model(path)
            index = np.asarray(members, np.int64)
            if model.heuristic_tribute:
                index = index[phase[index] == self.play_phase]
            if not index.size:
                continue
            if defer is not None and defer(model):
                deferred.append((model, index))
                continue
            obs, cand, offsets = _gather(rows, index)
            local = np.asarray(model.choose(obs, cand, offsets, phase[index]), np.int32)
            sizes = offsets[1:] - offsets[:-1]
            if local.shape != index.shape or (local < 0).any() or (local >= sizes).any():
                raise ValueError(f"{path}: invalid candidate indices")
            choices[index] = local
        return choices, deferred


def config_opponent(spec: str, reference, device: str = "cpu",
                    chunk_size: int = 32768, fallback: str | None = None,
                    pinned: bool = False) -> OpponentSource:
    """The opponent named by a non-league `PPOConfig.opponent`: "greedy",
    "frozen" (the frozen reference network itself), "frozen:<path>" with an
    already resolved path, or "follow:<directory>" (`FollowOpponent`, which
    needs `fallback`). A Stage A checkpoint plays through
    `FrozenModelOpponent`, anything else through `CheckpointOpponent`."""
    if spec == "greedy":
        return GreedyOpponent()
    if spec == "frozen":
        return FrozenModelOpponent(reference, device, chunk_size)
    if spec.startswith("follow:"):
        if not fallback:
            raise ValueError("a follow: opponent needs a fallback checkpoint")
        return FollowOpponent(spec[len("follow:"):], fallback, device, pinned=pinned)
    if not spec.startswith("frozen:"):
        raise ValueError(f"not a frozen, follow or greedy opponent: {spec}")
    from train.ckpt import load_checkpoint

    path = spec[len("frozen:"):]
    if load_checkpoint(path, "cpu").get("stage", "dmc") == "dmc":
        return FrozenModelOpponent.from_checkpoint(path, device, chunk_size)
    return CheckpointOpponent(path, device)
