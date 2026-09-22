"""Stage B opponent league (STAGE_B_TODO B7, DESIGN.md 8.4 item 4 and 7.3).

`League` is an `OpponentSource` (`train/opponents.py`). Each environment gets
one opponent for its whole match, drawn at `on_match_start` from a pool of:

- network policies, as `eval.policies.load_policy` specs: ``<path>`` (argmax;
  greedy Q for a DMC checkpoint, pruned argmax logit for a Stage B one),
  ``sample:<path>`` or ``sample=<T>:<path>`` (softmax over the pruned set);
- ``greedy``, the in-engine greedy bot;
- ``styled:<name>``, a fixed style of ``train.styles.FIXED_STYLES``;
- ``sampled-style``, a fresh style per match drawn from the TRAIN region of
  ``train.styles.StyleSpace`` (never the held-out region);
- learner snapshots registered with `add_snapshot`, which are network entries.

Styled entries write the style rows of the two non-learner seats with
`env.set_styles` and act with `styled_choice`; learner-seat rows stay at the
neutral style forever. Network entries are batched: all rows of environments
assigned to the same model go through one forward per model per step.

Sampling is prioritized by how often an entry beats the learner (an
exponential moving average of `on_match_end` results), with a floor and a
mixture with uniform, and is deterministic under `LeagueConfig.seed`. At most
`max_active_models` distinct network models are assigned at any time: when a
draw could exceed that, it is restricted to the already-active models plus the
non-network entries. A "model" is one spec string, so one checkpoint at two
temperatures counts twice. Loaded models live in an LRU cache.

Pool file (JSON)::

    {"config": {"seed": 1, "max_active_models": 4, ...},   # optional
     "entries": [
       {"spec": ".work/runpod/artifacts/pilot/final.pt", "name": "m1"},
       {"spec": "sample=1.0:.work/runpod/artifacts/pilot/final.pt", "weight": 0.5},
       {"spec": "greedy"},
       {"spec": "styled:bomb-happy"},
       {"spec": "sampled-style", "weight": 2.0}]}

`name` defaults to the spec; `weight` (default 1) multiplies the entry's
priority before the uniform mix. `config` keys are `LeagueConfig` fields.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence
import zlib

import numpy as np

from .opponents import OpponentRows

GREEDY = "greedy"
SAMPLED_STYLE = "sampled-style"
STYLED_PREFIX = "styled:"
KINDS = ("network", "greedy", "styled", "sampled_style")


@dataclass(frozen=True)
class LeagueConfig:
    seed: int = 0
    max_active_models: int = 4
    model_cache_size: int = 8
    # Prioritization: EMA of "entry beat the learner", floored, then mixed.
    ema_alpha: float = 0.05
    initial_beat_rate: float = 0.5
    weight_floor: float = 0.05
    priority_power: float = 1.0
    uniform_mix: float = 0.1
    # Learner snapshots: the PPO loop calls add_snapshot every `snapshot_every`
    # updates (see `should_snapshot`); the oldest is evicted past the cap.
    snapshot_every: int = 1000
    max_snapshots: int = 8
    # One pool entry per temperature per snapshot; None plays the argmax.
    snapshot_temperatures: tuple[float | None, ...] = (None,)
    snapshot_weight: float = 1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.max_active_models < 1:
            raise ValueError("max_active_models must be at least 1")
        if self.model_cache_size < self.max_active_models:
            raise ValueError("model_cache_size must hold every active model")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must lie in (0, 1]")
        if not 0.0 <= self.initial_beat_rate <= 1.0:
            raise ValueError("initial_beat_rate must lie in [0, 1]")
        if not 0.0 < self.weight_floor <= 1.0:
            raise ValueError("weight_floor must lie in (0, 1]")
        if not (math.isfinite(self.priority_power) and self.priority_power >= 0):
            raise ValueError("priority_power must be nonnegative and finite")
        if not 0.0 <= self.uniform_mix <= 1.0:
            raise ValueError("uniform_mix must lie in [0, 1]")
        if self.snapshot_every < 1 or self.max_snapshots < 1:
            raise ValueError("snapshot_every and max_snapshots must be positive")
        if not self.snapshot_temperatures:
            raise ValueError("snapshot_temperatures needs at least one entry")
        for t in self.snapshot_temperatures:
            if t is not None and not (math.isfinite(t) and t > 0):
                raise ValueError("snapshot temperatures must be positive or None")
        if not (math.isfinite(self.snapshot_weight) and self.snapshot_weight > 0):
            raise ValueError("snapshot_weight must be positive")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LeagueConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown LeagueConfig keys {sorted(unknown)}")
        data = dict(data)
        if "snapshot_temperatures" in data:
            data["snapshot_temperatures"] = tuple(data["snapshot_temperatures"])
        return cls(**data)


class NetworkModel(Protocol):
    """A loaded network opponent that decides a ragged batch in one call."""
    heuristic_tribute: bool

    def choose(self, obs: np.ndarray, cand: np.ndarray, offsets: np.ndarray,
               phase: np.ndarray) -> np.ndarray:
        """Local candidate index per row, int32."""
        ...


Loader = Callable[[str, "LeagueConfig", int], NetworkModel]


@dataclass
class Entry:
    name: str
    spec: str
    kind: str
    weight: float = 1.0
    snapshot: bool = False
    style: np.ndarray | None = None   # fixed styles only
    beat_ema: float = 0.5
    games: int = 0
    learner_wins: int = 0


def entry_kind(spec: str) -> str:
    if spec == GREEDY:
        return "greedy"
    if spec == SAMPLED_STYLE:
        return "sampled_style"
    if spec.startswith(STYLED_PREFIX):
        return "styled"
    if spec == "random":
        raise ValueError("the random policy is not a league opponent")
    return "network"


def load_pool(path: str | Path) -> tuple[LeagueConfig, list[dict[str, Any]]]:
    """Read a pool file; returns the config and the raw entry dicts."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not isinstance(data.get("entries"), list):
        raise ValueError("pool file needs an 'entries' list")
    for item in data["entries"]:
        if not isinstance(item, dict) or not isinstance(item.get("spec"), str):
            raise ValueError("every pool entry needs a 'spec' string")
        extra = set(item) - {"spec", "name", "weight"}
        if extra:
            raise ValueError(f"unknown pool entry keys {sorted(extra)}")
    return LeagueConfig.from_dict(data.get("config", {})), data["entries"]


class TorchModel:
    """Batched wrapper around a `load_policy` checkpoint policy."""

    def __init__(self, spec: str, device: str, seed: int) -> None:
        import torch

        from eval.policies import PrunedPolicy, load_policy

        policy = load_policy(spec, device=device)
        self.name = policy.name
        self.heuristic_tribute = bool(policy.heuristic_tribute)
        self.device = torch.device(device)
        self.pruned = policy.stage_b if isinstance(policy, PrunedPolicy) else None
        self.sample = bool(getattr(policy, "sample", False))
        self.model = policy.model
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)

    def choose(self, obs: np.ndarray, cand: np.ndarray, offsets: np.ndarray,
               phase: np.ndarray) -> np.ndarray:
        import torch

        from .model import select_actions

        obs_t = torch.as_tensor(obs, device=self.device)
        cand_t = torch.as_tensor(cand, device=self.device)
        off_t = torch.as_tensor(offsets.astype(np.int64), device=self.device)
        phase_t = torch.as_tensor(phase.astype(np.int64), device=self.device)
        with torch.inference_mode():
            if self.pruned is not None:
                step = self.pruned.act(obs_t, cand_t, off_t, phase_t,
                                       generator=self.generator, greedy=not self.sample)
                choice = step.choice
            else:
                scores = self.model.score_candidates(obs_t, cand_t, off_t, phase_t)
                choice = select_actions(scores, off_t)
        return choice.to("cpu").numpy().astype(np.int32)


def default_loader(spec: str, config: LeagueConfig, seed: int) -> NetworkModel:
    return TorchModel(spec, config.device, seed)


def _spec_seed(seed: int, spec: str) -> int:
    """Per-model generator seed, independent of load order."""
    sequence = np.random.SeedSequence([int(seed), zlib.crc32(spec.encode())])
    return int(sequence.generate_state(1, np.uint64)[0] >> np.uint64(1))


def _gather(rows: OpponentRows, index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """obs, cand and offsets of a row subset, as one contiguous ragged batch."""
    starts = rows.offsets[index].astype(np.int64)
    sizes = rows.offsets[index + 1].astype(np.int64) - starts
    offsets = np.zeros(len(index) + 1, np.int64)
    np.cumsum(sizes, out=offsets[1:])
    flat = np.repeat(starts - offsets[:-1], sizes) + np.arange(offsets[-1])
    return (np.ascontiguousarray(rows.obs[index]), np.ascontiguousarray(rows.cand[flat]),
            offsets)


class League:
    """Opponent pool and prioritized sampler; an `OpponentSource`."""

    def __init__(self, specs: Sequence[str | dict[str, Any]],
                 config: LeagueConfig = LeagueConfig(),
                 loader: Loader | None = None) -> None:
        import gd

        from .styles import StyleSpace, neutral

        self.config = config
        self.loader = loader or default_loader
        self.rng = np.random.default_rng(config.seed)
        self.play_phase = int(gd.Phase.Play)
        self.space = StyleSpace.default()
        self.neutral = neutral(self.space.dim)
        self.entries: list[Entry] = []
        for item in specs:
            item = {"spec": item} if isinstance(item, str) else dict(item)
            self._add_entry(item["spec"], item.get("name"), float(item.get("weight", 1.0)))
        if not self.entries:
            raise ValueError("the league needs at least one entry")
        self.snapshots: deque[tuple[str, list[Entry]]] = deque()
        self.cache: OrderedDict[str, NetworkModel] = OrderedDict()
        self.active: dict[str, int] = {}     # network spec -> environments using it
        self.env = None
        self.learner_team = np.zeros(0, np.int64)
        self.assigned: list[Entry | None] = []
        self.unresolved: list[deque[Entry]] = []
        self.styles = np.zeros((0, 4, self.space.dim), np.float32)
        self.forward_calls = 0

    @classmethod
    def from_file(cls, path: str | Path, loader: Loader | None = None,
                  **overrides: Any) -> "League":
        config, entries = load_pool(path)
        if overrides:
            config = LeagueConfig.from_dict({**asdict(config), **overrides})
        return cls(entries, config, loader)

    # --- pool -------------------------------------------------------------

    def _add_entry(self, spec: str, name: str | None, weight: float,
                   snapshot: bool = False) -> Entry:
        from .styles import fixed_style

        if not (math.isfinite(weight) and weight > 0):
            raise ValueError(f"entry weight must be positive: {spec}")
        kind = entry_kind(spec)
        name = name or spec
        if any(e.name == name for e in self.entries):
            raise ValueError(f"duplicate league entry name {name!r}")
        style = fixed_style(spec[len(STYLED_PREFIX):]) if kind == "styled" else None
        entry = Entry(name=name, spec=spec, kind=kind, weight=weight, snapshot=snapshot,
                      style=style, beat_ema=self.config.initial_beat_rate)
        self.entries.append(entry)
        return entry

    def should_snapshot(self, update: int) -> bool:
        return update > 0 and update % self.config.snapshot_every == 0

    def add_snapshot(self, path: str | Path, name: str | None = None) -> list[Entry]:
        """Register a learner checkpoint, one entry per snapshot temperature.

        `path` must not be overwritten later (use a per-update snapshot file,
        not `latest.pt`): models are loaded lazily on first use. Past
        `max_snapshots`, the oldest snapshot leaves the pool; environments
        already playing it finish their match first. Fixed entries never leave.
        """
        path = str(path)
        base = name or f"snapshot:{Path(path).name}"
        added = []
        for t in self.config.snapshot_temperatures:
            spec = path if t is None else f"sample={t:g}:{path}"
            label = base if t is None else f"{base}/T={t:g}"
            added.append(self._add_entry(spec, label, self.config.snapshot_weight, True))
        self.snapshots.append((path, added))
        while len(self.snapshots) > self.config.max_snapshots:
            _, old = self.snapshots.popleft()
            self.entries = [e for e in self.entries if all(e is not o for o in old)]
        return added

    # --- sampling ---------------------------------------------------------

    def weights(self, allowed: np.ndarray | None = None) -> np.ndarray:
        """Sampling probability per pool entry, optionally over a subset."""
        cfg = self.config
        beat = np.array([e.beat_ema for e in self.entries], np.float64)
        base = np.array([e.weight for e in self.entries], np.float64)
        mask = np.ones(len(self.entries), bool) if allowed is None else allowed
        priority = np.maximum(beat, cfg.weight_floor) ** cfg.priority_power * base * mask
        uniform = mask / mask.sum()
        return (1.0 - cfg.uniform_mix) * priority / priority.sum() + cfg.uniform_mix * uniform

    def _draw(self) -> Entry:
        allowed = None
        if len(self.active) >= self.config.max_active_models:
            allowed = np.array([e.kind != "network" or e.spec in self.active
                                for e in self.entries])
            if not allowed.any():
                raise RuntimeError("no pool entry fits the active-model cap; "
                                   "keep a non-network entry in the pool")
        index = int(self.rng.choice(len(self.entries), p=self.weights(allowed)))
        return self.entries[index]

    def _release(self, entry: Entry | None) -> None:
        if entry is None or entry.kind != "network":
            return
        self.active[entry.spec] -= 1
        if not self.active[entry.spec]:
            del self.active[entry.spec]

    def _model(self, spec: str) -> NetworkModel:
        if spec in self.cache:
            self.cache.move_to_end(spec)
            return self.cache[spec]
        model = self.loader(spec, self.config, _spec_seed(self.config.seed, spec))
        self.cache[spec] = model
        for old in list(self.cache):
            if len(self.cache) <= self.config.model_cache_size:
                break
            if old not in self.active:
                del self.cache[old]
        return model

    # --- OpponentSource ---------------------------------------------------

    def bind(self, env, learner_team: np.ndarray) -> None:
        team = np.asarray(learner_team, np.int64)
        if team.ndim != 1 or not np.isin(team, (0, 1)).all():
            raise ValueError("learner_team must be a vector of 0 and 1")
        self.env = env
        self.learner_team = team
        self.assigned = [None] * len(team)
        self.unresolved = [deque() for _ in team]
        self.active.clear()
        self.styles = np.ascontiguousarray(
            np.broadcast_to(self.neutral, (len(team), 4, self.space.dim)), np.float32)
        env.set_styles(self.styles)

    def on_match_start(self, env_ids: np.ndarray) -> None:
        from .styles import sample_style

        restyled = False
        for env_id in np.asarray(env_ids, np.int64).reshape(-1):
            e = int(env_id)
            self._release(self.assigned[e])
            entry = self._draw()
            self.assigned[e] = entry
            if len(self.unresolved[e]) >= 2:
                raise RuntimeError(f"environment {e}: matches start without ending")
            self.unresolved[e].append(entry)
            if entry.kind == "network":
                self.active[entry.spec] = self.active.get(entry.spec, 0) + 1
            if entry.kind == "styled":
                style = entry.style
            elif entry.kind == "sampled_style":
                style = sample_style(self.rng, self.space, "train")
            else:
                style = self.neutral
            opponent = 1 - int(self.learner_team[e])
            for seat in (opponent, opponent + 2):
                if not np.array_equal(self.styles[e, seat], style):
                    self.styles[e, seat] = style
                    restyled = True
        if restyled:
            # Refreshes styled_choice of the batch already pending (env.h).
            self.env.set_styles(self.styles)

    def act(self, rows: OpponentRows) -> np.ndarray:
        env_id = np.asarray(rows.env_id, np.int64)
        seat = np.asarray(rows.seat, np.int64)
        if len(env_id) and (seat % 2 == self.learner_team[env_id]).any():
            raise ValueError("act() received a learner-seat row")
        choice = np.array(rows.greedy_choice, np.int32, copy=True)
        styled = np.asarray(rows.styled_choice, np.int32)
        phase = np.asarray(rows.phase)
        groups: dict[str, list[int]] = {}
        for r, e in enumerate(env_id):
            entry = self.assigned[int(e)]
            if entry is None:
                raise RuntimeError(f"environment {int(e)} has no opponent; call on_match_start")
            if entry.kind in ("styled", "sampled_style"):
                choice[r] = styled[r]
            elif entry.kind == "network":
                groups.setdefault(entry.spec, []).append(r)
        for spec, members in groups.items():
            model = self._model(spec)
            index = np.asarray(members, np.int64)
            if model.heuristic_tribute:
                index = index[phase[index] == self.play_phase]
            if not len(index):
                continue
            obs, cand, offsets = _gather(rows, index)
            local = np.asarray(model.choose(obs, cand, offsets, phase[index]), np.int32)
            sizes = offsets[1:] - offsets[:-1]
            if local.shape != index.shape or (local < 0).any() or (local >= sizes).any():
                raise ValueError(f"{spec}: invalid candidate indices")
            self.forward_calls += 1
            choice[index] = local
        return choice

    def on_match_end(self, env_ids: np.ndarray, learner_won: np.ndarray) -> None:
        alpha = self.config.ema_alpha
        for env_id, won in zip(np.asarray(env_ids).reshape(-1),
                               np.asarray(learner_won).reshape(-1)):
            queue = self.unresolved[int(env_id)]
            if not queue:
                raise RuntimeError(f"environment {int(env_id)}: match ended before it started")
            # FIFO, so the result reaches the finished match's opponent whether
            # the loop reports the end before or after the next match's start.
            entry = queue.popleft()
            entry.games += 1
            entry.learner_wins += int(bool(won))
            entry.beat_ema += alpha * ((0.0 if won else 1.0) - entry.beat_ema)

    # --- logging ----------------------------------------------------------

    def stats(self) -> dict[str, float]:
        """Flat scalars for tensorboard: pool-level and per entry."""
        out: dict[str, float] = {"league/pool_size": float(len(self.entries)),
                                 "league/active_models": float(len(self.active)),
                                 "league/cached_models": float(len(self.cache)),
                                 "league/snapshots": float(len(self.snapshots))}
        for entry, weight in zip(self.entries, self.weights()):
            key = f"league/{entry.name}"
            out[f"{key}/games"] = float(entry.games)
            out[f"{key}/learner_win_rate"] = (entry.learner_wins / entry.games
                                              if entry.games else float("nan"))
            out[f"{key}/beat_ema"] = float(entry.beat_ema)
            out[f"{key}/weight"] = float(weight)
        return out
