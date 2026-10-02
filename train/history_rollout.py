"""Public match-event store, sequence rollout buffer and collector (STAGE_C T2).

Three pieces sit between ``gd.VecEnv`` and the history learner:

* ``MatchEventStore``: one ``PublicStream`` per ``(env_id, match_id)``. Drained
  ``PublicActionEvent``s are routed by that key, and so are pending decisions,
  so a match whose first decision arrives before any of its events (round 0
  has no tribute) still gets its own empty stream. Streams outlive the
  collection chunk that produced them: a match continues across PPO updates,
  and its earlier tokens stay available while any stored row still cites
  them. ``prune`` drops a match only when it is no longer the environment's
  current match and no retained row points at it.
* ``SequenceRolloutBuffer``: every learner row with its environment, match,
  round, seat, policy version, history prefix, observation, hidden counts
  (critic only), the full canonical candidate set in engine order, chosen
  index, target log-probability (``logp``, the collecting actor at temperature
  1 without epsilon), behaviour log-probability (``behaviour_logp``, the
  distribution the choice was drawn from; equal to ``logp`` unless the
  exploration floor is on), value, reward, done and phase.
  Trajectories are keyed ``(env, team, round)`` exactly as ``train/ppo.py``'s
  buffer keys them: the team's per-round return lands on the team's last
  stored row of the round, ``done`` is set there, all other rewards are zero
  and GAE never bootstraps across a round end. Only completed trajectories
  train; rows of rounds still in progress carry over to the next update.
* ``HistoryCollector``: the vector rollout. Per step it calls ``pending()``,
  drains public events into the store and finished rounds into the buffer,
  lets ``HistoryActor.act`` choose for play-phase rows of the learner policy
  and keeps the engine's ``greedy_choice`` for tribute and back-tribute rows
  (the declared exchange-only heuristic of DESIGN 1.3), stores the rows and
  steps the environment.

The store never reads an event's ``forced`` flag: ``PublicStream.append``
consumes only seat, encoded action, cards left, round index and phase.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence
import time

import gd
import numpy as np
import torch

from train.history_model import (HIDDEN_DIM, PLAY_PHASE, DecisionInputs, HistoryActor,
                                 PublicStream, StreamBatch)
from train.history_snapshot_batch import LAYOUT_FIELDS
from train.history_transfers import download_tensors as _download_tensors, upload_arrays

LEARNER = 0     # policy identity of the collecting learner in a seat assignment
NUM_SEATS = 4


# ---- event store ---------------------------------------------------------------

class MatchEventStore:
    """Raw public token streams keyed by ``(env_id, match_id)``."""

    def __init__(self) -> None:
        self.streams: dict[tuple[int, int], PublicStream] = {}
        self.current: dict[int, int] = {}

    def stream(self, env_id: int, match_id: int) -> PublicStream:
        """The stream of that match, created empty when first seen."""
        key = (int(env_id), int(match_id))
        stream = self.streams.get(key)
        if stream is None:
            latest = self.current.get(key[0], -1)
            if key[1] < latest:
                raise ValueError(f"match id went backwards for environment {key[0]}: "
                                 f"{key[1]} after {latest}")
            stream = PublicStream(key[1])
            self.streams[key] = stream
            self.current[key[0]] = key[1]
        return stream

    def ingest(self, events: Sequence[Any]) -> int:
        """Append drained public events, in order, to their match streams."""
        for event in events:
            self.stream(event.env_id, event.match_id).append(event)
        return len(events)

    def prefix(self, env_id: int, match_id: int) -> int:
        return self.stream(env_id, match_id).prefix

    def prune(self, cited: set[tuple[int, int]]) -> int:
        """Drop streams of past matches no retained row cites. Returns the count."""
        stale = [key for key in self.streams
                 if key not in cited and self.current.get(key[0]) != key[1]]
        for key in stale:
            del self.streams[key]
        return len(stale)

    @property
    def tokens(self) -> int:
        return sum(stream.prefix for stream in self.streams.values())

    def __len__(self) -> int:
        return len(self.streams)


# ---- GAE --------------------------------------------------------------------------

def compute_gae(values: np.ndarray, rewards: np.ndarray, dones: np.ndarray,
                gamma: float = 1.0, lam: float = 0.95,
                mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation along the last axis, ``[..., T]``.

    Same semantics as the Stage B buffer: ``dones[t]`` is a terminal step and
    nothing is bootstrapped across it, padding (``mask`` false) contributes
    nothing, and the value past the last column is zero, so a trajectory that
    ends with its round is terminal. Returns float32 ``(advantages, returns)``
    with ``returns = advantages + values``.
    """
    values = np.asarray(values, np.float64)
    rewards = np.asarray(rewards, np.float64)
    dones = np.asarray(dones, bool)
    if not (values.shape == rewards.shape == dones.shape):
        raise ValueError("values, rewards and dones must share a shape")
    valid = np.ones(values.shape, bool) if mask is None else np.asarray(mask, bool)
    if valid.shape != values.shape:
        raise ValueError("mask must match values")
    batch_shape = values.shape[:-1]
    advantages = np.zeros(values.shape, np.float64)
    next_adv = np.zeros(batch_shape, np.float64)
    next_value = np.zeros(batch_shape, np.float64)
    for t in range(values.shape[-1] - 1, -1, -1):
        live = 1.0 - dones[..., t]
        delta = rewards[..., t] + gamma * next_value * live - values[..., t]
        adv = delta + gamma * lam * live * next_adv
        keep = valid[..., t]
        adv = np.where(keep, adv, 0.0)
        advantages[..., t] = adv
        next_adv = adv
        next_value = np.where(keep, values[..., t], 0.0)
    returns = np.where(valid, advantages + values, 0.0)
    return advantages.astype(np.float32), returns.astype(np.float32)


# ---- rollout buffer ----------------------------------------------------------------

ROW_FIELDS = ("env", "match", "round", "seat", "version", "prefix", "phase", "traj",
              "chosen", "logp", "cand_count", "behaviour_logp")


def ragged_index(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Flat source indices of the ragged slices ``starts[i]:starts[i]+counts[i]``."""
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, np.int64)
    first = np.repeat(starts - (np.cumsum(counts) - counts), counts)
    return first + np.arange(total, dtype=np.int64)


# Per-row learner tensors of a PPO minibatch besides the actor inputs.
TRAINING_FIELDS = ("logp", "behaviour_logp", "advantage", "returns", "hidden")


@dataclass
class TrainingBatch:
    """One PPO minibatch on the learner device (``SequenceRolloutBuffer.training_batch``)."""
    inputs: DecisionInputs
    chosen: torch.Tensor                 # flat index of each row's chosen candidate
    fields: dict[str, torch.Tensor]      # TRAINING_FIELDS and extras, one row each


@dataclass
class Trajectory:
    env: int
    team: int
    match: int
    round: int
    rows: list[int] = field(default_factory=list)
    complete: bool = False
    reward: float = 0.0


class SequenceRolloutBuffer:
    """Learner rows grouped into ``(env, team, round)`` trajectories.

    Rows are appended per vector step as small array chunks and concatenated
    once by ``finalize``; that is fine for the CPU trainer and keeps every row
    addressable by one global index. ``values`` are written by the learner
    (current critic, at ``learn`` time) right before ``finalize``.
    """

    def __init__(self, obs_dim: int = int(gd.OBS_DIM), act_dim: int = int(gd.ACT_DIM)) -> None:
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)
        self.chunks: list[dict[str, np.ndarray]] = []
        self.n_rows = 0
        self.trajectories: list[Trajectory] = []
        self.open: dict[tuple[int, int], int] = {}
        self.data: dict[str, np.ndarray] | None = None
        self._match_layout: dict[bool, tuple[list[tuple[int, ...]], np.ndarray]] = {}
        self.samples = np.zeros(0, np.int64)
        self.value = np.zeros(0, np.float32)
        self.reward = np.zeros(0, np.float32)
        self.done = np.zeros(0, bool)
        self.advantage = np.zeros(0, np.float32)
        self.returns = np.zeros(0, np.float32)

    # -- collection -----------------------------------------------------------

    def add_step(self, *, keep: np.ndarray, env_id: np.ndarray, match_id: np.ndarray,
                 round_index: np.ndarray, seat: np.ndarray, phase: np.ndarray,
                 obs: np.ndarray, hidden: np.ndarray, cand: np.ndarray, offsets: np.ndarray,
                 chosen: np.ndarray, logp: np.ndarray, prefix: np.ndarray,
                 version: int, behaviour_logp: np.ndarray | None = None) -> int:
        """Store the batch rows where ``keep`` is true. ``obs``/``cand`` are the
        batch's arrays (uint8 or the engine's binary float32), ``offsets`` the
        batch's ragged candidate offsets, ``chosen`` the local candidate index
        of every batch row and ``prefix`` the history length each row read.
        ``behaviour_logp`` defaults to ``logp`` (on-policy sampling)."""
        rows = np.flatnonzero(np.asarray(keep, bool))
        if rows.size == 0:
            return 0
        env = np.asarray(env_id, np.int64)[rows]
        match = np.asarray(match_id, np.int64)[rows]
        rnd = np.asarray(round_index, np.int64)[rows]
        seats = np.asarray(seat, np.int64)[rows]
        offsets = np.asarray(offsets, np.int64)
        counts = offsets[rows + 1] - offsets[rows]
        chosen = np.asarray(chosen, np.int64)[rows]
        if ((chosen < 0) | (chosen >= counts)).any():
            raise ValueError("chosen index outside the row's candidate set")
        traj = np.empty(rows.size, np.int64)
        for i in range(rows.size):
            key = (int(env[i]), int(seats[i]) % 2)
            t = self.open.get(key)
            if t is None:
                t = len(self.trajectories)
                self.trajectories.append(Trajectory(key[0], key[1], int(match[i]), int(rnd[i])))
                self.open[key] = t
            record = self.trajectories[t]
            if record.match != int(match[i]) or record.round != int(rnd[i]):
                raise ValueError("finish_round was not called before the next round's decisions")
            record.rows.append(self.n_rows + i)
            traj[i] = t
        src = ragged_index(offsets[rows], counts)
        chunk = {
            "env": env, "match": match, "round": rnd, "seat": seats,
            "version": np.full(rows.size, int(version), np.int64),
            "prefix": np.asarray(prefix, np.int64)[rows],
            "phase": np.asarray(phase, np.int64)[rows], "traj": traj, "chosen": chosen,
            "logp": np.asarray(logp, np.float32)[rows], "cand_count": counts,
            "behaviour_logp": np.asarray(logp if behaviour_logp is None else behaviour_logp,
                                         np.float32)[rows],
            "obs": np.asarray(obs)[rows].astype(np.uint8, copy=True),
            "hidden": np.asarray(hidden).reshape(len(offsets) - 1, HIDDEN_DIM)[rows]
            .astype(np.uint8, copy=True),
            "cand": np.asarray(cand)[src].astype(np.uint8, copy=True),
        }
        self.chunks.append(chunk)
        self.n_rows += rows.size
        self.data = None
        return rows.size

    def finish_round(self, env_id: int, match_id: int, round_index: int, seat_return) -> None:
        """Close both teams' trajectories of a finished round with the team's
        return (``seat_return[team]``; seats 0/2 are team 0, 1/3 team 1)."""
        for team in (0, 1):
            t = self.open.get((int(env_id), team))
            if t is None:
                continue        # this team had no learner decision in the round
            record = self.trajectories[t]
            if record.match != int(match_id) or record.round != int(round_index):
                raise ValueError("finished round does not match the open trajectory")
            record.reward = float(seat_return[team])
            record.complete = True
            del self.open[(int(env_id), team)]

    def finish_rounds(self, results: Sequence[Any]) -> None:
        for r in results:
            self.finish_round(int(r.env_id), int(r.match_id), int(r.round_index), r.seat_return)

    # -- storage -----------------------------------------------------------

    def compact(self) -> dict[str, np.ndarray]:
        """All rows as contiguous arrays (cached until the next append)."""
        if self.data is None:
            self._match_layout.clear()
            if self.chunks:
                data = {name: np.concatenate([c[name] for c in self.chunks])
                        for name in (*ROW_FIELDS, "obs", "hidden", "cand")}
            else:
                data = {name: np.zeros(0, np.int64) for name in ROW_FIELDS}
                data["logp"] = np.zeros(0, np.float32)
                data["behaviour_logp"] = np.zeros(0, np.float32)
                data["obs"] = np.zeros((0, self.obs_dim), np.uint8)
                data["hidden"] = np.zeros((0, HIDDEN_DIM), np.uint8)
                data["cand"] = np.zeros((0, self.act_dim), np.uint8)
            data["cand_start"] = np.cumsum(data["cand_count"]) - data["cand_count"]
            self.data = data
        return self.data

    def __len__(self) -> int:
        return self.n_rows

    def completed_rows(self) -> np.ndarray:
        """Rows of completed trajectories, trajectory-major, time order within."""
        return np.asarray([r for t in self.trajectories if t.complete for r in t.rows], np.int64)

    def cited_matches(self) -> set[tuple[int, int]]:
        data = self.compact()
        return set(zip(data["env"].tolist(), data["match"].tolist()))

    # -- learning -----------------------------------------------------------

    def finalize(self, values: np.ndarray, gamma: float = 1.0, lam: float = 0.95) -> int:
        """GAE over completed trajectories. ``values`` is one value per stored
        row (``len(self)``); only completed rows are read. Returns the number
        of trainable samples."""
        values = np.asarray(values, np.float32)
        if values.shape != (self.n_rows,):
            raise ValueError("one value per stored row is required")
        self.compact()
        self.value = values
        self.reward = np.zeros(self.n_rows, np.float32)
        self.done = np.zeros(self.n_rows, bool)
        self.advantage = np.zeros(self.n_rows, np.float32)
        self.returns = np.zeros(self.n_rows, np.float32)
        complete = [t for t in self.trajectories if t.complete]
        for t in complete:
            last = t.rows[-1]
            self.reward[last] = t.reward
            self.done[last] = True
        self.samples = self.completed_rows()
        if not complete:
            return 0
        longest = max(len(t.rows) for t in complete)
        shape = (len(complete), longest)
        v = np.zeros(shape, np.float32)
        r = np.zeros(shape, np.float32)
        d = np.zeros(shape, bool)
        m = np.zeros(shape, bool)
        for i, t in enumerate(complete):
            rows = np.asarray(t.rows, np.int64)
            n = rows.size
            v[i, :n] = values[rows]
            r[i, :n] = self.reward[rows]
            d[i, :n] = self.done[rows]
            m[i, :n] = True
        adv, ret = compute_gae(v, r, d, gamma, lam, mask=m)
        for i, t in enumerate(complete):
            rows = np.asarray(t.rows, np.int64)
            self.advantage[rows] = adv[i, :rows.size]
            self.returns[rows] = ret[i, :rows.size]
        return int(self.samples.size)

    def match_groups(self) -> dict[tuple[int, int], np.ndarray]:
        """Completed sample rows grouped by ``(env, match)``."""
        keys, index = self._group_rows(self.samples)
        order = np.argsort(index, kind="stable")
        ends = np.cumsum(np.bincount(index, minlength=len(keys)))
        return {key: self.samples[order[start:end]]
                for key, start, end in zip(keys, np.r_[0, ends[:-1]], ends)}

    def _group_rows(self, rows: np.ndarray, round_streams: bool = False
                    ) -> tuple[list[tuple[int, ...]], np.ndarray]:
        """Match keys in first-row order, and each row's index into those keys.

        A trajectory belongs to one match and round. Factor those few keys
        once per compact buffer, then gather integer IDs for later minibatches
        instead of rebuilding Python tuples for every decision and epoch.
        """
        data = self.compact()
        layout = self._match_layout.get(round_streams)
        if layout is None:
            lookup: dict[tuple[int, ...], int] = {}
            trajectory_ids = np.empty(len(self.trajectories), np.int64)
            for i, trajectory in enumerate(self.trajectories):
                key = (trajectory.env, trajectory.match)
                if round_streams:
                    key += (trajectory.round,)
                trajectory_ids[i] = lookup.setdefault(key, len(lookup))
            layout = (list(lookup), trajectory_ids[data["traj"]])
            self._match_layout[round_streams] = layout
        keys, row_ids = layout
        ids, first, inverse = np.unique(row_ids[rows], return_index=True, return_inverse=True)
        order = np.argsort(first)
        remap = np.empty_like(order)
        remap[order] = np.arange(len(order))
        return [keys[i] for i in ids[order]], remap[inverse]

    def minibatches(self, matches_per_batch: int, rng: np.random.Generator
                    ) -> Iterator[np.ndarray]:
        """Row index arrays, each covering ``matches_per_batch`` whole matches."""
        groups = self.match_groups()
        keys = list(groups)
        order = rng.permutation(len(keys))
        for begin in range(0, len(keys), max(1, int(matches_per_batch))):
            picked = [keys[i] for i in order[begin:begin + max(1, int(matches_per_batch))]]
            yield np.concatenate([groups[key] for key in picked])

    def next_iteration(self) -> set[tuple[int, int]]:
        """Discard completed trajectories, carry rows of rounds still in progress
        and return the ``(env, match)`` keys those rows still cite."""
        data = self.compact()
        keep_traj = [i for i, t in enumerate(self.trajectories) if not t.complete]
        remap = {old: new for new, old in enumerate(keep_traj)}
        rows = np.asarray([r for i in keep_traj for r in self.trajectories[i].rows], np.int64)
        rows.sort()
        row_remap = {int(old): new for new, old in enumerate(rows.tolist())}
        kept = []
        for i in keep_traj:
            t = self.trajectories[i]
            kept.append(Trajectory(t.env, t.team, t.match, t.round,
                                   [row_remap[r] for r in t.rows], False, 0.0))
        self.trajectories = kept
        self.open = {key: remap[t] for key, t in self.open.items()}
        if rows.size:
            counts = data["cand_count"][rows]
            src = ragged_index(data["cand_start"][rows], counts)
            chunk = {name: data[name][rows] for name in (*ROW_FIELDS, "obs", "hidden")}
            chunk["traj"] = np.asarray([remap[int(t)] for t in data["traj"][rows]], np.int64)
            chunk["cand"] = data["cand"][src]
            self.chunks = [chunk]
        else:
            self.chunks = []
        self.n_rows = int(rows.size)
        self.data = None
        self.samples = np.zeros(0, np.int64)
        self.value = self.reward = self.advantage = self.returns = np.zeros(0, np.float32)
        self.done = np.zeros(0, bool)
        return self.cited_matches()

    # -- actor inputs -----------------------------------------------------------

    def decision_inputs(self, rows: np.ndarray, store: MatchEventStore, device,
                        streams: dict[tuple[int, int], PublicStream] | None = None
                        ) -> tuple[DecisionInputs, torch.Tensor]:
        """``DecisionInputs`` for stored rows against the store's current
        streams, plus the flat index of each row's chosen candidate. Each
        row's ``prefix`` is what it read at collection, however many tokens
        the match has gained since. ``streams`` overrides the store's streams
        per match (tests truncate them to check prefix invariance)."""
        batch = self.training_batch(rows, store, device, streams, fields=())
        return batch.inputs, batch.chosen

    def training_batch(self, rows: np.ndarray, store: MatchEventStore, device,
                       streams: dict[tuple[int, int], PublicStream] | None = None, *,
                       fields: Sequence[str] = TRAINING_FIELDS,
                       extra: dict[str, np.ndarray] | None = None,
                       length_groups: int = 0, width: int = 0,
                       round_streams: bool = False) -> "TrainingBatch":
        """``decision_inputs`` plus the rows' ``fields`` (stored row data, or
        ``advantage``/``returns`` from ``finalize``) and ``extra`` host arrays,
        all uploaded with one packed copy. ``length_groups > 0`` plans the
        learner's grouped encode (``DecisionInputs.match_groups``) for an actor
        of ``width``; streams are then only uploaded as far as a row reads."""
        if length_groups > 0 and width < 1:
            raise ValueError("length groups need the actor width for their cost model")
        data = self.compact()
        rows = np.asarray(rows, np.int64)
        unique, match_index = self._group_rows(rows, round_streams)
        if round_streams:
            # Planted-habit ROUND view: each row read its round's own stream.
            picked = [store.round_stream(*key) for key in unique]
        else:
            picked = [(streams or {}).get(key) or store.stream(*key) for key in unique]
        prefix = data["prefix"][rows]
        lengths = np.asarray([stream.prefix for stream in picked], np.int64)
        if np.any(prefix > lengths[match_index]):
            raise ValueError("a stored row cites more history than its match has")
        counts = data["cand_count"][rows]
        src = ragged_index(data["cand_start"][rows], counts)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        named = {name: (getattr(self, name) if name in ("advantage", "returns") else data[name])[rows]
                 for name in fields}
        named.update(extra or {})
        # Batched match attention's layout, planned here instead of on the
        # device: each row's rank among its match's rows, in row order.
        order = np.argsort(match_index, kind="stable")
        per_match = np.bincount(match_index, minlength=len(unique))
        starts = np.cumsum(per_match) - per_match
        match_rank = np.empty_like(match_index)
        match_rank[order] = np.arange(len(rows)) - starts[match_index[order]]
        arrays = [stream.arrays() for stream in picked]
        plan = []
        if length_groups > 0 and len(rows):
            from train.history_model import length_groups as plan_groups
            cited = np.zeros(len(unique), np.int64)
            np.maximum.at(cited, match_index, prefix)
            # Causal encoding: tokens past every row's prefix are never read.
            arrays = [tuple(a[:n] for a in arrs) for arrs, n in zip(arrays, cited.tolist())]
            local = np.zeros(len(unique), np.int64)
            for matches in plan_groups(cited, length_groups, width):
                local[matches] = np.arange(len(matches))
                members = np.flatnonzero(np.isin(match_index, matches))
                plan.append(((int(cited[matches].max()), int(per_match[matches].max())),
                             (matches.astype(np.int64), members, local[match_index[members]],
                              match_rank[members])))
        host = (*StreamBatch.host_arrays(arrays), match_index, prefix,
                data["obs"][rows], data["seat"][rows], data["cand"][src], offsets,
                offsets[:-1] + data["chosen"][rows], *named.values(), match_rank,
                *(array for _, group in plan for array in group))
        uploaded = upload_arrays(host, device)
        # Keep the existing floating-point fields at their original packed
        # offsets. Moving them by an odd number of int64 ranks changes their
        # CUDA vector alignment and can change reduction bits (e.g. advantages).
        fields_end = 11 + len(named)
        inputs = DecisionInputs(streams=StreamBatch(*uploaded[:4]), match_index=uploaded[4],
                                prefix=uploaded[5], obs=uploaded[6], seat=uploaded[7],
                                cand=uploaded[8], offsets=uploaded[9], match_rank=uploaded[fields_end],
                                match_slots=int(per_match.max(initial=0)))
        base = fields_end + 1
        if plan:
            from train.history_model import MatchGroup
            inputs.match_groups = [MatchGroup(*uploaded[base + 4 * g:base + 4 * g + 3], tokens,
                                              uploaded[base + 4 * g + 3], slots)
                                   for g, ((tokens, slots), _) in enumerate(plan)]
        return TrainingBatch(inputs, uploaded[10], dict(zip(named, uploaded[11:fields_end])))


# ---- collector -------------------------------------------------------------------

SeatPolicy = Callable[[int, int], Sequence[int]]


def all_learner(env_id: int, match_id: int) -> Sequence[int]:
    """Default seat assignment: current-policy copies in all four seats."""
    return (LEARNER,) * NUM_SEATS


# CollectStats fields written only in profile mode (timings and call shapes);
# everything else must be identical with profiling on and off.
PROFILE_STATS = ("phase_seconds", "group_phase_seconds", "policy_call_rows")


@dataclass
class CollectStats:
    steps: int = 0
    decisions: int = 0          # pending rows stepped (all seats, all phases)
    learner_rows: int = 0       # rows stored for PPO
    rounds: int = 0
    matches: int = 0
    team0_return: float = 0.0   # sum of seat_return[0] over finished rounds (zero-sum: team 1 is its negative)
    gain: float = 0.0           # sum of RoundResult.gain (levels won) over finished rounds
    prefix_sum: int = 0
    prefix_max: int = 0
    phase_seconds: dict[str, float] = field(default_factory=dict)
    # Profile mode only: the per-call phases split by identity group ("learner",
    # "snapshot") and a histogram {rows per policy call: calls} per group.
    group_phase_seconds: dict[str, dict[str, float]] = field(default_factory=dict)
    policy_call_rows: dict[str, dict[int, int]] = field(default_factory=dict)
    policy_batches: int = 0
    epsilon_picks: int = 0      # learner rows chosen by the exploration floor's uniform branch
    behaviour_entropy_sum: float = 0.0  # sum over learner rows of the entropy of pi_b

    @property
    def mean_prefix(self) -> float:
        return self.prefix_sum / self.learner_rows if self.learner_rows else 0.0


@dataclass
class _PolicyBatch:
    identity: int
    rows: np.ndarray
    keys: list[tuple[int, int]]
    streams: list[PublicStream]
    one_decision_per_stream: bool
    max_candidates: int


@dataclass
class _StepRows:
    """One vector step's pending rows as host arrays (collector-internal)."""
    env_id: np.ndarray
    match_id: np.ndarray
    round_index: np.ndarray
    seat: np.ndarray
    phase: np.ndarray
    offsets: np.ndarray
    counts: np.ndarray
    obs: np.ndarray
    cand: np.ndarray
    hidden: np.ndarray
    choices: np.ndarray           # engine greedy choice, replaced on play rows
    logp: np.ndarray
    behaviour_logp: np.ndarray
    prefix: np.ndarray            # public tokens each row may read
    identities: np.ndarray        # policy identity of each row's seat
    acting: np.ndarray | None = None   # learner play rows: stored for PPO


@dataclass
class _StepPlan:
    """Identity groups of one step and their inputs in upload order."""
    groups: list[_PolicyBatch] = field(default_factory=list)   # own actor call each
    merged: list[_PolicyBatch] = field(default_factory=list)   # one merged snapshot call
    host_fields: list[np.ndarray] = field(default_factory=list)
    layout: Any = None
    merged_actors: list[HistoryActor] = field(default_factory=list)
    slots: list[int] = field(default_factory=list)


class _PhaseTimer:
    """Profile mode: synchronized wall time per phase (and per identity group)."""

    def __init__(self, stats: CollectStats, device: torch.device, enabled: bool) -> None:
        self.stats, self.device, self.enabled = stats, device, enabled
        self.stamp = time.perf_counter() if enabled else 0.0

    def __call__(self, name: str, group: str | None = None) -> None:
        if not self.enabled:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        now = time.perf_counter()
        phases = self.stats.phase_seconds
        phases[name] = phases.get(name, 0.0) + now - self.stamp
        if group is not None:
            split = self.stats.group_phase_seconds.setdefault(group, {})
            split[name] = split.get(name, 0.0) + now - self.stamp
        self.stamp = now


class HistoryCollector:
    """Vector rollout of the history actor with heuristic tribute.

    ``env`` must have been built with ``log_public_actions=True`` and a
    ``log_env_limit`` covering every environment; ``reset()`` is called once,
    lazily, on the first step. ``seat_policy(env_id, match_id)`` records the
    policy identity of each seat when a match starts; only rows of the
    ``LEARNER`` identity are stored. ``resolve_policy`` supplies frozen history
    actors for the other identities; missing identities fail explicitly.
    ``temperature`` and ``epsilon`` shape the learner identity's sampling
    only (``HistoryActor.explore``); frozen snapshot seats always use
    ``HistoryActor.act`` at temperature 1.
    ``reuse_cache_lengths`` reuses uploaded prefixes for matching KV metadata;
    disable it to compare against the original separate lengths upload.
    ``private_graphs`` checks actor storage and structure once per synchronous
    ``collect`` call. Actor weights, structure and hooks must stay fixed during
    that call; normal PPO updates happen between calls. Direct ``step`` calls
    retain the full graph validation on every inference.
    ``batch_snapshot_policies`` (opt-in) evaluates all non-learner identities'
    play rows of a vector step in one merged actor call over stacked head
    weights (``train.history_snapshot_batch``). Each identity's public KV cache
    still encodes its own streams; the learner's call, rows and sampling are
    unchanged; snapshot seats bypass private graphs. Snapshot uniforms are
    drawn per identity in identity order after the learner, as before, so the
    generator advances identically; snapshot log-probabilities carry FP32
    reduction-order noise (acceptance tier 2). It may be switched between
    ``collect`` calls. ``batch_snapshot_encoder`` (opt-in, with the merged
    call and ``paged_cache``) also encodes those identities' public streams in
    one pass per step over stacked encoder weights
    (``train.history_paged_cache.merged_encode``), again tier 2 on snapshot
    seats only. ``prefill_learner_cache`` (opt-in, with the KV cache) rebuilds
    the learner's public cache for every current match at the start of
    ``collect`` in one batched pass per prefill chunk; without it each stream
    is rebuilt by the first step whose learner row reads it, in passes of its
    own. Same function of the same tokens in different batches: FP32
    reduction-order noise on learner seats (tier 2).
    """

    def __init__(self, env: Any, actor: HistoryActor, store: MatchEventStore,
                 buffer: SequenceRolloutBuffer, generator: torch.Generator | None = None,
                 device: str | torch.device = "cpu", seat_policy: SeatPolicy = all_learner,
                 record_choices: bool = False,
                 resolve_policy: Callable[[int], HistoryActor] | None = None,
                 assignment_log: Callable[[dict], None] | None = None,
                 kv_cache: bool = False, profile: bool = False,
                 temperature: float = 1.0, epsilon: float = 0.0,
                 reuse_cache_lengths: bool = True, private_graphs: bool = False,
                 private_graph_budget_mb: int = 512, private_graph_policy_budget_mb: int = 128,
                 triton_cache: bool = False,
                 triton_min_batch: int = 1,
                 batch_snapshot_policies: bool = False,
                 paged_cache: bool = False,
                 batch_snapshot_encoder: bool = False,
                 page_span: bool = False,
                 round_streams: bool = False,
                 row_style: Callable[[int, int], int] | None = None,
                 prefill_learner_cache: bool = False) -> None:
        self.env = env
        self.actor = actor
        self.store = store
        self.buffer = buffer
        self.generator = generator
        self.device = torch.device(device)
        self.seat_policy = seat_policy
        self.resolve_policy = resolve_policy
        self.assignment_log = assignment_log
        if kv_cache and actor.config.window:
            raise ValueError("rollout KV cache requires full history")
        if private_graphs and (not kv_cache or self.device.type != "cuda"):
            raise ValueError("private CUDA graphs require CUDA rollout and the full-history KV cache")
        if triton_cache and (not kv_cache or self.device.type != "cuda"):
            raise ValueError("Triton cache copies require CUDA rollout and the full-history KV cache")
        if paged_cache and (not kv_cache or triton_cache):
            raise ValueError("the paged KV cache requires the KV cache and replaces Triton copies")
        if batch_snapshot_encoder and not paged_cache:
            raise ValueError("merged snapshot encoding requires the paged KV cache")
        if page_span and (not paged_cache or private_graphs):
            raise ValueError("page-padded spans require the paged KV cache and no private graphs")
        if triton_min_batch < 1:
            raise ValueError("Triton minimum batch must be positive")
        if private_graph_budget_mb < 1 or private_graph_policy_budget_mb < 1:
            raise ValueError("private graph memory budget must be positive")
        # Planted-habit views of the learner (train/history_habit.py): ``round_streams``
        # reads a RoundEventStore's current-round stream; ``row_style`` gives the
        # opponents' z of a match for a style-input actor. Snapshot seats unchanged.
        if round_streams and not hasattr(store, "round_stream"):
            raise ValueError("round streams need a RoundEventStore")
        if actor.config.style_input and row_style is None:
            raise ValueError("a style-input learner needs the opponents' style per match")
        if actor.config.style_input and private_graphs:
            raise ValueError("private graphs do not capture the style input")
        if prefill_learner_cache and (not kv_cache or round_streams):
            raise ValueError("learner cache prefill requires the match-keyed KV cache")
        self.prefill_learner_cache = bool(prefill_learner_cache)
        self.round_streams = bool(round_streams)
        self.row_style = row_style
        self.kv_cache = kv_cache
        self.reuse_cache_lengths = reuse_cache_lengths
        self.private_graphs = private_graphs
        self.private_graph_budget_bytes = private_graph_budget_mb << 20
        self.private_graph_policy_budget_bytes = private_graph_policy_budget_mb << 20
        self.decision_graphs = {}
        self._graph_collecting = False
        self.triton_cache = triton_cache
        self.triton_min_batch = triton_min_batch
        self.paged_cache = paged_cache
        self.kv_pool = None             # shared by the snapshot identities' paged caches
        self.learner_pool = None        # the learner's own pages, returned before each learn
        self.batch_snapshot_encoder = bool(batch_snapshot_encoder)
        self.page_span = bool(page_span)
        self.batch_snapshot_policies = bool(batch_snapshot_policies)
        self.snapshot_heads = None      # stacked snapshot heads, created on first merged step
        if self.batch_snapshot_policies:
            from train.history_snapshot_batch import validate_actor
            validate_actor(actor)
        self.profile = profile
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.caches = {}
        self._frozen_stack: ExitStack | None = None
        self._frozen_caches: set = set()
        self.policy_decisions: dict[int, int] = {}
        self.assignments: dict[tuple[int, int], np.ndarray] = {}
        self._assignments_changed = False
        self.results: list[Any] = []            # RoundResults drained by the last collect()
        self.choice_log: list[np.ndarray] | None = [] if record_choices else None
        self.started = False
        self.version = 0

    def cache_metrics(self) -> dict:
        metrics = dict(bytes=sum(c.bytes for c in self.caches.values()),
                       entries=sum(len(c.entries) for c in self.caches.values()),
                       encoded_tokens=sum(c.encoded_tokens for c in self.caches.values()),
                       rebuilds=sum(c.rebuilds for c in self.caches.values()),
                       appends=sum(c.appends for c in self.caches.values()))
        pools = [pool for pool in (self.kv_pool, self.learner_pool) if pool is not None]
        if pools:
            metrics.update(pool_reserved_bytes=sum(pool.reserved_bytes for pool in pools),
                           pool_used_pages=sum(pool.used_pages for pool in pools),
                           pool_pages=sum(pool.pages for pool in pools),
                           pool_grows=sum(pool.grows for pool in pools))
        return metrics

    def invalidate_learner_cache(self) -> None:
        # Release before PPO allocates activations; rebuild with updated weights.
        if LEARNER in self.caches:
            self.caches[LEARNER].clear()
        if self.learner_pool is not None:
            self.learner_pool.reset()

    def _prune_caches(self) -> None:
        active = {}
        for key, seats in self.assignments.items():
            for identity in set(seats.tolist()):
                active.setdefault(identity, set()).add(key)
        for identity in list(self.caches):
            if identity == LEARNER and self.round_streams:
                continue    # round keys; cleared before every learn anyway
            if identity not in active:
                self.caches.pop(identity).clear()   # paged caches return their pages
            else:
                self.caches[identity].prune(active[identity])
        for identity in list(self.decision_graphs):
            if identity not in active:
                self.decision_graphs.pop(identity).clear()
        if self.snapshot_heads is not None:
            self.snapshot_heads.retain(set(active) - {LEARNER})

    @torch.no_grad()
    def _graph_log_probs(self, identity, actor, inputs, encoded):
        """Collector-owned graphs never become actor or checkpoint state."""
        from train.history_cuda_graphs import PrivateDecisionGraphs
        graph = self.decision_graphs.get(identity)
        if graph is not None and graph.actor is not actor:
            self.decision_graphs.pop(identity).clear()
            graph = None
        other_bytes = sum(g.bytes for key, g in self.decision_graphs.items() if key != identity)
        available = min(self.private_graph_policy_budget_bytes,
                        self.private_graph_budget_bytes - other_bytes)
        if available < 1:
            return actor.candidate_log_probs(inputs, encoded=encoded)
        if graph is None:
            graph = PrivateDecisionGraphs(actor, max_entries=32, max_bytes=available,
                                          admit_after=3, integer_alignment_agnostic=True)
            self.decision_graphs[identity] = graph
            if self._graph_collecting:
                graph.begin_interval()
        graph.set_memory_budget(available)
        return graph.log_probs(inputs, encoded)

    def graph_metrics(self) -> dict:
        return {str(identity): graph.metrics() for identity, graph in self.decision_graphs.items()}

    def release_snapshot_graphs(self) -> int:
        """Clear every non-learner identity's private graphs; returns how many.
        Their pool blocks become free allocator cache (empty_cache returns them)."""
        released = [identity for identity in self.decision_graphs if identity != LEARNER]
        for identity in released:
            self.decision_graphs.pop(identity).clear()
        return len(released)

    def snapshot_head_metrics(self) -> dict:
        return self.snapshot_heads.metrics() if self.snapshot_heads is not None else {}

    def _cache(self, identity: int, actor: HistoryActor):
        from train.history_inference import BatchedHistoryCache
        cache = self.caches.get(int(identity))
        if cache is None or cache.actor is not actor:
            if cache is not None:
                cache.clear()
            if self.paged_cache:
                from train.history_paged_cache import KVPagePool, PagedHistoryCache
                name = "learner_pool" if int(identity) == LEARNER else "kv_pool"
                if getattr(self, name) is None:
                    setattr(self, name, KVPagePool.for_actor(actor))
                cache = PagedHistoryCache(actor, getattr(self, name), page_span=self.page_span)
            elif self.triton_cache:
                from train.history_triton_cache import TritonHistoryCache
                cache = TritonHistoryCache(actor, min_batch=self.triton_min_batch)
            else:
                cache = BatchedHistoryCache(actor)
            self.caches[int(identity)] = cache
        if self._frozen_stack is not None and cache not in self._frozen_caches:
            self._frozen_stack.enter_context(cache.frozen_weights())
            self._frozen_caches.add(cache)
        return cache

    @contextmanager
    def frozen_weights(self):
        """Trainer-owned scope: weights stay fixed until collection finishes.

        Register caches lazily, including newly encountered snapshot identities.
        Retain exit checks for caches pruned or replaced during collection.
        Custom collectors need not enter this scope: their policy callbacks may
        still mutate weights between steps, with ordinary per-encode checks.
        """
        if self._frozen_stack is not None:
            raise RuntimeError("nested frozen-weight collection is unsupported")
        try:
            with ExitStack() as stack:
                self._frozen_stack = stack
                yield self
        finally:
            self._frozen_stack = None
            self._frozen_caches.clear()

    @torch.no_grad()
    def _prefill_learner(self, stats: CollectStats) -> None:
        """Encode every current match with a learner seat up to its drained
        events, all streams together. Events still pending in the engine are
        appended by the steps, as always."""
        mark = _PhaseTimer(stats, self.device, self.profile)
        keys = [key for key, seats in self.assignments.items()
                if (seats == LEARNER).any() and key in self.store.streams]
        if keys:
            self._cache(LEARNER, self.actor).prefill(keys, [self.store.streams[key]
                                                             for key in keys])
        mark("learner_prefill", "learner")

    def transfer_metrics(self) -> dict:
        return {str(identity): dict(cache.copy_stats) for identity, cache in self.caches.items()
                if hasattr(cache, "copy_stats")}

    def assignment(self, env_id: int, match_id: int) -> np.ndarray:
        key = (int(env_id), int(match_id))
        seats = self.assignments.get(key)
        if seats is None:
            seats = np.asarray(list(self.seat_policy(*key)), np.int64)
            if seats.shape != (NUM_SEATS,):
                raise ValueError("a seat assignment names four policies")
            if (seats < 0).any():
                raise ValueError("negative policy identity")
            if (seats != LEARNER).any() and self.resolve_policy is None:
                raise NotImplementedError("snapshot seats require a policy resolver")
            for identity in set(seats.tolist()) - {LEARNER}:
                if not isinstance(self.resolve_policy(identity), HistoryActor):
                    raise TypeError("training seats require a HistoryActor")
            seats.setflags(write=False)
            self.assignments[key] = seats
            self._assignments_changed = True
            if self.assignment_log is not None:
                self.assignment_log(dict(event="assignment", env=key[0], match=key[1],
                                         version=self.version, seats=seats.tolist()))
            stale = [k for k in self.assignments if k[0] == key[0] and k[1] < key[1]]
            for k in stale:
                del self.assignments[k]
        return seats

    @torch.no_grad()
    def _encode_streams(self, group: _PolicyBatch, actor: HistoryActor,
                        uploaded_prefix: torch.Tensor, prefix: np.ndarray
                        ) -> tuple[StreamBatch, torch.Tensor | None]:
        """A group's public streams: ``(metadata, encoded)`` through the KV cache,
        or the raw ``StreamBatch`` and ``None`` without it."""
        if not self.kv_cache:
            return StreamBatch.from_streams(group.streams, self.device), None
        lengths_hint = {}
        if self.reuse_cache_lengths and group.one_decision_per_stream:
            # With distinct keys, insertion order makes decision rows and
            # streams identical. Check the original host values before reusing
            # the prefix upload as stream metadata.
            host_lengths = tuple(prefix[group.rows].tolist())
            if host_lengths == tuple(stream.prefix for stream in group.streams):
                lengths_hint = dict(preuploaded_lengths=uploaded_prefix,
                                    host_lengths=host_lengths)
        return self._cache(group.identity, actor).encode(group.keys, group.streams,
                                                         **lengths_hint)

    @torch.no_grad()
    def _merged_snapshot_step(self, plan: _StepPlan, fields: dict[str, torch.Tensor],
                              prefix: np.ndarray, mark: _PhaseTimer):
        """Per-identity public encodes into one padded memory (or one merged
        encode), then one merged actor call and one Gumbel-max draw per row
        (uniforms per identity)."""
        from train.history_snapshot_batch import merged_log_probs, merged_sample
        layout, groups, actors = plan.layout, plan.merged, plan.merged_actors
        width = self.actor.config.width
        memory = None
        if self.batch_snapshot_encoder and self.kv_cache:
            from train.history_paged_cache import merged_encode
            parts = [(self._cache(group.identity, actor), group.keys, group.streams, slot)
                     for group, actor, slot in zip(groups, actors, plan.slots)]
            memory = merged_encode(parts, self.snapshot_heads, layout.length)
            mark("public_cache_or_collation", "snapshot")
            groups = ()     # memory is complete
        for group, actor, (begin, end), (first, last) in zip(
                groups, actors, layout.group_rows, layout.group_streams):
            stream_batch, encoded = self._encode_streams(group, actor, fields["prefix"][begin:end],
                                                         prefix)
            if encoded is None:
                encoded = actor.encode_batch(stream_batch)
            if encoded.dtype != torch.float32 or encoded.shape[0] != last - first:
                raise ValueError("merged snapshot inference needs FP32 memory, one row per stream")
            if memory is None:
                memory = encoded.new_zeros(layout.streams, layout.length, width)
            span = min(layout.length, encoded.shape[1])
            memory[first:last, :span].copy_(encoded[:, :span])
            del encoded
            mark("public_cache_or_collation", "snapshot")
        log_probs = merged_log_probs(self.snapshot_heads, layout, fields, memory)
        choice, chosen_logp = merged_sample(log_probs, layout, fields, self.generator)
        mark("actor_and_sampling", "snapshot")
        return choice, chosen_logp.float()

    # -- one vector step, in phases ---------------------------------------------

    def _drain(self, stats: CollectStats) -> None:
        """Contract order: public events, then ended rounds (before this batch's rows)."""
        self.store.ingest(self.env.drain_public_actions())
        results = self.env.drain_finished_rounds()
        self.buffer.finish_rounds(results)
        for r in results:
            self.results.append(r)
            stats.rounds += 1
            stats.matches += int(r.match_winner >= 0)
            stats.team0_return += float(r.seat_return[0])
            stats.gain += float(r.gain)

    def _read_rows(self, batch) -> _StepRows:
        """The pending rows as host arrays, with each row's policy identity and prefix."""
        n = int(batch.rows)
        if not n:
            raise RuntimeError("environment produced no pending decisions")
        offsets = np.asarray(batch.offsets, np.int64)
        rows = _StepRows(
            env_id=np.asarray(batch.env_id, np.int64), match_id=np.asarray(batch.match_id, np.int64),
            round_index=np.asarray(batch.round_index, np.int64),
            seat=np.asarray(batch.seat, np.int64), phase=np.asarray(batch.phase, np.int64),
            offsets=offsets, counts=offsets[1:] - offsets[:-1], obs=np.asarray(batch.obs),
            cand=np.asarray(batch.cand), hidden=np.asarray(batch.hidden_counts),
            choices=np.array(batch.greedy_choice, dtype=np.int32, copy=True),
            logp=np.zeros(n, np.float32), behaviour_logp=np.zeros(n, np.float32),
            prefix=np.zeros(n, np.int64), identities=np.zeros(n, np.int64))
        for i in range(n):
            seats = self.assignment(rows.env_id[i], rows.match_id[i])
            rows.identities[i] = seats[rows.seat[i]]
            if self.round_streams and rows.identities[i] == LEARNER:
                rows.prefix[i] = self.store.round_stream(
                    rows.env_id[i], rows.match_id[i], rows.round_index[i]).prefix
            else:
                rows.prefix[i] = self.store.stream(rows.env_id[i], rows.match_id[i]).prefix
        rows.acting = (rows.identities == LEARNER) & (rows.phase == PLAY_PHASE)
        if (self.kv_cache or self.snapshot_heads is not None) and self._assignments_changed:
            self._prune_caches()
            self._assignments_changed = False
        return rows

    def _plan(self, r: _StepRows) -> _StepPlan:
        """Play rows grouped by policy identity (learner first), their host inputs
        in upload order, and the merged snapshot layout if that path is on."""
        plan = _StepPlan()
        play = r.phase == PLAY_PHASE
        for identity in np.unique(r.identities[play]):
            rows = np.flatnonzero((r.identities == identity) & play)
            if self.round_streams and identity == LEARNER:
                from train.history_habit import ROUND_KEY_STRIDE
                triples = list(zip(r.env_id[rows].tolist(), r.match_id[rows].tolist(),
                                   r.round_index[rows].tolist()))
                keys = [(e, m * ROUND_KEY_STRIDE + k) for e, m, k in triples]
                unique = list(dict.fromkeys(keys))
                first = dict(zip(keys, triples))
                streams = [self.store.round_stream(*first[k]) for k in unique]
            else:
                keys = list(zip(r.env_id[rows].tolist(), r.match_id[rows].tolist()))
                unique = list(dict.fromkeys(keys))
                streams = [self.store.stream(*k) for k in unique]
            group = _PolicyBatch(int(identity), rows, unique, streams,
                                 len(keys) == len(unique), int(r.counts[rows].max()))
            if self.batch_snapshot_policies and identity != LEARNER:
                plan.merged.append(group)
                continue
            index = {key: i for i, key in enumerate(unique)}
            src = ragged_index(r.offsets[rows], r.counts[rows])
            local = np.concatenate(([0], np.cumsum(r.counts[rows])))
            plan.groups.append(group)
            plan.host_fields.extend((
                np.asarray([index[k] for k in keys], np.int64), r.prefix[rows],
                r.obs[rows].astype(np.uint8), r.seat[rows], r.cand[src].astype(np.uint8),
                local, np.repeat(np.arange(len(rows), dtype=np.int64), r.counts[rows])))
        if plan.merged:
            from train.history_snapshot_batch import SnapshotHeads, merged_layout
            if self.snapshot_heads is None:
                self.snapshot_heads = SnapshotHeads(encoder=self.batch_snapshot_encoder)
            plan.merged_actors = [self.resolve_policy(g.identity) for g in plan.merged]
            plan.slots = [self.snapshot_heads.slot(g.identity, actor)
                          for g, actor in zip(plan.merged, plan.merged_actors)]
            plan.layout = merged_layout(plan.merged, plan.slots, r.prefix, r.obs, r.seat, r.cand,
                                        r.offsets, r.counts, r.env_id, r.match_id)
            plan.host_fields.extend(plan.layout.arrays)
        return plan

    def _infer_group(self, group: _PolicyBatch, fields, r: _StepRows, stats: CollectStats,
                     mark: _PhaseTimer) -> tuple:
        """One identity's encode, actor call and sampling; the tensors to download."""
        identity = group.identity
        label = "learner" if identity == LEARNER else "snapshot"
        actor = self.actor if identity == LEARNER else self.resolve_policy(identity)
        stream_batch, encoded = self._encode_streams(group, actor, fields[1], r.prefix)
        mark("public_cache_or_collation", label)
        inputs = DecisionInputs(
            streams=stream_batch,
            match_index=fields[0], prefix=fields[1], obs=fields[2], seat=fields[3],
            cand=fields[4], offsets=fields[5], candidate_rows=fields[6],
            one_decision_per_stream=group.one_decision_per_stream)
        if actor.config.style_input:
            inputs.style = torch.as_tensor(
                np.asarray([self.row_style(e, m) for e, m in
                            zip(r.env_id[group.rows].tolist(), r.match_id[group.rows].tolist())],
                           np.int64), device=self.device)
        inference = {}
        if self.private_graphs:
            inference["inference_log_probs"] = self._graph_log_probs(identity, actor, inputs, encoded)
        if identity == LEARNER:
            sample = actor.explore(inputs, self.generator, temperature=self.temperature,
                                   epsilon=self.epsilon, encoded=encoded,
                                   max_candidates=group.max_candidates, **inference)
            downloads = (sample.choice, sample.logp.float(), sample.behaviour_logp.float(),
                         sample.uniform_pick.sum(), sample.behaviour_entropy.double().sum())
        else:
            choice, chosen_logp = actor.act(inputs, self.generator, encoded=encoded,
                                           max_candidates=group.max_candidates, **inference)
            downloads = (choice, chosen_logp.float())
        mark("actor_and_sampling", label)
        self._count_call(stats, label, [group])
        return downloads

    def _count_call(self, stats: CollectStats, label: str, groups: list[_PolicyBatch]) -> None:
        for group in groups:
            self.policy_decisions[group.identity] = (self.policy_decisions.get(group.identity, 0)
                                                     + len(group.rows))
        stats.policy_batches += 1
        if self.profile:
            size = sum(len(group.rows) for group in groups)
            sizes = stats.policy_call_rows.setdefault(label, {})
            sizes[size] = sizes.get(size, 0) + 1

    def _apply_downloads(self, pending: list, r: _StepRows, stats: CollectStats) -> None:
        # Policies do not depend on one another's actions until env.step().
        # Preserve their sampling order, then synchronize once for the whole
        # vector step instead of stalling after each identity's inference.
        downloaded = _download_tensors(tuple(t for _, _, tensors in pending for t in tensors),
                                       packed=self.device.type == "cuda")
        cursor = 0
        for identity, rows, tensors in pending:
            values = downloaded[cursor:cursor + len(tensors)]
            cursor += len(tensors)
            picked_choice, picked_logp = values[:2]
            if identity == LEARNER:
                r.behaviour_logp[rows] = values[2]
                stats.epsilon_picks += int(values[3])
                stats.behaviour_entropy_sum += float(values[4])
            if not (np.isfinite(picked_logp).all() and np.isfinite(r.behaviour_logp[rows]).all()):
                raise FloatingPointError("non-finite behaviour probability")
            r.choices[rows] = picked_choice.astype(np.int32)
            r.logp[rows] = picked_logp

    def _store(self, r: _StepRows, stats: CollectStats) -> None:
        if ((r.choices < 0) | (r.choices >= r.counts)).any():
            raise ValueError("a choice lies outside its candidate list")
        stored = self.buffer.add_step(
            keep=r.acting, env_id=r.env_id, match_id=r.match_id, round_index=r.round_index,
            seat=r.seat, phase=r.phase, obs=r.obs, hidden=r.hidden, cand=r.cand,
            offsets=r.offsets, chosen=r.choices, logp=r.logp, prefix=r.prefix,
            version=self.version, behaviour_logp=r.behaviour_logp)
        if stored:
            rows = np.flatnonzero(r.acting)
            stats.learner_rows += stored
            stats.prefix_sum += int(r.prefix[rows].sum())
            stats.prefix_max = max(stats.prefix_max, int(r.prefix[rows].max()))
        if self.choice_log is not None:
            self.choice_log.append(r.choices.copy())

    def step(self, stats: CollectStats | None = None) -> int:
        """One vector step. Returns the number of pending rows stepped."""
        stats = stats if stats is not None else CollectStats()
        mark = _PhaseTimer(stats, self.device, self.profile)
        if not self.started:
            self.env.reset()
            self.started = True
        batch = self.env.pending()
        mark("env_pending")
        self._drain(stats)
        mark("events_and_rounds")
        rows = self._read_rows(batch)
        mark("metadata_and_assignment")
        plan = self._plan(rows)
        mark("input_indexing")
        # All groups' decision inputs are known before inference. One aligned
        # upload preserves group/row order and avoids seven transfers per group.
        uploaded = upload_arrays(plan.host_fields, self.device)
        mark("decision_upload")
        pending = [(group.identity, group.rows,
                    self._infer_group(group, uploaded[7 * i:7 * (i + 1)], rows, stats, mark))
                   for i, group in enumerate(plan.groups)]
        if plan.layout is not None:
            # After the learner, as the per-identity calls were: identical
            # generator order. One merged actor call for every snapshot row.
            fields = dict(zip(LAYOUT_FIELDS, uploaded[7 * len(plan.groups):]))
            downloads = self._merged_snapshot_step(plan, fields, rows.prefix, mark)
            pending.append((-1, plan.layout.rows, downloads))
            self._count_call(stats, "snapshot", plan.merged)
        self._apply_downloads(pending, rows, stats)
        mark("decision_download")
        self._store(rows, stats)
        mark("buffer_and_counters")
        self.env.step(rows.choices)
        mark("env_step")
        stats.steps += 1
        stats.decisions += len(rows.choices)
        return len(rows.choices)

    def collect(self, steps: int, version: int | None = None) -> CollectStats:
        """``steps`` vector steps; ``version`` tags the stored rows' policy."""
        if version is not None:
            self.version = int(version)
        self.results.clear()
        stats = CollectStats()
        if self.batch_snapshot_encoder and not self.paged_cache:
            raise ValueError("merged snapshot encoding requires the paged KV cache")
        if (not self.batch_snapshot_policies or (self.snapshot_heads is not None and
                                                 self.snapshot_heads.encoder != self.batch_snapshot_encoder)):
            self.snapshot_heads = None      # switched off or re-laid out: release the stacked copy
        if self.batch_snapshot_policies:
            if self.snapshot_heads is None:
                from train.history_snapshot_batch import validate_actor, validate_encoder
                validate_actor(self.actor)
                if self.batch_snapshot_encoder:
                    validate_encoder(self.actor)
            # Snapshot seats no longer use private graphs; free their budget.
            self.release_snapshot_graphs()
        if self.prefill_learner_cache:
            if not self.kv_cache or self.round_streams:
                raise ValueError("learner cache prefill requires the match-keyed KV cache")
            self._prefill_learner(stats)
        if not self.private_graphs:
            for _ in range(int(steps)):
                self.step(stats)
            return stats
        if self._graph_collecting:
            raise RuntimeError("nested graph collection is unsupported")
        self._graph_collecting = True
        try:
            for graph in self.decision_graphs.values():
                graph.begin_interval()
            for _ in range(int(steps)):
                self.step(stats)
        finally:
            self._graph_collecting = False
            for graph in self.decision_graphs.values():
                graph.end_interval()
        return stats
