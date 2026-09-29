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
        data = self.compact()
        groups: dict[tuple[int, int], list[int]] = {}
        for row in self.samples.tolist():
            groups.setdefault((int(data["env"][row]), int(data["match"][row])), []).append(row)
        return {key: np.asarray(rows, np.int64) for key, rows in groups.items()}

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
                       extra: dict[str, np.ndarray] | None = None) -> "TrainingBatch":
        """``decision_inputs`` plus the rows' ``fields`` (stored row data, or
        ``advantage``/``returns`` from ``finalize``) and ``extra`` host arrays,
        all uploaded with one packed copy."""
        data = self.compact()
        rows = np.asarray(rows, np.int64)
        keys = list(zip(data["env"][rows].tolist(), data["match"][rows].tolist()))
        unique = list(dict.fromkeys(keys))
        index = {key: i for i, key in enumerate(unique)}
        picked = [(streams or {}).get(key) or store.stream(*key) for key in unique]
        prefix = data["prefix"][rows]
        for key, p in zip(keys, prefix.tolist()):
            if p > picked[index[key]].prefix:
                raise ValueError("a stored row cites more history than its match has")
        counts = data["cand_count"][rows]
        src = ragged_index(data["cand_start"][rows], counts)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        named = {name: (getattr(self, name) if name in ("advantage", "returns") else data[name])[rows]
                 for name in fields}
        named.update(extra or {})
        host = (*StreamBatch.host_arrays([stream.arrays() for stream in picked]),
                np.asarray([index[key] for key in keys], np.int64), prefix,
                data["obs"][rows], data["seat"][rows], data["cand"][src], offsets,
                offsets[:-1] + data["chosen"][rows], *named.values())
        uploaded = upload_arrays(host, device)
        inputs = DecisionInputs(streams=StreamBatch(*uploaded[:4]), match_index=uploaded[4],
                                prefix=uploaded[5], obs=uploaded[6], seat=uploaded[7],
                                cand=uploaded[8], offsets=uploaded[9])
        return TrainingBatch(inputs, uploaded[10], dict(zip(named, uploaded[11:])))


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
    ``collect`` calls.
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
                 batch_snapshot_policies: bool = False) -> None:
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
        if triton_min_batch < 1:
            raise ValueError("Triton minimum batch must be positive")
        if private_graph_budget_mb < 1 or private_graph_policy_budget_mb < 1:
            raise ValueError("private graph memory budget must be positive")
        self.kv_cache = kv_cache
        self.reuse_cache_lengths = reuse_cache_lengths
        self.private_graphs = private_graphs
        self.private_graph_budget_bytes = private_graph_budget_mb << 20
        self.private_graph_policy_budget_bytes = private_graph_policy_budget_mb << 20
        self.decision_graphs = {}
        self._graph_collecting = False
        self.triton_cache = triton_cache
        self.triton_min_batch = triton_min_batch
        self.batch_snapshot_policies = bool(batch_snapshot_policies)
        self.snapshot_heads = None      # stacked snapshot heads, created on first merged step
        if self.batch_snapshot_policies:
            from train.history_snapshot_batch import validate_actor
            validate_actor(actor)
        self.profile = profile
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.caches = {}
        self.policy_decisions: dict[int, int] = {}
        self.assignments: dict[tuple[int, int], np.ndarray] = {}
        self._assignments_changed = False
        self.results: list[Any] = []            # RoundResults drained by the last collect()
        self.choice_log: list[np.ndarray] | None = [] if record_choices else None
        self.started = False
        self.version = 0

    def cache_metrics(self) -> dict:
        return dict(bytes=sum(c.bytes for c in self.caches.values()),
                    entries=sum(len(c.entries) for c in self.caches.values()),
                    encoded_tokens=sum(c.encoded_tokens for c in self.caches.values()),
                    rebuilds=sum(c.rebuilds for c in self.caches.values()))

    def invalidate_learner_cache(self) -> None:
        # Release before PPO allocates activations; rebuild with updated weights.
        if LEARNER in self.caches:
            self.caches[LEARNER].clear()

    def _prune_caches(self) -> None:
        active = {}
        for key, seats in self.assignments.items():
            for identity in set(seats.tolist()):
                active.setdefault(identity, set()).add(key)
        for identity in list(self.caches):
            if identity not in active:
                del self.caches[identity]
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
            if self.triton_cache:
                from train.history_triton_cache import TritonHistoryCache
                cache = TritonHistoryCache(actor, min_batch=self.triton_min_batch)
            else:
                cache = BatchedHistoryCache(actor)
            self.caches[int(identity)] = cache
        return cache

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
    def _merged_snapshot_step(self, groups, actors, layout, fields, prefix, mark):
        """Per-identity public encodes into one padded memory, then one merged
        actor call and one Gumbel-max draw per row (uniforms per identity)."""
        from train.history_snapshot_batch import merged_log_probs, merged_sample
        width = self.actor.config.width
        memory = None
        for group, actor, (begin, end), (first, last) in zip(
                groups, actors, layout.group_rows, layout.group_streams):
            if self.kv_cache:
                cache = self._cache(group.identity, actor)
                lengths_hint = {}
                if self.reuse_cache_lengths and group.one_decision_per_stream:
                    host_lengths = tuple(prefix[group.rows].tolist())
                    if host_lengths == tuple(stream.prefix for stream in group.streams):
                        lengths_hint = dict(preuploaded_lengths=fields["prefix"][begin:end],
                                            host_lengths=host_lengths)
                _, encoded = cache.encode(group.keys, group.streams, **lengths_hint)
            else:
                encoded = actor.encode_batch(StreamBatch.from_streams(group.streams, self.device))
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

    def step(self, stats: CollectStats | None = None) -> int:
        """One vector step. Returns the number of pending rows stepped."""
        stats = stats if stats is not None else CollectStats()
        stamp = time.perf_counter() if self.profile else 0.0
        def mark(name, group=None):
            nonlocal stamp
            if self.profile:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                now = time.perf_counter()
                stats.phase_seconds[name] = stats.phase_seconds.get(name, 0.0) + now - stamp
                if group is not None:
                    split = stats.group_phase_seconds.setdefault(group, {})
                    split[name] = split.get(name, 0.0) + now - stamp
                stamp = now
        env = self.env
        if not self.started:
            env.reset()
            self.started = True
        batch = env.pending()
        mark("env_pending")
        # Contract order: events, ended rounds, then this batch's rows.
        self.store.ingest(env.drain_public_actions())
        results = env.drain_finished_rounds()
        self.buffer.finish_rounds(results)
        for r in results:
            self.results.append(r)
            stats.rounds += 1
            stats.matches += int(r.match_winner >= 0)
            stats.team0_return += float(r.seat_return[0])
            stats.gain += float(r.gain)
        mark("events_and_rounds")
        n = int(batch.rows)
        if not n:
            raise RuntimeError("environment produced no pending decisions")
        env_id = np.asarray(batch.env_id, np.int64)
        match_id = np.asarray(batch.match_id, np.int64)
        round_index = np.asarray(batch.round_index, np.int64)
        seat = np.asarray(batch.seat, np.int64)
        phase = np.asarray(batch.phase, np.int64)
        offsets = np.asarray(batch.offsets, np.int64)
        counts = offsets[1:] - offsets[:-1]
        obs = np.asarray(batch.obs)
        cand = np.asarray(batch.cand)
        hidden = np.asarray(batch.hidden_counts)
        choices = np.array(batch.greedy_choice, dtype=np.int32, copy=True)
        logp = np.zeros(n, np.float32)
        behaviour_logp = np.zeros(n, np.float32)
        prefix = np.zeros(n, np.int64)
        learner = np.zeros(n, bool)
        identities = np.zeros(n, np.int64)
        for i in range(n):
            seats = self.assignment(env_id[i], match_id[i])
            identities[i] = seats[seat[i]]
            learner[i] = identities[i] == LEARNER
            prefix[i] = self.store.stream(env_id[i], match_id[i]).prefix
        acting = learner & (phase == PLAY_PHASE)
        if (self.kv_cache or self.snapshot_heads is not None) and self._assignments_changed:
            self._prune_caches()
            self._assignments_changed = False
        mark("metadata_and_assignment")
        merge = self.batch_snapshot_policies
        groups, host_fields, merged_groups = [], [], []
        for identity in np.unique(identities[phase == PLAY_PHASE]):
            rows = np.flatnonzero((identities == identity) & (phase == PLAY_PHASE))
            keys = list(zip(env_id[rows].tolist(), match_id[rows].tolist()))
            unique = list(dict.fromkeys(keys))
            streams = [self.store.stream(*k) for k in unique]
            group = _PolicyBatch(int(identity), rows, unique, streams,
                                 len(keys) == len(unique), int(counts[rows].max()))
            if merge and identity != LEARNER:
                merged_groups.append(group)
                continue
            index = {key: i for i, key in enumerate(unique)}
            src = ragged_index(offsets[rows], counts[rows])
            local = np.concatenate(([0], np.cumsum(counts[rows])))
            groups.append(group)
            host_fields.extend((
                np.asarray([index[k] for k in keys], np.int64), prefix[rows],
                obs[rows].astype(np.uint8), seat[rows], cand[src].astype(np.uint8),
                local, np.repeat(np.arange(len(rows), dtype=np.int64), counts[rows])))
        layout = merged_actors = None
        if merged_groups:
            from train.history_snapshot_batch import SnapshotHeads, merged_layout
            if self.snapshot_heads is None:
                self.snapshot_heads = SnapshotHeads()
            merged_actors = [self.resolve_policy(g.identity) for g in merged_groups]
            slots = [self.snapshot_heads.slot(g.identity, actor)
                     for g, actor in zip(merged_groups, merged_actors)]
            layout = merged_layout(merged_groups, slots, prefix, obs, seat, cand, offsets,
                                   counts, env_id, match_id)
            host_fields.extend(layout.arrays)
        mark("input_indexing")
        # All groups' decision inputs are known before inference. One aligned
        # upload preserves group/row order and avoids seven transfers per group.
        uploaded = upload_arrays(host_fields, self.device)
        mark("decision_upload")
        pending_downloads = []
        for group_index, group in enumerate(groups):
            identity, rows = group.identity, group.rows
            label = "learner" if identity == LEARNER else "snapshot"
            fields = uploaded[7 * group_index:7 * (group_index + 1)]
            actor = self.actor if identity == LEARNER else self.resolve_policy(int(identity))
            encoded = None
            if self.kv_cache:
                cache = self._cache(identity, actor)
                lengths_hint = {}
                if self.reuse_cache_lengths and group.one_decision_per_stream:
                    # With distinct keys, insertion order makes decision rows
                    # and streams identical. Check the original host values
                    # before reusing the prefix upload as stream metadata.
                    host_lengths = tuple(prefix[rows].tolist())
                    if host_lengths == tuple(stream.prefix for stream in group.streams):
                        lengths_hint = dict(preuploaded_lengths=fields[1],
                                            host_lengths=host_lengths)
                stream_batch, encoded = cache.encode(group.keys, group.streams, **lengths_hint)
            else:
                stream_batch = StreamBatch.from_streams(group.streams, self.device)
            mark("public_cache_or_collation", label)
            inputs = DecisionInputs(
                streams=stream_batch,
                match_index=fields[0], prefix=fields[1], obs=fields[2], seat=fields[3],
                cand=fields[4], offsets=fields[5], candidate_rows=fields[6],
                one_decision_per_stream=group.one_decision_per_stream)
            inference = {}
            if self.private_graphs:
                inference["inference_log_probs"] = self._graph_log_probs(identity, actor, inputs, encoded)
            if identity == LEARNER:
                sample = actor.explore(inputs, self.generator, temperature=self.temperature,
                                       epsilon=self.epsilon, encoded=encoded,
                                       max_candidates=group.max_candidates, **inference)
                choice, chosen_logp = sample.choice, sample.logp
                downloads = (choice, chosen_logp.float(), sample.behaviour_logp.float(),
                             sample.uniform_pick.sum(), sample.behaviour_entropy.double().sum())
            else:
                choice, chosen_logp = actor.act(inputs, self.generator, encoded=encoded,
                                               max_candidates=group.max_candidates, **inference)
                downloads = (choice, chosen_logp.float())
            mark("actor_and_sampling", label)
            pending_downloads.append((int(identity), rows, downloads))
            self.policy_decisions[int(identity)] = self.policy_decisions.get(int(identity), 0) + len(rows)
            stats.policy_batches += 1
            if self.profile:
                sizes = stats.policy_call_rows.setdefault(label, {})
                sizes[len(rows)] = sizes.get(len(rows), 0) + 1
        if layout is not None:
            # After the learner, as the per-identity calls were: identical
            # generator order. One merged actor call for every snapshot row.
            fields = dict(zip(LAYOUT_FIELDS, uploaded[7 * len(groups):]))
            downloads = self._merged_snapshot_step(merged_groups, merged_actors, layout, fields,
                                                   prefix, mark)
            pending_downloads.append((-1, layout.rows, downloads))
            for group in merged_groups:
                self.policy_decisions[group.identity] = (self.policy_decisions.get(group.identity, 0)
                                                         + len(group.rows))
            stats.policy_batches += 1
            if self.profile:
                sizes = stats.policy_call_rows.setdefault("snapshot", {})
                sizes[len(layout.rows)] = sizes.get(len(layout.rows), 0) + 1
        # Policies do not depend on one another's actions until env.step().
        # Preserve their sampling order, then synchronize once for the whole
        # vector step instead of stalling after each identity's inference.
        all_downloaded = _download_tensors(
            tuple(t for _, _, tensors in pending_downloads for t in tensors),
            packed=self.device.type == "cuda")
        cursor = 0
        for identity, rows, tensors in pending_downloads:
            downloaded = all_downloaded[cursor:cursor + len(tensors)]
            cursor += len(tensors)
            picked_choice, picked_logp = downloaded[:2]
            if identity == LEARNER:
                behaviour_logp[rows] = downloaded[2]
                stats.epsilon_picks += int(downloaded[3])
                stats.behaviour_entropy_sum += float(downloaded[4])
            if not (np.isfinite(picked_logp).all() and np.isfinite(behaviour_logp[rows]).all()):
                raise FloatingPointError("non-finite behaviour probability")
            choices[rows] = picked_choice.astype(np.int32)
            logp[rows] = picked_logp
        mark("decision_download")
        if ((choices < 0) | (choices >= counts)).any():
            raise ValueError("a choice lies outside its candidate list")
        stored = self.buffer.add_step(
            keep=acting, env_id=env_id, match_id=match_id, round_index=round_index, seat=seat,
            phase=phase, obs=obs, hidden=hidden, cand=cand, offsets=offsets, chosen=choices,
            logp=logp, prefix=prefix, version=self.version, behaviour_logp=behaviour_logp)
        if stored:
            rows = np.flatnonzero(acting)
            stats.learner_rows += stored
            stats.prefix_sum += int(prefix[rows].sum())
            stats.prefix_max = max(stats.prefix_max, int(prefix[rows].max()))
        if self.choice_log is not None:
            self.choice_log.append(choices.copy())
        mark("buffer_and_counters")
        env.step(choices)
        mark("env_step")
        stats.steps += 1
        stats.decisions += n
        return n

    def collect(self, steps: int, version: int | None = None) -> CollectStats:
        """``steps`` vector steps; ``version`` tags the stored rows' policy."""
        if version is not None:
            self.version = int(version)
        self.results.clear()
        stats = CollectStats()
        if not self.batch_snapshot_policies:
            self.snapshot_heads = None      # switched off: release the stacked copy
        else:
            if self.snapshot_heads is None:
                from train.history_snapshot_batch import validate_actor
                validate_actor(self.actor)
            # Snapshot seats no longer use private graphs; free their budget.
            self.release_snapshot_graphs()
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
