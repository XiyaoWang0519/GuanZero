"""On-policy round buffer for Stage B PPO (DESIGN 8.4, STAGE_B_TODO B3).

Stores learner-seat decisions grouped into trajectories keyed by
(env, team, round). All storage is preallocated from `RolloutBufferConfig`
and reused across iterations: adding a batch of decisions only writes into
existing numpy arrays, there is no per-decision Python object.

Lifecycle of one PPO iteration:

1. `add_batch` once per vector step with the rows the learner acted on.
2. `finish_round` for every drained round result; it writes the team's round
   return (DESIGN 8.2 item 1) as the reward of that team's last step and sets
   `done` there. All other rewards stay zero, discount is 1.
3. `finalize(gamma, lam)` computes GAE over completed trajectories only.
4. `minibatches(...)` yields torch tensors, ragged candidates as a flat
   candidate tensor plus offsets, every completed sample exactly once per epoch.
5. `next_iteration()` clears completed trajectories. Trajectories whose round
   is still in progress are carried over (moved to the front of the storage)
   or dropped, per `carry_over`. A dropped round keeps being ignored until its
   result arrives, so no truncated trajectory is ever trained on.

Candidate features and observations are stored as uint8 because the v1
encoder emits binary planes (same choice as `train/buffer.py`). The critic
input is the actor observation concatenated with the three `hidden_counts`
rows `[3, 54]` flattened; the hidden rows are stored once per step and the
concatenation is assembled when a batch is read, so the observation is not
stored twice.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch

HIDDEN_DIM = 3 * 54


@dataclass
class RolloutBufferConfig:
    num_envs: int
    obs_dim: int
    act_dim: int
    max_steps: int = 262144          # learner decisions held at once
    max_candidates: int = 4194304    # candidate rows held at once
    max_trajectories: int = 65536    # (env, team, round) groups held at once
    carry_over: bool = True          # False drops rounds unfinished at iteration end

    def validate(self) -> None:
        for name in ("num_envs", "obs_dim", "act_dim", "max_steps",
                     "max_candidates", "max_trajectories"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


def compute_gae(values: np.ndarray, rewards: np.ndarray, dones: np.ndarray,
                gamma: float = 1.0, lam: float = 0.95,
                mask: np.ndarray | None = None,
                last_value: np.ndarray | float | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation along the last axis (time).

    Arrays are `[..., T]`. `dones[t]` marks a terminal step: nothing is
    bootstrapped across it. `mask[t]` false marks padding; padded positions get
    zero advantage and return and do not feed into earlier steps. `last_value`
    bootstraps past the final column (default 0, i.e. treat it as terminal).
    Returns `(advantages, returns)` as float32 with `returns = adv + values`.
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
    next_value = np.broadcast_to(
        np.asarray(0.0 if last_value is None else last_value, np.float64),
        batch_shape).copy()
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


class RolloutBuffer:
    """Preallocated whole-round storage for learner seats only."""

    def __init__(self, config: RolloutBufferConfig) -> None:
        config.validate()
        self.config = config
        s, c, k = config.max_steps, config.max_candidates, config.max_trajectories
        # Per-step storage.
        self.obs = np.zeros((s, config.obs_dim), np.uint8)
        self.hidden = np.zeros((s, HIDDEN_DIM), np.uint8)
        self.cand_start = np.zeros(s, np.int64)
        self.cand_count = np.zeros(s, np.int64)
        self.chosen = np.zeros(s, np.int64)
        self.logp = np.zeros(s, np.float32)
        self.value = np.zeros(s, np.float32)
        self.reward = np.zeros(s, np.float32)
        self.done = np.zeros(s, bool)
        self.phase = np.zeros(s, np.int8)
        self.seat = np.zeros(s, np.int8)
        self.traj = np.zeros(s, np.int64)
        self.advantage = np.zeros(s, np.float32)
        self.returns = np.zeros(s, np.float32)
        self.samples = np.zeros(s, np.int64)   # finalized step indices, first n_samples
        # Candidate pool, addressed by cand_start/cand_count.
        self.cand = np.zeros((c, config.act_dim), np.uint8)
        # Per-trajectory metadata.
        self.traj_env = np.zeros(k, np.int64)
        self.traj_team = np.zeros(k, np.int64)
        self.traj_len = np.zeros(k, np.int64)
        self.traj_last = np.zeros(k, np.int64)
        self.traj_complete = np.zeros(k, bool)
        self.traj_remap = np.zeros(k, np.int64)
        # Per (env, team) table of the open trajectory and its round key.
        shape = (config.num_envs, 2)
        self.open_traj = np.full(shape, -1, np.int64)
        self.open_match = np.zeros(shape, np.int64)
        self.open_round = np.zeros(shape, np.int64)
        self.dropped = np.zeros(shape, bool)
        self.drop_match = np.zeros(shape, np.int64)
        self.drop_round = np.zeros(shape, np.int64)
        self.n_steps = 0
        self.n_cand = 0
        self.n_traj = 0
        self.n_samples = 0
        self.finalized = False

    # ------------------------------------------------------------------ rollout
    def add_batch(self, *, learner: np.ndarray, env_id: np.ndarray,
                  match_id: np.ndarray, round_index: np.ndarray, seat: np.ndarray,
                  phase: np.ndarray, obs: np.ndarray, hidden_counts: np.ndarray,
                  cand: np.ndarray, offsets: np.ndarray, chosen: np.ndarray,
                  logp: np.ndarray, value: np.ndarray | None = None) -> int:
        """Store rows where `learner` is true. Arrays are batch-shaped like
        `VecEnv.pending()`; `cand`/`offsets` are the batch's ragged candidates
        and `chosen` is the index within each row's candidate slice.
        Returns the number of stored steps."""
        rows = np.flatnonzero(np.asarray(learner, bool))
        if rows.size == 0:
            return 0
        env = np.asarray(env_id, np.int64)[rows]
        team = np.asarray(seat, np.int64)[rows] % 2
        match = np.asarray(match_id, np.int64)[rows]
        rnd = np.asarray(round_index, np.int64)[rows]
        # Rounds dropped at an iteration boundary stay dropped until they end.
        dropped = self.dropped[env, team]
        if dropped.any():
            stale = dropped & ((self.drop_match[env, team] != match)
                               | (self.drop_round[env, team] != rnd))
            if stale.any():
                raise ValueError("a new round started before the dropped round finished")
            keep = ~dropped
            rows, env, team, match, rnd = rows[keep], env[keep], team[keep], match[keep], rnd[keep]
            if rows.size == 0:
                return 0
        slot = self.open_traj[env, team]
        opened = slot >= 0
        if (opened & ((self.open_match[env, team] != match)
                      | (self.open_round[env, team] != rnd))).any():
            raise ValueError("finish_round was not called before the next round's decisions")
        if (~opened).any():
            # A team may act twice in one batch (double tribute); open once.
            key = env[~opened] * 2 + team[~opened]
            unique_key, first = np.unique(key, return_index=True)
            fresh = unique_key.size
            if self.n_traj + fresh > self.config.max_trajectories:
                raise OverflowError("rollout buffer trajectory capacity exceeded")
            ids = np.arange(self.n_traj, self.n_traj + fresh)
            fe, ft = unique_key // 2, unique_key % 2
            self.open_traj[fe, ft] = ids
            self.open_match[fe, ft] = match[~opened][first]
            self.open_round[fe, ft] = rnd[~opened][first]
            self.traj_env[ids] = fe
            self.traj_team[ids] = ft
            self.traj_len[ids] = 0
            self.traj_complete[ids] = False
            self.n_traj += fresh
            slot = self.open_traj[env, team]
        offsets = np.asarray(offsets, np.int64)
        counts = offsets[rows + 1] - offsets[rows]
        n, total = rows.size, int(counts.sum())
        if self.n_steps + n > self.config.max_steps:
            raise OverflowError("rollout buffer step capacity exceeded")
        if self.n_cand + total > self.config.max_candidates:
            raise OverflowError("rollout buffer candidate capacity exceeded")
        chosen = np.asarray(chosen, np.int64)[rows]
        if ((chosen < 0) | (chosen >= counts)).any():
            raise ValueError("chosen index outside the row's candidate set")
        steps = slice(self.n_steps, self.n_steps + n)
        starts = self.n_cand + np.cumsum(counts) - counts
        src = _ragged_index(offsets[rows], counts, total)
        self.cand[self.n_cand:self.n_cand + total] = np.asarray(cand)[src]
        self.obs[steps] = np.asarray(obs)[rows]
        self.hidden[steps] = np.asarray(hidden_counts).reshape(len(offsets) - 1, HIDDEN_DIM)[rows]
        self.cand_start[steps] = starts
        self.cand_count[steps] = counts
        self.chosen[steps] = chosen
        self.logp[steps] = np.asarray(logp, np.float32)[rows]
        self.value[steps] = 0.0 if value is None else np.asarray(value, np.float32)[rows]
        self.reward[steps] = 0.0
        self.done[steps] = False
        self.phase[steps] = np.asarray(phase)[rows]
        self.seat[steps] = np.asarray(seat)[rows]
        self.traj[steps] = slot
        position = np.arange(self.n_steps, self.n_steps + n)
        np.add.at(self.traj_len, slot, 1)
        np.maximum.at(self.traj_last, slot, position)
        self.n_steps += n
        self.n_cand += total
        self.finalized = False
        return n

    def finish_round(self, env_id: int, match_id: int, round_index: int,
                     seat_return) -> None:
        """Close both teams' trajectories of a finished round. `seat_return`
        is the per-seat round return from `drain_finished_rounds`; seats 0/2
        are team 0 and seats 1/3 team 1, so `seat_return[team]` is the team's."""
        for team in (0, 1):
            if (self.dropped[env_id, team] and self.drop_match[env_id, team] == match_id
                    and self.drop_round[env_id, team] == round_index):
                self.dropped[env_id, team] = False
                continue
            t = int(self.open_traj[env_id, team])
            if t < 0:
                continue  # this team had no learner decisions in the round
            if (self.open_match[env_id, team] != match_id
                    or self.open_round[env_id, team] != round_index):
                raise ValueError("finished round does not match the open trajectory")
            last = int(self.traj_last[t])
            self.reward[last] = float(seat_return[team])
            self.done[last] = True
            self.traj_complete[t] = True
            self.open_traj[env_id, team] = -1
        self.finalized = False

    def finish_rounds(self, results) -> None:
        """Convenience over `VecEnv.drain_finished_rounds()` results."""
        for r in results:
            self.finish_round(int(r.env_id), int(r.match_id), int(r.round_index), r.seat_return)

    # ----------------------------------------------------------------- learning
    def critic_input(self, steps: np.ndarray) -> np.ndarray:
        """`[len(steps), obs_dim + 162]` uint8: observation then hidden rows."""
        return np.concatenate([self.obs[steps], self.hidden[steps]], axis=1)

    def pending_value_steps(self) -> np.ndarray:
        """Steps of completed trajectories, trajectory-major in time order."""
        n = self.n_steps
        idx = np.flatnonzero(self.traj_complete[self.traj[:n]])
        return idx[np.argsort(self.traj[idx], kind="stable")]

    def finalize(self, gamma: float = 1.0, lam: float = 0.95) -> int:
        """GAE over completed trajectories using the stored `value` slots.
        Returns the number of trainable samples."""
        order = self.pending_value_steps()
        m = order.size
        self.samples[:m] = order
        self.n_samples = m
        self.finalized = True
        if m == 0:
            return 0
        traj = self.traj[order]
        # Consecutive runs of equal trajectory id, time-ordered within each.
        boundary = np.flatnonzero(np.diff(traj)) + 1
        starts = np.concatenate(([0], boundary))
        lengths = np.diff(np.concatenate((starts, [m])))
        rows = np.repeat(np.arange(starts.size), lengths)
        cols = np.arange(m) - np.repeat(starts, lengths)
        shape = (starts.size, int(lengths.max()))
        values = np.zeros(shape, np.float32)
        rewards = np.zeros(shape, np.float32)
        dones = np.zeros(shape, bool)
        mask = np.zeros(shape, bool)
        values[rows, cols] = self.value[order]
        rewards[rows, cols] = self.reward[order]
        dones[rows, cols] = self.done[order]
        mask[rows, cols] = True
        adv, ret = compute_gae(values, rewards, dones, gamma, lam, mask=mask)
        self.advantage[order] = adv[rows, cols]
        self.returns[order] = ret[rows, cols]
        return m

    def minibatches(self, batch_size: int, rng: np.random.Generator,
                    device: torch.device | str = "cpu") -> Iterator[dict[str, torch.Tensor]]:
        """One epoch over finalized samples in random order, each exactly once.
        Candidates are flattened: `cand[offsets[i]:offsets[i+1]]` belong to
        sample `i` and `chosen[i]` indexes within that slice."""
        if not self.finalized:
            raise RuntimeError("finalize() must run before minibatches()")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        order = self.samples[rng.permutation(self.n_samples)]
        for begin in range(0, order.size, batch_size):
            yield self.gather(order[begin:begin + batch_size], device)

    def gather(self, steps: np.ndarray, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        counts = self.cand_count[steps]
        total = int(counts.sum())
        offsets = np.zeros(steps.size + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        src = _ragged_index(self.cand_start[steps], counts, total)

        def t(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return torch.from_numpy(np.ascontiguousarray(array)).to(device=device, dtype=dtype)

        obs = t(self.obs[steps], torch.float32)
        return {
            "steps": t(steps, torch.long),
            "obs": obs,
            "critic_obs": torch.cat([obs, t(self.hidden[steps], torch.float32)], dim=1),
            "cand": t(self.cand[src], torch.float32),
            "offsets": t(offsets, torch.long),
            "chosen": t(self.chosen[steps], torch.long),
            "logp": t(self.logp[steps], torch.float32),
            "value": t(self.value[steps], torch.float32),
            "advantage": t(self.advantage[steps], torch.float32),
            "returns": t(self.returns[steps], torch.float32),
            "phase": t(self.phase[steps], torch.long),
            "seat": t(self.seat[steps], torch.long),
        }

    # ---------------------------------------------------------------- iteration
    def next_iteration(self) -> None:
        """Discard completed trajectories; carry or drop in-progress rounds."""
        n = self.n_steps
        open_ids = np.flatnonzero(~self.traj_complete[:self.n_traj])
        if not self.config.carry_over or open_ids.size == 0:
            if open_ids.size:
                env, team = self.traj_env[open_ids], self.traj_team[open_ids]
                self.dropped[env, team] = True
                self.drop_match[env, team] = self.open_match[env, team]
                self.drop_round[env, team] = self.open_round[env, team]
                self.open_traj[env, team] = -1
            self.n_steps = self.n_cand = self.n_traj = 0
        else:
            keep = np.flatnonzero(~self.traj_complete[self.traj[:n]])  # time order
            k = keep.size
            counts = self.cand_count[keep]
            total = int(counts.sum())
            src = _ragged_index(self.cand_start[keep], counts, total)
            self.cand[:total] = self.cand[src]
            for array in (self.obs, self.hidden, self.chosen, self.logp, self.value,
                          self.phase, self.seat, self.traj, self.cand_count):
                array[:k] = array[keep]
            self.cand_start[:k] = np.cumsum(counts) - counts
            self.reward[:k] = 0.0
            self.done[:k] = False
            remap = self.traj_remap
            remap[open_ids] = np.arange(open_ids.size)
            for array in (self.traj_env, self.traj_team, self.traj_len):
                array[:open_ids.size] = array[open_ids]
            self.traj_complete[:open_ids.size] = False
            self.traj[:k] = remap[self.traj[:k]]
            self.traj_last[:open_ids.size] = 0
            np.maximum.at(self.traj_last, self.traj[:k], np.arange(k))
            live = self.open_traj >= 0
            self.open_traj[live] = remap[self.open_traj[live]]
            self.n_steps, self.n_cand, self.n_traj = k, total, open_ids.size
        self.n_samples = 0
        self.finalized = False

    def clear(self) -> None:
        """Forget everything, including open and dropped rounds."""
        self.open_traj.fill(-1)
        self.dropped.fill(False)
        self.n_steps = self.n_cand = self.n_traj = self.n_samples = 0
        self.finalized = False

    def storage(self) -> dict[str, np.ndarray]:
        """Every preallocated array, for capacity and reuse checks."""
        return {name: value for name, value in vars(self).items()
                if isinstance(value, np.ndarray)}


def _ragged_index(starts: np.ndarray, counts: np.ndarray, total: int) -> np.ndarray:
    """Flat source indices for slices `[starts[i], starts[i] + counts[i])`."""
    local = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(starts, counts) + local
