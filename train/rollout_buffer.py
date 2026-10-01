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

from train.advantages import compute_gae

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


class RolloutBuffer:
    """Preallocated whole-round storage for learner seats only."""

    @classmethod
    def shared(cls, config: RolloutBufferConfig) -> tuple["RolloutBuffer", dict[str, torch.Tensor]]:
        """A buffer whose arrays live in shared memory, plus those arrays as
        tensors. Another process passed the tensors (torch.multiprocessing
        shares them without copying) rebuilds a view with
        `RolloutBuffer(config, arrays={k: t.numpy() ...})`; the Python-side
        counters (`n_steps`, `n_cand`, `n_traj`) are per view and must be
        handed over explicitly. See `train/ppo_actors.py`."""
        buffer = cls(config)
        tensors = {}
        for name, array in buffer.storage().items():
            tensor = torch.from_numpy(array).clone().share_memory_()
            tensors[name] = tensor
            setattr(buffer, name, tensor.numpy())
        return buffer, tensors

    def __init__(self, config: RolloutBufferConfig,
                 arrays: dict[str, np.ndarray] | None = None) -> None:
        """`arrays` adopts existing storage (a shared-memory view) in place of
        freshly allocated arrays; names, shapes and dtypes must match."""
        self._init_storage(config)
        for name, array in (arrays or {}).items():
            mine = getattr(self, name)
            if not isinstance(mine, np.ndarray) or mine.shape != array.shape or mine.dtype != array.dtype:
                raise ValueError(f"adopted array {name} does not match the buffer config")
            setattr(self, name, array)

    def _init_storage(self, config: RolloutBufferConfig) -> None:
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
        self.policy_keep = np.zeros(s, bool)
        self.returns = np.zeros(s, np.float32)
        self.samples = np.zeros(s, np.int64)   # finalized step indices, first n_samples
        # Candidate pool, addressed by cand_start/cand_count, and per candidate
        # the frozen reference's log-probability over the stored (pruned) set.
        self.cand = np.zeros((c, config.act_dim), np.uint8)
        self.ref_logp = np.zeros(c, np.float32)
        # Per-trajectory metadata.
        self.traj_env = np.zeros(k, np.int64)
        self.traj_team = np.zeros(k, np.int64)
        self.traj_len = np.zeros(k, np.int64)
        self.traj_last = np.zeros(k, np.int64)
        self.traj_complete = np.zeros(k, bool)
        self.traj_remap = np.zeros(k, np.int64)
        # Finish position (0 first .. 3 last) of every seat, set at round end.
        self.traj_finish = np.zeros((k, 4), np.int64)
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
        self.staged = None    # StagedSamples, see stage()

    # ------------------------------------------------------------------ rollout
    def add_batch(self, *, learner: np.ndarray, env_id: np.ndarray,
                  match_id: np.ndarray, round_index: np.ndarray, seat: np.ndarray,
                  phase: np.ndarray, obs: np.ndarray, hidden_counts: np.ndarray,
                  cand: np.ndarray, offsets: np.ndarray, chosen: np.ndarray,
                  logp: np.ndarray, value: np.ndarray | None = None,
                  ref_logp: np.ndarray | None = None,
                  cand_index: np.ndarray | None = None,
                  obs_index: np.ndarray | None = None) -> int:
        """Store rows where `learner` is true. Arrays are batch-shaped like
        `VecEnv.pending()`; `cand`/`offsets` are the batch's ragged candidates
        and `chosen` is the index within each row's candidate slice.
        `ref_logp`, aligned with the ragged candidates, is the frozen
        reference's log-probability of each candidate (zero when omitted).
        `cand_index`, when given, maps each ragged candidate to its row of
        `cand`, so a caller can store a subset of a larger candidate array
        without materialising it first; otherwise candidate `j` is `cand[j]`.
        `obs_index` similarly maps each batch row to its row of `obs`, avoiding
        an intermediate observation gather before learner/drop filtering.
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
        pool = slice(self.n_cand, self.n_cand + total)
        self.cand[pool] = np.asarray(cand)[src if cand_index is None
                                           else np.asarray(cand_index)[src]]
        self.ref_logp[pool] = 0.0 if ref_logp is None else np.asarray(ref_logp, np.float32)[src]
        self.obs[steps] = np.asarray(obs)[rows if obs_index is None
                                        else np.asarray(obs_index)[rows]]
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
                     seat_return, order=None) -> None:
        """Close both teams' trajectories of a finished round. `seat_return`
        is the per-seat round return from `drain_finished_rounds`; seats 0/2
        are team 0 and seats 1/3 team 1, so `seat_return[team]` is the team's.
        `order`, the round's finish order (`RoundResult.order`), fills
        `traj_finish` for the auxiliary finish loss."""
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
            if order is not None:
                self.traj_finish[t, list(order)] = np.arange(4)
            self.reward[last] = float(seat_return[team])
            self.done[last] = True
            self.traj_complete[t] = True
            self.open_traj[env_id, team] = -1
        self.finalized = False

    def finish_rounds(self, results) -> None:
        """Convenience over `VecEnv.drain_finished_rounds()` results."""
        for r in results:
            self.finish_round(int(r.env_id), int(r.match_id), int(r.round_index), r.seat_return,
                              r.order)

    # ----------------------------------------------------------------- learning
    def critic_input(self, steps: np.ndarray) -> np.ndarray:
        """`[len(steps), obs_dim + 162]` uint8: observation then hidden rows."""
        return np.concatenate([self.obs[steps], self.hidden[steps]], axis=1)

    def pending_value_steps(self) -> np.ndarray:
        """Steps of completed trajectories, trajectory-major in time order."""
        n = self.n_steps
        idx = np.flatnonzero(self.traj_complete[self.traj[:n]])
        return idx[np.argsort(self.traj[idx], kind="stable")]

    def stage(self, device: torch.device | str) -> "StagedSamples":
        """Upload the completed trajectories' features to `device` once for
        this iteration (see `StagedSamples`). `critic_input`-shaped chunks and
        `gather` then read from the device copy. Dropped by `next_iteration`
        and `clear`."""
        self.staged = StagedSamples(self, device)
        return self.staged

    def finalize(self, gamma: float = 1.0, lam: float = 0.95) -> int:
        """GAE over completed trajectories using the stored `value` slots.
        Returns the number of trainable samples."""
        order = self.pending_value_steps()
        m = order.size
        self.samples[:m] = order
        self.n_samples = m
        self.finalized = True
        if self.staged is not None:
            self.staged.scalars = None    # re-uploaded by the first gather
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
        staged = self.staged
        if (staged is not None and self.finalized and staged.device == torch.device(device)
                and staged.covers(steps)):
            if staged.scalars is None:
                staged.refresh_scalars(self)
            return staged.gather(steps)
        counts = self.cand_count[steps]
        total = int(counts.sum())
        offsets = np.zeros(steps.size + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        src = _ragged_index(self.cand_start[steps], counts, total)
        device = torch.device(device)

        def t(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return to_device(array, device, dtype)

        obs = t(self.obs[steps], torch.float32)
        return {
            "steps": t(steps, torch.long),
            "obs": obs,
            "critic_obs": torch.cat([obs, t(self.hidden[steps], torch.float32)], dim=1),
            "cand": t(self.cand[src], torch.float32),
            "ref_logp": t(self.ref_logp[src], torch.float32),
            "finish": t(self.traj_finish[self.traj[steps], self.seat[steps]], torch.long),
            "offsets": t(offsets, torch.long),
            "chosen": t(self.chosen[steps], torch.long),
            "logp": t(self.logp[steps], torch.float32),
            "value": t(self.value[steps], torch.float32),
            "advantage": t(self.advantage[steps], torch.float32),
            "policy_keep": t(self.policy_keep[steps], torch.bool),
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
            self.ref_logp[:total] = self.ref_logp[src]
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
        self.staged = None

    def clear(self) -> None:
        """Forget everything, including open and dropped rounds."""
        self.open_traj.fill(-1)
        self.dropped.fill(False)
        self.n_steps = self.n_cand = self.n_traj = self.n_samples = 0
        self.finalized = False
        self.staged = None

    def storage(self) -> dict[str, np.ndarray]:
        """Every preallocated array, for capacity and reuse checks."""
        return {name: value for name, value in vars(self).items()
                if isinstance(value, np.ndarray)}


class StagedSamples:
    """The completed trajectories' rows on the learner's device, uploaded once
    per iteration instead of once per minibatch per epoch.

    Rows are in `pending_value_steps()` order, which `finalize` reuses as the
    samples order. Features stay uint8 on the device and are widened per
    chunk or minibatch; `gather(steps)` selects rows and their ragged
    candidates on the device from host-computed indices, so a minibatch
    costs an index upload of a few hundred kilobytes rather than a host
    gather and upload of tens of megabytes. Every tensor it returns holds
    exactly the values `RolloutBuffer.gather` would, in the same order.
    The per-sample scalars written by `finalize` and the advantage filter
    (value, advantage, returns, policy_keep) are uploaded by
    `refresh_scalars` once they exist."""

    def __init__(self, buffer: RolloutBuffer, device: torch.device | str) -> None:
        self.device = torch.device(device)
        steps = buffer.pending_value_steps()
        self.steps = steps
        self.position = np.full(buffer.n_steps, -1, np.int64)
        self.position[steps] = np.arange(steps.size)
        counts = buffer.cand_count[steps]
        self.cand_count = counts
        self.cand_start = np.cumsum(counts) - counts
        src = _ragged_index(buffer.cand_start[steps], counts, int(counts.sum()))
        self.obs = to_device(buffer.obs[steps], self.device, torch.uint8)
        self.hidden = to_device(buffer.hidden[steps], self.device, torch.uint8)
        self.cand = to_device(buffer.cand[src], self.device, torch.uint8)
        self.ref_logp = to_device(buffer.ref_logp[src], self.device, torch.float32)
        self.scalars: dict[str, torch.Tensor] | None = None

    def critic_input(self, begin: int, end: int) -> torch.Tensor:
        """`[end - begin, obs_dim + 162]` float32 of rows `begin:end`."""
        return torch.cat([self.obs[begin:end].to(torch.float32),
                          self.hidden[begin:end].to(torch.float32)], dim=1)

    def refresh_scalars(self, buffer: RolloutBuffer) -> None:
        """Upload the per-sample scalars after `finalize` (and the advantage
        filter). The finalized samples must be exactly the staged rows."""
        steps = self.steps
        if not buffer.finalized or not np.array_equal(buffer.samples[:buffer.n_samples], steps):
            raise RuntimeError("staged rows do not match the finalized samples")

        def t(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
            return to_device(array, self.device, dtype)

        self.scalars = {
            "finish": t(buffer.traj_finish[buffer.traj[steps], buffer.seat[steps]], torch.long),
            "chosen": t(buffer.chosen[steps], torch.long),
            "logp": t(buffer.logp[steps], torch.float32),
            "value": t(buffer.value[steps], torch.float32),
            "advantage": t(buffer.advantage[steps], torch.float32),
            "policy_keep": t(buffer.policy_keep[steps], torch.bool),
            "returns": t(buffer.returns[steps], torch.float32),
            "phase": t(buffer.phase[steps], torch.long),
            "seat": t(buffer.seat[steps], torch.long),
        }

    def covers(self, steps: np.ndarray) -> bool:
        """Whether every requested step is a staged (completed) row."""
        steps = np.asarray(steps)
        return bool(steps.size and (steps < self.position.size).all()
                    and (self.position[steps] >= 0).all())

    def gather(self, steps: np.ndarray) -> dict[str, torch.Tensor]:
        """`RolloutBuffer.gather(steps)` from the device copy."""
        if self.scalars is None:
            raise RuntimeError("refresh_scalars() must run before gather()")
        if not self.covers(steps):
            raise ValueError("a requested step is not a staged sample")
        pos = self.position[steps]
        counts = self.cand_count[pos]
        total = int(counts.sum())
        offsets = np.zeros(steps.size + 1, np.int64)
        np.cumsum(counts, out=offsets[1:])
        src = _ragged_index(self.cand_start[pos], counts, total)
        pos_t = to_device(pos, self.device, torch.long)
        src_t = to_device(src, self.device, torch.long)
        obs = self.obs.index_select(0, pos_t).to(torch.float32)
        hidden = self.hidden.index_select(0, pos_t).to(torch.float32)
        result = {
            "steps": to_device(steps, self.device, torch.long),
            "obs": obs,
            "critic_obs": torch.cat([obs, hidden], dim=1),
            "cand": self.cand.index_select(0, src_t).to(torch.float32),
            "ref_logp": self.ref_logp.index_select(0, src_t),
            "offsets": to_device(offsets, self.device, torch.long),
        }
        for name, tensor in self.scalars.items():
            result[name] = tensor.index_select(0, pos_t)
        return result


def to_device(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """`array` as a `dtype` tensor on `device`, widened after the transfer.

    A CUDA host-to-device copy that also changes dtype converts on the host
    and moves the wide type, four times the bytes for the uint8 features, so
    the copy keeps the stored dtype and goes through pinned memory without
    blocking the host; the conversion runs on the device."""
    value = torch.from_numpy(np.ascontiguousarray(array))
    if device.type == "cuda":
        return value.pin_memory().to(device, non_blocking=True).to(dtype)
    return value.to(device=device, dtype=dtype)


def _ragged_index(starts: np.ndarray, counts: np.ndarray, total: int) -> np.ndarray:
    """Flat source indices for slices `[starts[i], starts[i] + counts[i])`."""
    local = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    return np.repeat(starts, counts) + local
