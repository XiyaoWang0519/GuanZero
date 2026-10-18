"""Growable row storage of ``SequenceRolloutBuffer`` against the chunk buffer it replaced.

``ReferenceRolloutBuffer`` below is the chunk-and-concatenate implementation
(per-step chunk dicts, ``np.concatenate`` in ``compact``, dict remaps in
``next_iteration``, per-trajectory GAE loops) kept verbatim. Randomized
add/finish/finalize/minibatch/carry-over sequences drive both in lockstep and
must agree bit for bit on every array, trajectory, sample and minibatch order;
a tiny production-option trainer must end with identical weights, optimizer
state and metrics whichever buffer, value download and snapshot digest it uses.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import weakref
from typing import Any, Sequence

import numpy as np
import pytest
import torch

import train.history_population as population_module
import train.history_ppo as ppo_module
import train.history_rollout as rollout
from train.history_model import HIDDEN_DIM, HistoryPolicyConfig, fresh_player
from train.history_population import weights_digest
from train.history_ppo import HistoryPPOConfig, HistoryTrainer
from train.history_rollout import (ROW_FIELDS, SequenceRolloutBuffer, Trajectory, compute_gae,
                                   ragged_index)

OBS_DIM, ACT_DIM = 23, 11      # narrow rows keep the randomized runs fast


# ---- the replaced implementation, verbatim -------------------------------------------

class ReferenceRolloutBuffer(SequenceRolloutBuffer):
    """Per-step chunks concatenated by ``compact`` (the implementation before
    the growable storage). Only ``training_batch``/``decision_inputs`` are shared."""

    def __init__(self, obs_dim: int = int(rollout.gd.OBS_DIM),
                 act_dim: int = int(rollout.gd.ACT_DIM)) -> None:
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

    def add_step(self, *, keep, env_id, match_id, round_index, seat, phase, obs, hidden, cand,
                 offsets, chosen, logp, prefix, version, behaviour_logp=None) -> int:
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
        for team in (0, 1):
            t = self.open.get((int(env_id), team))
            if t is None:
                continue
            record = self.trajectories[t]
            if record.match != int(match_id) or record.round != int(round_index):
                raise ValueError("finished round does not match the open trajectory")
            record.reward = float(seat_return[team])
            record.complete = True
            del self.open[(int(env_id), team)]

    def finish_rounds(self, results: Sequence[Any]) -> None:
        for r in results:
            self.finish_round(int(r.env_id), int(r.match_id), int(r.round_index), r.seat_return)

    def compact(self) -> dict[str, np.ndarray]:
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

    def completed_rows(self) -> np.ndarray:
        return np.asarray([r for t in self.trajectories if t.complete for r in t.rows], np.int64)

    def cited_matches(self) -> set[tuple[int, int]]:
        data = self.compact()
        return set(zip(data["env"].tolist(), data["match"].tolist()))

    def finalize(self, values: np.ndarray, gamma: float = 1.0, lam: float = 0.95) -> int:
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
        keys, index = self._group_rows(self.samples)
        order = np.argsort(index, kind="stable")
        ends = np.cumsum(np.bincount(index, minlength=len(keys)))
        return {key: self.samples[order[start:end]]
                for key, start, end in zip(keys, np.r_[0, ends[:-1]], ends)}

    def _group_rows(self, rows, round_streams=False):
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

    def minibatches(self, matches_per_batch, rng):
        groups = self.match_groups()
        keys = list(groups)
        order = rng.permutation(len(keys))
        for begin in range(0, len(keys), max(1, int(matches_per_batch))):
            picked = [keys[i] for i in order[begin:begin + max(1, int(matches_per_batch))]]
            yield np.concatenate([groups[key] for key in picked])

    def next_iteration(self) -> set[tuple[int, int]]:
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


@torch.no_grad()
def reference_refresh_values(self, chunk: int = 4096) -> np.ndarray:
    """``HistoryTrainer.refresh_values`` before the single download."""
    data = self.buffer.compact()
    n = len(self.buffer)
    values = np.zeros(n, np.float32)
    for begin in range(0, n, chunk):
        obs = torch.as_tensor(data["obs"][begin:begin + chunk], device=self.device)
        hidden = torch.as_tensor(data["hidden"][begin:begin + chunk], device=self.device)
        values[begin:begin + chunk] = self.critic(obs, hidden).float().cpu().numpy()
    return values


def reference_weights_digest(state: dict[str, torch.Tensor]) -> str:
    """``weights_digest`` before the packed download."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


# ---- comparison helpers ------------------------------------------------------------

def assert_bitwise(a: np.ndarray, b: np.ndarray, what: str = "") -> None:
    a, b = np.asarray(a), np.asarray(b)
    assert a.dtype == b.dtype and a.shape == b.shape, (what, a.dtype, b.dtype, a.shape, b.shape)
    assert a.tobytes() == b.tobytes(), what


def assert_same_state(new: SequenceRolloutBuffer, ref: ReferenceRolloutBuffer) -> None:
    assert len(new) == len(ref) == new.n_rows
    data, expected = new.compact(), ref.compact()
    assert list(data) == list(expected)
    for name in expected:
        assert_bitwise(data[name], expected[name], name)
    assert [asdict(t) for t in new.trajectories] == [asdict(t) for t in ref.trajectories]
    assert new.open == ref.open
    for name in ("samples", "value", "reward", "done", "advantage", "returns"):
        assert_bitwise(getattr(new, name), getattr(ref, name), name)
    assert_bitwise(new.completed_rows(), ref.completed_rows(), "completed_rows")
    assert new.cited_matches() == ref.cited_matches()


class Result:
    def __init__(self, env: int, match: int, rnd: int, seat_return) -> None:
        self.env_id, self.match_id, self.round_index, self.seat_return = env, match, rnd, seat_return


class Driver:
    """Random vector steps: each environment plays rounds of a match, a random
    subset of rows is the learner's, rounds end in random order (sometimes
    between steps, with matches advancing) and inputs alternate between uint8
    and the engine's float32."""

    def __init__(self, seed: int, envs: int,
                 end_chances: Sequence[float] = (0.0, 0.0, 0.05, 0.1, 0.3, 1.0)) -> None:
        self.rng = np.random.default_rng(seed)
        self.envs = envs
        self.end_chances = end_chances
        self.match = np.zeros(envs, np.int64)
        self.round = np.zeros(envs, np.int64)

    def step(self, version: int) -> dict[str, Any]:
        rng = self.rng
        env = np.repeat(np.arange(self.envs), rng.integers(0, 3, self.envs))
        rng.shuffle(env)
        n = env.size
        counts = rng.integers(1, 9, n)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        values = np.array([0, 1, 1, 0, 2, 7.0, 0.5, 254.9, 255])
        floats = rng.random() < 0.5
        obs = rng.choice(values, (n, OBS_DIM)) if floats else rng.integers(0, 256, (n, OBS_DIM))
        cand = (rng.choice(values, (int(offsets[-1]), ACT_DIM)) if floats
                else rng.integers(0, 256, (int(offsets[-1]), ACT_DIM)))
        hidden = rng.integers(0, 6, (n, 3, HIDDEN_DIM // 3))
        dtype = np.float32 if floats else np.uint8
        keep = rng.random(n) < rng.choice([0.0, 0.5, 1.0])
        logp = rng.standard_normal(n).astype(np.float32)
        return dict(
            keep=keep, env_id=env, match_id=self.match[env], round_index=self.round[env],
            seat=rng.integers(0, 4, n), phase=rng.integers(0, 3, n), obs=obs.astype(dtype),
            hidden=hidden.astype(dtype if rng.random() < 0.5 else np.uint8),
            cand=cand.astype(dtype), offsets=offsets,
            chosen=(rng.integers(0, 1 << 20, n) % counts).astype(np.int32), logp=logp,
            prefix=rng.integers(0, 500, n), version=version,
            behaviour_logp=(None if rng.random() < 0.3
                            else (logp + rng.standard_normal(n).astype(np.float32))))

    def finished(self) -> list[Result]:
        rng = self.rng
        ended = rng.permutation(self.envs)
        ended = ended[rng.random(self.envs) < rng.choice(self.end_chances)]
        results = []
        for e in ended.tolist():
            reward = float(rng.choice([1, 2, 3, -1, -2, -3])) * (1 + rng.random() / 3)
            results.append(Result(e, int(self.match[e]), int(self.round[e]),
                                  (reward, -reward, reward, -reward)))
            if rng.random() < 0.3:
                self.match[e] += 1
                self.round[e] = 0
            else:
                self.round[e] += 1
        return results


def run_lockstep(seed: int, updates: int, envs: int, keep_views: bool, **driver) -> None:
    driver = Driver(seed, envs, **driver)
    new = SequenceRolloutBuffer(OBS_DIM, ACT_DIM)
    ref = ReferenceRolloutBuffer(OBS_DIM, ACT_DIM)
    kept: list[tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = []
    for update in range(updates):
        for _ in range(int(driver.rng.integers(0, 24))):
            results = driver.finished()
            if driver.rng.random() < 0.5:
                new.finish_rounds(results)
                ref.finish_rounds(results)
            else:
                for r in results:
                    new.finish_round(r.env_id, r.match_id, r.round_index, r.seat_return)
                    ref.finish_round(r.env_id, r.match_id, r.round_index, r.seat_return)
            step = driver.step(update)
            assert new.add_step(**step) == ref.add_step(**step)
            if driver.rng.random() < 0.3:
                assert_same_state(new, ref)
        values = driver.rng.standard_normal(len(ref)).astype(np.float32)
        gamma, lam = float(driver.rng.choice([1.0, 0.99])), float(driver.rng.choice([0.95, 1.0]))
        assert new.finalize(values.copy(), gamma, lam) == ref.finalize(values.copy(), gamma, lam)
        assert_same_state(new, ref)
        for round_streams in (False, True):
            a, b = new._group_rows(new.samples, round_streams), ref._group_rows(ref.samples,
                                                                                round_streams)
            assert a[0] == b[0]
            assert_bitwise(a[1], b[1], "group ids")
        groups, expected = new.match_groups(), ref.match_groups()
        assert list(groups) == list(expected)
        for key in expected:
            assert_bitwise(groups[key], expected[key], f"group {key}")
        for epoch in range(2):
            size = int(driver.rng.integers(1, 5))
            batches = list(new.minibatches(size, np.random.default_rng((seed, update, epoch))))
            reference = list(ref.minibatches(size, np.random.default_rng((seed, update, epoch))))
            assert len(batches) == len(reference)
            for x, y in zip(batches, reference):
                assert_bitwise(x, y, "minibatch")
        if keep_views:
            # A caller holding compact() arrays keeps the rows of their time.
            live = new.compact()
            kept.append((live, {k: v.copy() for k, v in live.items()}))
        assert new.next_iteration() == ref.next_iteration()
        assert_same_state(new, ref)
        for live, copied in kept:
            for name in copied:
                assert_bitwise(live[name], copied[name], f"kept {name}")


# ---- randomized lockstep -----------------------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_randomized_lockstep_matches_chunk_buffer(seed):
    run_lockstep(seed, updates=10, envs=int(np.random.default_rng(seed).integers(1, 9)),
                 keep_views=seed % 2 == 1)


@pytest.mark.parametrize("seed", range(2))
def test_long_rounds_carry_over_many_updates(seed):
    # Trajectories of dozens of rows, most of them carried over several updates.
    run_lockstep(50 + seed, updates=20, envs=6, keep_views=seed == 1, end_chances=(0.0, 0.03))


@pytest.mark.parametrize("seed", range(3))
def test_lockstep_with_tiny_capacities_grows_and_swaps(seed, monkeypatch):
    # Growth on nearly every append and spares smaller than the live storage.
    monkeypatch.setattr(rollout, "_MIN_ROWS", 1)
    monkeypatch.setattr(rollout, "_MIN_CANDIDATES", 1)
    run_lockstep(100 + seed, updates=8, envs=5, keep_views=seed == 1)


def test_empty_buffers_and_empty_updates_match():
    new, ref = SequenceRolloutBuffer(), ReferenceRolloutBuffer()
    assert_same_state(new, ref)
    assert new.finalize(np.zeros(0, np.float32)) == ref.finalize(np.zeros(0, np.float32)) == 0
    assert list(new.minibatches(2, np.random.default_rng(0))) == []
    assert new.next_iteration() == ref.next_iteration() == set()
    assert_same_state(new, ref)
    for name, value in new.compact().items():
        assert value.flags.c_contiguous and len(value) == 0, name


def test_storage_is_reused_unless_a_view_survives():
    driver = Driver(7, 4)
    buffer = SequenceRolloutBuffer(OBS_DIM, ACT_DIM)

    def fill(version):
        for _ in range(6):
            buffer.finish_rounds(driver.finished())
            buffer.add_step(**driver.step(version))
        buffer.finalize(np.zeros(len(buffer), np.float32))

    fill(0)
    first = weakref.ref(buffer._rows["obs"])     # weak: a strong one would pin the storage
    buffer.next_iteration()
    second = weakref.ref(buffer._rows["obs"])
    assert second() is not first()
    fill(1)
    buffer.next_iteration()
    assert buffer._rows["obs"] is first(), "the warm spare storage is reused"
    held = buffer.compact()["obs"]      # a view of the first storage, kept across carry-overs
    snapshot = held.copy()
    fill(2)
    buffer.next_iteration()
    assert buffer._rows["obs"] is second()
    fill(3)
    buffer.next_iteration()
    assert buffer._rows["obs"] is not first() and buffer._rows["obs"] is not second()
    np.testing.assert_array_equal(held, snapshot)


def nonempty_step(driver: Driver) -> dict[str, Any]:
    while True:
        step = driver.step(0)
        if len(step["keep"]):
            step["keep"][:] = True
            return step


def test_validation_errors_are_unchanged():
    for cls in (SequenceRolloutBuffer, ReferenceRolloutBuffer):
        buffer = cls(OBS_DIM, ACT_DIM)
        step = nonempty_step(Driver(3, 2))
        bad = dict(step, chosen=np.asarray(step["chosen"]) + 100)
        with pytest.raises(ValueError, match="chosen index"):
            buffer.add_step(**bad)
        buffer.add_step(**step)
        moved = dict(step, round_index=np.asarray(step["round_index"]) + 1)
        with pytest.raises(ValueError, match="finish_round was not called"):
            buffer.add_step(**moved)
    for cls in (SequenceRolloutBuffer, ReferenceRolloutBuffer):
        buffer = cls(OBS_DIM, ACT_DIM)
        buffer.add_step(**nonempty_step(Driver(4, 1)))
        with pytest.raises(ValueError, match="does not match"):
            buffer.finish_round(0, 0, 5, (1., -1., 1., -1.))
        index = next(iter(buffer.open.values()))
        buffer.trajectories[index].complete = True      # open, yet marked complete
        with pytest.raises(KeyError):
            buffer.next_iteration()


# ---- snapshot digest -------------------------------------------------------------------

def digest_cases(device):
    actor, critic = fresh_player(HistoryPolicyConfig(width=16, layers=1, heads=2,
                                                     aux_heads="next,belief,outcome"), 3)
    base = torch.arange(24, dtype=torch.float32, device=device).reshape(4, 6)
    odd = {"transposed": base.t(), "strided": base[:, ::2], "scalar": torch.tensor(2.5, device=device),
           "empty": torch.zeros(0, 3, device=device), "flags": base > 3,
           "counts": torch.arange(5, dtype=torch.int64, device=device),
           "half": base.to(torch.float16), "grad": torch.ones(3, device=device, requires_grad=True)}
    return [{k: v.to(device) for k, v in actor.state_dict().items()},
            {k: v.to(device) for k, v in critic.state_dict().items()}, odd, {}]


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))])
def test_packed_digest_equals_per_tensor_digest(device):
    for state in digest_cases(device):
        assert weights_digest(state) == reference_weights_digest(state)
    mixed = {"a": torch.ones(3), "b": torch.zeros(2, device=device)}
    assert weights_digest(mixed) == reference_weights_digest(mixed)


# ---- trainer: values, then a whole production-option run ------------------------------------

PRODUCTION = dict(
    width=16, layers=1, heads=2, response_mode="auxiliary", aux_heads="next,belief,outcome",
    aux_coef=0.1, entropy=0.03, epochs=2, lr=3e-4, rollout_epsilon=0.0,
    snapshot_updates=2, population_recent=4, population_archive_every=0,
    causal_sdpa=True, rollout_kv_cache=True, rollout_batched_attention=True,
    rollout_wide_projection=True, rollout_paged_cache=True, rollout_page_span=True,
    batch_snapshot_policies=True, batch_snapshot_encoder=True,
    rollout_prefill_learner_cache=True, learner_length_groups=2, learner_batched_attention=True,
    num_envs=8, steps_per_update=64, minibatch_matches=2, seed=2026101803,
    torch_threads=1, num_threads=1, checkpoint_updates=10**9)
TIMING_KEYS = {"collection_phase_seconds", "collection_group_phase_seconds", "decisions_per_sec",
               "collect_seconds", "learn_seconds", "learn_decisions_per_sec",
               "learn_exposures_per_sec", "learner_collect_decisions_per_sec", "elapsed_seconds",
               "cuda_trim_seconds"}


@pytest.fixture
def deterministic():
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1])


def test_refresh_values_single_download_is_bitwise(tmp_path, deterministic):
    trainer = HistoryTrainer(HistoryPPOConfig(**dict(PRODUCTION, updates=1)), tmp_path / "v",
                             device="cpu")
    trainer.collect()
    assert len(trainer.buffer) > 20
    for chunk in (4096, 7, len(trainer.buffer), 1):
        assert_bitwise(trainer.refresh_values(chunk), reference_refresh_values(trainer, chunk),
                       f"chunk {chunk}")
    trainer.buffer = SequenceRolloutBuffer()
    assert_bitwise(trainer.refresh_values(), reference_refresh_values(trainer), "empty")


def without_timings(line: dict[str, Any]) -> dict[str, Any]:
    line = {k: v for k, v in line.items() if k not in TIMING_KEYS}
    line["cuda_trim"] = {point: {k: v for k, v in record.items() if k != "seconds"}
                         for point, record in line.get("cuda_trim", {}).items()}
    return line


def run_trainer(path, updates: int, reference: bool, monkeypatch) -> dict[str, Any]:
    with monkeypatch.context() as patch:
        if reference:
            patch.setattr(ppo_module, "SequenceRolloutBuffer", ReferenceRolloutBuffer)
            patch.setattr(HistoryTrainer, "refresh_values", reference_refresh_values)
            patch.setattr(population_module, "weights_digest", reference_weights_digest)
        torch.manual_seed(0)
        trainer = HistoryTrainer(HistoryPPOConfig(**dict(PRODUCTION, updates=updates)), path,
                                 device="cpu")
        assert isinstance(trainer.buffer, ReferenceRolloutBuffer) == reference
        trainer.collector.choice_log = []
        lines = [trainer.update() for _ in range(updates)]
    return dict(
        lines=[without_timings(line) for line in lines],
        choices=[c.copy() for c in trainer.collector.choice_log],
        rows={k: v.copy() for k, v in trainer.buffer.compact().items()},
        trajectories=[asdict(t) for t in trainer.buffer.trajectories],
        open=dict(trainer.buffer.open),
        actor={k: v.clone() for k, v in trainer.actor.state_dict().items()},
        critic={k: v.clone() for k, v in trainer.critic.state_dict().items()},
        optimizers=[trainer.actor_optimizer.state_dict()["state"],
                    trainer.critic_optimizer.state_dict()["state"]],
        generator=trainer.generator.get_state().clone(),
        rng=trainer.rng.bit_generator.state,
        # The lineage name carries a per-run identity; digests and updates must agree.
        population={i: {k: v for k, v in meta.items() if k != "lineage"}
                    for i, meta in trainer.population.metadata.items()},
        streams={key: stream.prefix for key, stream in trainer.store.streams.items()})


def test_production_trainer_is_bitwise_with_the_reference_buffer(tmp_path, deterministic,
                                                                   monkeypatch):
    updates = 6
    new = run_trainer(tmp_path / "new", updates, False, monkeypatch)
    ref = run_trainer(tmp_path / "ref", updates, True, monkeypatch)
    assert sum(line["carried_rows"] > 0 for line in ref["lines"]) >= 3
    assert sum(line["update_samples"] > 0 for line in ref["lines"]) >= 3
    assert len(ref["population"]) >= 2, "snapshots (and their digests) were taken"
    assert new["lines"] == ref["lines"]
    assert len(new["choices"]) == len(ref["choices"]) > 0
    for x, y in zip(new["choices"], ref["choices"]):
        assert_bitwise(x, y, "choices")
    assert list(new["rows"]) == list(ref["rows"])
    for name in ref["rows"]:
        assert_bitwise(new["rows"][name], ref["rows"][name], name)
    for key in ("trajectories", "open", "rng", "population", "streams"):
        assert new[key] == ref[key], key
    for model in ("actor", "critic"):
        assert list(new[model]) == list(ref[model])
        for name in ref[model]:
            assert torch.equal(new[model][name], ref[model][name]), (model, name)
    for a, b in zip(new["optimizers"], ref["optimizers"]):
        assert a.keys() == b.keys()
        for index in a:
            for name in a[index]:
                assert torch.equal(torch.as_tensor(a[index][name]),
                                   torch.as_tensor(b[index][name])), (index, name)
    assert torch.equal(new["generator"], ref["generator"])
