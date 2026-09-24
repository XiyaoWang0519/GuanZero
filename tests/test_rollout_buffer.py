"""Stage B on-policy round buffer: GAE, masking, learner filter, reuse (B3)."""
import numpy as np
import pytest
import torch

import gd
from train.rollout_buffer import HIDDEN_DIM, RolloutBuffer, RolloutBufferConfig, compute_gae

OBS, ACT = 6, 4


def make_buffer(num_envs=4, carry_over=True, **kw):
    return RolloutBuffer(RolloutBufferConfig(
        num_envs=num_envs, obs_dim=OBS, act_dim=ACT, max_steps=kw.get("steps", 512),
        max_candidates=kw.get("cands", 4096), max_trajectories=kw.get("trajs", 64),
        carry_over=carry_over))


def fake_batch(rng, env_id, seat, round_index, match_id=None, learner=None, values=None):
    n = len(env_id)
    counts = rng.integers(1, 5, size=n)
    offsets = np.concatenate(([0], np.cumsum(counts)))
    cand = rng.integers(0, 2, size=(offsets[-1], ACT), dtype=np.uint8)
    return dict(
        learner=np.ones(n, bool) if learner is None else np.asarray(learner, bool),
        env_id=np.asarray(env_id), seat=np.asarray(seat),
        match_id=np.zeros(n, np.int64) if match_id is None else np.asarray(match_id),
        round_index=np.asarray(round_index), phase=np.full(n, 3),
        obs=rng.integers(0, 2, size=(n, OBS), dtype=np.uint8),
        hidden_counts=rng.integers(0, 3, size=(n, 3, 54), dtype=np.uint8),
        cand=cand, offsets=offsets, chosen=rng.integers(0, counts),
        logp=rng.normal(size=n).astype(np.float32),
        value=rng.normal(size=n).astype(np.float32) if values is None else np.asarray(values, np.float32))


def mc_returns(rewards):
    return np.cumsum(rewards[::-1])[::-1]


def test_lambda_one_is_monte_carlo_return_minus_value():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(3, 7))
    rewards = np.zeros((3, 7))
    rewards[:, -1] = [3, -2, 1]
    dones = np.zeros((3, 7), bool)
    dones[:, -1] = True
    adv, ret = compute_gae(values, rewards, dones, gamma=1.0, lam=1.0)
    expected = np.stack([mc_returns(r) for r in rewards])
    np.testing.assert_allclose(adv, expected - values, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret, expected, rtol=1e-5, atol=1e-5)


def test_lambda_zero_is_one_step_td_error():
    rng = np.random.default_rng(1)
    values = rng.normal(size=9)
    rewards = rng.normal(size=9)
    dones = np.zeros(9, bool)
    dones[[3, 8]] = True  # two episodes back to back
    adv, _ = compute_gae(values, rewards, dones, gamma=0.9, lam=0.0)
    next_values = np.append(values[1:], 0.0) * (~dones)
    np.testing.assert_allclose(adv, rewards + 0.9 * next_values - values, rtol=1e-5, atol=1e-6)


def test_padding_is_masked_and_does_not_leak():
    values = np.array([[0.5, 0.2, 0.1, 9.0, 9.0], [1.0, -1.0, 0.3, 0.2, 0.4]])
    rewards = np.array([[0, 0, 2, 7, 7], [0, 0, 0, 0, -1]], float)
    dones = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 1]], bool)
    mask = np.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], bool)
    adv, ret = compute_gae(values, rewards, dones, 1.0, 0.8, mask=mask)
    short, short_ret = compute_gae(values[0, :3], rewards[0, :3], dones[0, :3], 1.0, 0.8)
    np.testing.assert_allclose(adv[0, :3], short, atol=1e-6)
    np.testing.assert_allclose(ret[0, :3], short_ret, atol=1e-6)
    assert not adv[0, 3:].any() and not ret[0, 3:].any()


def test_interleaved_rounds_get_team_return_and_partial_rounds_are_excluded():
    rng = np.random.default_rng(2)
    buf = make_buffer()
    # env 0: team 0 acts 3 times, team 1 twice; env 1: team 0 acts twice and
    # its round never ends in this iteration.
    schedule = [([0, 1], [0, 0]), ([0, 1], [1, 2]), ([0], [2]), ([0], [3]), ([0], [0])]
    for envs, seats in schedule:
        buf.add_batch(**fake_batch(rng, envs, seats, round_index=np.zeros(len(envs))))
    buf.finish_round(0, 0, 0, [2, -2, 2, -2])
    assert buf.finalize(gamma=1.0, lam=1.0) == 5
    steps = buf.samples[:buf.n_samples]
    assert set(buf.traj_env[buf.traj[steps]].tolist()) == {0}
    for team, ret in ((0, 2.0), (1, -2.0)):
        own = steps[buf.traj_team[buf.traj[steps]] == team]
        assert np.all(np.diff(own) > 0)  # time order inside the trajectory
        assert buf.reward[own[-1]] == ret and buf.done[own[-1]] and not buf.reward[own[:-1]].any()
        np.testing.assert_allclose(buf.returns[own], ret)
        np.testing.assert_allclose(buf.advantage[own], ret - buf.value[own], atol=1e-6)
    open_step = np.flatnonzero(buf.traj_env[buf.traj[:buf.n_steps]] == 1)
    assert open_step.size == 2 and not np.isin(open_step, steps).any()


def test_learner_only_filtering_with_real_env():
    env = gd.VecEnv(num_envs=6, num_threads=1, seed=11)
    env.reset()
    first = env.pending()
    buf = RolloutBuffer(RolloutBufferConfig(num_envs=6, obs_dim=first.obs.shape[1],
                                            act_dim=first.cand.shape[1],
                                            max_steps=20000, max_candidates=400000,
                                            max_trajectories=2000))
    learner_team = np.array([0, 1, 0, 1, 0, 1])
    stored, team_returns = 0, {}
    for _ in range(300):
        batch = env.pending()
        results = env.drain_finished_rounds()
        for r in results:
            for team in (0, 1):
                t = int(buf.open_traj[int(r.env_id), team])
                if t >= 0:
                    team_returns[t] = float(r.seat_return[team])
                    assert r.seat_return[team] == r.seat_return[team + 2]
        buf.finish_rounds(results)
        env.drain_public_actions()
        seat = np.asarray(batch.seat)
        env_ids = np.asarray(batch.env_id)
        learner = seat % 2 == learner_team[env_ids]
        choices = np.array(batch.greedy_choice, dtype=np.int32, copy=True)
        stored += buf.add_batch(learner=learner, env_id=env_ids, match_id=batch.match_id,
                                round_index=batch.round_index, seat=seat, phase=batch.phase,
                                obs=batch.obs, hidden_counts=batch.hidden_counts,
                                cand=batch.cand, offsets=batch.offsets, chosen=choices,
                                logp=np.zeros(batch.rows, np.float32))
        env.step(choices)
    n = buf.n_steps
    assert stored == n > 0
    assert np.all(buf.seat[:n] % 2 == learner_team[buf.traj_env[buf.traj[:n]]])
    assert np.all(buf.traj_team[buf.traj[:n]] == learner_team[buf.traj_env[buf.traj[:n]]])
    assert buf.finalize() > 0 and team_returns
    for t, ret in team_returns.items():
        assert buf.reward[buf.traj_last[t]] == ret and buf.done[buf.traj_last[t]]
    batch = next(buf.minibatches(10**6, np.random.default_rng(0)))
    assert batch["critic_obs"].shape[1] == buf.config.obs_dim + HIDDEN_DIM
    steps = batch["steps"].numpy()
    # Round end reward: only the last step of each trajectory is nonzero.
    assert np.all(buf.reward[steps][~buf.done[steps]] == 0)


def test_ragged_minibatches_cover_each_sample_once():
    rng = np.random.default_rng(3)
    buf = make_buffer(num_envs=8)
    for step in range(6):
        buf.add_batch(**fake_batch(rng, np.arange(8), np.full(8, step % 4), np.zeros(8)))
    for e in range(8):
        buf.finish_round(e, 0, 0, [1, -1, 1, -1])
    total = buf.finalize(lam=0.9)
    assert total == 48
    seen = []
    for batch in buf.minibatches(7, np.random.default_rng(4)):
        steps = batch["steps"].numpy()
        seen.extend(steps)
        offsets = batch["offsets"].numpy()
        assert offsets[-1] == batch["cand"].shape[0]
        for i, s in enumerate(steps):
            start, count = buf.cand_start[s], buf.cand_count[s]
            np.testing.assert_array_equal(batch["cand"][offsets[i]:offsets[i + 1]].numpy(),
                                          buf.cand[start:start + count])
            assert 0 <= batch["chosen"][i] < count
        torch.testing.assert_close(batch["critic_obs"][:, OBS:], torch.from_numpy(
            buf.hidden[steps]).float())
    assert sorted(seen) == sorted(buf.samples[:total].tolist())
    assert len(seen) == len(set(seen)) == total


def test_carry_over_keeps_unfinished_round():
    rng = np.random.default_rng(5)
    buf = make_buffer(carry_over=True)
    first = fake_batch(rng, [0, 1], [0, 0], [0, 0])
    buf.add_batch(**first)
    buf.finish_round(1, 0, 0, [1, -1, 1, -1])
    buf.finalize()
    buf.next_iteration()
    assert buf.n_steps == 1 and buf.n_traj == 1
    np.testing.assert_array_equal(buf.obs[0], first["obs"][0])
    count = buf.cand_count[0]
    np.testing.assert_array_equal(buf.cand[:count], first["cand"][:first["offsets"][1]])
    buf.add_batch(**fake_batch(rng, [0], [2], [0]))
    buf.finish_round(0, 0, 0, [3, -3, 3, -3])
    assert buf.finalize(lam=1.0) == 2
    steps = buf.samples[:2]
    np.testing.assert_allclose(buf.returns[steps], 3.0)
    np.testing.assert_allclose(buf.advantage[steps], 3.0 - buf.value[steps], atol=1e-6)


def test_drop_mode_discards_unfinished_round_until_it_ends():
    rng = np.random.default_rng(6)
    buf = make_buffer(carry_over=False)
    buf.add_batch(**fake_batch(rng, [0], [0], [0]))
    buf.finalize()
    buf.next_iteration()
    assert buf.n_steps == 0
    # Rest of the dropped round is ignored, not stored as a truncated round.
    assert buf.add_batch(**fake_batch(rng, [0], [2], [0])) == 0
    buf.finish_round(0, 0, 0, [2, -2, 2, -2])
    assert buf.finalize() == 0
    # The next round is recorded normally.
    assert buf.add_batch(**fake_batch(rng, [0], [0], [1])) == 1
    buf.finish_round(0, 0, 1, [1, -1, 1, -1])
    assert buf.finalize() == 1


def test_round_mismatch_is_rejected():
    rng = np.random.default_rng(7)
    buf = make_buffer()
    buf.add_batch(**fake_batch(rng, [0], [0], [0]))
    with pytest.raises(ValueError):
        buf.add_batch(**fake_batch(rng, [0], [0], [1]))


@pytest.mark.parametrize("dtype", [np.float32, np.uint8])
def test_indexed_features_preserve_filtering_and_drop_state(dtype):
    rng = np.random.default_rng(9)
    ordinary = make_buffer(carry_over=False)
    indexed = make_buffer(carry_over=False)
    # env 0 is still open when the iteration ends: its later rows must be
    # dropped before either feature index is applied.
    opening = fake_batch(rng, [0], [0], [0])
    for buffer in (ordinary, indexed):
        buffer.add_batch(**opening)
        buffer.finalize()
        buffer.next_iteration()
    batch = fake_batch(rng, [0, 1, 2, 3], [0, 2, 0, 2], [0, 0, 0, 0],
                       learner=[True, True, False, True])
    obs_index = np.array([3, 0, 4, 1])
    obs_pool = np.zeros((6, OBS), dtype=dtype)
    obs_pool[obs_index] = batch["obs"]
    cand_index = rng.permutation(len(batch["cand"]))
    cand_pool = np.empty_like(batch["cand"], dtype=dtype)
    cand_pool[cand_index] = batch["cand"]
    batch["ref_logp"] = rng.normal(size=len(batch["cand"])).astype(np.float32)
    assert ordinary.add_batch(**batch) == 2
    assert indexed.add_batch(**dict(batch, obs=obs_pool, obs_index=obs_index,
                                    cand=cand_pool, cand_index=cand_index)) == 2
    for buffer in (ordinary, indexed):
        for env in (0, 1, 3):
            buffer.finish_round(env, 0, 0, [2, -2, 2, -2], [2, 0, 1, 3])
        buffer.finalize()
    for name, expected in ordinary.storage().items():
        np.testing.assert_array_equal(indexed.storage()[name], expected, err_msg=name)
    expected = ordinary.gather(ordinary.samples[:ordinary.n_samples])
    actual = indexed.gather(indexed.samples[:indexed.n_samples])
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_storage_is_reused_without_growth():
    rng = np.random.default_rng(8)
    for carry in (True, False):
        buf = make_buffer(num_envs=4, carry_over=carry, steps=256, cands=2048, trajs=32)
        before = {k: (id(v), v.nbytes, v.ctypes.data) for k, v in buf.storage().items()}
        round_of = np.zeros(4, np.int64)
        for it in range(200):
            for step in range(8):
                buf.add_batch(**fake_batch(rng, np.arange(4), np.full(4, step % 4), round_of))
                done = rng.random(4) < 0.2
                for e in np.flatnonzero(done):
                    buf.finish_round(int(e), 0, int(round_of[e]), [1, -1, 1, -1])
                round_of = round_of + done
            buf.finalize()
            for _ in buf.minibatches(64, rng):
                pass
            buf.next_iteration()
            assert buf.n_steps <= buf.config.max_steps
        after = {k: (id(v), v.nbytes, v.ctypes.data) for k, v in buf.storage().items()}
        assert after == before
