"""M0 task 8: the vectorized environment plays 1,000 matches with random
choices and keeps every buffer consistent."""
import numpy as np

import gd


def test_vecenv_plays_1000_matches():
    rng = np.random.default_rng(0)
    env = gd.VecEnv(num_envs=64, num_threads=4, seed=1234)
    env.reset()
    finished_rounds = 0
    finished_matches = 0
    steps = 0
    while finished_matches < 1000 and steps < 5_000_000:
        b = env.pending()
        n = b.offsets.shape[0] - 1
        if n == 0:
            break
        assert b.obs.shape == (n, gd.OBS_DIM)
        assert b.cand.shape[1] == gd.ACT_DIM
        assert b.offsets[0] == 0 and b.offsets[-1] == b.cand.shape[0]
        assert np.all(np.diff(b.offsets) >= 1), "every decision needs a candidate"
        assert np.all(np.isfinite(b.obs))
        widths = np.diff(b.offsets)
        choices = (rng.random(n) * widths).astype(np.int32)
        env.step(choices)
        steps += 1
        for r in env.drain_finished_rounds():
            finished_rounds += 1
            assert sorted(r.order) == [0, 1, 2, 3] or r.num_finished_seats == 3
            assert r.gain in (1, 2, 3)
            assert sum(r.seat_return) == 0
            if r.match_winner >= 0:
                finished_matches += 1
    assert finished_matches >= 1000, (finished_matches, finished_rounds, steps)


def test_vecenv_is_deterministic():
    def run(seed):
        env = gd.VecEnv(num_envs=8, num_threads=1, seed=seed)
        env.reset()
        rng = np.random.default_rng(0)
        out = []
        for _ in range(500):
            b = env.pending()
            n = b.offsets.shape[0] - 1
            if n == 0:
                break
            widths = np.diff(b.offsets)
            ch = (rng.random(n) * widths).astype(np.int32)
            out.append(float(b.obs.sum()))
            env.step(ch)
        return out

    assert run(5) == run(5)
    assert run(5) != run(6)


def test_fork_copies_a_decision_point():
    env = gd.VecEnv(num_envs=4, num_threads=1, seed=99)
    env.reset()
    b = env.pending()
    assert b.offsets.shape[0] - 1 > 0
    new_ids = env.fork(int(b.env_id[0]), 3)
    assert len(new_ids) == 3
    b2 = env.pending()
    rows = {int(e): i for i, e in enumerate(b2.env_id)}
    src = rows[int(b.env_id[0])]
    for nid in new_ids:
        i = rows[int(nid)]
        assert np.array_equal(b2.obs[i], b2.obs[src])
