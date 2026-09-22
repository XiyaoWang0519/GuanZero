"""`VecEnv.set_styles` while a batch is pending (STAGE_B_TODO B7 contract fix).

A match can end and the next one open inside one `pending()`. The opponent
source restyles that environment afterwards, in `on_match_start`, so the
pending `styled_choice` must follow the new styles, through the numpy view the
caller already holds, exactly as a fresh environment with those styles gives.
"""
from __future__ import annotations

import numpy as np

import gd

ENVS = 6
SEED = 4242


def styles(kind: str, envs: int = ENVS) -> np.ndarray:
    out = np.tile(np.asarray(gd.StyleParams.neutral().to_array(), np.float32), (envs, 4, 1))
    if kind == "hot":
        for e in range(envs):
            for seat in range(4):
                p = gd.StyleParams.neutral()
                p.bomb_threshold = 0.0
                p.lead_high_bias = 1.0 if seat % 2 else -1.0
                p.temperature = 1.0 + 0.1 * e
                out[e, seat] = np.asarray(p.to_array(), np.float32)
    return np.ascontiguousarray(out)


def test_restyle_after_match_restart_matches_fresh_env():
    live = gd.VecEnv(num_envs=ENVS, num_threads=2, seed=SEED, encode=False)
    fresh = gd.VecEnv(num_envs=ENVS, num_threads=2, seed=SEED, encode=False)
    live.reset()
    fresh.reset()
    live.set_styles(styles("neutral"))
    fresh.set_styles(styles("hot"))
    last = np.zeros(ENVS, np.int64)
    restarts = changed = 0
    for _ in range(200_000):
        a, b = live.pending(), fresh.pending()
        env_id, match_id = np.asarray(a.env_id), np.asarray(a.match_id)
        restarted = match_id != last[env_id]
        last[env_id] = match_id
        if restarted.any():
            restarts += int(restarted.sum())
            held = a.styled_choice          # a view taken before the restyle
            stale = np.array(held, copy=True)
            live.set_styles(styles("hot"))
            np.testing.assert_array_equal(held, np.asarray(b.styled_choice))
            changed += int((held != stale).sum())
            live.clear_styles()
            np.testing.assert_array_equal(held, np.asarray(a.greedy_choice))
            live.set_styles(styles("neutral"))
        choice = np.array(a.greedy_choice, dtype=np.int32, copy=True)
        live.step(choice)
        fresh.step(choice)
        live.drain_finished_rounds()
        fresh.drain_finished_rounds()
        if restarts >= 10:
            break
    assert restarts >= 10
    assert changed > 0
