"""Training-boundary checks: trajectory identities, privileged labels and logs."""

import gc

import numpy as np
import pytest

import gd


def _deal(hands, leader=0):
    deal = gd.DealSpec()
    deal.hands = hands
    deal.level = 12
    deal.leader = leader
    return deal


def test_hidden_labels_are_relative_and_do_not_enter_observations():
    hands = [gd.cards(s) for s in ("S3 D3", "S4 D4", "S5 D5", "S6 D6")]
    first = gd.VecEnv(1)
    first.reset([_deal(hands)])
    a = first.pending()
    expected = np.array([np.bincount(h, minlength=54) for h in hands[1:]], dtype=np.uint8)
    np.testing.assert_array_equal(a.hidden_counts[0], expected)
    assert a.hidden_counts.shape == (1, 3, 54)
    assert a.hidden_counts.dtype == np.uint8
    assert a.greedy_choice.dtype == np.int32
    second = gd.VecEnv(1)
    second.reset([_deal([hands[0], hands[3], hands[2], hands[1]])])
    b = second.pending()
    np.testing.assert_array_equal(a.obs, b.obs)
    np.testing.assert_array_equal(a.cand, b.cand)
    assert not np.array_equal(a.hidden_counts, b.hidden_counts)


def test_round_result_identifies_the_source_trajectory():
    env = gd.VecEnv(4, num_threads=2, seed=7)
    env.reset()
    completed = []
    for _ in range(400):
        batch = env.pending()
        completed.extend(env.drain_finished_rounds())
        if len(completed) >= 4:
            break
        env.step(batch.greedy_choice)
    assert len(completed) >= 4
    seen = set()
    for result in completed:
        key = (result.env_id, result.match_id, result.round_index)
        assert key not in seen
        seen.add(key)
        assert 0 <= result.env_id < 4
        assert result.match_id >= 0
        assert result.round_index >= 0
        assert sum(result.seat_return) == 0
        assert sorted(result.order) == [0, 1, 2, 3]


def test_logging_preserves_decisions_and_includes_forced_passes():
    regular = gd.VecEnv(4, num_threads=2, seed=19)
    logged = gd.VecEnv(4, num_threads=2, seed=19,
                       log_public_actions=True, log_env_limit=2)
    regular.reset()
    logged.reset()
    forced = 0
    previous = {}
    for _ in range(120):
        a, b = regular.pending(), logged.pending()
        for field in ("obs", "cand", "offsets", "seat", "greedy_choice", "hidden_counts"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
        for event in logged.drain_public_actions():
            assert 0 <= event.env_id < 2
            assert event.encoded_action.shape == (gd.ACT_DIM,)
            assert not event.encoded_action[gd.ACT_TRIBUTE_FLAGS:].any()
            if event.phase == gd.Phase.Play:
                key = (event.env_id, event.match_id, event.round_index)
                assert event.step == previous.get(key, -1) + 1
                previous[key] = event.step
            if event.forced:
                forced += 1
                assert event.action.is_pass
            assert 0 <= event.cards_left <= 28
        regular.step(a.greedy_choice)
        logged.step(b.greedy_choice)
    assert forced > 0
    assert regular.drain_public_actions() == []


def test_invalid_choices_fail_atomically_and_require_fresh_batch():
    env = gd.VecEnv(3, seed=99)
    with pytest.raises(RuntimeError, match="reset"):
        env.pending()
    env.reset()
    batch = env.pending()
    original = batch.obs.copy()
    with pytest.raises(ValueError, match="one index"):
        env.step(np.zeros(2, dtype=np.int32))
    with pytest.raises(IndexError, match="choice"):
        env.step(np.array([0, 0, -1], dtype=np.int32))
    with pytest.raises(ValueError, match="one-dimensional"):
        env.step(np.zeros((3, 1), dtype=np.int32))
    with pytest.raises(TypeError):
        env.step(np.zeros(3, dtype=np.float32))
    np.testing.assert_array_equal(env.pending().obs, original)
    env.step(np.zeros(3, dtype=np.int32))
    with pytest.raises(RuntimeError, match="pending"):
        env.step(np.zeros(3, dtype=np.int32))
    env.reset()
    with pytest.raises(RuntimeError, match="pending"):
        env.step(np.zeros(3, dtype=np.int32))


@pytest.mark.parametrize("kwargs", [{"num_envs": 0}, {"num_envs": -1},
                                  {"num_envs": 1, "num_threads": 0},
                                  {"num_envs": 1, "log_env_limit": -2}])
def test_invalid_env_dimensions(kwargs):
    with pytest.raises(ValueError):
        gd.VecEnv(**kwargs)


def test_fork_validates_bounds_and_invalidates_previous_batch():
    env = gd.VecEnv(2)
    env.reset()
    env.pending()
    for idx in (-1, 2):
        with pytest.raises(IndexError):
            env.fork(idx, 1)
    with pytest.raises(ValueError):
        env.fork(0, -1)
    env.fork(0, 1)
    with pytest.raises(RuntimeError):
        env.step(np.zeros(2, dtype=np.int32))
    assert env.pending().rows == 3


def test_views_are_read_only_and_keep_environment_alive():
    env = gd.VecEnv(2)
    env.reset()
    batch = env.pending()
    obs = batch.obs
    original = obs.copy()
    for field in ("obs", "cand", "hidden_counts", "greedy_choice", "match_id"):
        assert not getattr(batch, field).flags.writeable
    del env, batch
    gc.collect()
    np.testing.assert_array_equal(obs, original)


def test_no_encoding_retains_metadata_and_empty_feature_arrays():
    env = gd.VecEnv(2, encode=False)
    env.reset()
    batch = env.pending()
    assert batch.obs.shape == (0, gd.OBS_DIM)
    assert batch.cand.shape == (0, gd.ACT_DIM)
    assert batch.hidden_counts.shape == (2, 3, 54)
    env.step(batch.greedy_choice)


@pytest.mark.parametrize("field,value", [
    ("level", -1), ("level", 256), ("leader", 4), ("owner", 2),
    ("team_levels", [0]), ("team_levels", [0, 13]), ("fails", []),
    ("prev_order", [0, 0, 1, 2]), ("hands", [[0], [1], [2]]),
    ("hands", [[54], [1], [2], [3]]), ("hands", [[0, 0, 0], [1], [2], [3]]),
])
def test_deal_fields_reject_invalid_input(field, value):
    with pytest.raises(ValueError):
        setattr(gd.DealSpec(), field, value)
