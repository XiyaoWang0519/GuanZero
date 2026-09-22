"""Behaviour of the style-parameterised heuristic bot (M2_TODO task 1).

Each sweep drives four styled seats through a few thousand rounds and checks
that the statistic the parameter is supposed to control moves monotonically.
The histograms are printed so that the numbers, not only the assertion, are in
the test output.
"""

from __future__ import annotations

import numpy as np
import pytest

import gd

ROUNDS = 900
NUM_ENVS = 96
THREADS = 4

POWER_TYPES = {"Single", "Pair", "Triple", "FullHouse", "Bomb"}
WINDOW_SPAN = {"Straight": 9, "StraightFlush": 9, "Tube": 11, "Plate": 12}


def neutral_styles(num_envs: int = NUM_ENVS) -> np.ndarray:
    """All four seats of every environment at the greedy-equivalent style."""
    row = gd.StyleParams.neutral().to_array()
    return np.tile(row.astype(np.float32), (num_envs, 4, 1))


def play_rank(action) -> float:
    """How high a play sits inside its own ordering, normalised to [0, 1]."""
    if action.type in ("Bomb", "StraightFlush", "JokerBomb"):
        return 1.0
    if action.type in POWER_TYPES:
        return action.key / 14.0
    if action.type in WINDOW_SPAN:
        return action.key / WINDOW_SPAN[action.type]
    return 0.0


def run_styled(styles: np.ndarray, seed: int, rounds: int = ROUNDS) -> dict:
    """Play `rounds` rounds with every seat driven by its style; collect stats.

    The lead/follow split comes from the pending batch, where a following row is
    exactly the one that also offers a pass (RULES.md 5). Bomb timing comes from
    the actor's own hand size before the play.
    """
    env = gd.VecEnv(num_envs=styles.shape[0], num_threads=THREADS, seed=seed,
                    encode=False, log_public_actions=True)
    env.reset()
    env.set_styles(np.ascontiguousarray(styles, dtype=np.float32))

    bomb_fraction: list[float] = []      # actor's cards in hand when bombing
    lead_ranks: list[float] = []
    follow_ranks: list[float] = []
    lead_types: dict[str, int] = {}
    leads = 0
    done = 0
    play_phase = int(gd.Phase.Play)
    while done < rounds:
        batch = env.pending()
        choices = np.ascontiguousarray(batch.styled_choice, dtype=np.int32)
        phases = np.asarray(batch.phase)
        for row in range(batch.rows):
            if phases[row] != play_phase:
                continue
            cands = env.row_actions(row)
            picked = cands[int(choices[row])]
            if picked.type == "Pass":
                continue
            leading = not any(c.type == "Pass" for c in cands)
            if leading:
                leads += 1
                lead_ranks.append(play_rank(picked))
                lead_types[picked.type] = lead_types.get(picked.type, 0) + 1
            else:
                follow_ranks.append(play_rank(picked))
        env.step(choices)
        done += len(env.drain_finished_rounds())
        for event in env.drain_public_actions():
            action = event.action
            if action.type not in ("Bomb", "StraightFlush", "JokerBomb"):
                continue
            # cards_left is the count after the play, so the actor held
            # cards_left + len(cards) of its 27 dealt cards when it bombed.
            bomb_fraction.append((event.cards_left + len(action.cards)) / 27.0)

    return {
        "bomb_fraction": float(np.mean(bomb_fraction)) if bomb_fraction else float("nan"),
        "bombs": len(bomb_fraction),
        "mean_lead_rank": float(np.mean(lead_ranks)) if lead_ranks else float("nan"),
        "mean_follow_rank": float(np.mean(follow_ranks)) if follow_ranks else float("nan"),
        "lead_type_fraction": {t: n / max(1, leads) for t, n in lead_types.items()},
        "leads": leads,
    }


def assert_monotone(name: str, values: list[float], increasing: bool) -> None:
    order = "increasing" if increasing else "decreasing"
    print(f"\n{name}: {[round(v, 4) for v in values]} (expected {order})")
    diffs = np.diff(values)
    if increasing:
        assert (diffs >= -1e-9).all(), f"{name} is not monotone {order}: {values}"
    else:
        assert (diffs <= 1e-9).all(), f"{name} is not monotone {order}: {values}"
    assert abs(values[-1] - values[0]) > 1e-6, f"{name} did not move at all: {values}"


def test_style_dim_and_neutral_layout():
    assert gd.STYLE_DIM == 18
    neutral = gd.StyleParams.neutral().to_array()
    assert neutral.dtype == np.float32
    assert neutral.shape == (gd.STYLE_DIM,)
    assert neutral[gd.STYLE_BOMB_THRESHOLD] == 1.0
    assert neutral[gd.STYLE_PARTNER_WEIGHT] == 1.0
    assert neutral[gd.STYLE_FOLLOW_AGGRESSION] == 0.0
    assert neutral[gd.STYLE_LEAD_HIGH_BIAS] == 0.0
    assert neutral[gd.STYLE_TEMPERATURE] == 0.0
    # The per-type block has one slot per play type, in enum order.
    assert gd.STYLE_FOLLOW_AGGRESSION - gd.STYLE_TYPE_PREF == 13


def test_bomb_threshold_moves_bomb_timing():
    """Lower threshold bombs earlier, so more cards are still in hand."""
    fractions, counts = [], []
    for value in (1.0, 0.75, 0.5, 0.25, 0.0):
        styles = neutral_styles()
        styles[:, :, gd.STYLE_BOMB_THRESHOLD] = value
        stats = run_styled(styles, seed=101)
        fractions.append(stats["bomb_fraction"])
        counts.append(stats["bombs"])
    print(f"\nbomb_threshold 1.0 -> 0.0, bombs played: {counts}")
    # Threshold 1 is the greedy rule and bombs only at the very end.
    assert_monotone("mean fraction of hand remaining when bombing", fractions,
                    increasing=True)
    assert counts[-1] > counts[0]


def test_follow_aggression_moves_following_rank():
    ranks = []
    for value in (0.0, 0.25, 0.5, 0.75, 1.0):
        styles = neutral_styles()
        styles[:, :, gd.STYLE_FOLLOW_AGGRESSION] = value
        ranks.append(run_styled(styles, seed=202)["mean_follow_rank"])
    assert_monotone("mean rank of following plays", ranks, increasing=True)


def test_lead_high_bias_moves_lead_rank():
    ranks = []
    for value in (-1.0, -0.5, 0.0, 0.5, 1.0):
        styles = neutral_styles()
        styles[:, :, gd.STYLE_LEAD_HIGH_BIAS] = value
        ranks.append(run_styled(styles, seed=303)["mean_lead_rank"])
    assert_monotone("mean lead rank", ranks, increasing=True)


def test_type_preference_moves_lead_type_frequency():
    """One slot of the per-type block: leading pairs."""
    slot = gd.STYLE_TYPE_PREF + 2         # Type::Pair is enum index 2
    fractions = []
    for value in (-2.0, -1.0, 0.0, 1.0, 2.0):
        styles = neutral_styles()
        styles[:, :, slot] = value
        stats = run_styled(styles, seed=404)
        fractions.append(stats["lead_type_fraction"].get("Pair", 0.0))
        print(f"  pref {value:+.1f} lead histogram: "
              + ", ".join(f"{t}={f:.3f}" for t, f in
                          sorted(stats["lead_type_fraction"].items())))
    assert_monotone("fraction of leads that are pairs", fractions, increasing=True)


def test_neutral_styles_reproduce_greedy_choice():
    env = gd.VecEnv(num_envs=64, num_threads=THREADS, seed=11, encode=False)
    env.reset()
    env.set_styles(neutral_styles(64))
    compared = 0
    for _ in range(400):
        batch = env.pending()
        greedy = np.asarray(batch.greedy_choice)
        styled = np.asarray(batch.styled_choice)
        assert np.array_equal(greedy, styled), "neutral style differs from greedy"
        compared += batch.rows
        env.step(np.ascontiguousarray(greedy, dtype=np.int32))
        env.drain_finished_rounds()
    assert compared > 5000
    print(f"\nneutral styles matched greedy on {compared} pending rows")


def test_unset_styles_give_greedy_choice():
    env = gd.VecEnv(num_envs=32, num_threads=2, seed=5, encode=False)
    env.reset()
    for _ in range(50):
        batch = env.pending()
        assert np.array_equal(np.asarray(batch.greedy_choice),
                              np.asarray(batch.styled_choice))
        env.step(np.ascontiguousarray(batch.greedy_choice, dtype=np.int32))
        env.drain_finished_rounds()


def test_non_neutral_styles_differ_on_some_rows():
    styles = neutral_styles(64)
    styles[:, :, gd.STYLE_BOMB_THRESHOLD] = 0.0
    styles[:, :, gd.STYLE_LEAD_HIGH_BIAS] = 1.0
    styles[:, :, gd.STYLE_FOLLOW_AGGRESSION] = 1.0
    styles[:, :, gd.STYLE_PARTNER_WEIGHT] = 0.0
    env = gd.VecEnv(num_envs=64, num_threads=THREADS, seed=11, encode=False)
    env.reset()
    env.set_styles(styles)
    differing = total = 0
    for _ in range(200):
        batch = env.pending()
        greedy = np.asarray(batch.greedy_choice)
        styled = np.asarray(batch.styled_choice)
        differing += int((greedy != styled).sum())
        total += batch.rows
        env.step(np.ascontiguousarray(styled, dtype=np.int32))
        env.drain_finished_rounds()
    print(f"\nstyled differed from greedy on {differing} of {total} rows")
    assert differing > 0


def test_clear_styles_returns_to_greedy():
    env = gd.VecEnv(num_envs=16, num_threads=2, seed=9, encode=False)
    env.reset()
    styles = neutral_styles(16)
    styles[:, :, gd.STYLE_LEAD_HIGH_BIAS] = 1.0
    env.set_styles(styles)
    env.pending()
    env.clear_styles()
    for _ in range(20):
        batch = env.pending()
        assert np.array_equal(np.asarray(batch.greedy_choice),
                              np.asarray(batch.styled_choice))
        env.step(np.ascontiguousarray(batch.greedy_choice, dtype=np.int32))
        env.drain_finished_rounds()


def test_set_styles_validates_shape():
    env = gd.VecEnv(num_envs=8, num_threads=1, seed=1, encode=False)
    env.reset()
    with pytest.raises(Exception):
        env.set_styles(np.zeros((8, 4, gd.STYLE_DIM + 1), dtype=np.float32))
    with pytest.raises(Exception):
        env.set_styles(np.zeros((7, 4, gd.STYLE_DIM), dtype=np.float32))
    with pytest.raises(Exception):
        env.set_styles(np.zeros((8, 4), dtype=np.float32))


def test_styled_choice_is_deterministic_under_temperature():
    styles = neutral_styles(32)
    styles[:, :, gd.STYLE_TEMPERATURE] = 0.8
    picks = []
    for _ in range(2):
        env = gd.VecEnv(num_envs=32, num_threads=3, seed=77, encode=False)
        env.reset()
        env.set_styles(styles)
        trace = []
        for _ in range(60):
            batch = env.pending()
            choice = np.ascontiguousarray(batch.styled_choice, dtype=np.int32)
            trace.append(choice.copy())
            env.step(choice)
            env.drain_finished_rounds()
        picks.append(np.concatenate(trace))
    assert np.array_equal(picks[0], picks[1])
    # Temperature really does sample: some choice must differ from the argmax.
    cold = neutral_styles(32)
    env = gd.VecEnv(num_envs=32, num_threads=3, seed=77, encode=False)
    env.reset()
    env.set_styles(cold)
    cold_trace = []
    for _ in range(60):
        batch = env.pending()
        choice = np.ascontiguousarray(batch.styled_choice, dtype=np.int32)
        cold_trace.append(choice.copy())
        env.step(choice)
        env.drain_finished_rounds()
    assert not np.array_equal(picks[0], np.concatenate(cold_trace))


def test_engine_styled_matches_greedy_at_neutral():
    engine = gd.Engine()
    state = gd.MatchState()
    engine.new_match(state, 4242)
    neutral = gd.StyleParams.neutral().to_array()
    checked = 0
    seed = 4242
    for _ in range(2000):
        if state.phase == gd.Phase.RoundEnd:
            engine.end_round(state)
            if state.winner >= 0:
                seed += 1
                engine.new_match(state, seed)
            else:
                engine.begin_round(state)
            continue
        cands = engine.legal_actions(state)
        if not cands:
            seed += 1
            engine.new_match(state, seed)
            continue
        assert engine.styled(state, neutral, 0) == engine.greedy(state, 0)
        checked += 1
        engine.apply(state, cands[engine.greedy(state, 0)])
    assert checked > 1000
    print(f"\nEngine.styled matched Engine.greedy on {checked} decisions")
