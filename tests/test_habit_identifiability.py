"""Ideal-observer identifiability: feature parity, exact LLR, match plumbing."""
import math

import gd
import numpy as np
import torch

from bench.habit_identifiability import feature, llr, play_match, tilt
from bench.history_value_tasks import stream_from
from train.history_habit import style_feature
from train.history_model import DecisionInputs, StreamBatch
from tests.test_history_value_tasks import fixture, small_actor


def test_feature_matches_the_training_style_feature():
    state, arrays, _ = fixture()
    engine = gd.Engine()
    legal = engine.legal_actions(state)
    sb = StreamBatch.from_arrays([arrays], 'cpu')
    inputs = DecisionInputs(sb, torch.zeros(1, dtype=torch.long), sb.lengths,
                            torch.tensor(np.asarray(state.observation(0))[None]), torch.tensor([0]),
                            torch.tensor(np.asarray([engine.encode_action(a, state, 0) for a in legal])),
                            torch.tensor([0, len(legal)]), one_decision_per_stream=True)
    for axis in ('pass', 'bomb', 'lead_single'):
        np.testing.assert_array_equal(feature(legal, axis), style_feature(inputs, axis).numpy())


def test_llr_equals_log_ratio_of_the_two_tilted_policies():
    p = np.array([.5, .2, .2, .1])
    f = np.array([1., 0., 1., 0.])
    for b in (.5, 2., 4.):
        plus, minus = tilt(p, f, b), tilt(p, f, -b)
        for a in range(4):
            assert math.isclose(llr(p, f, b, a), math.log(plus[a] / minus[a]), rel_tol=1e-12)
    assert llr(p, np.zeros(4), 3., 1) == 0.0     # no featured action: no evidence


def test_match_rows_have_valid_posteriors_and_round_resets():
    rows = play_match(small_actor(), 'pass', 4.0, 1, seed=11)
    assert rows and all(0 <= r['post_round'] <= 1 and 0 <= r['post_full'] <= 1 for r in rows)
    for r in rows:
        if r['opp_events'] == 0:
            assert r['post_round'] == .5          # no opponent move yet this round
    first = [r for r in rows if r['round'] == 0]
    assert all(math.isclose(r['post_round'], r['post_full']) for r in first)


def test_sigmoid_is_stable_for_extreme_evidence():
    from bench.habit_identifiability import sigmoid
    assert sigmoid(-1e4) == 0.0 and sigmoid(1e4) == 1.0 and sigmoid(0) == .5
