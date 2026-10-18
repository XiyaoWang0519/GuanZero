"""Whole-round habit value: team rollout parity, planner choice and fork logic."""
import numpy as np

from bench.history_habit_round import choose, play_deal, team_rollout
from bench.history_value_tasks import rollout
from tests.test_history_habit_value import styled_pair
from tests.test_history_value_tasks import fixture, small_actor


def test_team_rollout_matches_the_seat_zero_reference():
    actor = small_actor()
    state, arrays, cand = fixture()
    pair = styled_pair(actor, 'pass', 2.0)
    seeds = np.array([5, 6, 7])
    np.testing.assert_array_equal(team_rollout(actor, pair, state, arrays, cand, seeds),
                                  rollout(actor, pair, state, arrays, cand, seeds))


def test_choose_keeps_blueprint_unless_margin_is_cleared():
    q = np.zeros((2, 3, 4, 2))
    q[0, 1, :, 0] = .015          # below margin under type 0
    q[1, 2, :, 0] = .5            # large gain under type 1 only
    assert choose(q, np.array([1., 0.]), .02) == 0
    assert choose(q, np.array([0., 1.]), .02) == 2
    assert choose(q, np.array([.5, .5]), .02) == 2   # .25 average still clears it
    q[1, 2, :, 0] = .03
    assert choose(q, np.array([.5, .5]), .02) == 0   # .015 average does not


def test_identical_types_never_fork_and_give_zero_difference():
    actor = small_actor()
    pair = styled_pair(actor, 'pass', 0.0)    # b = 0: both "types" are the base model
    for z in (0, 1):
        r = play_deal(actor, pair, deal_seed=101 + z, z=z, candidates=2, worlds=2, margin=0., seed=3)
        assert not r['forked'] and r['disagreements'] == 0
        assert r['oracle'] == r['blind'] and r['searches'] > 0


def test_distinct_types_play_a_complete_deal():
    actor = small_actor()
    pair = styled_pair(actor, 'pass', 8.0)
    r = play_deal(actor, pair, deal_seed=7, z=1, candidates=3, worlds=2, margin=0., seed=3)
    assert set(r['oracle']) == {'win', 'ret', 'searches'} and set(r['blind']) == {'win', 'ret', 'searches'}
    if not r['forked']:
        assert r['oracle'] == r['blind']


def test_trigger_skips_search_until_a_hand_is_small():
    actor = small_actor()
    pair = styled_pair(actor, 'pass', 8.0)
    never = play_deal(actor, pair, deal_seed=9, z=0, candidates=2, worlds=2, margin=0., seed=3, trigger=-1)
    assert never['searches'] == 0 and not never['forked'] and never['oracle'] == never['blind']
    late = play_deal(actor, pair, deal_seed=9, z=0, candidates=2, worlds=2, margin=0., seed=3, trigger=10)
    assert late['oracle']['searches'] > 0 and late['blind']['searches'] > 0
