"""C3 v0 information boundary and bounded rollout checks."""
import random

import gd
import pytest

from eval.policies import GreedyPolicy
from eval.search import SearchConfig, SearchPolicy


def late_state(seed=33):
    engine = gd.Engine()
    state = gd.MatchState()
    engine.new_match(state, seed)
    for _ in range(200):
        assert state.phase == gd.Phase.Play
        unseen = 108 - len(state.hand(state.to_move)) - sum(len(state.played(s)) for s in range(4))
        if unseen <= 12 and len(engine.legal_actions(state)) >= 2:
            return engine, state
        actions = engine.legal_actions(state)
        engine.apply(state, actions[engine.greedy(state)])
    raise AssertionError("fixture did not reach a searchable endgame")


def test_sampler_conserves_cards_and_public_seat_sizes():
    _, state = late_state()
    original = state.serialize()
    seat = state.to_move
    for seed in range(20):
        sample = state.determinize_uniform(seat, seed)
        assert sample.hand(seat) == state.hand(seat)
        assert [len(sample.hand(s)) for s in range(4)] == [len(state.hand(s)) for s in range(4)]
        counts = [0] * 54
        for s in range(4):
            for card in sample.hand(s) + sample.played(s):
                counts[card] += 1
        assert counts == [2] * 54
    assert state.serialize() == original


def test_same_visible_information_same_sample_and_search_choice():
    engine, state = late_state()
    seat = state.to_move
    alternate_truth = state.determinize_uniform(seat, 991)
    assert state.observation(seat).tolist() == alternate_truth.observation(seat).tolist()
    for seed in range(10):
        a = state.determinize_uniform(seat, seed)
        b = alternate_truth.determinize_uniform(seat, seed)
        assert [a.hand(s) for s in range(4)] == [b.hand(s) for s in range(4)]
    config = SearchConfig(time_ms=2000, max_worlds=2, max_actions=3)
    choices = []
    for source in (state, alternate_truth):
        policy = SearchPolicy(GreedyPolicy(), config)
        actions = engine.legal_actions(source)
        choices.append(policy.select(engine, source, actions, random.Random(765)))
        assert policy.completed == 1 and policy.worlds_completed == [2]
    assert choices[0] == choices[1]


def test_clone_play_matches_serial_replay_and_time_fallback():
    engine, state = late_state()
    world = state.determinize_uniform(state.to_move, 73)
    copy = gd.MatchState.deserialize(world.serialize())
    for _ in range(50):
        if world.phase == gd.Phase.RoundEnd:
            break
        action = engine.legal_actions(world)[0]
        engine.apply(world, action)
        engine.apply(copy, action)
        assert world.hash() == copy.hash()
    actions = engine.legal_actions(state)
    policy = SearchPolicy(GreedyPolicy(), SearchConfig(time_ms=0.000001,
                                                       min_worlds=2, max_worlds=2))
    rng = random.Random(17)
    assert policy.select(engine, state, actions, rng) == engine.greedy(state)
    assert policy.fallbacks == 1 and policy.completed == 0


def test_custom_blueprint_cannot_inspect_live_hidden_hands():
    class CheatingPolicy(GreedyPolicy):
        pass

    with pytest.raises(TypeError, match="audited built-in"):
        SearchPolicy(CheatingPolicy())


@pytest.mark.parametrize("field,value", [
    ("time_ms", float("inf")), ("time_ms", float("nan")),
    ("prior_blueprint", float("inf")), ("kl_temperature", float("inf")),
    ("value_margin", float("nan")),
])
def test_nonfinite_search_parameters_are_rejected(field, value):
    with pytest.raises(ValueError, match="invalid search configuration"):
        SearchConfig(**{field: value})
