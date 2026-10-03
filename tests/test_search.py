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


def test_search_spec_takes_config_overrides():
    from eval.policies import load_policy

    assert load_policy("search:greedy").config == SearchConfig()
    tuned = load_policy("search@unseen_threshold=30,time_ms=250.0:greedy").config
    assert (tuned.unseen_threshold, tuned.time_ms) == (30, 250.0)
    assert tuned.max_worlds == SearchConfig().max_worlds
    for bad in ("search@no_such_field=1:greedy", "search@unseen_threshold=:greedy",
                "search@unseen_threshold=30:"):
        with pytest.raises(ValueError):
            load_policy(bad)


def test_history_search_isolates_branches_and_hidden_truth():
    import numpy as np
    import torch
    from eval.history_policy import HistoryPolicy, apply_and_observe, resolve_forced_passes
    from train.history_model import HistoryPolicyConfig, fresh_player

    torch.set_num_threads(1)
    actor, _ = fresh_player(HistoryPolicyConfig(width=16, layers=1, heads=2,
        action_width=16, fusion_width=16, critic_width=16, critic_layers=1), seed=3)
    base = HistoryPolicy(actor)
    search = SearchPolicy(base, SearchConfig(time_ms=10000, max_worlds=2, max_actions=3))
    search.start_match()
    engine, state = gd.Engine(), gd.MatchState()
    engine.auto_pass = False
    engine.new_match(state, 33)
    for _ in range(300):
        resolve_forced_passes(engine, state, [search])
        assert state.phase == gd.Phase.Play
        actions = engine.legal_actions(state)
        unseen = 108 - len(state.hand(state.to_move)) - sum(len(state.played(s)) for s in range(4))
        if unseen <= 12 and len(actions) >= 2:
            break
        apply_and_observe(engine, state, actions[engine.greedy(state)], [search])
    else:
        raise AssertionError('no late decision')
    before = [a.copy() for a in base.stream.arrays()]
    serialized = state.serialize()
    alternate = state.determinize_uniform(state.to_move, 991)
    choices = []
    for source in (state, alternate):
        choices.append(search.select(engine, source, engine.legal_actions(source), random.Random(765)))
        for expected, actual in zip(before, base.stream.arrays()):
            np.testing.assert_array_equal(expected, actual)
        assert state.serialize() == serialized
        assert engine.auto_pass is False
    assert search.completed == 2 and choices[0] == choices[1]
    # An expired budget also preserves the live stream and root decision.
    fallback = SearchPolicy(base, SearchConfig(time_ms=0.000001))
    assert fallback.select(engine, state, actions, random.Random(7)) == base.select(engine, state, actions, random.Random(7))
    assert fallback.fallbacks == 1
    for expected, actual in zip(before, base.stream.arrays()):
        np.testing.assert_array_equal(expected, actual)


def test_history_search_checkpoint_duplicate_lifecycle(tmp_path):
    import torch
    from eval.policies import load_policy
    from eval.duplicate import generate_deals, play_duplicate
    from train.history_model import HistoryPolicyConfig, fresh_player, checkpoint_payload, save_history_checkpoint

    torch.set_num_threads(1)
    actor, critic = fresh_player(HistoryPolicyConfig(width=16, layers=1, heads=2,
        action_width=16, fusion_width=16, critic_width=16, critic_layers=1), seed=3)
    path = tmp_path / 'history.pt'
    save_history_checkpoint(path, checkpoint_payload(actor, critic, lineage='test', seed=3))
    search = load_policy(f'search@time_ms=0.000001:{path}')
    opponent = load_policy(str(path))
    score = play_duplicate(generate_deals(1, 13)[0], search, opponent, 5)
    assert score.pair_difference == 0
    assert search.blueprint.matches_started == opponent.matches_started == 2
    assert search.blueprint.stream is not opponent.stream


@pytest.mark.parametrize('batch_size', [1, 5, 64])
def test_all_actions_batched_search_matches_scalar(batch_size):
    import numpy as np
    import torch
    from eval.history_policy import HistoryPolicy, apply_and_observe, resolve_forced_passes
    from train.history_model import HistoryPolicyConfig, fresh_player
    torch.set_num_threads(1)
    actor, _ = fresh_player(HistoryPolicyConfig(width=16, layers=1, heads=2,
        action_width=16, fusion_width=16, critic_width=16, critic_layers=1), seed=3)
    base = HistoryPolicy(actor)
    base.start_match()
    engine, state = gd.Engine(), gd.MatchState()
    engine.auto_pass = False
    engine.new_match(state, 33)
    for _ in range(300):
        resolve_forced_passes(engine, state, [base])
        actions = engine.legal_actions(state)
        unseen = 108-len(state.hand(state.to_move))-sum(len(state.played(s)) for s in range(4))
        if unseen <= 12 and len(actions) > 4:
            break
        apply_and_observe(engine, state, actions[engine.greedy(state)], [base])
    else:
        raise AssertionError('no wide late root')
    original = [a.copy() for a in base.stream.arrays()]
    results = []
    for size in (0, batch_size):
        policy = SearchPolicy(base, SearchConfig(max_actions=0, max_worlds=4, min_worlds=4,
            max_rollout_steps=0, time_ms=60000, selection='mean', rollout_batch_size=size))
        chosen = policy.select(engine, state, actions, random.Random(71))
        record = policy.decisions[0]
        assert record['candidates'] == record['legal_actions'] == len(actions)
        assert record['worlds'] == 4
        results.append((chosen, record['means']))
        for old, current in zip(original, base.stream.arrays()):
            np.testing.assert_array_equal(old, current)
        assert not engine.auto_pass
    assert results[0] == results[1]


def _tiny_belief_actor(seed=3):
    from train.history_model import HistoryPolicyConfig, fresh_player
    return fresh_player(HistoryPolicyConfig(width=16, layers=1, heads=2, action_width=16,
                                            fusion_width=16, critic_width=16, critic_layers=1,
                                            aux_heads="belief"), seed=seed)[0]


def test_belief_sampler_needs_a_belief_head_and_valid_config():
    from eval.history_policy import HistoryPolicy
    from train.history_model import HistoryPolicyConfig, fresh_player

    with pytest.raises(ValueError, match="invalid search configuration"):
        SearchConfig(sampler="posterior")
    with pytest.raises(ValueError, match="invalid search configuration"):
        SearchConfig(sampler="belief", belief_floor=1.5)
    with pytest.raises(TypeError, match="belief head"):
        SearchPolicy(GreedyPolicy(), SearchConfig(sampler="belief"))
    plain = fresh_player(HistoryPolicyConfig(width=16, layers=1, heads=2, action_width=16,
                                             fusion_width=16, critic_width=16, critic_layers=1), 3)[0]
    with pytest.raises(TypeError, match="belief head"):
        SearchPolicy(HistoryPolicy(plain), SearchConfig(sampler="belief"))
    SearchPolicy(HistoryPolicy(_tiny_belief_actor()), SearchConfig(sampler="belief"))


def test_belief_sampler_worlds_follow_the_head_and_stay_legal(monkeypatch):
    import numpy as np
    import torch
    from eval.history_policy import HistoryPolicy, apply_and_observe, resolve_forced_passes

    torch.set_num_threads(1)
    actor = _tiny_belief_actor()
    base = HistoryPolicy(actor)
    search = SearchPolicy(base, SearchConfig(sampler="belief", time_ms=10000, max_worlds=3,
                                             max_actions=2, min_worlds=1))
    search.start_match()
    engine, state = gd.Engine(), gd.MatchState()
    engine.auto_pass = False
    engine.new_match(state, 44)
    for _ in range(60):
        resolve_forced_passes(engine, state, [search])
        actions = engine.legal_actions(state)
        if len(actions) >= 2 and sum(len(state.played(s)) for s in range(4)) > 20:
            break
        apply_and_observe(engine, state, actions[engine.greedy(state)], [search])
    seat = state.to_move
    # Point the head at the truth: for every card, the relative seats holding its
    # unseen copies. A seat the head rules out never receives a copy; counts are conserved.
    truth = np.zeros((54, 3))
    for r in range(1, 4):
        for card in state.hand((seat + r) % 4):
            truth[card, r - 1] += 1
    logits = torch.where(torch.as_tensor(truth) > 0, 20.0, -20.0).float()[None]
    monkeypatch.setattr(actor, "belief_logits", lambda decision: logits)
    actions = engine.legal_actions(state)
    search._root_log_probs(engine, state, actions)
    assert search._world_weights is not None and len(search._world_weights) == 162
    # Per-card marginals carry no joint constraint, so a seat can fill up before
    # all its true cards are dealt and the remainder falls back to other seats;
    # the worlds must still be legal and place far more copies right than a
    # uniform shuffle does.
    def placed_right(world):
        hits = total = 0
        for r in range(1, 4):
            for card in world.hand((seat + r) % 4):
                hits += truth[card, r - 1] > 0
                total += 1
        return hits / total

    rng = random.Random(5)
    belief_scores, uniform_scores = [], []
    for _ in range(10):
        world = search.sample_world(state, seat, rng)
        assert world.hand(seat) == state.hand(seat)
        for r in range(1, 4):
            assert len(world.hand((seat + r) % 4)) == len(state.hand((seat + r) % 4))
        total = sorted(c for s in range(4) for c in world.hand(s)) + sorted(
            c for s in range(4) for c in world.played(s))
        assert total == sorted(c for s in range(4) for c in state.hand(s)) + sorted(
            c for s in range(4) for c in state.played(s))
        belief_scores.append(placed_right(world))
        uniform_scores.append(placed_right(state.determinize_uniform(seat, rng.getrandbits(64))))
    assert np.mean(belief_scores) > 0.85 > np.mean(uniform_scores) + 0.3
    # End to end: the root pass computes the weights and the search completes.
    config = SearchConfig(sampler="belief", trigger="hand", hand_threshold=27, time_ms=10000,
                          max_worlds=2, max_actions=2, min_worlds=1)
    search = SearchPolicy(base, config)
    search.start_match()
    search.select(engine, state, actions, random.Random(9))
    assert search.completed == 1 and search.decisions[-1]["sampler"] == "belief"
