"""Evaluation contracts: paired units, fixed hands, legal probes and inference."""
from collections import Counter
from dataclasses import asdict
import json
import random

import gd
import numpy as np
import pytest

from eval.arena import evaluate_checkpoint_belief, evaluate_matches, main, wilson_interval
from eval.duplicate import (RoundScore, bootstrap_interval, evaluate_duplicates,
                            generate_deals, play_duplicate, play_round)
from eval.elo import expected_score, refresh_elo, update_elo
from eval.policies import GreedyPolicy, ModelPolicy, RandomPolicy, choose_action, load_policy
from eval.probes import build_probes, evaluate_probes


def test_fixed_deals_are_repeatable_complete_double_decks():
    first, again = generate_deals(6, seed=42), generate_deals(6, seed=42)
    for left, right in zip(first, again):
        assert left.hands == right.hands
        assert (left.level, left.leader) == (right.level, right.leader)
        assert all(len(hand) == 27 for hand in left.hands)
        assert Counter(card for hand in left.hands for card in hand) == Counter({c: 2 for c in range(54)})
        assert left.owner == -1
    assert generate_deals(1, seed=43)[0].hands != first[0].hands
    assert all(d.level == 12 for d in generate_deals(3, seed=1, level=12))


def test_duplicate_swaps_policies_without_rotating_deal(monkeypatch):
    from eval import duplicate

    deal = generate_deals(1, seed=19)[0]
    observed = []

    def record(engine, state, policies, seed):
        observed.append(([state.hand(seat) for seat in range(4)], state.to_move,
                         state.level, tuple(p.name for p in policies), seed))
        return RoundScore((0, 1, 2, 3), 0, 2, (2, -2, 2, -2), 0)

    monkeypatch.setattr(duplicate, "play_round", record)
    result = play_duplicate(deal, GreedyPolicy(), RandomPolicy(), seed=83)
    assert len(observed) == 2
    assert observed[0][:3] == observed[1][:3] == (deal.hands, deal.leader, deal.level)
    assert observed[0][3] == ("greedy", "random")
    assert observed[1][3] == ("random", "greedy")
    assert observed[0][4] == observed[1][4] == 83
    assert result.pair_difference == 0


def test_duplicate_self_play_cancels_exactly_and_order_reversal_negates():
    deals = generate_deals(4, seed=2)
    random_policy, greedy = RandomPolicy(), GreedyPolicy()
    for deal in deals:
        same = play_duplicate(deal, random_policy, random_policy, seed=55)
        assert same.first == same.swapped
        assert same.levels_per_round == 0
        ab = play_duplicate(deal, greedy, random_policy, seed=55)
        ba = play_duplicate(deal, random_policy, greedy, seed=55)
        assert ab.pair_difference == -ba.pair_difference
        assert ab.levels_per_round == ab.pair_difference / 2


def test_paired_bootstrap_and_report_use_deals_as_sampling_units():
    assert bootstrap_interval([2, 2, 2], samples=100) == (2, 2)
    assert bootstrap_interval([-3, -1, 0, 2, 3], seed=8, samples=500) == bootstrap_interval(
        [-3, -1, 0, 2, 3], seed=8, samples=500)
    report = evaluate_duplicates(RandomPolicy(), RandomPolicy(), generate_deals(4, seed=4),
                                 seed=3, bootstrap_samples=100)
    assert report["rounds"] == 8
    assert report["pair_scores"] == [0, 0, 0, 0]
    assert report["bootstrap_95_ci"] == [0, 0]
    assert report["banker_rate"] == 0.5
    assert report["double_win_rate"] == report["opponent_double_win_rate"]
    with pytest.raises(ValueError):
        bootstrap_interval([])


def test_match_wilson_extremes_and_known_midpoint():
    lower, upper = wilson_interval(50, 100)
    assert lower == pytest.approx(0.4038315304)
    assert upper == pytest.approx(0.5961684696)
    assert wilson_interval(0, 10)[0] == pytest.approx(0)
    assert wilson_interval(10, 10)[1] == pytest.approx(1)
    assert wilson_interval(10, 10)[0] < 1
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_full_matches_terminate_and_alternate_agent_seats():
    report = evaluate_matches(GreedyPolicy(), RandomPolicy(), count=4, seed=37)
    assert [r["agent_team"] for r in report["results"]] == [0, 1, 0, 1]
    assert [r["seed"] for r in report["results"]] == [37, 38, 39, 40]
    assert report["rounds"] == sum(r["rounds"] for r in report["results"])
    assert report["wins"] == sum(r["agent_won"] for r in report["results"])
    assert all(r["winner"] in (0, 1) and r["rounds"] > 0 for r in report["results"])
    assert 0 <= report["double_win_rate"] <= report["banker_rate"] <= 1
    with pytest.raises(RuntimeError, match="exceeded"):
        evaluate_matches(GreedyPolicy(), GreedyPolicy(), count=1, max_rounds=1)


def test_net_level_metric_uses_zero_return_for_failed_owned_a_round():
    deal = gd.DealSpec()
    deal.hands = [gd.cards(hand) for hand in ("S3", "S4", "S2 D2", "S5")]
    deal.level = 12
    deal.team_levels = [12, 5]
    deal.owner = 0
    deal.leader = 0
    state = gd.MatchState()
    engine = gd.Engine()
    engine.set_deal(state, deal)
    score = play_round(engine, state, (GreedyPolicy(), GreedyPolicy()))
    assert score.order == (0, 1, 3, 2)
    assert score.winning_team == 0 and score.gain == 1
    assert score.net_gain(0) == score.net_gain(1) == 0
    assert state.fails == [1, 0] and state.winner == -1


def test_all_seven_probe_groups_offer_nontrivial_legal_good_choices():
    probes = build_probes()
    assert {probe.group for probe in probes} == set(range(1, 8))
    assert len(probes) == 8
    for probe in probes:
        actions = probe.engine.legal_actions(probe.state)
        assert any(probe.good(action) for action in actions), probe.name
        assert any(not probe.good(action) for action in actions), probe.name
        seen = Counter(card for seat in range(4)
                       for card in probe.state.hand(seat) + probe.state.played(seat))
        assert max(seen.values()) <= 2
        for action in actions:
            copy = gd.MatchState.deserialize(probe.state.serialize())
            # Every supplied candidate is accepted from this exact position.
            probe.engine.apply(copy, action)


def test_probes_report_actual_failure_and_heuristic_tribute():
    report = evaluate_probes(GreedyPolicy(), repeats=2)
    cases = {case["name"]: case for case in report["cases"]}
    assert cases["let_partner_hold"]["pass_rate"] == 1
    assert cases["tribute_preserve_straight_flush"]["pass_rate"] == 1
    # This is an actual weakness of the current greedy bot. Keep the probe
    # independent of its implementation so training can reveal improvement.
    assert cases["opponent_one_cover_high"]["pass_rate"] == 0
    assert report["mean_group_pass_rate"] < 1
    assert "heuristic tribute" in report["tribute_note"]


def test_invalid_policy_choice_is_not_silently_clamped():
    class BadPolicy:
        name = "bad"

        def select(self, engine, state, actions, rng):
            return len(actions)

    probe = build_probes()[0]
    with pytest.raises(ValueError, match="outside legal set"):
        choose_action(BadPolicy(), probe.engine, probe.state, random.Random(0))
    with pytest.raises(RuntimeError, match="exceeded"):
        play_round(probe.engine, probe.state, (GreedyPolicy(), RandomPolicy()), max_decisions=0)


def test_elo_draws_conservation_and_duplicate_result_direction():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1900, 1500) == pytest.approx(10 / 11)
    assert update_elo(1500, 1500, 0.5) == (1500, 1500)
    win, loss = update_elo(1500, 1500, 1)
    assert win > 1500 > loss
    assert win + loss == 3000
    report = {"agent": "a", "opponent": "b", "duplicate": {"pair_scores": [3, 1, 0]}}
    ratings = refresh_elo([report])
    assert ratings["a"] > ratings["b"]
    assert sum(ratings.values()) == pytest.approx(3000)


def test_arena_cli_writes_self_contained_report(tmp_path):
    output = tmp_path / "arena.json"
    main(["--deals", "2", "--matches", "1", "--bootstrap-samples", "100",
          "--seed", "123", "--output", str(output)])
    report = json.loads(output.read_text())
    assert report["protocol"] == "internal-house"
    assert report["duplicate"]["deals"] == 2
    assert report["match"]["matches"] == 1
    assert len(report["probes"]["groups"]) == 7
    assert "not OpenGuanDan" in report["baseline_note"]


def test_model_policy_supplies_only_actor_observation_and_ragged_candidates():
    torch = pytest.importorskip("torch")

    class SpyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def score_candidates(self, obs, cand, offsets, phase):
            self.seen = (obs.clone(), cand.clone(), offsets.clone(), phase.clone())
            return torch.arange(len(cand), dtype=torch.float32, device=cand.device)

    model = SpyModel()
    policy = ModelPolicy(model)
    probe = build_probes()[0]
    actions = probe.engine.legal_actions(probe.state)
    chosen = policy.select(probe.engine, probe.state, actions, random.Random(0))
    assert chosen == len(actions) - 1
    obs, cand, offsets, phase = model.seen
    np.testing.assert_array_equal(obs.numpy()[0], probe.state.observation(probe.state.to_move))
    assert cand.shape == (len(actions), gd.ACT_DIM)
    assert offsets.tolist() == [0, len(actions)]
    assert phase.tolist() == [int(gd.Phase.Play)]
    assert not model.training
    soft_policy = ModelPolicy(model, margin=1.0)
    picks = [soft_policy.select(probe.engine, probe.state, actions, random.Random(seed))
             for seed in range(20)]
    assert set(picks) == {len(actions) - 2, len(actions) - 1}
    assert soft_policy.select(probe.engine, probe.state, actions, random.Random(8)) == picks[8]
    with pytest.raises(ValueError, match="margin"):
        ModelPolicy(model, margin=-1)
    tribute = build_probes()[-1]
    model.seen = None
    choice = policy.select(tribute.engine, tribute.state,
                           tribute.engine.legal_actions(tribute.state), random.Random(0))
    assert model.seen is None
    assert tribute.good(tribute.engine.legal_actions(tribute.state)[choice])


def test_training_checkpoint_loads_into_evaluation_policy(tmp_path):
    torch = pytest.importorskip("torch")
    from train.ckpt import rng_state, save_checkpoint
    from train.model import GuandanModel, ModelConfig

    config = ModelConfig(obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM,
                         state_width=8, state_layers=1, action_width=8,
                         action_layers=1, fusion_width=8, fusion_layers=1)
    model = GuandanModel(config)
    path = tmp_path / "agent.pt"
    save_checkpoint(path, {"model_config": asdict(config), "model": model.state_dict(),
                           "optimizer": torch.optim.Adam(model.parameters()).state_dict(),
                           "config": {}, "progress": {}, "rng": rng_state(np.random.default_rng(0))})
    policy = load_policy(str(path))
    assert policy.name.startswith("agent.pt@")
    assert load_policy(str(path)).name == policy.name
    for expected, restored in zip(model.parameters(), policy.model.parameters()):
        torch.testing.assert_close(expected, restored)
    probe = build_probes()[0]
    actions = probe.engine.legal_actions(probe.state)
    expected = ModelPolicy(model).select(probe.engine, probe.state, actions, random.Random(0))
    assert policy.select(probe.engine, probe.state, actions, random.Random(0)) == expected
    with torch.no_grad():
        next(model.parameters()).add_(1)
    save_checkpoint(path, {"model_config": asdict(config), "model": model.state_dict(),
                           "optimizer": {}, "config": {}, "progress": {}, "rng": {}})
    assert load_policy(str(path)).name != policy.name


def test_checkpoint_belief_metrics_from_real_engine_rounds(tmp_path):
    torch = pytest.importorskip("torch")
    from train.buffer import Decision
    from train.logs import save_round
    from train.model import GuandanModel, ModelConfig

    # Real states and privileged labels; two independent match groups permit
    # the exact same match split as the offline architecture experiment.
    logs = tmp_path / "logs"
    engine = gd.Engine()
    for seed in range(2):
        state = gd.MatchState()
        engine.new_match(state, seed + 11)
        records = []
        while state.phase != gd.Phase.RoundEnd:
            actions = engine.legal_actions(state)
            action = actions[engine.greedy(state)]
            seat = state.to_move
            records.append(Decision(
                obs=state.observation(seat).astype(np.uint8),
                action=engine.encode_action(action, state, seat).astype(np.uint8),
                hidden=np.asarray([np.bincount(state.hand((seat + rel) % 4), minlength=54)
                                   for rel in (1, 2, 3)], dtype=np.uint8),
                seat=seat, phase=int(state.phase), prefix=0,
            ))
            engine.apply(state, action)
        save_round(logs / f"round-{seed:08d}.npz", records, [], group=f"evaluation-match-{seed}")
    model = GuandanModel(ModelConfig(state_width=8, state_layers=1, action_width=8,
                                     action_layers=1, fusion_width=8, fusion_layers=1))
    with torch.no_grad():
        model.hidden_head.weight.zero_()
        model.hidden_head.bias.zero_()
    policy = ModelPolicy(model)
    report = evaluate_checkpoint_belief(policy, logs, batch_size=7)
    assert report["evaluated_matches"] == report["excluded_matches"] == 1
    assert report["evaluated_rounds"] == 1
    assert report["log_loss"] == pytest.approx(np.log(3), abs=1e-6)
    assert report["checkpoint_training_overlap"] == "not_verified"
    stages = report["by_stage_and_seat"]
    assert set(stages) == {"early", "middle", "late"}
    assert sum(stage["partner"]["decisions"] for stage in stages.values()) == report["evaluated_decisions"]
    for stage in stages.values():
        assert set(stage) == {"lho", "partner", "rho"}
        assert stage["partner"]["decisions"] > 0
        for metric in stage.values():
            assert metric["log_loss"] == pytest.approx(np.log(3), abs=1e-6)
    with pytest.raises(ValueError, match="checkpoint"):
        evaluate_checkpoint_belief(RandomPolicy(), logs)
    with pytest.raises(SystemExit) as error:
        main(["--agent", "greedy", "--belief-logs", str(logs), "--deals", "1"])
    assert error.value.code == 2


def test_independent_belief_collector_never_trains_and_preserves_groups(tmp_path):
    pytest.importorskip("torch")
    from eval.collect_belief import collect_belief
    from train.belief_probe import load_rounds
    from train.ckpt import save_checkpoint
    from train.model import GuandanModel, ModelConfig

    config = ModelConfig(state_width=8, state_layers=1, action_width=8,
                         action_layers=1, fusion_width=8, fusion_layers=1)
    model = GuandanModel(config)
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, {"model_config": asdict(config), "model": model.state_dict(),
                                 "optimizer": {}, "config": {"seed": 7}, "progress": {}, "rng": {}})
    original = checkpoint.read_bytes()
    output = tmp_path / "independent"
    result = collect_belief(checkpoint, output, rounds=2, num_envs=2, seed=99, max_seconds=30)
    assert result["purpose"] == "evaluation_only"
    assert result["status"] == "complete"
    assert result["learner_updates"] == 0
    assert result["collected_rounds"] == 2
    assert checkpoint.read_bytes() == original
    assert sorted(path.name for path in output.iterdir()) == [
        "matches.json", "provenance.json", "round-00000000.npz", "round-00000001.npz"]
    excluded, selected = load_rounds(output)
    assert len(excluded) == len(selected) == 1
    assert excluded[0]["group"] != selected[0]["group"]
    for record in excluded + selected:
        np.testing.assert_array_equal(record["hidden"].sum(1),
                                      record["obs"][:, 108:162] + record["obs"][:, 162:216])
        # A logged decision sees the history strictly before its own action.
        for seat, prefix in zip(record["seat"], record["prefix"]):
            assert record["tokens"][prefix, :4].argmax() == seat
    metrics = evaluate_checkpoint_belief(load_policy(str(checkpoint)), output)
    assert np.isfinite(metrics["log_loss"])
    assert metrics["checkpoint_training_overlap"] == "independent_frozen_checkpoint_collection"
    with pytest.raises(FileExistsError):
        collect_belief(checkpoint, output, rounds=2, num_envs=2, seed=99)
    with pytest.raises(ValueError, match="seed"):
        collect_belief(checkpoint, tmp_path / "same-seed", rounds=2, seed=7)
    incomplete = tmp_path / "timeout"
    with pytest.raises(TimeoutError):
        collect_belief(checkpoint, incomplete, rounds=2, seed=101, max_seconds=1e-12)
    assert json.loads((incomplete / "provenance.json").read_text())["status"] == "incomplete"
