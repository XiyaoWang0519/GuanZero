"""Stage A2 evaluation isolates tribute changes from frozen play strength."""
from collections import Counter
from dataclasses import asdict
import copy
import json

import gd
import pytest

torch = pytest.importorskip("torch")

from eval.policies import ModelPolicy, load_policy, model_digest
from eval.tribute import (assert_same_play_model, evaluate_tribute_duplicates,
                         generate_tribute_deals, main)
from train.ckpt import save_checkpoint
from train.model import GuandanModel, ModelConfig


@pytest.fixture
def models():
    torch.set_num_threads(1)
    torch.manual_seed(91)
    model = GuandanModel(ModelConfig(state_width=8, state_layers=1,
        action_width=8, action_layers=1, fusion_width=8, fusion_layers=1))
    return (ModelPolicy(copy.deepcopy(model), "learned", heuristic_tribute=False),
            ModelPolicy(copy.deepcopy(model), "heuristic"))


def _checkpoint(path, model, **metadata):
    payload = {"model_config": asdict(model.config), "model": model.state_dict(),
               "optimizer": {}, "config": {"seed": 11, "action_mode": "canonical"},
               "progress": {}, "rng": {}}
    save_checkpoint(path, {**payload, **metadata})
    return str(path)


def _deal(hands, previous=(0, 1, 2, 3)):
    deal = gd.DealSpec()
    deal.hands = [gd.cards(hand) for hand in hands]
    deal.level = 12
    deal.team_levels = [12, 12]
    deal.owner = -1
    deal.leader = -1
    deal.prev_order = list(previous)
    return deal


def test_generated_tribute_deals_are_reproducible_complete_and_unfiltered():
    a, b = generate_tribute_deals(100, 872), generate_tribute_deals(100, 872)
    anti, categories = 0, set()
    for left, right in zip(a, b):
        assert (left.hands, left.level, left.prev_order) == (right.hands, right.level, right.prev_order)
        assert left.leader == left.owner == -1
        assert sorted(left.prev_order) == list(range(4))
        assert all(len(hand) == 27 for hand in left.hands)
        assert Counter(card for hand in left.hands for card in hand) == Counter({c: 2 for c in range(54)})
        state = gd.MatchState()
        gd.Engine().set_deal(state, left)
        anti += state.phase == gd.Phase.Play
        categories.add((left.prev_order[0] % 2 == left.prev_order[1] % 2,
                        left.prev_order[0] % 2 == left.prev_order[3] % 2))
    assert 0 < anti < 100
    assert categories == {(True, False), (False, False), (False, True)}
    assert generate_tribute_deals(1, 873)[0].hands != a[0].hands
    with pytest.raises(ValueError, match="positive"):
        generate_tribute_deals(0, 0)


def test_checkpoint_marker_controls_tribute_behavior_and_identity(tmp_path, models):
    model = models[0].model
    path = tmp_path / "model.pt"
    baseline = load_policy(_checkpoint(path, model))
    assert baseline.heuristic_tribute and baseline.stage == "dmc"
    learned = load_policy(_checkpoint(path, model, stage="a2", tribute_policy="learned",
        base_checkpoint_id=baseline.checkpoint_id, base_training_seed=11,
        config={"seed": 22, "action_mode": "full"},
        tribute_fit={"dataset_provenance": {"seed": 33}}))
    assert not learned.heuristic_tribute and learned.stage == "a2"
    assert learned.name != baseline.name and "/tribute=learned" in learned.name
    assert learned.checkpoint_id == baseline.checkpoint_id == model_digest(model.state_dict())
    assert learned.training_seed == 22 and learned.base_training_seed == 11
    assert learned.action_mode == "full"
    assert learned.collection_seed == 33


def test_collection_seed_is_rejected_independently_of_optimizer_seed(tmp_path, models):
    model = models[0].model
    baseline = load_policy(_checkpoint(tmp_path / "base.pt", model))
    learned = load_policy(_checkpoint(tmp_path / "a2.pt", model, stage="a2", tribute_policy="learned",
        base_checkpoint_id=baseline.checkpoint_id, base_training_seed=11,
        config={"seed": 22}, tribute_fit={"dataset_provenance": {"seed": 33}}))
    assert learned.training_seed == 22 and learned.collection_seed == 33
    with pytest.raises(ValueError, match="collection seeds"):
        evaluate_tribute_duplicates(learned, baseline, generate_tribute_deals(1, 33), seed=33)


def test_library_checks_base_identity_even_when_shared_tensors_match(models):
    agent, baseline = models
    agent.base_checkpoint_id = "1" * 64
    baseline.checkpoint_id = "2" * 64
    with pytest.raises(ValueError, match="base checkpoint identity"):
        assert_same_play_model(agent, baseline)
    baseline.checkpoint_id = agent.base_checkpoint_id
    assert_same_play_model(agent, baseline)


def test_probe_note_matches_actual_tribute_policy(models):
    from eval.probes import evaluate_probes

    learned, baseline = models
    assert "learned tribute heads" in evaluate_probes(learned)["tribute_note"]
    assert "heuristic tribute" in evaluate_probes(baseline)["tribute_note"]


@pytest.mark.parametrize("metadata", [
    {"stage": "unknown"}, {"tribute_policy": "learned"},
    {"stage": "a2"}, {"stage": "a2", "tribute_policy": "unknown"},
    {"stage": "a2", "tribute_policy": "learned"},
    {"stage": "a2", "tribute_policy": "learned", "base_checkpoint_id": "bogus"},
    {"config": {"action_mode": "unrecognized"}},
])
def test_unsupported_or_incomplete_marker_is_rejected(tmp_path, models, metadata):
    with pytest.raises(ValueError):
        load_policy(_checkpoint(tmp_path / "invalid.pt", models[0].model, **metadata))


@pytest.mark.parametrize("prefix", ["state_tower.", "action_tower.", "phase_heads.3.", "hidden_head.", "finish_head."])
def test_backbone_assertion_rejects_every_nontribute_component(models, prefix):
    agent, baseline = models
    assert_same_play_model(agent, baseline)
    with torch.no_grad():
        next(parameter for name, parameter in agent.model.named_parameters()
             if name.startswith(prefix)).add_(1)
    with pytest.raises(ValueError, match="not frozen"):
        assert_same_play_model(agent, baseline)


def test_tribute_heads_can_change_but_play_mode_cannot(models):
    agent, baseline = models
    with torch.no_grad():
        for name, parameter in agent.model.named_parameters():
            if name.startswith(("phase_heads.1.", "phase_heads.2.")):
                parameter.add_(3)
    assert_same_play_model(agent, baseline)
    baseline.margin = 0.1
    with pytest.raises(ValueError, match="argmax"):
        assert_same_play_model(agent, baseline)
    baseline.margin = 0
    baseline.action_mode = "full"
    with pytest.raises(ValueError, match="action modes"):
        assert_same_play_model(agent, baseline)


def test_anti_tribute_only_reports_zero_without_false_payoff_claim(models):
    deal = _deal(["S3 D3", "S4 D4", "S6 D6", "HR HR S9"])
    report = evaluate_tribute_duplicates(*models, [deal], seed=73, bootstrap_samples=30)
    assert report["anti_tribute_deals"] == 1
    assert report["pair_scores"] == [0]
    assert report["bootstrap_95_ci"] == [0, 0]
    assert report["deals_with_actual_choices"] == report["deals_with_changed_choices"] == 0
    assert report["conclusion"] == "no_multi_candidate_tribute_decisions"
    assert not report["payoff_supported"]
    assert report["deployment_recommendation"] == "retain_heuristic"
    assert report["results"][0]["first"] == report["results"][0]["swapped"]
    assert all(not count["decisions"] for who in report["phase_counts"].values() for count in who.values())


def test_forced_exchange_counted_separately_from_real_choices(models):
    deal = _deal(["S3", "S4", "S6", "SK"])
    report = evaluate_tribute_duplicates(*models, [deal], seed=73, bootstrap_samples=30)
    assert report["anti_tribute_deals"] == 0
    assert report["deals_with_actual_choices"] == 0
    assert report["deals_with_changed_choices"] == 0
    for who in report["phase_counts"].values():
        for count in who.values():
            assert count == {"decisions": 1, "multiple_candidates": 0,
                             "single_candidate": 1, "different_from_heuristic": 0}


def test_duplicate_starts_from_exact_same_deal_and_swaps_assignment(monkeypatch, models):
    from eval import tribute
    from eval.duplicate import RoundScore

    deal = _deal(["S3 D3 S4", "HR S6 D6", "S7 D7 S8", "SB S9 D9"], (0, 2, 1, 3))
    observed = []

    def record(engine, state, agent, opponent, agent_team, seed):
        observed.append(([state.hand(seat) for seat in range(4)], state.phase,
                         state.to_move, agent_team, seed))
        return RoundScore((0, 1, 2, 3), 0, 2, (2, -2, 2, -2), 0), {
            "phase_counts": tribute._phase_counts(), "choices": []}

    monkeypatch.setattr(tribute, "play_tribute_leg", record)
    report = tribute.evaluate_tribute_duplicates(*models, [deal], 74, 30)
    assert observed[0][:3] == observed[1][:3] == (deal.hands, gd.Phase.Tribute, 1)
    assert [row[3:] for row in observed] == [(0, 74), (1, 74)]
    assert report["pair_scores"] == [0]
    assert report["breakdown"]["double"]["deals"] == 1


def test_actual_phase_counts_and_partner_exchange_are_reported(models):
    single = _deal(["S3 D3 S8", "S4 D4 S9", "ST SJ SQ SK SA DA", "S5 D5 S6"], (0, 1, 3, 2))
    double = _deal(["S3 D3 S4", "HR S6 D6", "S7 D7 S8", "SB S9 D9"], (0, 2, 1, 3))
    report = evaluate_tribute_duplicates(*models, [single, double], seed=73, bootstrap_samples=30)
    again = evaluate_tribute_duplicates(*models, [single, double], seed=73, bootstrap_samples=30)
    assert report == again
    assert report["breakdown"]["single_partner"]["deals"] == 1
    assert report["breakdown"]["double"]["deals"] == 1
    assert report["deals_with_actual_choices"] == 2
    assert sum(count["decisions"] for who in report["phase_counts"].values() for count in who.values()) == 12
    for phase in report["phase_counts"]["opponent"].values():
        assert phase["different_from_heuristic"] == 0
    assert report["conclusion"] == "insufficient_independent_deals"
    assert not report["payoff_supported"]


def test_evaluation_requires_independent_seed_and_previous_order(models):
    agent, opponent = models
    agent.training_seed, agent.base_training_seed = 11, 12
    opponent.training_seed = 13
    deals = generate_tribute_deals(1, 4)
    for seed in (11, 12, 13):
        with pytest.raises(ValueError, match="training seeds"):
            evaluate_tribute_duplicates(agent, opponent, deals, seed)
    deals[0].leader = 0
    with pytest.raises(ValueError, match="leader=-1"):
        evaluate_tribute_duplicates(agent, opponent, deals, 4)
    with pytest.raises(ValueError, match="at least one"):
        evaluate_tribute_duplicates(agent, opponent, [], 4)


def test_cli_reports_provenance_and_rejects_wrong_base(tmp_path, models):
    model = models[0].model
    base = _checkpoint(tmp_path / "base.pt", model)
    learned = _checkpoint(tmp_path / "a2.pt", model, stage="a2", tribute_policy="learned",
                          base_checkpoint_id=model_digest(model.state_dict()), base_training_seed=11,
                          config={"seed": 12, "action_mode": "canonical"})
    output = tmp_path / "eval.json"
    args = ["--agent", learned, "--opponent", base, "--deals", "2", "--seed", "44",
            "--bootstrap-samples", "30", "--output", str(output)]
    main(args)
    report = json.loads(output.read_text())
    assert report["protocol"] == "internal-house-tribute-duplicate"
    assert report["play_model_exactly_equal"]
    assert report["base_training_seed"] == 11 and report["agent_training_seed"] == 12
    assert report["duplicate"]["deals"] == 2 and report["duplicate"]["rounds"] == 4
    with torch.no_grad():
        next(model.parameters()).add_(1)
    _checkpoint(tmp_path / "base.pt", model)
    with pytest.raises(SystemExit):
        main(args)
