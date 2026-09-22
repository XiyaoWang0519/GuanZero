"""Counterfactual truth, private-input boundaries and frozen-head optimization."""
from dataclasses import asdict
import json
from pathlib import Path
import random
from types import SimpleNamespace

import gd
import numpy as np
import pytest
import torch

from eval.duplicate import play_round
from eval.policies import GreedyPolicy, ModelPolicy, load_policy
from eval.probes import build_probes
from train.ckpt import load_checkpoint, rng_state, save_checkpoint
from train.dmc import TrainConfig, Trainer
from train.model import GuandanModel, ModelConfig
from train.tribute import HEAD_PREFIXES, fit, load_dataset, split_matches
from train.tribute_data import branch_returns, collect, engine_source_digest, label_position


def _base(path):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model = GuandanModel(ModelConfig(state_width=16, state_layers=1,
        action_width=8, action_layers=1, fusion_width=16, fusion_layers=1))
    save_checkpoint(path, {"model_config": asdict(model.config), "model": model.state_dict(),
        "optimizer": torch.optim.Adam(model.parameters()).state_dict(),
        "config": {"seed": 7, "action_mode": "canonical"}, "progress": {"updates": 1},
        "rng": rng_state(np.random.default_rng(7))})
    return load_policy(str(path))


@pytest.mark.parametrize("model_policy", [False, True])
def test_counterfactual_returns_match_serial_real_engine_and_keep_source(tmp_path, model_policy):
    probe = build_probes()[-1]
    engine, state = probe.engine, probe.state
    policy = _base(tmp_path / "base.pt") if model_policy else GreedyPolicy()
    source = state.serialize()
    actor = state.to_move
    expected = []
    for action in engine.legal_actions(state):
        branch = gd.MatchState.deserialize(source)
        engine.apply(branch, action)
        expected.append(play_round(engine, branch, (policy, policy), seed=82).seat_return[actor])
    case = label_position(engine, state, policy, seed=82, group="match-1")
    np.testing.assert_array_equal(case["returns"], expected)
    np.testing.assert_allclose(case["advantages"], np.asarray(expected) - np.mean(expected))
    assert state.serialize() == source
    assert case["obs"].shape == (gd.OBS_DIM,) and "hidden" not in case
    assert len(case["returns"]) == len(engine.legal_actions(state))
    np.testing.assert_array_equal(branch_returns(engine, state, policy, seed=82), expected)


def test_all_double_tribute_decisions_preserve_pending_choices_and_main_path():
    engine = gd.Engine()
    deal = gd.DealSpec()
    deal.hands = [gd.cards(s) for s in ["S3 D3 S8", "SK DK S4 D4", "C5 D5 C8", "SA DA S6 D6"]]
    deal.level = 5
    deal.team_levels = [5, 5]
    deal.owner = 0
    deal.prev_order = [0, 2, 1, 3]
    state = gd.MatchState()
    engine.set_deal(state, deal)
    untouched = gd.MatchState.deserialize(state.serialize())
    phases = []
    policy = GreedyPolicy()
    while state.phase in (gd.Phase.Tribute, gd.Phase.BackTribute):
        source = state.serialize()
        case = label_position(engine, state, policy, seed=91)
        assert state.serialize() == source
        assert len(case["returns"]) == len(engine.legal_actions(state))
        phases.append(int(state.phase))
        engine.apply(state, engine.legal_actions(state)[engine.greedy(state)])
        engine.apply(untouched, engine.legal_actions(untouched)[engine.greedy(untouched)])
        assert state.serialize() == untouched.serialize()
    assert phases == [1, 1, 2, 2]


def test_label_uses_original_actors_seat_return_not_finish_award():
    probe = build_probes()[-1]
    # Real legal branches, with the engine's terminal zero-return convention
    # injected to isolate which target the labeler reads (gain is not return).
    class ZeroReturn:
        def __getattr__(self, name):
            return getattr(probe.engine, name)
        def end_round(self, state):
            probe.engine.end_round(state)
            return SimpleNamespace(seat_return=[0, 0, 0, 0], gain=3, winning_team=0)
    values = branch_returns(ZeroReturn(), probe.state, GreedyPolicy())
    np.testing.assert_array_equal(values, np.zeros(len(values)))


def test_hidden_other_hands_never_enter_exchange_actor_features():
    engine = gd.Engine()
    hands = ["S3 D3 S8", "S4 D4 S9", "S5 D5 S6", "ST SJ SQ SK SA DA"]
    cases = []
    for variant in (hands, [hands[1], hands[0], hands[2], hands[3]]):
        deal = gd.DealSpec()
        deal.hands = [gd.cards(s) for s in variant]
        deal.level = 5
        deal.team_levels = [5, 5]
        deal.prev_order = [0, 1, 2, 3]
        state = gd.MatchState()
        engine.set_deal(state, deal)
        assert state.to_move == 3
        cases.append(label_position(engine, state, GreedyPolicy()))
    np.testing.assert_array_equal(cases[0]["obs"], cases[1]["obs"])
    np.testing.assert_array_equal(cases[0]["cand"], cases[1]["cand"])


def _dataset(path, base_id):
    path.mkdir()
    probe = build_probes()[-1]
    cases = [label_position(probe.engine, probe.state, GreedyPolicy())]
    probe.engine.apply(probe.state, probe.engine.legal_actions(probe.state)[probe.engine.greedy(probe.state)])
    cases.append(label_position(probe.engine, probe.state, GreedyPolicy()))
    for group in range(5):
        for phase, case in enumerate(cases):
            np.savez_compressed(path / f"position-{group * 2 + phase:08d}.npz",
                                **{**case, "group": np.asarray(f"match-{group}")})
    provenance = {"purpose": "tribute_counterfactual_training", "status": "complete",
        "base_checkpoint_id": base_id, "positions": 10, "seed": 932,
        "play_mode": "fp32_argmax", "other_tribute": "heuristic", "action_mode": "canonical",
        "engine_source_sha256": engine_source_digest()}
    (path / "provenance.json").write_text(json.dumps(provenance))


def test_actual_fit_freezes_play_weights_and_tags_incompatible_optimizer(tmp_path):
    source = tmp_path / "base.pt"
    policy = _base(source)
    original = source.read_bytes()
    dataset = tmp_path / "dataset"
    _dataset(dataset, policy.checkpoint_id)
    report = fit(source, dataset, tmp_path / "fit", steps=30, batch_positions=4, seed=23)
    result = load_checkpoint(tmp_path / "fit/tribute.pt")
    assert report["frozen_weights_unchanged"]
    assert set(report["train_matches"]).isdisjoint(report["holdout_matches"])
    assert source.read_bytes() == original
    changed = []
    for name, value in policy.model.state_dict().items():
        if not torch.equal(value, result["model"][name]):
            changed.append(name)
            assert name.startswith(HEAD_PREFIXES)
    assert any(n.startswith("phase_heads.1.") for n in changed)
    assert any(n.startswith("phase_heads.2.") for n in changed)
    learned = load_policy(str(tmp_path / "fit/tribute.pt"))
    assert not learned.heuristic_tribute and learned.base_checkpoint_id == policy.checkpoint_id
    probe = build_probes()[0]
    a = policy.select(probe.engine, probe.state, probe.engine.legal_actions(probe.state), random.Random(3))
    b = learned.select(probe.engine, probe.state, probe.engine.legal_actions(probe.state), random.Random(3))
    assert a == b
    config = TrainConfig(model=asdict(policy.model.config), num_envs=2, num_threads=1,
                         log_envs=0, tensorboard=False, replay_capacity=2048)
    with pytest.raises(ValueError, match="Stage A checkpoint"):
        Trainer(config, tmp_path / "bad-resume", resume=tmp_path / "fit/tribute.pt")
    with pytest.raises(FileExistsError):
        fit(source, dataset, tmp_path / "fit", steps=1)


def test_rejects_incomplete_mismatched_corrupt_and_single_match_data(tmp_path):
    _dataset(tmp_path / "dataset", "a" * 64)
    with pytest.raises(ValueError, match="frozen base"):
        load_dataset(tmp_path / "dataset", "b" * 64)
    cases, _, _ = load_dataset(tmp_path / "dataset", "a" * 64)
    with pytest.raises(ValueError, match="distinct"):
        split_matches([{**c, "group": "only"} for c in cases])
    p = tmp_path / "dataset/provenance.json"
    manifest = json.loads(p.read_text())
    manifest["status"] = "incomplete"
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="complete"):
        load_dataset(tmp_path / "dataset", "a" * 64)
    manifest["status"] = "complete"
    manifest["engine_source_sha256"] = "b" * 64
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="encoder source"):
        load_dataset(tmp_path / "dataset", "a" * 64)


def test_fit_deadline_covers_loading_and_mutable_base_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "base.pt"
    policy = _base(source)
    _dataset(tmp_path / "dataset", policy.checkpoint_id)
    with pytest.raises(TimeoutError, match="walltime"):
        fit(source, tmp_path / "dataset", tmp_path / "timeout", max_seconds=1e-12)
    from train import tribute
    real_load = tribute.load_checkpoint
    def changed(*args, **kwargs):
        value = real_load(*args, **kwargs)
        value["model"]["phase_heads.3.1.bias"] += 1
        return value
    monkeypatch.setattr(tribute, "load_checkpoint", changed)
    with pytest.raises(ValueError, match="changed while loading"):
        fit(source, tmp_path / "dataset", tmp_path / "changed", steps=1)


def test_branch_bounds_and_stochastic_play_are_rejected(tmp_path):
    probe = build_probes()[-1]
    with pytest.raises(RuntimeError, match="decision bound"):
        branch_returns(probe.engine, probe.state, GreedyPolicy(), max_decisions=1)
    policy = _base(tmp_path / "base.pt")
    policy.margin = .01
    with pytest.raises(ValueError, match="argmax"):
        branch_returns(probe.engine, probe.state, policy)


def test_collection_timeout_marks_provenance_incomplete(tmp_path):
    source = tmp_path / "base.pt"
    _base(source)
    with pytest.raises(TimeoutError):
        collect(source, tmp_path / "data", positions=2, max_seconds=1e-12)
    assert json.loads((tmp_path / "data/provenance.json").read_text())["status"] == "incomplete"
    with pytest.raises(FileExistsError):
        collect(source, tmp_path / "data")
    with pytest.raises(ValueError, match="training seed"):
        collect(source, tmp_path / "other", seed=7)
