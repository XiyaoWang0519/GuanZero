"""STAGE_B_TODO B0: fixed styled opponents and the baseline harness on tiny counts."""
from dataclasses import asdict
import json
from pathlib import Path
import random

import gd
import numpy as np
import pytest

from eval.duplicate import play_round
from eval.policies import GreedyPolicy, StyledPolicy, load_policy
from eval.probes import build_probes
from train.styles import FIXED_STYLES, fixed_style

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_style_vectors_change_only_the_documented_slots():
    neutral = gd.StyleParams.neutral().to_array()
    expected = {
        "bomb-happy": {gd.STYLE_BOMB_THRESHOLD: 0.0},
        "bomb-shy": {gd.STYLE_BOMB_THRESHOLD: 1.1},
        "high-lead": {gd.STYLE_LEAD_HIGH_BIAS: 1.0},
        # Type::Single is enum index 1.
        "low-lead": {gd.STYLE_LEAD_HIGH_BIAS: -1.0, gd.STYLE_TYPE_PREF + 1: 1.0},
    }
    assert set(FIXED_STYLES) == set(expected)
    for name, changes in expected.items():
        vector = fixed_style(name)
        assert vector.dtype == np.float32 and vector.shape == (gd.STYLE_DIM,)
        want = neutral.copy()
        for slot, value in changes.items():
            want[slot] = value
        np.testing.assert_array_equal(vector, want, err_msg=name)
        assert vector[gd.STYLE_TEMPERATURE] == 0.0
    with pytest.raises(ValueError):
        fixed_style("bomb-curious")


def test_styled_spec_loads_a_legal_deterministic_policy():
    policy = load_policy("styled:high-lead")
    assert isinstance(policy, StyledPolicy) and policy.name == "styled:high-lead"
    np.testing.assert_array_equal(policy.style, fixed_style("high-lead"))
    with pytest.raises(ValueError):
        load_policy("styled:nope")
    with pytest.raises(ValueError):
        StyledPolicy(np.zeros(3))
    for probe in build_probes():
        actions = probe.engine.legal_actions(probe.state)
        picks = {policy.select(probe.engine, probe.state, actions, random.Random(seed))
                 for seed in range(3)}
        assert len(picks) == 1 and 0 <= picks.pop() < len(actions)


def test_neutral_styled_policy_plays_exactly_like_greedy():
    engine = gd.Engine()
    neutral = StyledPolicy(gd.StyleParams.neutral().to_array(), "neutral")
    for seed in range(3):
        states = []
        for policies in ((neutral, neutral), (GreedyPolicy(), GreedyPolicy())):
            state = gd.MatchState()
            engine.new_match(state, seed)
            states.append(play_round(engine, state, policies, seed))
        assert states[0] == states[1]


def test_every_fixed_style_plays_differently_from_greedy():
    from eval.stage_b_baseline import style_divergence

    report = style_divergence(rounds=150, seed=0)
    assert report["decisions"] > 1000
    for name, entry in report["styles"].items():
        assert entry["differs"] > 0, name
    # The bomb styles only differ where the bomb gate binds; the lead styles
    # change many leads.
    rates = {name: entry["rate"] for name, entry in report["styles"].items()}
    assert rates["bomb-shy"] < rates["bomb-happy"]


def tiny_checkpoint(path: Path) -> Path:
    torch = pytest.importorskip("torch")
    from train.ckpt import rng_state, save_checkpoint
    from train.model import GuandanModel, ModelConfig

    config = ModelConfig(obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM, state_width=8, state_layers=1,
                         action_width=8, action_layers=1, fusion_width=8, fusion_layers=1)
    model = GuandanModel(config)
    save_checkpoint(path, {"model_config": asdict(config), "model": model.state_dict(),
                           "optimizer": torch.optim.Adam(model.parameters()).state_dict(),
                           "config": {"seed": 3}, "progress": {},
                           "rng": rng_state(np.random.default_rng(0))})
    return path


def test_baseline_cli_writes_every_pair_with_intervals(tmp_path):
    from eval.stage_b_baseline import OPPONENTS, main

    checkpoint = tiny_checkpoint(tmp_path / "final.pt")
    output = tmp_path / "baseline.json"
    main(["--checkpoint", str(checkpoint), "--deals", "2", "--matches", "1", "--seed", "5",
          "--bootstrap-samples", "20", "--throughput-seconds", "0", "--workers", "1",
          "--out", str(output)])
    report = json.loads(output.read_text())
    policy = load_policy(str(checkpoint))
    assert report["checkpoint"]["model_digest"] == policy.checkpoint_id
    assert report["seeds"]["deal_seed"] == 5
    assert report["requested"] == {"deals": 2, "matches": 1, "bootstrap_samples": 20}
    assert set(report["pairs"]) == set(OPPONENTS)
    for spec, pair in report["pairs"].items():
        dup, match = pair["duplicate"], pair["match"]
        assert dup["deals"] == 2 and dup["rounds"] == 4
        lo, hi = dup["bootstrap_95_ci"]
        assert lo <= dup["mean_net_levels_per_round"] <= hi
        assert match["matches"] == 1 and 0 <= match["wilson_95_ci"][0] <= match["wilson_95_ci"][1] <= 1
        assert "results" not in dup and "results" not in match
        if spec.startswith("styled:"):
            assert pair["opponent"]["style_vector"] == fixed_style(spec[7:]).tolist()
    # Same deterministic policy in both legs of every deal.
    assert report["pairs"]["self"]["duplicate"]["mean_net_levels_per_round"] == 0
    assert report["runtime"]["under_budget"] is True
    assert "dmc_throughput" not in report
    assert report["host"]["logical_cpus"] >= 1


def test_baseline_results_do_not_depend_on_worker_count(tmp_path):
    from eval.stage_b_baseline import run_baseline

    checkpoint = tiny_checkpoint(tmp_path / "final.pt")
    runs = [run_baseline(checkpoint, 5, 3, 9, 20, ("greedy", "styled:low-lead"),
                         workers=workers, log=lambda _: None) for workers in (1, 2, 1)]
    for spec in ("greedy", "styled:low-lead"):
        pairs = [run["pairs"][spec] for run in runs]
        for pair in pairs:
            pair.pop("seconds")
        assert pairs[0] == pairs[1] == pairs[2]
    assert runs[1]["runtime"]["workers"] == 2


def test_dmc_throughput_uses_the_real_trainer():
    pytest.importorskip("torch")
    from eval.stage_b_baseline import measure_dmc_throughput

    result = measure_dmc_throughput(ROOT / "train/configs/smoke.json", 30.0)
    assert result["updates"] >= 1 and result["decisions"] > 0
    assert result["rollout_decisions_per_second"] > 0 and result["updates_per_second"] > 0


def test_checkpoint_path_falls_back_to_the_main_checkout(tmp_path, monkeypatch):
    from eval import stage_b_baseline

    main = tmp_path / "main"
    worktree = main / ".claude" / "worktrees" / "agent-x"
    worktree.mkdir(parents=True)
    artifact = main / ".work" / "final.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"x")
    monkeypatch.setattr(stage_b_baseline, "ROOT", worktree)
    monkeypatch.chdir(worktree)
    assert stage_b_baseline.resolve_path(Path(".work/final.pt")) == artifact.resolve()
    with pytest.raises(FileNotFoundError):
        stage_b_baseline.resolve_path(Path(".work/missing.pt"))
