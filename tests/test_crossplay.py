"""STAGE_B_TODO B8x: per-seat cross-play lineups on tiny counts."""
from dataclasses import asdict
import json
from pathlib import Path
import random

import gd
import numpy as np
import pytest

from eval.duplicate import generate_deals, play_duplicate, play_duplicate_teams
from eval.policies import GreedyPolicy, load_policy


class RecordingPolicy:
    """Greedy play that records every seat it was asked to act for."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.seats: list[int] = []

    def select(self, engine, state, actions, rng: random.Random) -> int:
        self.seats.append(state.to_move)
        return engine.greedy(state)


def tiny_checkpoint(path: Path, seed: int = 0) -> Path:
    torch = pytest.importorskip("torch")
    from train.ckpt import rng_state, save_checkpoint
    from train.model import GuandanModel, ModelConfig

    torch.manual_seed(seed)
    config = ModelConfig(obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM, state_width=8, state_layers=1,
                         action_width=8, action_layers=1, fusion_width=8, fusion_layers=1)
    model = GuandanModel(config)
    save_checkpoint(path, {"model_config": asdict(config), "model": model.state_dict(),
                           "optimizer": torch.optim.Adam(model.parameters()).state_dict(),
                           "config": {"seed": seed}, "progress": {},
                           "rng": rng_state(np.random.default_rng(0))})
    return path


def test_partner_sits_two_seats_from_the_model_on_both_legs():
    """Replay each leg of a duplicate alone and check who acted where."""
    from eval.duplicate import play_round_seats

    deal = generate_deals(1, seed=4)[0]
    score = play_duplicate_teams(deal, *[(GreedyPolicy(), GreedyPolicy())] * 2, seed=7)
    x, p, r0, r1 = (RecordingPolicy(n) for n in ("x", "p", "r0", "r1"))
    for s, seating in enumerate(((x, r0, p, r1), (r0, x, r1, p))):
        for policy in (x, p, r0, r1):
            policy.seats.clear()
        engine = gd.Engine(gd.RuleConfig.house())
        state = gd.MatchState()
        engine.set_deal(state, deal)
        replay = play_round_seats(engine, state, seating, seed=7)
        # Recorders play greedy, so the leg replays the all-greedy duplicate.
        assert replay == (score.first, score.swapped)[s]
        assert set(x.seats) == {s} and set(p.seats) == {s + 2}
        assert set(r0.seats) == {1 - s} and set(r1.seats) == {3 - s}
    # And play_duplicate_teams itself uses exactly that seating.
    x, p, r0, r1 = (RecordingPolicy(n) for n in ("x", "p", "r0", "r1"))
    assert play_duplicate_teams(deal, (x, p), (r0, r1), seed=7) == score
    assert set(x.seats) == {0, 1} and set(p.seats) == {2, 3}
    # Leg 1 is played first: seats 0 then 1 for the model, 2 then 3 for its partner.
    assert x.seats == sorted(x.seats) and p.seats == sorted(p.seats)


def test_self_pair_equals_the_existing_duplicate_evaluation(tmp_path):
    from eval.crossplay import evaluate_lineup
    from eval.duplicate import evaluate_duplicates

    checkpoint = str(tiny_checkpoint(tmp_path / "x.pt"))
    deals = generate_deals(4, seed=21)
    for opponent in ("greedy", "styled:bomb-happy"):
        ours = evaluate_lineup((checkpoint, checkpoint), (opponent, opponent), 4, 21, 50)
        theirs = evaluate_duplicates(load_policy(checkpoint), load_policy(opponent), deals, 21, 50)
        assert ours["pair_scores"] == theirs["pair_scores"]
        assert ours["bootstrap_95_ci"] == theirs["bootstrap_95_ci"]
        assert ours["double_win_rate"] == theirs["double_win_rate"]
    greedy = GreedyPolicy()
    for deal in deals[:2]:
        assert (play_duplicate_teams(deal, (greedy, greedy), (greedy, greedy), 5)
                == play_duplicate(deal, greedy, greedy, 5))


def test_results_do_not_depend_on_worker_count(tmp_path):
    from eval.crossplay import run_crossplay

    x = str(tiny_checkpoint(tmp_path / "x.pt", seed=1))
    ref = str(tiny_checkpoint(tmp_path / "ref.pt", seed=2))
    runs = [run_crossplay(x, ("greedy", "m1"), ref, 5, 9, 20, workers=workers, log=lambda _: None)
            for workers in (1, 2)]
    for run in runs:
        for key in ("runtime", "host"):
            run.pop(key)
        for entry in [run["self_pair"], *(v[k] for v in run["partners"].values() for k in ("X+P", "P+P"))]:
            entry.pop("seconds")
    assert runs[0] == runs[1]


def test_crossplay_cli_smoke(tmp_path):
    from eval.crossplay import main

    x = tiny_checkpoint(tmp_path / "x.pt", seed=1)
    ref = tiny_checkpoint(tmp_path / "ref.pt", seed=2)
    extra = tiny_checkpoint(tmp_path / "extra.pt", seed=3)
    output = tmp_path / "crossplay.json"
    main(["--model", str(x), "--reference", str(ref), "--partners", "greedy", "styled:low-lead", "m1",
          "--extra-partners", str(extra), "--deals", "3", "--seed", "5", "--bootstrap-samples", "20",
          "--workers", "1", "--out", str(output)])
    report = json.loads(output.read_text())
    assert report["model"]["model_digest"] == load_policy(str(x)).checkpoint_id
    assert report["reference_team"]["model_digest"] == load_policy(str(ref)).checkpoint_id
    assert list(report["partners"]) == ["greedy", "styled:low-lead", "m1", str(extra)]
    self_pair = report["self_pair"]["mean_net_levels_per_round"]
    for spec, entry in report["partners"].items():
        xp, pp = entry["X+P"]["mean_net_levels_per_round"], entry["P+P"]["mean_net_levels_per_round"]
        assert entry["vs_self"]["mean"] == pytest.approx(xp - self_pair)
        assert entry["partner_lift"]["mean"] == pytest.approx(xp - pp)
        lo, hi = entry["vs_self"]["bootstrap_95_ci"]
        assert lo <= hi
        assert entry["X+P"]["deals"] == 3 and entry["X+P"]["rounds"] == 6
        assert 0 <= entry["X+P"]["double_win_rate"] <= 1
    # The reference playing itself in duplicate deals is exactly zero.
    assert report["partners"]["m1"]["P+P"]["mean_net_levels_per_round"] == 0
    assert report["runtime"]["lineups_played"] == 1 + 2 * 4
