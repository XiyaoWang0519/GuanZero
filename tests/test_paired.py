"""eval/paired.py and the per-deal records of eval/danlm/arena.py (no DanLM needed)."""
import json

import numpy as np
import pytest
import torch

from eval import lineage_eval, paired
from eval.danlm.arena import Divergences, LockstepRound, score_duplicate
from train.history_model import (HistoryPolicyConfig, checkpoint_payload, fresh_player,
                                 save_history_checkpoint)

TINY = HistoryPolicyConfig(width=16, layers=1, heads=2, action_width=16, fusion_width=16,
                           critic_width=16, critic_layers=1)


def leg(deal: int, side: str, reward: int | None, first: int = 0, failure: str | None = None,
        tribute: str = "none") -> dict:
    """One arena leg record; ``reward`` is our team's, ``first`` the first finisher."""
    ours = (0, 2) if side == "a" else (1, 3)
    rewards = [0] * 4
    if reward is not None:
        for seat in range(4):
            rewards[seat] = reward if seat in ours else -reward
    counts = {"our_candidate_unmatched": 2}
    if failure:
        counts[failure] = 1
    return {"id": f"deal-{deal}/{side}", "status": "mirror_failed" if failure else "ok",
            "seats": [], "finish_order": [first, (first + 1) % 4, (first + 2) % 4, (first + 3) % 4],
            "rewards": rewards, "our_order": None, "our_returns": None, "decisions": 50,
            "tribute": tribute, "finish_snapshots": [],
            "failure": {"kind": failure, "detail": "x", "decision": 7, "trick_reading": None}
            if failure else None,
            "divergences": {"counts": counts, "samples": {}}}


def arena_report(legs: dict[int, tuple], name: str, **identity) -> dict:
    """An arena duplicate report from {deal: (reward_a, reward_b, failure_a, failure_b)}."""
    records = []
    for deal, (ra, rb, fa, fb) in legs.items():
        records.append(leg(deal, "a", None if fa else ra, first=0 if (ra or 0) > 0 else 1,
                           failure=fa))
        records.append(leg(deal, "b", None if fb else rb, first=1 if (rb or 0) > 0 else 0,
                           failure=fb))
    report = {"mode": "duplicate", "checkpoint": name, "checkpoint_name": name,
              "danlm": "ckpts/x.pt", "seed": 5, "deals": len(legs), "tribute_fraction": 0.5,
              "deal_set_sha256": "d" * 64}
    report.update(identity)
    report.update(score_duplicate(records, len(legs), 5, 200))
    return report


def test_score_duplicate_keeps_every_deal_and_its_failures():
    report = arena_report({0: (2, -1, None, None), 1: (1, 3, None, None),
                           2: (1, 1, "their_choice_missing", None)}, "m")
    assert report["scored_deals"] == 2 and report["excluded_deals_mirror_failed"] == 1
    assert report["excluded_deal_ids"] == [2]
    assert report["failed_legs_by_kind"] == {"their_choice_missing": 1}
    assert report["levels_per_round"] == pytest.approx(((2 - 1) / 2 + (1 + 3) / 2) / 2)
    assert report["round_win_rate"] == 3 / 4
    rows = report["per_deal"]
    assert [r["deal"] for r in rows] == [0, 1, 2]
    assert rows[0]["score"] == 0.5 and rows[0]["round_wins"] == 1
    assert rows[2]["scored"] is False and rows[2]["score"] is None
    assert rows[2]["legs"][0]["failure"]["kind"] == "their_choice_missing"
    assert rows[2]["legs"][1]["reward"] == 1 and rows[2]["legs"][1]["status"] == "ok"
    assert rows[0]["legs"][0]["divergences"] == {"our_candidate_unmatched": 2}


def test_failure_record_names_the_reading_earlier_in_the_trick():
    lockstep = LockstepRound.__new__(LockstepRound)
    lockstep.div, lockstep.round_id, lockstep.decisions = Divergences(), "r", 12
    lockstep.trick_reading = "theirs seat 1 Single(H5) -> Single(H5)"
    assert lockstep.fail("their_choice_missing", "seat 2 Single(S6)") == "mirror_failed"
    assert lockstep.failure == {"kind": "their_choice_missing", "detail": "seat 2 Single(S6)",
                                "decision": 12,
                                "trick_reading": "theirs seat 1 Single(H5) -> Single(H5)"}
    assert lockstep.div.counts == {"their_choice_missing": 1}


def test_paired_difference_on_common_deals_with_exclusions():
    a = arena_report({0: (2, -1, None, None), 1: (1, 3, None, None), 2: (1, 1, None, None),
                      3: (-2, -2, None, "to_move_mismatch")}, "A")
    b = arena_report({0: (1, -1, None, None), 1: (1, 1, None, None),
                      2: (1, 1, "their_choice_missing", None), 3: (-1, -1, None, None)}, "B")
    result = paired.compare(paired.deal_scores(a), paired.deal_scores(b), seed=1, samples=500)
    assert result["common_deals"] == 2 and result["deals"] == 4
    # Deal 0: 0.5 - 0.0, deal 1: 2.0 - 1.0.
    assert result["difference"]["mean"] == pytest.approx(0.75)
    assert result["a_levels_per_round"] == pytest.approx(1.25)
    assert result["excluded"]["a_only"] == [3] and result["excluded"]["b_only"] == [2]
    assert result["excluded"]["reasons"]["2"] == {"a": None, "b": ["their_choice_missing"]}
    assert result["difference_bound_if_excluded_scored"] == pytest.approx(
        [(1.5 - 12) / 4, (1.5 + 12) / 4])
    lo, hi = result["difference"]["bootstrap_95_ci"]
    assert 0.5 <= lo <= hi <= 1.0
    assert result["deal_set_verified"] is True


def test_paired_refuses_different_deal_sets_and_old_reports():
    a = arena_report({0: (1, 1, None, None), 1: (1, 1, None, None)}, "A")
    b = arena_report({0: (1, 1, None, None), 1: (1, 1, None, None)}, "B", seed=6)
    with pytest.raises(ValueError, match="same deals"):
        paired.compare(paired.deal_scores(a), paired.deal_scores(b))
    old = dict(a)
    del old["per_deal"]
    with pytest.raises(ValueError, match="per_deal"):
        paired.deal_scores(old)


def test_paired_lineage_reports_against_one_baseline(tmp_path):
    torch.set_num_threads(1)
    paths = {}
    for name, seed in (("x", 1), ("y", 2), ("z", 3)):
        actor, critic = fresh_player(TINY, seed)
        paths[name] = tmp_path / f"{name}.pt"
        save_history_checkpoint(paths[name], checkpoint_payload(actor, critic, lineage="t",
                                                                seed=seed))
    outputs = {}
    for name in ("x", "y"):
        outputs[name] = tmp_path / f"{name}-vs-z.json"
        assert lineage_eval.main(["--candidate", str(paths[name]), "--baseline", str(paths["z"]),
                                  "--output", str(outputs[name]), "--deals", "12", "--seed", "4",
                                  "--batch-size", "6", "--threads", "1"]) == 0
    x, y = (json.loads(outputs[n].read_text()) for n in ("x", "y"))
    assert x["excluded_deals"] == 0 and x["deal_set_sha256"] == y["deal_set_sha256"]
    result_path = tmp_path / "x-minus-y.json"
    assert paired.main([str(outputs["x"]), str(outputs["y"]), "--output", str(result_path),
                        "--samples", "300"]) == 0
    result = json.loads(result_path.read_text())
    expected = np.array(x["duplicates"]["pair_scores"]) - np.array(y["duplicates"]["pair_scores"])
    assert result["common_deals"] == 12 and result["kind"] == "lineage_eval"
    assert result["difference"]["mean"] == pytest.approx(expected.mean())
    assert result["a_levels_per_round"] == pytest.approx(x["mean_net_levels_per_round"])
    assert result["difference_bound_if_excluded_scored"] == pytest.approx(
        [expected.mean(), expected.mean()])
    same = paired.compare(paired.deal_scores(x), paired.deal_scores(x))
    assert same["difference"]["mean"] == 0 and same["deals_with_different_score"] == 0
    # A run against another baseline is a different yardstick.
    other = dict(y, baseline_sha256="0" * 64)
    with pytest.raises(ValueError, match="same deals"):
        paired.compare(paired.deal_scores(x), paired.deal_scores(other))
