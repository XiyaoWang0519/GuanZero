"""Stage C C1(a): the candidate pruning audit."""
import gd
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eval.pruning_audit import (BOMB_SIZES, TYPE_NAMES, audit, segment_ranks)  # noqa: E402
from train.model import GuandanModel, ModelConfig  # noqa: E402
from train.policy import PolicyConfig, StageBPolicy  # noqa: E402

SMALL = ModelConfig(obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM, state_width=16, state_layers=1,
                    action_width=16, action_layers=1, fusion_width=16, fusion_layers=1)


def test_segment_ranks_match_brute_force_with_stable_ties():
    scores = torch.tensor([0.5, 2.0, 2.0, -1.0, 3.0, 1.0, 1.0, 1.0])
    offsets = torch.tensor([0, 4, 5, 8])
    ranks = segment_ranks(scores, offsets).tolist()
    expected = []
    for a, b in zip(offsets[:-1].tolist(), offsets[1:].tolist()):
        segment = scores[a:b].tolist()
        order = sorted(range(len(segment)), key=lambda i: (-segment[i], i))
        rank = [0] * len(segment)
        for r, i in enumerate(order):
            rank[i] = r
        expected.extend(rank)
    assert ranks == expected == [2, 0, 1, 3, 0, 0, 1, 2]


def test_audit_invariants_on_a_tiny_policy(tmp_path):
    torch.manual_seed(1)
    policy = StageBPolicy.from_model(GuandanModel(SMALL), PolicyConfig(top_k=6))
    with torch.no_grad():
        next(policy.net.parameters()).add_(0.3)  # net diverges from the reference
    report = audit(tmp_path / "none.pt", decisions=300, num_envs=8, threads=1,
                   seed=11, policy=policy, coverage_ks=(2, 6, 12, 1000))
    assert report["status"] == "complete" and report["play_decisions"] >= 300
    assert report["top_k"] == 6 and report["checkpoint_file_sha256"] is None
    # Coverage is monotone in k and total at a k above every candidate set.
    fractions = [report["coverage_of_full_argmax_at_top_k"][str(k)]["all"]["fraction"]
                 for k in (2, 6, 12, 1000)]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    # A divergent decision is exactly one whose full argmax is not covered at top_k.
    divergent = report["full_argmax_outside_pruned_set"]["all"]["fraction"]
    assert divergent == pytest.approx(1.0 - fractions[1])
    # The chosen action always lies inside the pruned set: rank < top_k, or pass.
    chosen = report["chosen_reference_rank"]["histogram"]
    assert all(int(r) < 6 or r == "0" for r in chosen) or report["by_type"]["Pass"]["chosen"] > 0
    assert sum(chosen.values()) == report["play_decisions"]
    # Group tables sum to the totals; pruned counts never exceed totals.
    groups = report["decisions_with_pruning"]
    assert sum(v["count"] for k, v in groups.items() if k != "all") == groups["all"]["count"]
    for name in TYPE_NAMES:
        row = report["by_type"][name]
        assert 0 <= row["pruned"] <= row["candidates"]
    assert report["by_type"]["Pass"]["pruned"] == 0
    assert report["by_type"]["Tribute"]["candidates"] == 0
    assert set(report["by_bomb_size"]) == {str(s) for s in BOMB_SIZES}
    sizes = report["candidate_set_size"]
    assert sizes["count"] == report["play_decisions"] and sizes["max"] >= 1
