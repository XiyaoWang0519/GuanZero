"""Unit checks for aux-loss balance screen helpers (no fixture required)."""
import torch

from bench.aux_loss_balance_screen import cosine, flat_encoder, summarize


def test_cosine_orthogonal_and_aligned():
    a = torch.tensor([1.0, 0.0], dtype=torch.float64)
    b = torch.tensor([0.0, 1.0], dtype=torch.float64)
    c = torch.tensor([2.0, 0.0], dtype=torch.float64)
    assert cosine(a, b) == 0.0
    assert abs(cosine(a, c) - 1.0) < 1e-12
    assert cosine(a, -c) == -1.0


def test_flat_encoder_respects_mask():
    params = [torch.ones(2), torch.ones(3)]
    grads = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])]
    vec = flat_encoder(grads, params, [True, False])
    assert vec.tolist() == [1.0, 2.0]


def test_summarize_flags_stable_next_conflict():
    def row(dom: float, cos: float) -> dict:
        return {
            "weighted_encoder_grad_norm": {
                "ppo": 1.0, "next": dom, "belief": 0.1, "outcome": 0.1},
            "dominance_vs_ppo": {
                "next_over_ppo": dom, "belief_over_ppo": 0.1, "outcome_over_ppo": 0.1},
            "pairwise_encoder_cosine": {
                a: {b: (cos if {a, b} == {"next", "ppo"} else 0.5)
                    for b in ("ppo", "next", "belief", "outcome")}
                for a in ("ppo", "next", "belief", "outcome")},
            "conflict": {
                "ppo_vs_next": cos < 0,
                "ppo_vs_belief": False,
                "ppo_vs_outcome": False,
                "next_vs_belief": False,
                "next_vs_outcome": False,
                "belief_vs_outcome": False,
            },
            "next_coef_ablations": [
                dict(next_coef=0.1, cosine_with_ppo=-0.5,
                     encoder_grad_norm=10.0, ppo_encoder_fraction=0.1),
                dict(next_coef=0.0, cosine_with_ppo=0.9,
                     encoder_grad_norm=1.1, ppo_encoder_fraction=0.9),
            ],
        }

    summary = summarize([row(50.0, -0.4), row(40.0, -0.2)])
    assert summary["next_dominates_and_conflicts_ppo"] is True
    assert summary["fixtures_next_conflicts_ppo"] == 2
    assert "warranted" in summary["recommendation"]
    assert summary["train_ablation_proposal"]["knob"] == "next_coef"

    # Majority conflict still warrants ablations even if one batch is aligned.
    mixed = summarize([row(50.0, -0.4), row(40.0, -0.2), row(30.0, 0.05)])
    assert mixed["next_dominates_and_conflicts_ppo"] is False
    assert mixed["next_ppo_conflict_rate"] == 2 / 3
    assert "warranted" in mixed["recommendation"]
