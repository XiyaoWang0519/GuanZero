"""Stratified entropy diagnostic: strata, lead/follow split and bounds."""
import math

import numpy as np

from eval import history_entropy
from train.history_ppo import HistoryPPOConfig, HistoryTrainer


def test_random_policy_entropy_is_bounded_and_stratified(tmp_path):
    trainer = HistoryTrainer(HistoryPPOConfig(width=32, layers=1, heads=4, num_envs=4,
                                              steps_per_update=16, seed=2, updates=0), tmp_path)
    path = trainer.save(tmp_path / "initial.pt")
    report = history_entropy.measure(path, envs=4, steps=48)
    assert report["all"]["decisions"] > 0 and report["single_candidate_excluded"] >= 0
    assert report["lead"]["decisions"] + report["follow"]["decisions"] == report["all"]["decisions"]
    assert sum(s["decisions"] for s in report["strata"].values()) == report["all"]["decisions"]
    two = report["strata"]["follow_n2"]
    if two["decisions"]:
        assert 0 < two["mean_entropy"] <= math.log(2) + 1e-9
    assert 0 < report["all"]["mean_normalized_entropy"] <= 1 + 1e-9


def test_summary_excludes_single_candidates_and_counts_near_deterministic():
    values = dict(entropy=np.array([0.0, 0.01, 0.69, 1.0]), top=np.array([1.0, 0.999, 0.5, 0.4]),
                  count=np.array([1, 2, 2, 4]), follow=np.array([True, True, True, False]))
    summary = history_entropy.summarize(values)
    assert summary["single_candidate_excluded"] == 1
    assert summary["follow"]["decisions"] == 2 and summary["lead"]["decisions"] == 1
    assert summary["follow"]["near_deterministic_fraction"] == 0.5
    assert summary["strata"]["lead_n3-5"]["decisions"] == 1
