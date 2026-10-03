"""eval/lineage_eval.py: a checkpoint against itself scores exactly zero; output fields."""
import json

import torch

from eval import lineage_eval
from train.history_model import (HistoryPolicyConfig, checkpoint_payload, fresh_player,
                                 save_history_checkpoint)

TINY = HistoryPolicyConfig(width=16, layers=1, heads=2, action_width=16, fusion_width=16,
                           critic_width=16, critic_layers=1)


def test_identical_checkpoints_score_zero_and_report_fields(tmp_path):
    torch.set_num_threads(1)
    actor, critic = fresh_player(TINY, 5)
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    save_history_checkpoint(a, checkpoint_payload(actor, critic, lineage="t", seed=5))
    save_history_checkpoint(b, checkpoint_payload(actor, critic, lineage="t", seed=5))
    out = tmp_path / "out.json"
    assert lineage_eval.main(["--candidate", str(a), "--baseline", str(b), "--output", str(out),
                              "--deals", "6", "--seed", "3", "--batch-size", "4", "--threads", "1"]) == 0
    report = json.loads(out.read_text())
    assert report["mean_net_levels_per_round"] == 0.0 and report["deals"] == 6
    assert report["candidate_sha256"] == report["baseline_sha256"]
    assert len(report["duplicates"]["pair_scores"]) == 6
    assert report["evaluator"]["backend"] == "batched" and report["evaluator"]["kv_cache"] is True
