"""Small end-to-end check of the portable overnight evaluator."""
from dataclasses import asdict
import json

import gd
import torch

from bench.overnight_eval import main, paired_delta
from train.ckpt import save_checkpoint
from train.model import GuandanModel, ModelConfig
from train.styles import StyleSpace


def test_paired_delta_uses_deal_order():
    a = {"pair_scores": [2, -1, 0]}
    b = {"pair_scores": [1, -2, 3]}
    result = paired_delta(a, b, 5, 30)
    assert result["pair_differences"] == [1, 1, -3]
    assert result["mean_levels_per_round"] == -1 / 3


def test_tiny_checkpoint_evaluation_is_complete_and_replayable(tmp_path):
    torch.manual_seed(9)
    model_config = ModelConfig(obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM,
                               state_width=16, state_layers=1, action_width=16,
                               action_layers=1, fusion_width=16, fusion_layers=1)
    checkpoint = tmp_path / "tiny.pt"
    save_checkpoint(checkpoint, {"model_config": asdict(model_config),
                                 "model": GuandanModel(model_config).state_dict(),
                                 "optimizer": {}, "config": {}, "progress": {}, "rng": {}})
    output = tmp_path / "result.json"
    arguments = ["--candidate", str(checkpoint), "--control", str(checkpoint),
                 "--output", str(output), "--device", "cpu", "--deals", "2",
                 "--matches", "2", "--seed", "71", "--batch-size", "2",
                 "--engine-threads", "1", "--heldout-styles", "1",
                 "--bootstrap-samples", "30"]
    assert main(arguments) == 0
    first = json.loads(output.read_text())
    assert first["status"] == "complete"
    assert first["evaluation"]["backend"] == "batched"
    assert first["evaluation"]["torch_threads"] == 2
    assert first["candidate"]["model_digest"] == first["control"]["model_digest"]
    assert len(first["h2h"]["duplicate"]["pair_scores"]) == 2
    assert len(first["h2h"]["duplicate"]["results"]) == 2
    assert len(first["h2h"]["matches"]["results"]) == 2
    assert len(first["suite"]) == 6
    space = StyleSpace.default()
    heldout = first["suite"]["heldout:0"]
    assert space.in_heldout(heldout["style_vector"])
    for cell in first["suite"].values():
        assert len(cell["candidate"]["results"]) == 2
        assert len(cell["control"]["pair_scores"]) == 2
        assert len(cell["paired_delta"]["pair_differences"]) == 2
        assert cell["paired_delta"]["pair_differences"] == [0, 0]
    assert main(arguments) == 0
    second = json.loads(output.read_text())
    assert first["h2h"]["duplicate"]["pair_scores"] == second["h2h"]["duplicate"]["pair_scores"]
    assert first["suite"]["heldout:0"]["style_vector"] == second["suite"]["heldout:0"]["style_vector"]
