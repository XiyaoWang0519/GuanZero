"""train/checkpoint_average.py: uniform means, refusals, loading and no resume."""
import json

import pytest
import torch

from eval import lineage_eval
from eval.policies import load_policy
from train.checkpoint_average import average_checkpoints, average_state_dicts, main
from train.history_model import (HistoryPolicyConfig, checkpoint_payload, fresh_player,
                                 load_history_checkpoint, save_history_checkpoint)
from train.history_ppo import HistoryPPOConfig, HistoryTrainer

TINY = HistoryPolicyConfig(width=16, layers=1, heads=2, action_width=16, fusion_width=16,
                           critic_width=16, critic_layers=1)


def write(path, config=TINY, seed=1, lineage="t", updates=0, weights_seed=None):
    actor, critic = fresh_player(config, seed if weights_seed is None else weights_seed)
    save_history_checkpoint(path, checkpoint_payload(actor, critic, lineage=lineage, seed=seed,
                                                     progress={"updates": updates,
                                                               "decisions": 10 * updates}))
    return path


def test_average_is_the_uniform_mean_of_actor_and_critic(tmp_path):
    paths = [write(tmp_path / f"u{i}.pt", seed=i, updates=i) for i in (1, 2, 3, 4)]
    record = average_checkpoints(paths, tmp_path / "avg.pt")
    assert record["count"] == 4 and [x["updates"] for x in record["inputs"]] == [1, 2, 3, 4]
    loaded = [load_history_checkpoint(p)[2] for p in paths]
    actor, critic, payload = load_history_checkpoint(tmp_path / "avg.pt")
    for key, value in actor.state_dict().items():
        expected = torch.stack([p["model"][key].double() for p in loaded]).mean(0).float()
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    for key, value in critic.state_dict().items():
        expected = torch.stack([p["critic"][key].double() for p in loaded]).mean(0).float()
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert payload["optimizer"] == {} and payload["rng"] == {}
    assert payload["progress"]["updates"] == 4 and payload["lineage"] == "t"
    assert payload["averaged_from"]["method"] == "uniform"
    assert load_policy(str(tmp_path / "avg.pt")).stage == "history_ppo"


def test_average_of_equal_weights_plays_exactly_like_the_input(tmp_path):
    torch.set_num_threads(1)
    a = write(tmp_path / "a.pt", seed=1, weights_seed=7)
    b = write(tmp_path / "b.pt", seed=2, weights_seed=7)
    assert main([str(a), str(b), "--output", str(tmp_path / "avg.pt")]) == 0
    out = tmp_path / "eval.json"
    assert lineage_eval.main(["--candidate", str(tmp_path / "avg.pt"), "--baseline", str(a),
                              "--output", str(out), "--deals", "6", "--seed", "3",
                              "--batch-size", "6", "--threads", "1"]) == 0
    assert json.loads(out.read_text())["mean_net_levels_per_round"] == 0.0


def test_average_refuses_mismatched_or_repeated_inputs(tmp_path):
    a = write(tmp_path / "a.pt", seed=1)
    b = write(tmp_path / "b.pt", seed=2)
    with pytest.raises(ValueError, match="at least two"):
        average_checkpoints([a], tmp_path / "x.pt")
    with pytest.raises(ValueError, match="listed twice"):
        average_checkpoints([a, a], tmp_path / "x.pt")
    with pytest.raises(ValueError, match="lineage"):
        average_checkpoints([a, write(tmp_path / "c.pt", seed=3, lineage="other")],
                            tmp_path / "x.pt")
    wide = HistoryPolicyConfig(width=32, layers=1, heads=2, action_width=16, fusion_width=16,
                               critic_width=16, critic_layers=1)
    with pytest.raises(ValueError, match="architecture"):
        average_checkpoints([a, write(tmp_path / "w.pt", config=wide, seed=4)], tmp_path / "x.pt")
    average_checkpoints([a, b], tmp_path / "avg.pt")
    with pytest.raises(ValueError, match="already an average"):
        average_checkpoints([a, tmp_path / "avg.pt"], tmp_path / "x.pt")
    assert not (tmp_path / "x.pt").exists()


def test_non_floating_tensors_must_agree():
    one = {"w": torch.ones(2), "n": torch.tensor([1, 2])}
    two = {"w": torch.zeros(2), "n": torch.tensor([1, 3])}
    with pytest.raises(ValueError, match="non-floating"):
        average_state_dicts([one, two])
    out = average_state_dicts([one, dict(two, n=torch.tensor([1, 2]))])
    assert torch.equal(out["w"], torch.full((2,), 0.5)) and torch.equal(out["n"], one["n"])


def test_trainer_refuses_to_resume_an_average(tmp_path):
    config = HistoryPPOConfig(width=32, layers=1, heads=4, num_envs=4, steps_per_update=90,
                              seed=3, updates=1, epochs=1, minibatch_matches=2)
    a = write(tmp_path / "a.pt", config=config.policy_config(), seed=1)
    b = write(tmp_path / "b.pt", config=config.policy_config(), seed=2)
    average_checkpoints([a, b], tmp_path / "avg.pt")
    with pytest.raises(ValueError, match="evaluation only"):
        HistoryTrainer(config, tmp_path / "run", resume=tmp_path / "avg.pt")
