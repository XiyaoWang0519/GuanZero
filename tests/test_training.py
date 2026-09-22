"""Learner correctness, real self-play updates, and checkpoint recovery."""
from dataclasses import replace
import json
from pathlib import Path
import random
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

import gd
from train.buffer import Decision, ReplayBuffer
from train.ckpt import load_checkpoint, restore_rng, rng_state, save_checkpoint
from train.dmc import TrainConfig, Trainer
from train.model import GuandanModel, ModelConfig, select_actions


def tiny_config(**overrides):
    data = json.loads((Path(__file__).parents[1] / "train/configs/smoke.json").read_text())
    return TrainConfig(**{**data, **overrides})


def test_ragged_scoring_matches_individual_states_and_gradients():
    torch.set_num_threads(1)
    torch.manual_seed(8)
    model = GuandanModel(ModelConfig(obs_dim=9, act_dim=5, state_width=16,
                                    action_width=8, fusion_width=8))
    obs, cand = torch.randn(3, 9), torch.randn(9, 5)
    offsets, phase = torch.tensor([0, 2, 8, 9]), torch.tensor([3, 1, 2])
    batched = model.score_candidates(obs, cand, offsets, phase, chunk_size=3)
    separate = torch.cat([model(obs[i:i+1].repeat(int(offsets[i+1]-offsets[i]), 1),
                                cand[offsets[i]:offsets[i+1]],
                                phase[i:i+1].repeat(int(offsets[i+1]-offsets[i])))["q"]
                          for i in range(3)])
    torch.testing.assert_close(batched, separate)
    batched.sum().backward()
    assert all(next(model.phase_heads[str(p)].parameters()).grad.abs().sum() > 0
               for p in (1, 2, 3))


def test_ragged_selection_handles_ties_and_exploration():
    scores = torch.tensor([2., 2., -5., 4., 0., 1.])
    offsets = torch.tensor([0, 2, 3, 6])
    assert select_actions(scores, offsets).tolist() == [0, 0, 0]
    generator = torch.Generator().manual_seed(12)
    seen = set()
    for _ in range(50):
        choices = select_actions(scores, offsets, epsilon=1, generator=generator)
        assert torch.all(choices >= 0) and torch.all(choices < offsets.diff())
        seen.add(tuple(choices.tolist()))
    assert len(seen) == 6
    for _ in range(20):
        assert select_actions(scores, offsets, margin=.1)[2] == 0


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_near_best_sampling_accepts_mixed_precision_scores(dtype):
    scores = torch.tensor([1., 1., -5., 4., 0., 1.], dtype=dtype)
    offsets = torch.tensor([0, 2, 3, 6])
    generator = torch.Generator().manual_seed(24)
    first_choices = set()
    for _ in range(30):
        choices = select_actions(scores, offsets, margin=.1, generator=generator)
        first_choices.add(int(choices[0]))
        assert choices[1:].tolist() == [0, 0]
    assert first_choices == {0, 1}


def test_replay_targets_follow_acting_team_and_terminal_order():
    replay = ReplayBuffer(3, 2, 1)
    decisions = [Decision(np.array([i, 0]), np.array([0]), np.zeros((3, 54)),
                          seat=i, phase=3) for i in range(4)]
    replay.add_round(decisions, [2, -2, 2, -2], [2, 0, 3, 1])
    assert len(replay) == 3
    for i in range(3):
        seat = int(replay.arrays["obs"][i, 0])
        assert replay.arrays["returns"][i] == (2 if seat % 2 == 0 else -2)
        assert replay.arrays["finish"][i] == [2, 0, 3, 1].index(seat)
    replay.clear()
    with pytest.raises(ValueError):
        replay.sample(1, np.random.default_rng())


def test_atomic_checkpoint_failure_retains_previous_file(tmp_path, monkeypatch):
    path = tmp_path / "latest.pt"
    save_checkpoint(path, {"sentinel": 123})
    original = path.read_bytes()
    def fail(payload, stream):
        stream.write(b"partial")
        raise OSError("simulated disk failure")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError):
        save_checkpoint(path, {"sentinel": 456})
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_restores_python_numpy_and_torch_rngs():
    rng = np.random.default_rng(5)
    saved = rng_state(rng)
    expected = (random.random(), rng.random(), torch.rand(3))
    restore_rng(saved, rng)
    actual = (random.random(), rng.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])


def test_actual_dmc_updates_only_network_play_and_resumes(tmp_path):
    config = tiny_config(max_updates=2, greedy_start=1.0, greedy_end=1.0, snapshot_updates=2)
    trainer = Trainer(config, tmp_path)
    initial = {name: param.detach().clone() for name, param in trainer.model.named_parameters()}
    progress = trainer.run()
    assert progress["updates"] == 2 and progress["rounds"] > 0 and progress["samples"] > 0
    baseline = load_checkpoint(tmp_path / "checkpoints/step-000000000.pt")
    assert baseline["progress"]["updates"] == 0
    for name, value in initial.items():
        torch.testing.assert_close(baseline["model"][name], value)
    assert any(not torch.equal(initial[n], p) for n, p in trainer.model.named_parameters()
               if n.startswith("state_tower"))
    # Stage A never updates tribute heads.
    assert all(torch.equal(initial[n], p) for n, p in trainer.model.named_parameters()
               if n.startswith(("phase_heads.1.", "phase_heads.2.")))
    for key, decisions in trainer.pending.items():
        greedy = trainer.greedy_teams[key[:2]]
        assert all(d.phase == 3 and d.seat % 2 != greedy for d in decisions)
    checkpoint = load_checkpoint(tmp_path / "latest.pt")
    snapshot = tmp_path / "checkpoints/step-000000002.pt"
    snapshot_bytes = snapshot.read_bytes()
    assert checkpoint["optimizer"]["state"]
    resumed = Trainer(replace(config, max_updates=4), tmp_path, resume=tmp_path / "latest.pt")
    assert not resumed.pending and len(resumed.replay) == 0
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name])
    assert resumed.run()["updates"] == 4
    assert resumed.progress["resumes"] == 1
    assert snapshot.read_bytes() == snapshot_bytes
    with pytest.raises(ValueError, match="differs at"):
        Trainer(replace(config, learning_rate=.05), tmp_path, resume=tmp_path / "latest.pt")


def test_sigterm_saves_a_resumable_real_trainer_checkpoint(tmp_path):
    root = Path(__file__).parents[1]
    environment = dict(os.environ, PYTHONPATH=f"{root / 'python'}:{root}")
    command = [sys.executable, "-m", "train.dmc", "--config", "train/configs/smoke.json",
               "--run-dir", str(tmp_path), "--max-updates", "100000", "--max-seconds", "30"]
    with (tmp_path / "process.log").open("w") as log:
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 15
            while not (tmp_path / "metrics.jsonl").exists() and process.poll() is None:
                if time.monotonic() > deadline:
                    pytest.fail("trainer did not reach learning")
                time.sleep(.02)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=10) == 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    saved = load_checkpoint(tmp_path / "latest.pt")
    assert saved["progress"]["updates"] > 0
    resumed = Trainer(tiny_config(max_updates=saved["progress"]["updates"] + 1),
                      tmp_path, resume=tmp_path / "latest.pt")
    assert resumed.run()["updates"] == saved["progress"]["updates"] + 1


def test_rollout_copies_engine_buffers_and_has_correct_private_labels(tmp_path):
    trainer = Trainer(tiny_config(rollout_steps=1, log_envs=0), tmp_path)
    batch = trainer.env.pending()
    expected = np.array(batch.obs, copy=True)
    hidden = np.array(batch.hidden_counts, copy=True)
    seats = np.array(batch.seat, copy=True)
    trainer.collect()
    records = {key[0]: values[0] for key, values in trainer.pending.items()}
    assert records
    trainer.env.pending()
    for row, record in records.items():
        np.testing.assert_array_equal(record.obs, expected[row])
        np.testing.assert_array_equal(record.hidden, hidden[row])
        assert record.seat == seats[row]
        # All unseen cards are partitioned among three hidden hands.
        unseen = record.obs[108:162] + record.obs[162:216]
        np.testing.assert_array_equal(record.hidden.sum(0), unseen)


def test_failed_learner_does_not_overwrite_good_checkpoint(tmp_path, monkeypatch):
    trainer = Trainer(tiny_config(), tmp_path)
    def fail():
        with torch.no_grad():
            next(trainer.model.parameters()).fill_(float("nan"))
        raise FloatingPointError("injected learner failure")
    monkeypatch.setattr(trainer, "learn", fail)
    with pytest.raises(FloatingPointError):
        trainer.run()
    checkpoint = load_checkpoint(tmp_path / "latest.pt")
    assert checkpoint["progress"]["updates"] == 0
    assert all(torch.isfinite(value).all() for value in checkpoint["model"].values())


def test_tensorboard_contains_real_learner_metrics(tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    trainer = Trainer(tiny_config(tensorboard=True, max_updates=1), tmp_path)
    trainer.run()
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert events.Scalars("loss")[0].step == 1
    assert np.isfinite(events.Scalars("belief_loss")[0].value)
    runtime = json.loads((tmp_path / "runtime-000.json").read_text())
    assert runtime["device"] == "cpu" and runtime["bf16_enabled"] is False


@pytest.mark.parametrize("changes", [dict(checkpoint_seconds=601), dict(num_envs=0),
                                     dict(epsilon_start=1.1), dict(hidden_weight=-1),
                                     dict(max_seconds=float("nan"))])
def test_reject_invalid_config(changes):
    with pytest.raises(ValueError):
        tiny_config(**changes).validate()
