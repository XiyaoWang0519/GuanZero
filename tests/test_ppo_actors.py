"""B5b synchronous actor processes: no policy lag, learner reads shared shards."""
from dataclasses import asdict
import json

import gd
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train.ckpt import save_checkpoint  # noqa: E402
from train.model import GuandanModel, ModelConfig  # noqa: E402
from train.opponents import GreedyOpponent  # noqa: E402
from train.ppo import PPOConfig, PPOTrainer, policy_terms  # noqa: E402

SMALL = ModelConfig(obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM, state_width=32, state_layers=2,
                    action_width=16, action_layers=1, fusion_width=16, fusion_layers=2)


def init_checkpoint(path):
    torch.manual_seed(11)
    save_checkpoint(path, {"model_config": asdict(SMALL), "model": GuandanModel(SMALL).state_dict(),
                           "optimizer": {}, "config": {"action_mode": "canonical", "seed": 1},
                           "progress": {"updates": 0}, "rng": {}})
    return path


def config(init, **overrides) -> PPOConfig:
    base = dict(init_checkpoint=str(init), critic_init="", critic_width=16, critic_layers=1,
                opponent="frozen", num_envs=8, num_threads=1, torch_threads=1,
                rollout_steps=60, epochs=2, minibatch_size=96, top_k=4, temperature=0.5,
                max_updates=100, max_seconds=600, checkpoint_seconds=60, snapshot_updates=100,
                tensorboard=False)
    base.update(overrides)
    return PPOConfig(**base)


@pytest.fixture
def actor_trainer(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    trainer = PPOTrainer(config(init, num_envs=8, actor_processes=2, rollout_steps=80,
                                policy_lr=1e-3, num_threads=1), tmp_path / "run")
    yield trainer
    trainer.close()


def test_actor_updates_have_no_policy_lag(actor_trainer):
    trainer = actor_trainer
    assert len(trainer.buffers) == 2 and trainer.env is None
    records = [trainer.update() for _ in range(3)]
    assert all(r["update_samples"] > 0 for r in records)
    assert trainer.progress["decisions"] == 3 * 80 * 8
    assert records[0]["first_ratio_max_deviation"] < 1e-5
    for record in records:
        assert np.isfinite(record["policy_loss"]) and np.isfinite(record["value_loss"])
    # Rounds still in progress are carried over (as in-process); the actors
    # move them to the front of their shard before collecting, so every step
    # after them was drawn in this rollout, by the weights published for it.
    carried = [int((~b.traj_complete[b.traj[:b.n_steps]]).sum()) for b in trainer.buffers]
    trainer.collect()
    fresh = 0
    for buffer, old in zip(trainer.buffers, carried):
        steps = np.arange(old, buffer.n_steps)
        fresh += steps.size
        mb = buffer.gather(steps, trainer.device)
        with torch.no_grad():
            log_prob, _, _ = policy_terms(trainer.policy, mb["obs"], mb["cand"], mb["offsets"],
                                          mb["phase"], mb["chosen"], trainer.phase_code)
        torch.testing.assert_close(log_prob, mb["logp"], rtol=0, atol=1e-5)
    assert fresh > 100
    digest = float(sum(p.detach().double().sum() for p in trainer.net.parameters()))
    assert trainer.actors.request("weights_digest") == pytest.approx([digest, digest], rel=1e-12)


def test_actor_shards_use_every_environment_and_both_teams(actor_trainer):
    trainer = actor_trainer
    trainer.collect()
    for buffer in trainer.buffers:
        steps = slice(0, buffer.n_steps)
        teams = set((buffer.seat[steps] % 2).tolist())
        envs = set(buffer.traj_env[:buffer.n_traj].tolist())
        assert teams == {0, 1} and envs == {0, 1, 2, 3}
        # learner_team is env % 2 within each shard
        assert np.array_equal(buffer.traj_team[:buffer.n_traj], buffer.traj_env[:buffer.n_traj] % 2)


def test_actor_runs_are_reproducible_under_a_seed(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    results = []
    for name in ("a", "b"):
        trainer = PPOTrainer(config(init, num_envs=4, actor_processes=2, rollout_steps=150,
                                    num_threads=1), tmp_path / name)
        try:
            results.append([trainer.update() for _ in range(2)])
        finally:
            trainer.close()
    assert results[0][-1]["update_samples"] > 0
    for first, second in zip(*results):
        for key in ("decisions", "learner_decisions", "rounds", "update_samples",
                    "policy_loss", "value_loss", "entropy"):
            if key not in first:
                continue
            assert first[key] == pytest.approx(second[key], rel=1e-6), key


def test_filtered_actor_update_uses_one_threshold_for_all_shards(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    trainer = PPOTrainer(config(init, num_envs=4, actor_processes=2, rollout_steps=150,
                                advantage_filter_quantile=0.8,
                                advantage_filter_min_magnitude=0.05), tmp_path / "run")
    try:
        stats = trainer.update()
        assert stats["update_samples"] > 0
        assert 0 < stats["policy_filter_kept_fraction"] < 1
        assert np.isfinite(stats["policy_loss"])
    finally:
        trainer.close()


def test_actor_processes_reject_an_opponent_object_and_bad_shards(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    with pytest.raises(ValueError):
        PPOTrainer(config(init, actor_processes=2), tmp_path / "x", opponent=GreedyOpponent())
    with pytest.raises(ValueError):
        config(init, num_envs=6, actor_processes=4).validate()


def test_actor_run_checkpoints_and_resumes(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    run = tmp_path / "run"
    first = PPOTrainer(config(init, num_envs=4, actor_processes=2, rollout_steps=40,
                              num_threads=1, max_updates=2), run)
    first.run()
    assert not any(p.is_alive() for p in first.actors.processes)
    resumed = PPOTrainer(config(init, num_envs=4, actor_processes=2, rollout_steps=40,
                                num_threads=1, max_updates=3), run, resume=run / "latest.pt")
    resumed.run()
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["updates"] for r in rows] == [1, 2, 3]
