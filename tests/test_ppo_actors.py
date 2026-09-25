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


def _rollout_and_weights(trainer, updates):
    rollouts = []
    learn = trainer.learn

    def learn_and_record():
        rollouts.append([{k: getattr(b, k)[:b.n_steps].copy() for k in ("obs", "chosen", "logp")}
                         for b in trainer.buffers])
        return learn()

    trainer.learn = learn_and_record
    stats = [trainer.update() for _ in range(updates)]
    weights = {k: v.detach().clone() for k, v in trainer.net.state_dict().items()}
    return rollouts, stats, weights


def test_pipeline_reproduces_the_actor_processes(tmp_path):
    """rollout_pipeline builds the actor-process shards (same seeds, same
    per-shard work) in this process, so on CPU both give the same rollouts,
    statistics and weights; only the host scheduling differs."""
    init = init_checkpoint(tmp_path / "init.pt")
    runs = []
    for name, shards in (("actors", dict(actor_processes=2)), ("pipeline", dict(rollout_pipeline=2))):
        trainer = PPOTrainer(config(init, num_envs=8, rollout_steps=80, policy_lr=1e-3, **shards),
                             tmp_path / name)
        try:
            runs.append(_rollout_and_weights(trainer, 3))
        finally:
            trainer.close()
    (rollout_a, stats_a, weights_a), (rollout_p, stats_p, weights_p) = runs
    for update_a, update_p in zip(rollout_a, rollout_p):
        for shard_a, shard_p in zip(update_a, update_p):
            for key in shard_a:
                assert np.array_equal(shard_a[key], shard_p[key]), key
    for first, second in zip(stats_a, stats_p):
        for key in ("decisions", "learner_decisions", "rounds", "update_samples",
                    "policy_loss", "value_loss", "entropy", "first_ratio_max_deviation"):
            if key in first:
                assert first[key] == second[key], key
    for key in weights_a:
        assert torch.equal(weights_a[key], weights_p[key]), key


def test_pipeline_shards_deal_different_games_without_policy_lag(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    trainer = PPOTrainer(config(init, num_envs=8, rollout_pipeline=2, rollout_steps=80),
                         tmp_path / "run")
    try:
        assert len(trainer.buffers) == 2 and trainer.env is None
        records = [trainer.update() for _ in range(2)]
        assert trainer.progress["decisions"] == 2 * 80 * 8
        assert records[0]["first_ratio_max_deviation"] < 1e-5
        # Each shard deals from its own seed: no shard replays the other's games.
        a, b = trainer.buffers
        n = min(a.n_steps, b.n_steps)
        assert n > 50 and not np.array_equal(a.obs[:n], b.obs[:n])
        trainer.collect()   # shards take the learner's weights at the start of a collect
        digest = float(sum(p.detach().double().sum() for p in trainer.net.parameters()))
        assert trainer.actors.request("weights_digest") == pytest.approx([digest, digest],
                                                                         rel=1e-12)
    finally:
        trainer.close()


def test_pipeline_checks_its_configuration(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    for bad in (dict(rollout_pipeline=1), dict(rollout_pipeline=2, actor_processes=2),
                dict(num_envs=6, rollout_pipeline=4)):
        with pytest.raises(ValueError):
            config(init, **bad).validate()
    with pytest.raises(ValueError):
        PPOTrainer(config(init, rollout_pipeline=2), tmp_path / "x", opponent=GreedyOpponent())


def test_pipeline_run_checkpoints_and_resumes(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    run = tmp_path / "run"
    PPOTrainer(config(init, num_envs=4, rollout_pipeline=2, rollout_steps=40,
                      max_updates=2), run).run()
    PPOTrainer(config(init, num_envs=4, rollout_pipeline=2, rollout_steps=40,
                      max_updates=3), run, resume=run / "latest.pt").run()
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["updates"] for r in rows] == [1, 2, 3]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (per-shard streams)")
def test_pipeline_on_cuda_matches_the_actor_processes(tmp_path):
    """The CUDA path runs each shard on its own stream; its sampled actions
    must match the actor processes' (the same shards, one per process)."""
    init = init_checkpoint(tmp_path / "init.pt")
    runs = []
    for name, shards in (("actors", dict(actor_processes=2)), ("pipeline", dict(rollout_pipeline=2))):
        trainer = PPOTrainer(config(init, num_envs=8, rollout_steps=80, **shards),
                             tmp_path / name, device="cuda")
        try:
            runs.append(_rollout_and_weights(trainer, 2))
        finally:
            trainer.close()
    for update_a, update_p in zip(runs[0][0], runs[1][0]):
        for shard_a, shard_p in zip(update_a, update_p):
            assert np.array_equal(shard_a["chosen"], shard_p["chosen"])
            np.testing.assert_allclose(shard_a["logp"], shard_p["logp"], rtol=0, atol=1e-5)
