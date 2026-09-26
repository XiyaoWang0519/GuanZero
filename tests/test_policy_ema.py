"""PPOConfig.policy_ema (STAGE_C_TODO C2(c)): averaged policy weights for evaluation only."""
import pytest

torch = pytest.importorskip("torch")

from eval.policies import load_policy  # noqa: E402
from train.ckpt import load_checkpoint  # noqa: E402
from train.ppo import PPOTrainer  # noqa: E402

from test_ppo_actors import config, init_checkpoint  # noqa: E402


def test_ema_follows_the_recurrence_and_never_feeds_back(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    runs = {}
    for name, decay in (("off", 0.0), ("on", 0.5)):
        trainer = PPOTrainer(config(init, num_envs=4, rollout_steps=60, policy_lr=1e-3,
                                    policy_ema=decay), tmp_path / name)
        seen = []
        if decay:
            update = trainer.update_policy_ema

            def record():
                seen.append({k: v.detach().clone() for k, v in trainer.net.state_dict().items()})
                update()

            trainer.update_policy_ema = record
        trainer.update()
        trainer.update()
        runs[name] = (trainer, seen)
    (off, _), (on, seen) = runs["off"], runs["on"]
    # Training is identical with the average on: it is read, never written back.
    for key, value in off.net.state_dict().items():
        assert torch.equal(value, on.net.state_dict()[key]), key
    assert len(seen) == on.progress["optimizer_steps"] > 2
    expected = {k: v.clone() for k, v in seen[0].items()}
    for weights in seen[1:]:
        for key, value in weights.items():
            if value.is_floating_point():
                expected[key] = 0.5 * expected[key] + 0.5 * value
            else:
                expected[key] = value.clone()
    for key, value in expected.items():
        torch.testing.assert_close(on.policy_ema_state[key], value, rtol=0, atol=1e-6)


def test_ema_checkpoint_loads_as_a_policy_and_survives_resume(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    run = tmp_path / "run"
    first = PPOTrainer(config(init, num_envs=4, rollout_steps=40, policy_lr=1e-3,
                              policy_ema=0.9, max_updates=2), run)
    first.run()
    assert (run / "latest-ema.pt").exists()
    ema = load_policy(str(run / "latest-ema.pt"))
    latest = load_policy(str(run / "latest.pt"))
    assert ema.checkpoint_id != latest.checkpoint_id
    saved = load_checkpoint(run / "latest.pt", "cpu")["policy_ema_model"]
    resumed = PPOTrainer(config(init, num_envs=4, rollout_steps=40, policy_lr=1e-3,
                                policy_ema=0.9, max_updates=3), run, resume=run / "latest.pt")
    for key, value in saved.items():
        assert torch.equal(resumed.policy_ema_state[key], value), key


def test_ema_is_off_by_default_and_validated(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    run = tmp_path / "run"
    PPOTrainer(config(init, num_envs=4, rollout_steps=40, max_updates=1), run).run()
    assert not (run / "latest-ema.pt").exists()
    assert "policy_ema_model" not in load_checkpoint(run / "latest.pt", "cpu")
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError):
            config(init, policy_ema=bad).validate()
