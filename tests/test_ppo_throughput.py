"""B5b throughput path: same actions, log-probs, advantages and losses as B5."""
from dataclasses import asdict

import gd
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from train.ckpt import save_checkpoint  # noqa: E402
from train.model import GuandanModel, ModelConfig  # noqa: E402
from train.policy import segment_log_softmax  # noqa: E402
from train.ppo import (PPOConfig, PPOTrainer, Uploader, cached_reference_kl,  # noqa: E402
                       policy_terms, reference_kl)

SMALL = ModelConfig(obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM, state_width=32, state_layers=2,
                    action_width=16, action_layers=1, fusion_width=16, fusion_layers=2)


def init_checkpoint(path, learned_tribute: bool = False):
    torch.manual_seed(11)
    model = GuandanModel(SMALL)
    payload = {"model_config": asdict(SMALL), "model": model.state_dict(), "optimizer": {},
               "config": {"action_mode": "canonical", "seed": 1}, "progress": {"updates": 0},
               "rng": {}}
    if learned_tribute:
        payload.update(stage="a2", tribute_policy="learned")
    save_checkpoint(path, payload)
    return path


def config(init, **overrides) -> PPOConfig:
    base = dict(init_checkpoint=str(init), critic_init="", critic_width=16, critic_layers=1,
                opponent="frozen", num_envs=8, num_threads=2, torch_threads=1,
                rollout_steps=60, epochs=2, minibatch_size=96, top_k=4, temperature=0.5,
                max_updates=100, max_seconds=600, checkpoint_seconds=60, snapshot_updates=100,
                tensorboard=False)
    base.update(overrides)
    return PPOConfig(**base)


def pair(tmp_path, init, **overrides):
    return [PPOTrainer(config(init, fast_rollout=fast, **overrides), tmp_path / f"run-{fast}")
            for fast in (False, True)]


STORED = ("obs", "hidden", "cand_start", "cand_count", "chosen", "phase", "seat", "traj",
          "reward", "done")


def assert_same_rollout(old: PPOTrainer, new: PPOTrainer) -> None:
    a, b = old.buffer, new.buffer
    assert (a.n_steps, a.n_cand, a.n_traj) == (b.n_steps, b.n_cand, b.n_traj)
    assert a.n_steps > 50
    for name in STORED:
        assert np.array_equal(getattr(a, name)[:a.n_steps], getattr(b, name)[:b.n_steps]), name
    assert np.array_equal(a.cand[:a.n_cand], b.cand[:b.n_cand])
    np.testing.assert_allclose(a.logp[:a.n_steps], b.logp[:b.n_steps], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(a.ref_logp[:a.n_cand], b.ref_logp[:b.n_cand], rtol=1e-5, atol=1e-5)
    assert old.progress == new.progress


@pytest.mark.parametrize("opponent", ["frozen", "greedy"])
def test_fast_rollout_samples_the_same_actions_as_b5(tmp_path, opponent):
    init = init_checkpoint(tmp_path / "init.pt")
    old, new = pair(tmp_path, init, opponent=opponent, rollout_steps=150)
    assert new.fused_opponent == (opponent == "frozen") and not old.fused_opponent
    for _ in range(2):
        old.collect()
        new.collect()
        assert_same_rollout(old, new)
        old_stats, new_stats = old.learn(), new.learn()
        a, b = old.buffer, new.buffer
        assert a.n_samples == b.n_samples > 0
        idx = a.samples[:a.n_samples]
        np.testing.assert_allclose(a.advantage[idx], b.advantage[idx], rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(a.returns[idx], b.returns[idx], rtol=1e-4, atol=1e-5)
        for key, value in old_stats.items():
            assert new_stats[key] == pytest.approx(value, rel=1e-3, abs=1e-5), key
        old.buffer.next_iteration()
        new.buffer.next_iteration()
        old.progress["updates"] += 1
        new.progress["updates"] += 1
    # Adam moves near-zero gradient entries by about lr whatever their size, so
    # float-level gradient differences show up as weight differences of order
    # lr; the gradients themselves are compared in the test below.
    lr_scale = old.config.policy_lr * old.progress["optimizer_steps"]
    for p, q in zip(old.net.parameters(), new.net.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=lr_scale)


def test_cached_kl_gives_the_b5_gradient(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    trainer = PPOTrainer(config(init, policy_lr=1e-2, rollout_steps=100), tmp_path / "run")
    trainer.update()          # move the policy away from the reference
    trainer.collect()
    trainer.refresh_values()
    assert trainer.buffer.finalize(1.0, 0.95) > 0
    mb = next(trainer.buffer.minibatches(10**6, np.random.default_rng(0)))
    grads = []
    for fast in (False, True):
        trainer.config.fast_rollout = fast
        trainer.net.zero_grad(set_to_none=True)
        terms = trainer.minibatch_loss(mb, kl_coef=1.0)
        terms["policy_total"].backward()
        grads.append([p.grad.clone() for p in trainer.net.parameters() if p.grad is not None])
        assert float(terms["kl_ref"].detach()) > 0
    for old, new in zip(*grads):
        torch.testing.assert_close(old, new, rtol=1e-4, atol=1e-7)


def test_fast_rollout_with_learned_tribute_heads(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt", learned_tribute=True)
    old, new = pair(tmp_path, init, tribute_policy="learned",
                    rollout_steps=80)
    old.collect()
    new.collect()
    assert_same_rollout(old, new)
    assert (new.buffer.phase[:new.buffer.n_steps] != int(gd.Phase.Play)).any()


def test_cached_reference_kl_equals_recomputed(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    trainer = PPOTrainer(config(init, policy_lr=1e-2), tmp_path / "run")
    trainer.update()          # the policy moves away from the reference
    trainer.collect()
    trainer.refresh_values()
    assert trainer.buffer.finalize(1.0, 0.95) > 0
    mb = next(trainer.buffer.minibatches(10**6, np.random.default_rng(0)))
    with torch.no_grad():
        _, _, log_probs = policy_terms(trainer.policy, mb["obs"], mb["cand"], mb["offsets"],
                                       mb["phase"], mb["chosen"], trainer.phase_code)
        recomputed = reference_kl(trainer.policy, log_probs, mb["obs"], mb["cand"],
                                  mb["offsets"], mb["phase"], trainer.phase_code)
        ref = trainer.policy.reference.score_candidates(mb["obs"], mb["cand"], mb["offsets"],
                                                        mb["phase"], phase_code=trainer.phase_code)
    assert float(recomputed.mean()) > 0
    torch.testing.assert_close(mb["ref_logp"], segment_log_softmax(ref / 0.5, mb["offsets"]),
                               rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(cached_reference_kl(log_probs, mb["ref_logp"], mb["offsets"]),
                               recomputed, rtol=1e-4, atol=1e-6)


def test_non_finite_loss_raises_at_epoch_end(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    trainer = PPOTrainer(config(init), tmp_path / "run")
    trainer.collect()
    with torch.no_grad():
        next(trainer.critic.parameters()).fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        trainer.learn()


def test_uploader_is_zero_copy_on_cpu_and_widens():
    upload = Uploader(torch.device("cpu"))
    source = np.arange(6, dtype=np.float32)
    source.setflags(write=False)
    view = upload("x", source, torch.uint8, torch.float32)
    assert view.data_ptr() == source.__array_interface__["data"][0]
    widened = upload("y", np.arange(3, dtype=np.int32), torch.int64, torch.long)
    assert widened.dtype == torch.long and widened.tolist() == [0, 1, 2]
