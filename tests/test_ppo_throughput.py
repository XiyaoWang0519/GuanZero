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
    assert upload.host_arrays["x"] is source
    widened = upload("y", np.arange(3, dtype=np.int32), torch.int64, torch.long)
    assert widened.dtype == torch.long and widened.tolist() == [0, 1, 2]


def test_narrow_host_staging_preserves_learning_and_rng(tmp_path):
    class NarrowHostUploader(Uploader):
        """Exercise the uint8 storage source on a host without CUDA."""

        def __call__(self, name, array, wire, dtype):
            result = super().__call__(name, array, wire, dtype)
            if wire == torch.uint8:
                self.host_arrays[name] = np.asarray(array, dtype=np.uint8)
            return result

    init = init_checkpoint(tmp_path / "init.pt")
    old, new = [PPOTrainer(config(init, rollout_steps=100), tmp_path / name)
                for name in ("wide", "narrow")]
    new.upload = NarrowHostUploader(new.device)
    for _ in range(2):
        old.collect()
        new.collect()
        assert old.learn() == new.learn()
        for name, expected in old.buffer.storage().items():
            np.testing.assert_array_equal(new.buffer.storage()[name], expected, err_msg=name)
        for a, b in ((old.net, new.net), (old.critic, new.critic)):
            for key, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
        for a, b in ((old.policy_optimizer, new.policy_optimizer),
                     (old.critic_optimizer, new.critic_optimizer)):
            left, right = a.state_dict(), b.state_dict()
            assert left["param_groups"] == right["param_groups"]
            for parameter, state in left["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, right["state"][parameter][key], rtol=0, atol=0)
        assert old.rng.bit_generator.state == new.rng.bit_generator.state
        assert torch.equal(old.generator.get_state(), new.generator.get_state())
        old.buffer.next_iteration()
        new.buffer.next_iteration()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_uploader_retains_current_narrow_host_features():
    upload = Uploader(torch.device("cuda"))
    for shape in ((3, 4), (2, 4), (8, 4)):
        source = (np.arange(np.prod(shape)).reshape(shape) % 2).astype(np.float32)
        value = upload("obs", source, torch.uint8, torch.float32)
        torch.cuda.synchronize()
        staged = upload.host_arrays["obs"]
        assert staged.dtype == np.uint8 and staged.shape == source.shape
        assert staged.ctypes.data == upload.pinned["obs"].data_ptr()
        np.testing.assert_array_equal(staged, source)
        torch.testing.assert_close(value.cpu(), torch.from_numpy(source), rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Learner-side device staging and the reference skip (September 24 pass).
# Both must leave every stored row, every statistic that enters the loss and
# every weight bitwise unchanged; see docs/reports/perf-learner-2026-09-24.md.

def state(trainer) -> list[dict[str, torch.Tensor]]:
    return [{k: v.detach().clone() for k, v in m.state_dict().items()}
            for m in (trainer.net, trainer.critic)]


def assert_same_state(a, b) -> None:
    for left, right in zip(state(a), state(b)):
        for key, value in left.items():
            assert torch.equal(value, right[key]), key


def rows_snapshot(trainer) -> list[dict[str, np.ndarray]]:
    out = []
    for buffer in trainer.buffers:
        n, c = buffer.n_steps, buffer.n_cand
        out.append({**{k: getattr(buffer, k)[:n].copy() for k in STORED},
                    "logp": buffer.logp[:n].copy(), "cand": buffer.cand[:c].copy(),
                    "ref_logp": buffer.ref_logp[:c].copy()})
    return out


def capture_rollouts(trainer, updates: int) -> list[list[dict[str, np.ndarray]]]:
    """Run `updates` updates, returning the rows each one learned from."""
    captured = []
    learn = trainer.learn

    def learn_and_capture():
        captured.append(rows_snapshot(trainer))
        return learn()

    trainer.learn = learn_and_capture
    for _ in range(updates):
        trainer.update()
    return captured


@pytest.mark.parametrize("actors", [0, 2])
def test_learn_on_device_reproduces_host_gathers_exactly(tmp_path, actors):
    init = init_checkpoint(tmp_path / "init.pt")
    host, device = [PPOTrainer(config(init, learn_on_device=on, skip_unpruned_reference=False,
                                      actor_processes=actors, num_threads=1, rollout_steps=100),
                               tmp_path / f"run-{on}") for on in (False, True)]
    staged_used = []
    learn = device.learn

    def learn_and_check():
        record = learn()
        staged_used.append(all(b.staged is not None and b.staged.scalars is not None
                               for b in device.buffers))
        return record

    device.learn = learn_and_check
    try:
        learned = 0
        for _ in range(3):
            left, right = host.update(), device.update()
            assert left["update_samples"] == right["update_samples"]
            if left["update_samples"]:
                learned += 1
                for key in ("policy_loss", "value_loss", "entropy", "kl_ref", "hidden_loss",
                            "finish_loss", "approx_kl"):
                    assert left[key] == right[key], key
            assert_same_state(host, device)
        assert learned >= 2 and staged_used == [True] * 3
    finally:
        host.close()
        device.close()


def test_staged_gather_equals_host_gather_for_any_subset(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    trainer = PPOTrainer(config(init, learn_on_device=False), tmp_path / "run")
    trainer.collect()
    trainer.refresh_values()
    buffer = trainer.buffer
    assert buffer.finalize(1.0, 0.95) > 0
    rng = np.random.default_rng(3)
    samples = buffer.samples[:buffer.n_samples]
    subset = rng.choice(samples, size=min(40, samples.size), replace=False)
    expected = buffer.gather(subset)
    staged = buffer.stage("cpu")
    assert staged.covers(subset) and not staged.covers(np.array([buffer.n_steps]))
    actual = buffer.gather(subset)
    assert set(actual) == set(expected)
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key
    # Rows that are not finalized samples fall back to the host gather.
    unfinished = np.setdiff1d(np.arange(buffer.n_steps), samples)
    if unfinished.size:
        fallback = buffer.gather(unfinished[:5])
        assert torch.equal(fallback["obs"], torch.from_numpy(buffer.obs[unfinished[:5]]).float())
    buffer.next_iteration()
    assert buffer.staged is None


def test_reference_skip_changes_no_row_choice_or_weight(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    full, skip = [PPOTrainer(config(init, kl_coef=0.0, skip_unpruned_reference=on),
                             tmp_path / f"run-{on}") for on in (False, True)]
    top_k = full.config.top_k
    # The skip run's act() receives zeros for the candidates it did not score:
    # exactly the rows with at most top_k candidates, which prune whole.
    act, calls = skip.policy.act, []

    def act_and_record(obs, cand, offsets, phase, **kw):
        ref = kw["ref_scores"]
        raw = (offsets[1:] - offsets[:-1]).numpy()
        unscored = (ref[offsets[:-1]] == 0).numpy()
        calls.append((raw, unscored))
        return act(obs, cand, offsets, phase, **kw)

    skip.policy.act = act_and_record
    seen_skipped = seen_scored = False
    for a, b in zip(capture_rollouts(full, 3), capture_rollouts(skip, 3)):
        for x, y in zip(a, b):
            for key in STORED + ("logp", "cand"):
                assert np.array_equal(x[key], y[key]), key
            assert np.isfinite(x["ref_logp"]).all()
            # A stored row's candidates are all NaN (skipped) or all finite
            # (scored). The stored count is the pruned one: a skipped row kept
            # all of its at most top_k candidates, a scored row kept at least
            # top_k (top_k plus pass when the pass ranked below).
            for row in range(y["cand_count"].size):
                s, c = y["cand_start"][row], y["cand_count"][row]
                nan = np.isnan(y["ref_logp"][s:s + c])
                assert nan.all() or not nan.any()
                assert c <= top_k if nan.all() else c >= top_k
                seen_skipped |= bool(nan.all())
                seen_scored |= not nan.any()
    assert seen_skipped and seen_scored
    assert calls and all(np.array_equal(unscored, raw <= top_k) for raw, unscored in calls)
    assert any((raw > top_k).any() for raw, _ in calls)
    assert_same_state(full, skip)
    counters = {k: v for k, v in full.progress.items() if k != "elapsed_seconds"}
    assert counters == {k: v for k, v in skip.progress.items() if k != "elapsed_seconds"}


def test_reference_skip_waits_for_the_kl_term_to_reach_zero(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    trainer = PPOTrainer(config(init, kl_coef=0.1, kl_anneal_updates=2), tmp_path / "run")
    trainer.collect()
    assert trainer.kl_coef() > 0 and trainer.learner_reference_all
    assert np.isfinite(trainer.buffer.ref_logp[:trainer.buffer.n_cand]).all()
    trainer.buffer.next_iteration()
    trainer.progress["updates"] = 2
    assert trainer.kl_coef() == 0
    trainer.collect()
    assert not trainer.learner_reference_all
    b = trainer.buffer
    first = b.ref_logp[b.cand_start[:b.n_steps]]
    assert np.isnan(first).any() and np.isfinite(first).any()
    assert (b.cand_count[:b.n_steps][np.isnan(first)] <= trainer.config.top_k).all()
    record = trainer.learn()
    assert np.isfinite(record["kl_ref"]) and np.isfinite(record["policy_loss"])


def test_partial_reference_loss_and_gradient_ignore_skipped_rows(tmp_path):
    init = init_checkpoint(tmp_path / "init.pt")
    trainer = PPOTrainer(config(init, kl_coef=0.0), tmp_path / "run")
    trainer.collect()
    trainer.refresh_values()
    assert trainer.buffer.finalize(1.0, 0.95) > 0
    mb = next(trainer.buffer.minibatches(10**6, np.random.default_rng(0)))
    assert torch.isnan(mb["ref_logp"]).any() and torch.isfinite(mb["ref_logp"]).any()
    zeroed = {**mb, "ref_logp": torch.where(torch.isfinite(mb["ref_logp"]), mb["ref_logp"],
                                             torch.zeros(()))}
    grads = []
    for batch, partial in ((mb, True), (zeroed, False)):
        trainer.net.zero_grad(set_to_none=True)
        terms = trainer.minibatch_loss(batch, 0.0, partial)
        assert torch.isfinite(terms["kl_ref"])
        terms["policy_total"].backward()
        grads.append({"terms": terms,
                      "grads": [p.grad.clone() for p in trainer.net.parameters()
                                if p.grad is not None]})
    a, b = grads
    for key in ("policy_total", "policy_loss", "entropy", "hidden_loss", "finish_loss",
                "value_loss", "approx_kl"):
        assert torch.equal(a["terms"][key], b["terms"][key]), key
    for x, y in zip(a["grads"], b["grads"]):
        assert torch.equal(x, y)
