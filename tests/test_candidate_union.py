"""C1 candidate-support contracts, including one real short PPO update."""
from dataclasses import asdict

import gd
import numpy as np
import pytest
import torch

from eval.batched import EvalConfig, play_duplicate_batch
from eval.duplicate import generate_deals, play_duplicate_teams
from eval.policies import GreedyPolicy, PrunedPolicy, load_policy
from train.ckpt import save_checkpoint
from train.model import GuandanModel, ModelConfig
from train.policy import (PASS_FEATURE, PolicyConfig, StageBPolicy, pass_mask,
                          prune_candidates, segment_log_softmax)
from train.ppo import PPOConfig, PPOTrainer, cached_reference_kl


SMALL = ModelConfig(obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM, state_width=16,
                    state_layers=1, action_width=16, action_layers=1,
                    fusion_width=16, fusion_layers=1)


def test_uniform_union_support_and_default_rng_identity():
    scores = torch.arange(10, dtype=torch.float32)
    offsets = torch.tensor([0, 10])
    passed = torch.zeros(10, dtype=torch.bool)
    passed[0] = True
    g = torch.Generator().manual_seed(31)
    before = g.get_state().clone()
    default, _ = prune_candidates(scores, offsets, 2, passed, generator=g)
    assert default.tolist() == [0, 8, 9]
    assert torch.equal(before, g.get_state())
    draws = np.zeros(7, np.int64)
    for seed in range(1400):
        keep, off = prune_candidates(scores, offsets, 2, passed, extra=2,
                                     generator=torch.Generator().manual_seed(seed))
        chosen = keep.tolist()
        assert chosen == sorted(set(chosen)) and off.tolist() == [0, 5]
        assert {0, 8, 9}.issubset(chosen)
        for index in set(chosen) - {0, 8, 9}:
            draws[index - 1] += 1
    # Every remaining legal move has comparable inclusion probability.
    assert draws.min() > 320 and draws.max() < 480
    with pytest.raises(ValueError, match="seeded generator"):
        prune_candidates(scores, offsets, 2, passed, extra=2)
    with pytest.raises(ValueError, match="nonnegative integer"):
        PolicyConfig(candidate_mode="union", candidate_extra=1.5)


def test_union_act_choose_reference_kl_and_checkpoint(tmp_path):
    torch.manual_seed(5)
    model = GuandanModel(SMALL)
    config = PolicyConfig(top_k=2, temperature=0.8,
                          candidate_mode="union", candidate_extra=3)
    policy = StageBPolicy.from_model(model, config)
    obs = torch.rand(2, gd.OBS_DIM)
    cand = torch.rand(15, gd.ACT_DIM)
    cand[:, PASS_FEATURE] = 0
    cand[0, PASS_FEATURE] = 1
    offsets = torch.tensor([0, 10, 15])
    phase = torch.full((2,), int(gd.Phase.Play))
    g1, g2 = torch.Generator().manual_seed(12), torch.Generator().manual_seed(12)
    step = policy.act(obs, cand, offsets, phase, generator=g1)
    choice, finite = policy.choose(obs, cand, offsets, phase, generator=g2)
    assert bool(finite) and torch.equal(choice, step.choice)
    assert torch.equal(g1.get_state(), g2.get_state())
    assert step.pruned_offsets.tolist() in ([0, 5, 10], [0, 6, 11])
    assert bool(pass_mask(cand[step.keep_index[:6]]).any())
    logp, entropy = policy.evaluate(obs, cand[step.keep_index], step.pruned_offsets,
                                    phase, step.pruned_choice)
    torch.testing.assert_close(logp, step.log_prob)
    torch.testing.assert_close(entropy, step.entropy)
    with torch.no_grad():
        ref = policy.reference.score_candidates(obs, cand[step.keep_index],
                                                step.pruned_offsets, phase)
        expected_ref = segment_log_softmax(ref / config.temperature, step.pruned_offsets)
        torch.testing.assert_close(step.ref_log_probs, expected_ref)
        own = segment_log_softmax(step.logits, step.pruned_offsets)
        torch.testing.assert_close(cached_reference_kl(own, expected_ref,
                                                       step.pruned_offsets), torch.zeros(2))
    path = tmp_path / "union.pt"
    save_checkpoint(path, policy.checkpoint_payload(optimizer={}, config={}, progress={}, rng={}))
    loaded = load_policy(str(path))
    assert loaded.stage_b.config == config
    assert loaded.stage_b.config.candidate_mode == "union"
    payload = policy.checkpoint_payload(optimizer={}, config={}, progress={}, rng={})
    payload["policy_config"].pop("candidate_mode")
    payload["policy_config"].pop("candidate_extra")
    from train.policy import policy_from_payload
    assert policy_from_payload(payload).config.candidate_mode == "top_k"


def test_zero_extra_preserves_default_sampling_trajectory():
    torch.manual_seed(19)
    model = GuandanModel(SMALL)
    base = StageBPolicy.from_model(model, PolicyConfig(top_k=3))
    zero = StageBPolicy.from_model(model, PolicyConfig(top_k=3, candidate_mode="union",
                                                     candidate_extra=0))
    obs = torch.rand(3, gd.OBS_DIM)
    cand = torch.rand(25, gd.ACT_DIM)
    offsets = torch.tensor([0, 9, 17, 25])
    phase = torch.full((3,), int(gd.Phase.Play))
    first, second = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    a = base.act(obs, cand, offsets, phase, generator=first)
    b = zero.act(obs, cand, offsets, phase, generator=second)
    assert torch.equal(a.keep_index, b.keep_index)
    assert torch.equal(a.choice, b.choice)
    assert torch.equal(a.log_prob, b.log_prob)
    assert torch.equal(first.get_state(), second.get_state())


def test_tiny_ppo_union_stores_support_and_batched_evaluates(tmp_path):
    torch.manual_seed(9)
    model = GuandanModel(SMALL)
    initial = tmp_path / "initial.pt"
    save_checkpoint(initial, {"model_config": asdict(SMALL), "model": model.state_dict(),
                              "optimizer": {}, "config": {"action_mode": "canonical"},
                              "progress": {}, "rng": {}})
    config = PPOConfig(init_checkpoint=str(initial), critic_init="", critic_width=16,
                       critic_layers=1, opponent="greedy", num_envs=4, num_threads=1,
                       torch_threads=1, rollout_steps=384, epochs=1, minibatch_size=64,
                       top_k=3, candidate_mode="union", candidate_extra=2,
                       temperature=1.0, max_updates=1, max_seconds=60,
                       checkpoint_seconds=60, snapshot_updates=100, tensorboard=False)
    assert config.buffer_config().max_candidates == config.buffer_config().max_steps * 6
    assert PPOConfig(**asdict(config)).candidate_extra == 2
    assert PolicyConfig(**asdict(PolicyConfig(top_k=config.top_k,
                                            candidate_mode=config.candidate_mode,
                                            candidate_extra=config.candidate_extra))).candidate_extra == 2
    trainer = PPOTrainer(config, tmp_path / "run")
    try:
        trainer.collect()
        buffer = trainer.buffer
        assert buffer.n_steps
        assert buffer.cand_count[:buffer.n_steps].max() <= 6
        assert (buffer.cand_count[:buffer.n_steps] > config.top_k + 1).any()
        stored = buffer.gather(np.arange(min(buffer.n_steps, 32)))
        with torch.no_grad():
            logp, _ = trainer.policy.evaluate(stored["obs"], stored["cand"],
                                               stored["offsets"], stored["phase"], stored["chosen"])
            torch.testing.assert_close(logp, stored["logp"], atol=1e-5, rtol=1e-5)
            ref = trainer.policy.reference.score_candidates(
                stored["obs"], stored["cand"], stored["offsets"], stored["phase"])
            expected = segment_log_softmax(ref / config.temperature, stored["offsets"])
            torch.testing.assert_close(expected, stored["ref_logp"], atol=1e-5, rtol=1e-5)
        buffer.next_iteration()
        result = trainer.update()
        assert result["updates"] == 1
        trainer.save()
        loaded = load_policy(str(tmp_path / "run" / "latest.pt"))
        assert loaded.stage_b.config.candidate_extra == 2
        assert isinstance(loaded, PrunedPolicy)
        deals = generate_deals(2, 51)
        args = (deals, (loaded, loaded), (GreedyPolicy(), GreedyPolicy()))
        first = play_duplicate_batch(*args, seed=17, config=EvalConfig(batch_size=2))
        assert play_duplicate_batch(*args, seed=17, config=EvalConfig(batch_size=2)) == first
        scalar = [play_duplicate_teams(deal, args[1], args[2], 17 + i)
                  for i, deal in enumerate(deals)]
        assert [play_duplicate_teams(deal, args[1], args[2], 17 + i)
                for i, deal in enumerate(deals)] == scalar
    finally:
        trainer.close()


def test_union_support_crosses_actor_ipc(tmp_path):
    torch.manual_seed(13)
    model = GuandanModel(SMALL)
    initial = tmp_path / "initial.pt"
    save_checkpoint(initial, {"model_config": asdict(SMALL), "model": model.state_dict(),
                              "optimizer": {}, "config": {"action_mode": "canonical"},
                              "progress": {}, "rng": {}})
    config = PPOConfig(init_checkpoint=str(initial), critic_init="", critic_width=16,
                       critic_layers=1, opponent="greedy", num_envs=4, num_threads=1,
                       torch_threads=1, rollout_steps=128, epochs=1, minibatch_size=64,
                       actor_processes=2, top_k=3, candidate_mode="union", candidate_extra=2,
                       temperature=1.0, max_updates=1, max_seconds=60,
                       checkpoint_seconds=60, snapshot_updates=100, tensorboard=False)
    trainer = PPOTrainer(config, tmp_path / "run")
    try:
        trainer.collect()
        assert len(trainer.buffers) == 2
        for buffer in trainer.buffers:
            assert buffer.n_steps > 0
            assert buffer.cand_count[:buffer.n_steps].max() <= 6
            assert (buffer.cand_count[:buffer.n_steps] > config.top_k + 1).any()
            stored = buffer.gather(np.arange(min(buffer.n_steps, 24)))
            with torch.no_grad():
                logp, _ = trainer.policy.evaluate(stored["obs"], stored["cand"],
                                                   stored["offsets"], stored["phase"],
                                                   stored["chosen"])
                torch.testing.assert_close(logp, stored["logp"], atol=1e-5, rtol=1e-5)
    finally:
        trainer.close()
