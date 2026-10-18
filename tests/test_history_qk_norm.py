"""Config-gated private cross-attention Q/K normalization (``private_qk_norm``).

The default ("none") must leave the architecture record, the parameters and
the outputs of the current lineage untouched. The "query" and "qk" arms add
LayerNorms without consuming random numbers, run through every private
attention path, and bound the attention logits that saturate in the main
lineage (docs/reports/private-attention-diagnosis-2026-10-15.md).
"""
import gd
import numpy as np
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from eval.botzone.numpy_actor import NumpyHistoryActor
from eval.private_attention_audit import Stress, apply_stress, batch_inputs, capture, collect_sample
from train.history_model import (HistoryActor, HistoryPolicyConfig, checkpoint_payload,
                                 config_record, fresh_player, load_history_checkpoint,
                                 save_history_checkpoint)
from train.history_ppo import HistoryPPOConfig, HistoryTrainer, build_parser, config_from_args
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer
from train.history_snapshot_batch import validate_actor

BASE = dict(width=32, layers=1, heads=4)
ARMS = ("query", "qk")


@pytest.fixture(scope="module")
def sample():
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE), 3)
    return collect_sample(actor, envs=3, steps=160, seed=5, matches=3, minibatch_matches=3,
                          rows_per_match=20)


def test_default_record_and_parameters_unchanged():
    config = HistoryPolicyConfig(**BASE)
    assert config.private_qk_norm == "none"
    assert "private_qk_norm" not in config_record(config)
    actor, critic = fresh_player(config, 0)
    assert actor.query_norm is None and actor.q_norm is None and actor.k_norm is None
    assert not any(key.startswith(("query_norm.", "q_norm.", "k_norm."))
                   for key in actor.state_dict())
    payload = checkpoint_payload(actor, critic, lineage="x", seed=0)
    assert "private_qk_norm" not in payload["model_config"]
    with pytest.raises(ValueError, match="private_qk_norm"):
        HistoryPolicyConfig(**BASE, private_qk_norm="rms")


@pytest.mark.parametrize("arm", ARMS)
def test_arms_keep_initialisation_and_add_only_norms(arm):
    base_actor, base_critic = fresh_player(HistoryPolicyConfig(**BASE), 11)
    actor, critic = fresh_player(HistoryPolicyConfig(**BASE, private_qk_norm=arm), 11)
    base, state = base_actor.state_dict(), actor.state_dict()
    added = set(state) - set(base)
    expected = {"query_norm"} if arm == "query" else {"q_norm", "k_norm"}
    assert {key.split(".")[0] for key in added} == expected
    for key, value in base.items():
        assert torch.equal(state[key], value), key
    for (name, a), (_, b) in zip(base_critic.state_dict().items(), critic.state_dict().items()):
        assert torch.equal(a, b), name
    assert config_record(actor.config)["private_qk_norm"] == arm


def test_default_outputs_bitwise_and_arms_identity_free(sample):
    """Same weights: the default path is the old computation; arms change it."""
    batch = sample[0]
    inputs = batch_inputs(batch)
    base, _ = fresh_player(HistoryPolicyConfig(**BASE), 2)
    with torch.no_grad():
        reference = base.candidate_log_probs(inputs)
        width, heads = base.config.width, base.config.heads
        q = base.q_proj(base.private(inputs.obs.float()) + base.seat(inputs.seat))
        torch.testing.assert_close(
            base.project_query(base.private(inputs.obs.float()) + base.seat(inputs.seat)),
            q.view(len(q), heads, width // heads), rtol=0, atol=0)
        for arm in ARMS:
            actor, _ = fresh_player(HistoryPolicyConfig(**BASE, private_qk_norm=arm), 2)
            assert not torch.equal(actor.candidate_log_probs(inputs), reference)


@pytest.mark.parametrize("arm", ("none",) + ARMS)
def test_attention_paths_agree(arm, sample):
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE, private_qk_norm=arm), 4)
    batch = sample[0]
    inputs = batch_inputs(batch)
    with torch.no_grad():
        looped = actor.decision_states(None, inputs)
        actor.batched_match_attention = True
        batched = actor.decision_states(None, inputs)
        actor.batched_match_attention = False
        explicit = capture(actor, inputs)
        encoded = actor.encode_batch(inputs.streams)
        keys, values = actor.project_keys_values(encoded)
        query = actor.private(inputs.obs.float()) + actor.seat(inputs.seat)
        rows = torch.arange(len(query))
        streams = encoded[inputs.match_index]
        allowed = torch.arange(encoded.shape[1])[None] <= inputs.prefix[:, None]
        per_row = torch.cat([actor._attend(query[i:i + 1], keys[inputs.match_index[i:i + 1]],
                                           values[inputs.match_index[i:i + 1]], allowed[i:i + 1])
                             for i in rows])
        k, v = actor.project_keys_values(streams)
        independent = actor._attend_independent(query, k, v, allowed)
    torch.testing.assert_close(batched, looped, rtol=1e-5, atol=1e-5)
    assert explicit["state_error"] < 1e-5
    torch.testing.assert_close(independent, per_row, rtol=1e-5, atol=1e-5)


def test_windowed_actor_uses_key_norm():
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE, window=8, private_qk_norm="qk"), 1)
    stream = torch.randn(3, 9, BASE["width"]) * 50
    keys, values = actor.project_keys_values(stream)
    raw_keys, raw_values = actor.kv_proj(stream).chunk(2, dim=-1)
    torch.testing.assert_close(values, raw_values)
    per_head = keys.unflatten(-1, (BASE["heads"], -1))
    torch.testing.assert_close(per_head.mean(-1), torch.zeros(per_head.shape[:-1]),
                               rtol=0, atol=1e-5)
    assert not torch.allclose(keys, raw_keys)


def test_qk_bounds_logits_where_the_baseline_saturates(sample):
    """Trained-regime numerics on identical weights: the baseline becomes exactly
    one-hot with exact-zero q/k gradients; qk stays soft with nonzero gradients."""
    stress = Stress(query_mean=2700.0, q_gain=200.0, k_gain=100.0)
    batch = sample[0]
    inputs = batch_inputs(batch)
    results = {}
    for arm in ("none", "qk"):
        actor, _ = fresh_player(HistoryPolicyConfig(**BASE, private_qk_norm=arm), 6)
        apply_stress(actor, stress)
        seen = batch["prefix"].numpy() >= 1
        max_prob = capture(actor, inputs)["max_prob"][seen]
        actor.zero_grad()
        with sdpa_kernel(SDPBackend.MATH):
            loss = actor.candidate_log_probs(inputs)[batch["chosen"]].sum()
        loss.backward()
        q_grad = float(actor.q_proj.weight.grad.abs().max())
        results[arm] = (max_prob, q_grad)
    base_prob, base_grad = results["none"]
    qk_prob, qk_grad = results["qk"]
    assert (base_prob > 1 - 1e-6).mean() > 0.95
    assert base_grad == 0.0
    assert qk_prob.max() < 0.5
    assert qk_grad > 0.0


@pytest.mark.parametrize("arm", ARMS)
def test_checkpoint_round_trip(arm, tmp_path):
    actor, critic = fresh_player(HistoryPolicyConfig(**BASE, private_qk_norm=arm), 8)
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(0.01)
    path = tmp_path / "x.pt"
    save_history_checkpoint(path, checkpoint_payload(actor, critic, lineage="ab", seed=8))
    loaded, _, payload = load_history_checkpoint(path)
    assert payload["model_config"]["private_qk_norm"] == arm
    assert loaded.config.private_qk_norm == arm
    for key, value in actor.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)


def test_refused_by_paths_that_bypass_the_helpers():
    actor = HistoryActor(HistoryPolicyConfig(**BASE, private_qk_norm="qk"))
    with pytest.raises(ValueError, match="Q/K"):
        validate_actor(actor)
    validate_actor(HistoryActor(HistoryPolicyConfig(**BASE)))
    with pytest.raises(ValueError, match="Q/K"):
        NumpyHistoryActor({}, dict(width=32, heads=4, layers=1, max_rounds=16,
                                   private_qk_norm="query"))
    with pytest.raises(ValueError, match="Q/K"):
        HistoryPPOConfig(private_qk_norm="qk", batch_snapshot_policies=True)


def test_trainer_flag_and_one_update(tmp_path):
    assert config_from_args(build_parser().parse_args(["--output", "x"])).private_qk_norm == "none"
    args = build_parser().parse_args(["--output", "x", "--private-qk-norm", "qk"])
    config = config_from_args(args)
    assert config.private_qk_norm == "qk" and config.policy_config().private_qk_norm == "qk"
    config = HistoryPPOConfig(width=32, layers=1, heads=4, num_envs=3, steps_per_update=120,
                              seed=3, updates=1, epochs=1, minibatch_matches=2,
                              private_qk_norm="qk")
    trainer = HistoryTrainer(config, tmp_path / "qk")
    assert trainer.actor.q_norm is not None
    trainer.update()
    assert all(torch.isfinite(p).all() for p in trainer.actor.parameters())
    trainer.save()
    resumed = HistoryTrainer(config, tmp_path / "qk", resume=str(tmp_path / "qk" / "latest.pt"))
    assert resumed.actor.config.private_qk_norm == "qk"
    for key, value in trainer.actor.state_dict().items():
        assert torch.equal(resumed.actor.state_dict()[key], value)


def test_kv_cache_rollout_matches_recompute():
    """The collector's cached inference reads the same normalized keys."""
    actor, _ = fresh_player(HistoryPolicyConfig(**BASE, private_qk_norm="qk"), 9)
    env = gd.VecEnv(num_envs=2, num_threads=1, seed=4, log_public_actions=True, log_env_limit=2)
    store, buffer = MatchEventStore(), SequenceRolloutBuffer()
    HistoryCollector(env, actor, store, buffer, torch.Generator().manual_seed(1),
                     kv_cache=True).collect(60)
    rows = np.arange(len(buffer))
    inputs, chosen = buffer.decision_inputs(rows, store, "cpu")
    with torch.no_grad():
        recomputed = actor.candidate_log_probs(inputs)[chosen].numpy()
    np.testing.assert_allclose(recomputed, buffer.compact()["logp"], rtol=1e-4, atol=1e-5)
