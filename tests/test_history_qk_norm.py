"""Private-attention norms (``query_norm``, ``qk_norm``; history_model).

Both flags default off and the off path must stay bit for bit the existing
one on all three private-attention paths (single query masked SDPA, batched
independent rows, ``_attend_by_match``). On, the paths agree with a plain
reference implementation and with each other, forward and gradients are
finite, and the norms receive gradient. Inserting them into a trained actor
or lineage (``extend_actor``, resume with ``--resume-set``) keeps every other
weight and Adam moment, records the change and resets the attention; the
merged snapshot head and the population follow the same architecture.
"""
from dataclasses import replace
import math

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from eval.botzone.export import export
from train.history_model import (QK_NORM_EPS, QK_NORM_REFERENCE_KEYS, HistoryActor,
                                 HistoryPolicyConfig, checkpoint_payload, config_record,
                                 extend_actor, fresh_player, load_history_checkpoint,
                                 rms_normalize, save_history_checkpoint)
from train.history_ppo import (HistoryPPOConfig, HistoryTrainer, build_parser, config_from_args,
                               parse_resume_overrides)
from train.history_snapshot_batch import SnapshotHeads, head_parameters
from test_history_model import Rollout
from test_history_snapshot_batch import SPECS, Step, merged, reference

FLAG_SETS = [dict(query_norm=True), dict(qk_norm=True), dict(query_norm=True, qk_norm=True)]
SMALL = dict(width=32, layers=1, heads=4, action_width=16, fusion_width=16,
             critic_width=16, critic_layers=1)


def small_config(**overrides) -> HistoryPolicyConfig:
    return HistoryPolicyConfig(**{**SMALL, **overrides})


def trainer_config(**overrides) -> HistoryPPOConfig:
    values = dict(width=32, layers=1, heads=4, num_envs=4, steps_per_update=90, seed=3,
                  updates=1, epochs=1, minibatch_matches=2)
    values.update(overrides)
    return HistoryPPOConfig(**values)


@pytest.fixture(scope="module")
def rollout() -> Rollout:
    return Rollout(num_envs=3, steps=160, capture=(0, 60, 159))


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA private-attention norms")
    return request.param


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def loud_actor(config: HistoryPolicyConfig, seed: int) -> HistoryActor:
    """A fresh actor whose private query is large and mean-dominated, like the
    trained lineage's (norms in the thousands), so the norms visibly matter."""
    actor = fresh_player(config, seed)[0]
    with torch.no_grad():
        actor.private[-2].weight.mul_(40)
        actor.private[-2].bias.add_(60)
    return actor


# ---- configuration -----------------------------------------------------------------

def test_defaults_off_record_cli_and_resume_overrides():
    plain = small_config()
    assert plain.query_norm is False and plain.qk_norm is False
    assert "query_norm" not in config_record(plain) and "qk_norm" not in config_record(plain)
    assert config_record(small_config(query_norm=True))["query_norm"] is True
    assert config_record(small_config(qk_norm=True))["qk_norm"] is True
    assert "query_norm" not in config_record(small_config(qk_norm=True))
    assert HistoryPPOConfig().query_norm is False and HistoryPPOConfig().qk_norm is False
    args = build_parser().parse_args(["--output", "x"])
    assert config_from_args(args).policy_config() == HistoryPolicyConfig(width=64, layers=2)
    args = build_parser().parse_args(["--output", "x", "--query-norm", "--qk-norm"])
    policy = config_from_args(args).policy_config()
    assert policy.query_norm and policy.qk_norm
    assert parse_resume_overrides(["query_norm=true", "qk_norm=0"]) == {"query_norm": True,
                                                                        "qk_norm": False}
    # A checkpoint record round-trips through the config.
    assert HistoryPolicyConfig(**config_record(policy)) == policy
    actor = HistoryActor(policy)
    assert actor.private_norm_parameter_names() == {"query_norm.weight", "query_norm.bias",
                                                    "qk_temperature.log_temperature"}
    assert HistoryActor(plain).private_norm_parameter_names() == set()


def test_off_adds_no_parameters_and_norm_init_is_as_documented():
    plain, _ = fresh_player(small_config(), 5)
    for flags in FLAG_SETS:
        torch.manual_seed(5)
        with_norms, critic = fresh_player(small_config(**flags), 5)
        # Every pre-existing parameter and the critic are initialised as without.
        for name, value in plain.state_dict().items():
            assert torch.equal(with_norms.state_dict()[name], value), name
        plain_critic = fresh_player(small_config(), 5)[1].state_dict()
        for name, value in critic.state_dict().items():
            assert torch.equal(plain_critic[name], value), name
        added = set(with_norms.state_dict()) - set(plain.state_dict())
        assert added == with_norms.private_norm_parameter_names()
        if flags.get("query_norm"):
            assert torch.equal(with_norms.query_norm.weight, torch.ones(32))
            assert torch.equal(with_norms.query_norm.bias, torch.zeros(32))
        if flags.get("qk_norm"):
            depth = 32 // 4
            expected = math.log(QK_NORM_REFERENCE_KEYS) / math.sqrt(depth)
            assert torch.allclose(with_norms.qk_temperature(), torch.full((4,), expected))
            # A perfect match among QK_NORM_REFERENCE_KEYS keys gets half the mass.
            logit = expected * math.sqrt(depth)
            share = math.exp(logit) / (math.exp(logit) + QK_NORM_REFERENCE_KEYS - 1)
            assert abs(share - 0.5) < 1e-3


# ---- the three attention paths -----------------------------------------------------

def reference_attention(actor: HistoryActor, query, encoded, match_index, prefix):
    """Plain per-row implementation of the private attention with the norms
    written out explicitly; row by row, so it is independent of the batching."""
    cfg = actor.config
    heads, depth = cfg.heads, cfg.width // cfg.heads
    out = []
    for i in range(len(query)):
        row = query[i:i + 1]
        if cfg.query_norm:
            row = F.layer_norm(row, (cfg.width,), actor.query_norm.weight, actor.query_norm.bias,
                               actor.query_norm.eps)
        q = actor.q_proj(row).view(heads, depth)
        keys, values = actor.kv_proj(encoded[match_index[i], :prefix[i] + 1]).chunk(2, dim=-1)
        k = keys.view(-1, heads, depth)
        if cfg.qk_norm:
            q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + QK_NORM_EPS)
            q = q * actor.qk_temperature()[:, None]
            k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + QK_NORM_EPS)
        scores = torch.einsum("hd,shd->hs", q, k) / math.sqrt(depth)
        weights = torch.softmax(scores, dim=-1)
        attended = torch.einsum("hs,shd->hd", weights, values.view(-1, heads, depth))
        out.append(actor.out_proj(attended.reshape(1, cfg.width)))
    return torch.cat(out)


def attention_problem(seed: int, device="cpu", width: int = 32, matches: int = 3, length: int = 13,
                      n: int = 9):
    g = torch.Generator().manual_seed(seed)
    query = torch.randn(n, width, generator=g) * 50 + 100          # mean-dominated, like trained
    encoded = torch.randn(matches, length, width, generator=g)
    match_index = torch.randint(matches, (n,), generator=g)
    prefix = torch.randint(length, (n,), generator=g)
    return tuple(x.to(device) for x in (query, encoded, match_index, prefix))


def three_paths(actor: HistoryActor, query, encoded, match_index, prefix):
    keys, values = actor.kv_proj(encoded).chunk(2, dim=-1)
    positions = torch.arange(encoded.shape[1], device=encoded.device)
    single = torch.cat([actor._attend(query[i:i + 1], keys[m:m + 1], values[m:m + 1],
                                      positions[None] <= prefix[i])
                        for i, m in enumerate(match_index.tolist())])
    by_match = actor._attend_by_match(query, keys, values, match_index, prefix)
    own_k, own_v = actor.kv_proj(encoded[match_index]).chunk(2, dim=-1)
    independent = actor._attend_independent(query, own_k, own_v, positions[None] <= prefix[:, None])
    return single, by_match, independent


def legacy_attend(actor: HistoryActor, query, keys, values, allowed):
    """``HistoryActor._attend`` as committed before the norms (d05d75a), shared stream."""
    width, heads = actor.config.width, actor.config.heads
    depth = width // heads
    q = actor.q_proj(query).view(-1, heads, depth)
    k = keys.view(keys.shape[0], keys.shape[1], heads, depth)
    v = values.view(values.shape[0], values.shape[1], heads, depth)
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k[0].transpose(0, 1)[None], v[0].transpose(0, 1)[None],
        attn_mask=allowed[None, None])[0].transpose(0, 1)
    return actor.out_proj(out.reshape(-1, width))


@pytest.mark.parametrize("seed", [1, 2])
def test_off_is_the_previous_code_bit_for_bit(seed, device):
    actor = loud_actor(small_config(), seed).to(device)
    query, encoded, match_index, prefix = attention_problem(seed, device)
    keys, values = actor.kv_proj(encoded).chunk(2, dim=-1)
    positions = torch.arange(encoded.shape[1], device=device)
    legacy = torch.cat([legacy_attend(actor, query[i:i + 1], keys[m:m + 1], values[m:m + 1],
                                      positions[None] <= prefix[i])
                        for i, m in enumerate(match_index.tolist())])
    single, by_match, independent = three_paths(actor, query, encoded, match_index, prefix)
    assert torch.equal(single, legacy)
    # Un-normalised logits of a loud query are huge (the saturated regime of the
    # diagnosis), so reduction order moves the outputs far more than with norms.
    expected = reference_attention(actor, query, encoded, match_index, prefix)
    for value in (single, by_match, independent):
        assert torch.allclose(value, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("flags", FLAG_SETS, ids=lambda f: "+".join(sorted(f)))
@pytest.mark.parametrize("seed", [1, 2])
def test_on_matches_reference_on_all_paths_with_finite_gradients(flags, seed, device):
    actor = loud_actor(small_config(**flags), seed).to(device)
    problem = attention_problem(seed, device)
    expected = reference_attention(actor, *problem)
    paths = three_paths(actor, *problem)
    for name, value in zip(("single", "by_match", "independent"), paths):
        assert value.isfinite().all(), name
        assert torch.allclose(value, expected, rtol=1e-4, atol=1e-5), name
    for value in paths:
        actor.zero_grad()
        value.square().sum().backward(retain_graph=True)
        for pname, parameter in actor.named_parameters():
            if parameter.grad is not None:
                assert parameter.grad.isfinite().all(), pname
        # The inserted norms are live parameters: they receive gradient.
        if flags.get("query_norm"):
            assert actor.query_norm.weight.grad.abs().sum() > 0
        if flags.get("qk_norm"):
            assert actor.qk_temperature.log_temperature.grad.abs().sum() > 0
            # q/k see a gradient too: the normalised logits are not saturated.
            assert actor.q_proj.weight.grad.abs().sum() > 0


def test_qk_norm_logits_are_bounded_by_the_temperature():
    actor = loud_actor(small_config(qk_norm=True), 7)
    query, encoded, match_index, prefix = attention_problem(7)
    heads, depth = 4, 8
    q = actor._query_heads(query)
    k = actor._key_heads(actor.kv_proj(encoded).chunk(2, dim=-1)[0].view(3, -1, heads, depth))
    scores = torch.einsum("nhd,mshd->nmhs", q, k) / math.sqrt(depth)
    bound = actor.qk_temperature() * math.sqrt(depth)
    assert (scores.abs() <= bound[None, None, :, None] * (1 + 1e-5)).all()
    assert torch.allclose(rms_normalize(k).pow(2).mean(-1), torch.ones(3, 13, heads), atol=1e-4)
    # Tiny keys stay finite: the epsilon keeps a zero row at zero.
    assert torch.equal(rms_normalize(torch.zeros(2, heads, 8)), torch.zeros(2, heads, 8))


# ---- real decisions: rollout, batched paths, window control ------------------------

@pytest.mark.parametrize("flags", FLAG_SETS, ids=lambda f: "+".join(sorted(f)))
def test_real_decisions_finite_across_layouts(rollout, flags):
    actor = loud_actor(small_config(**flags), 11).train()
    for snapshot in (rollout.step0, rollout.last):
        inputs = snapshot.inputs()
        log_probs = actor.candidate_log_probs(inputs)
        assert log_probs.isfinite().all() and log_probs.shape == (inputs.cand.shape[0],)
        loss = -log_probs[inputs.offsets[:-1]].mean()
        actor.zero_grad()
        loss.backward()
        assert all(p.grad.isfinite().all() for p in actor.parameters() if p.grad is not None)
        # Opt-in batched rollout and learner paths: same values up to reduction order.
        with torch.no_grad():
            actor.batched_private_attention = True
            batched = actor.candidate_log_probs(inputs)
            actor.batched_private_attention = False
            actor.batched_match_attention = True
            inputs_many = replace(inputs, one_decision_per_stream=False)
            by_match = actor.candidate_log_probs(inputs_many)
            actor.batched_match_attention = False
        assert torch.allclose(batched, log_probs, rtol=1e-4, atol=1e-5)
        assert torch.allclose(by_match, log_probs, rtol=1e-4, atol=1e-5)
    # The windowed control shares ``_attend``.
    windowed = loud_actor(small_config(window=8, **flags), 11)
    assert windowed.candidate_log_probs(rollout.last.inputs()).isfinite().all()


# ---- insertion into a trained actor and lineage ------------------------------------

def test_extend_actor_inserts_norms_keeps_weights_and_resets_attention(rollout):
    base = loud_actor(small_config(aux_heads="outcome"), 13)
    inputs = rollout.last.inputs()
    with torch.no_grad():
        before = base.candidate_log_probs(inputs)
    for flags in FLAG_SETS:
        extended = extend_actor(base, replace(base.config, **flags))
        for name, value in base.state_dict().items():
            assert torch.equal(extended.state_dict()[name], value), name
        added = set(extended.state_dict()) - set(base.state_dict())
        assert added == extended.private_norm_parameter_names()
        with torch.no_grad():
            after = extended.candidate_log_probs(inputs)
        assert after.isfinite().all()
        # Documented: the inserted norm changes the attention, hence the policy.
        assert not torch.allclose(after, before, atol=1e-3)
        # Nothing is ever removed; nothing else may change.
        with pytest.raises(ValueError, match="removed"):
            extend_actor(extended, base.config)
        with pytest.raises(ValueError, match="only"):
            extend_actor(base, replace(base.config, width=64, **flags))
    # Heads and norms can be added together, in one step.
    both = extend_actor(loud_actor(small_config(), 13),
                        small_config(aux_heads="next,outcome", query_norm=True, qk_norm=True))
    assert both.aux_head_names == ("next", "outcome") and both.config.qk_norm


def test_resume_inserts_norms_records_change_and_keeps_adam_moments(tmp_path):
    output = tmp_path / "plain"
    trainer = HistoryTrainer(trainer_config(updates=2, snapshot_updates=1,
                                            aux_heads="outcome", aux_coef=0.1), output)
    trainer.run()
    assert trainer.population.models, "a snapshot must exist to exercise the population load"
    checkpoint = output / "latest.pt"
    saved_actor, _, payload = load_history_checkpoint(checkpoint)
    assert "query_norm" not in payload["model_config"]
    resumed = HistoryTrainer(trainer_config(updates=3), tmp_path / "norms", resume=checkpoint,
                             resume_overrides={"query_norm": True, "qk_norm": True})
    assert resumed.actor.config.query_norm and resumed.actor.config.qk_norm
    assert resumed.actor.aux_head_names == ("outcome",)
    assert resumed.config_changes[-1]["changes"] == {"qk_norm": [False, True],
                                                     "query_norm": [False, True]}
    for name, value in saved_actor.state_dict().items():
        assert torch.equal(resumed.actor.state_dict()[name], value), name
    # Adam moments follow the parameter names; the norms start without state.
    state = resumed.actor_optimizer.state_dict()["state"]
    names = [n for n, _ in resumed.actor.named_parameters()]
    saved_names = [n for n, _ in saved_actor.named_parameters()]
    assert len(state) == len(saved_names)
    for saved_index, name in enumerate(saved_names):
        moments = payload["optimizer"]["actor"]["state"][saved_index]
        assert torch.equal(state[names.index(name)]["exp_avg"], moments["exp_avg"]), name
    for name in resumed.actor.private_norm_parameter_names():
        assert names.index(name) not in state
    # Resident snapshots take the learner's architecture, with the fresh norms.
    assert len(resumed.population.models) == len(payload["population"]["snapshots"])
    for model in resumed.population.models.values():
        assert model.config == resumed.actor.config
        assert torch.equal(model.qk_temperature.log_temperature,
                           resumed.actor.qk_temperature.log_temperature)
        assert model.config.qk_norm and "norms_inserted" in resumed.population.metadata[
            next(iter(resumed.population.metadata))]
    line = resumed.update()
    assert np.isfinite(line["policy_loss"]) and np.isfinite(line["aux_loss"])
    assert torch.isfinite(resumed.actor.qk_temperature.log_temperature).all()
    resumed.save()
    reloaded, _, record = load_history_checkpoint(tmp_path / "norms" / "latest.pt")
    assert record["model_config"]["query_norm"] is True and record["model_config"]["qk_norm"] is True
    assert reloaded.config.query_norm and reloaded.config.qk_norm
    again = HistoryTrainer(trainer_config(updates=4), tmp_path / "again",
                           resume=tmp_path / "norms" / "latest.pt")
    # A plain resume adds no record: the insertion stays the last change, at update 2.
    assert again.actor.config.qk_norm and len(again.config_changes) == 1
    assert again.config_changes[-1]["at_update"] == 2
    with pytest.raises(ValueError, match="removed"):
        HistoryTrainer(trainer_config(updates=4), tmp_path / "removed",
                       resume=tmp_path / "norms" / "latest.pt",
                       resume_overrides={"qk_norm": False})


def test_resume_adds_heads_after_norms_maps_moments_by_name(tmp_path):
    """Norms are registered after the heads: a lineage that trained with norms and
    later adds heads has its saved parameters out of prefix order."""
    output = tmp_path / "norms-first"
    trainer = HistoryTrainer(trainer_config(updates=2, query_norm=True, qk_norm=True), output)
    trainer.run()
    saved_actor, _, payload = load_history_checkpoint(output / "latest.pt")
    resumed = HistoryTrainer(trainer_config(updates=3), tmp_path / "heads", resume=output / "latest.pt",
                             resume_overrides={"aux_heads": "outcome", "aux_coef": 0.1})
    names = [n for n, _ in resumed.actor.named_parameters()]
    saved_names = [n for n, _ in saved_actor.named_parameters()]
    assert names.index("qk_temperature.log_temperature") != saved_names.index("qk_temperature.log_temperature")
    state = resumed.actor_optimizer.state_dict()["state"]
    for saved_index, name in enumerate(saved_names):
        moments = payload["optimizer"]["actor"]["state"][saved_index]
        assert torch.equal(state[names.index(name)]["exp_avg_sq"], moments["exp_avg_sq"]), name
    assert np.isfinite(resumed.update()["policy_loss"])


# ---- merged snapshot head, Botzone export -----------------------------------------

@pytest.mark.parametrize("flags", FLAG_SETS, ids=lambda f: "+".join(sorted(f)))
@pytest.mark.parametrize("name", ["mixed", "multi-decision", "single-row-identity"])
def test_merged_snapshot_head_matches_each_identity(flags, name, device):
    torch.use_deterministic_algorithms(True)
    try:
        step = Step(SPECS[name], seed=list(SPECS).index(name), shorter_prefix=name == "multi-decision")
        config = HistoryPolicyConfig(width=32, layers=1, heads=4, action_width=16, fusion_width=24,
                                     critic_width=16, response_mode="auxiliary", **flags)
        policies = {i: loud_actor(config, 40 + i).to(device) for i, _, _ in SPECS[name]}
        groups, layout, fields, log_probs = merged(step, policies, device)
        expected = reference(step, policies, groups, device)
        assert log_probs.isfinite().all()
        assert float((log_probs - expected).abs().max()) < 2e-5
        heads = SnapshotHeads()
        heads.slot(1, policies[next(iter(policies))])
        assert set(heads.tensors) >= ({"n0_w", "n0_b"} if flags.get("query_norm") else set())
        assert set(heads.tensors) >= ({"k_b", "qk_t"} if flags.get("qk_norm") else set())
        # Norm parameters are part of the slot's invalidation signature.
        assert all(any(p is t for t in head_parameters(policies[1]))
                   for n, p in policies[1].named_parameters()
                   if n in policies[1].private_norm_parameter_names())
        with pytest.raises(ValueError, match="one architecture"):
            heads.slot(2, loud_actor(replace(config, query_norm=False, qk_norm=False), 3).to(device))
    finally:
        torch.use_deterministic_algorithms(False)


def test_botzone_export_refuses_norm_actors(tmp_path):
    actor, critic = fresh_player(small_config(query_norm=True), 2)
    path = tmp_path / "norms.pt"
    save_history_checkpoint(path, checkpoint_payload(actor, critic, lineage="test", seed=2))
    with pytest.raises(ValueError, match="query_norm"):
        export(path, tmp_path / "norms.npz")
