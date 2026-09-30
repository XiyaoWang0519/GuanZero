"""Planted-habit diagnostic (train/history_habit.py): styled pack, views, init, resume.

The central check: what each view's learner read at collection is exactly
what PPO recomputes from stored rows (on-policy log-probabilities match), so
FULL, ROUND and ORACLE differ only in what the learner may see.
"""
import numpy as np
import pytest
import torch

from train.history_habit import (IDENTITY_STYLE, RoundEventStore, StyledActor, style_feature)
from train.history_model import (DecisionInputs, HistoryActor, HistoryPolicyConfig, StreamBatch,
                                 checkpoint_payload, config_record, fresh_player,
                                 load_history_checkpoint, save_history_checkpoint,
                                 segment_log_softmax)
from train.history_ppo import HistoryPPOConfig, HistoryTrainer

ARCH = dict(width=32, layers=1, heads=4, response_mode="auxiliary")


@pytest.fixture(scope="module")
def base_checkpoint(tmp_path_factory):
    """A small trained-architecture checkpoint used as both pack base and init."""
    path = tmp_path_factory.mktemp("base") / "base.pt"
    trainer = HistoryTrainer(HistoryPPOConfig(**ARCH, num_envs=4, steps_per_update=60, seed=5,
                                              updates=1, epochs=1, minibatch_matches=2),
                             path.parent / "run")
    trainer.update()
    trainer.save(path)
    return path


def habit_config(base, view, **overrides):
    values = dict(**ARCH, num_envs=6, steps_per_update=120, seed=11, updates=1, epochs=1,
                  minibatch_matches=2, snapshot_updates=0, habit_pack=str(base),
                  habit_init=str(base), habit_view=view, habit_axis="lead_single",
                  habit_strength=1.0)
    values.update(overrides)
    return HistoryPPOConfig(**values)


def toy_inputs(cand_types, decisions):
    """DecisionInputs whose candidates carry only a type one-hot."""
    cand = torch.zeros(len(cand_types), 154)
    cand[torch.arange(len(cand_types)), 108 + torch.as_tensor(cand_types)] = 1
    offsets = torch.as_tensor(decisions)
    n = len(offsets) - 1
    return DecisionInputs(streams=None, match_index=torch.zeros(n, dtype=torch.long),
                          prefix=torch.zeros(n, dtype=torch.long), obs=torch.zeros(n, 1),
                          seat=torch.zeros(n, dtype=torch.long), cand=cand, offsets=offsets)


def test_style_features_follow_the_declared_axes():
    # decision 0 follows (has Pass): pass, single, bomb; decision 1 leads: single, pair, SF
    inputs = toy_inputs([0, 1, 8, 1, 2, 9], [0, 3, 6])
    assert style_feature(inputs, "pass").tolist() == [1, 0, 0, 0, 0, 0]
    assert style_feature(inputs, "bomb").tolist() == [0, 0, 1, 0, 0, 1]
    assert style_feature(inputs, "lead_single").tolist() == [0, 0, 0, 1, 0, 0]


def test_styled_actor_tilts_only_the_feature_mass():
    actor = StyledActor(HistoryPolicyConfig(width=16, layers=1, heads=2))
    inputs = toy_inputs([1, 2, 3, 0, 1], [0, 3, 5])
    base = segment_log_softmax(torch.randn(5), inputs.rows, inputs.decisions)
    for z in (-1, 1):
        styled = actor.set_style("lead_single", 1.5, z).styled_log_probs(inputs, base).exp()
        p = base.exp()
        # leading decision: single tilted by exp(1.5 z), the others keep their ratio
        q0 = p[0] / p[:3].sum()
        expected = q0 * np.exp(1.5 * z) / (q0 * np.exp(1.5 * z) + 1 - q0)
        assert torch.allclose(styled[0], expected, atol=1e-6)
        assert torch.allclose(styled[1] / styled[2], p[1] / p[2], atol=1e-5)
        assert torch.allclose(styled[3:], p[3:], atol=1e-6)   # following: untouched
        assert torch.allclose(styled[:3].sum(), torch.tensor(1.0), atol=1e-6)


def test_style_input_is_recorded_only_when_on_and_starts_neutral():
    off = HistoryPolicyConfig(width=16, layers=1, heads=2)
    assert "style_input" not in config_record(off)
    actor, critic = fresh_player(off, 0)
    assert "style_input" not in checkpoint_payload(actor, critic, lineage="x", seed=0)["model_config"]
    on = HistoryPolicyConfig(width=16, layers=1, heads=2, style_input=True)
    assert config_record(on)["style_input"] is True
    oracle = HistoryActor(on)
    missing, _ = oracle.load_state_dict(actor.state_dict(), strict=False)
    assert missing == ["style_embedding.weight"]
    # Every other parameter keeps its registration order: the new one is last.
    assert [n for n, _ in oracle.named_parameters()][:-1] == [n for n, _ in actor.named_parameters()]


def test_pack_assignment_one_learner_team_and_match_fixed_style(base_checkpoint, tmp_path):
    trainer = HistoryTrainer(habit_config(base_checkpoint, "full"), tmp_path)
    trainer.collect()
    assert trainer.collector.assignments
    for (env, match), seats in trainer.collector.assignments.items():
        learner = [s for s in range(4) if seats[s] == 0]
        assert learner in ([0, 2], [1, 3])
        others = {int(seats[s]) for s in range(4) if seats[s] != 0}
        assert len(others) == 1
        assert IDENTITY_STYLE[others.pop()] == trainer.population.style(env, match)
    rows = trainer.buffer.compact()
    for env, match, seat in zip(rows["env"], rows["match"], rows["seat"]):
        assert trainer.collector.assignments[(int(env), int(match))][int(seat)] == 0
    pack = trainer.population.models
    assert {m.style_z for m in pack.values()} == {-1, 1}
    assert all(not p.requires_grad for m in pack.values() for p in m.parameters())


PRODUCTION = dict(rollout_kv_cache=True, causal_sdpa=True, rollout_batched_attention=True,
                  learner_batched_attention=True, learner_length_groups=4)


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("view", ["full", "round", "oracle"])
def test_collected_log_probs_match_the_learner_recompute(base_checkpoint, tmp_path, view, fast):
    trainer = HistoryTrainer(habit_config(base_checkpoint, view, **(PRODUCTION if fast else {})),
                             tmp_path)
    trainer.collect()
    data = trainer.buffer.compact()
    rows = np.arange(len(trainer.buffer))
    assert len(rows) > 50
    log_prob, _ = trainer.recompute_log_probs(rows)
    np.testing.assert_allclose(log_prob.detach().numpy(), data["logp"][rows], atol=2e-5)
    if view == "round":
        store = trainer.store
        assert isinstance(store, RoundEventStore)
        for env, match, rnd, prefix in zip(data["env"], data["match"], data["round"],
                                           data["prefix"]):
            stream = store.round_stream(int(env), int(match), int(rnd))
            assert prefix <= stream.prefix
            assert (stream.rounds[:prefix] == rnd).all()
        full = [store.stream(int(e), int(m)).prefix for e, m in zip(data["env"], data["match"])]
        assert (data["prefix"] <= np.asarray(full)).all()
        assert (data["prefix"][data["round"] > 0] < np.asarray(full)[data["round"] > 0]).all()


def test_oracle_reads_the_style_and_other_views_do_not(base_checkpoint, tmp_path):
    trainer = HistoryTrainer(habit_config(base_checkpoint, "oracle"), tmp_path)
    assert trainer.actor.config.style_input
    trainer.collect()
    rows = np.arange(min(len(trainer.buffer), 64))
    before, _ = trainer.recompute_log_probs(rows)
    # The zero-initialised embedding leaves the initial policy equal to the source's.
    source, _, _ = load_history_checkpoint(base_checkpoint)
    batch = trainer.buffer.training_batch(rows, trainer.store, "cpu", fields=())
    base = source.candidate_log_probs(batch.inputs)[batch.chosen]
    torch.testing.assert_close(before, base, atol=1e-6, rtol=0)
    with torch.no_grad():
        trainer.actor.style_embedding.weight.normal_()
    after, _ = trainer.recompute_log_probs(rows)
    assert not torch.allclose(before, after)
    full = HistoryTrainer(habit_config(base_checkpoint, "full"), tmp_path / "full")
    assert not full.actor.config.style_input and not hasattr(full.actor, "style_embedding")


def test_init_keeps_source_weights_and_optimizer(base_checkpoint, tmp_path):
    source, source_critic, payload = load_history_checkpoint(base_checkpoint)
    trainer = HistoryTrainer(habit_config(base_checkpoint, "full", lr=1e-5), tmp_path)
    for name, value in source.state_dict().items():
        assert torch.equal(trainer.actor.state_dict()[name], value)
    for name, value in source_critic.state_dict().items():
        assert torch.equal(trainer.critic.state_dict()[name], value)
    saved = payload["optimizer"]["actor"]["state"]
    state = trainer.actor_optimizer.state_dict()["state"]
    assert set(state) == set(saved)
    assert torch.equal(state[0]["exp_avg"], saved[0]["exp_avg"])
    assert trainer.actor_optimizer.param_groups[0]["lr"] == 1e-5
    assert trainer.lineage.startswith("habit-full-")
    assert trainer.habit_init["update"] == payload["progress"]["updates"]


@pytest.mark.parametrize("view", ["round", "oracle"])
def test_habit_update_save_and_resume(base_checkpoint, tmp_path, view):
    trainer = HistoryTrainer(habit_config(base_checkpoint, view, updates=2), tmp_path)
    line = trainer.update()
    assert line["update_samples"] > 0 and np.isfinite(line["policy_loss"])
    path = trainer.save()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["habit"]["view"] == view and payload["habit_init"]["path"] == str(base_checkpoint)
    assert payload["population"]["pack"]["axis"] == "lead_single"
    resumed = HistoryTrainer(habit_config(base_checkpoint, view, updates=2), tmp_path,
                             resume=path)
    assert resumed.lineage == trainer.lineage
    assert resumed.actor.config.style_input == (view == "oracle")
    assert resumed.update()["update_samples"] > 0


def test_habit_configuration_guards(base_checkpoint):
    with pytest.raises(ValueError):
        habit_config(base_checkpoint, "full", snapshot_updates=2)
    with pytest.raises(ValueError):
        habit_config(base_checkpoint, "sideways")
    with pytest.raises(ValueError):
        HistoryPPOConfig(**ARCH, habit_view="round")
    with pytest.raises(ValueError):
        HistoryPPOConfig(**ARCH, habit_init="x.pt")
    with pytest.raises(ValueError):
        habit_config(base_checkpoint, "oracle", rollout_kv_cache=True, rollout_private_graphs=True)


@pytest.mark.parametrize("view", ["round", "oracle"])
def test_ppo_minibatch_starts_on_policy(base_checkpoint, tmp_path, view):
    """The learner's own minibatch path (length groups, labels, style) reproduces
    the behaviour probabilities before any step: ratio 1."""
    trainer = HistoryTrainer(habit_config(base_checkpoint, view, **PRODUCTION), tmp_path)
    trainer.collect()
    trainer.buffer.finalize(trainer.refresh_values())
    rows = next(iter(trainer.minibatches()))
    stats = trainer.minibatch_loss(rows)
    assert float(stats["ratio_deviation"]) < 1e-4
    assert np.isfinite(float(stats["response_loss"]))


@pytest.mark.parametrize("view", ["full", "round", "oracle"])
def test_pack_evaluation_runs_each_view(base_checkpoint, tmp_path, view):
    from eval.history_habit_eval import evaluate
    trainer = HistoryTrainer(habit_config(base_checkpoint, view), tmp_path)
    path = trainer.save()
    result = evaluate(path, base_checkpoint, view, "lead_single", 1.0, matches=1, envs=2,
                      seed=4, steps_per_chunk=200)
    assert result["meta"]["matches"] >= 1 and len(result["ret"]) == result["meta"]["rounds"]
    assert set(np.unique(result["z"])) <= {-1, 1}
    assert set(np.unique(result["team"])) <= {0, 1}
    if view != "oracle":
        with pytest.raises(ValueError):
            evaluate(path, base_checkpoint, "oracle", "lead_single", 1.0, 1, 2, 4)
