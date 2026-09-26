"""Exploration floor for learner seats: pi_b = (1 - eps) softmax(logits / T) + eps / n.

Defaults (T = 1, eps = 0, cap = 1) must reproduce the pre-change sampler and
learner bit for bit; otherwise the buffer keeps the target log-probability in
``logp`` and the sampled one in ``behaviour_logp``, and the learner weights
each advantage by min(cap, exp(logp - behaviour_logp)).
"""
import math

import gd
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from train.history_model import (HIDDEN_DIM, DecisionInputs, ExplorationSample, HistoryActor,
                                 HistoryPolicyConfig, fresh_player)
from train.history_ppo import HistoryPPOConfig, HistoryTrainer, build_parser, config_from_args
from train.history_rollout import (ROW_FIELDS, HistoryCollector, MatchEventStore,
                                   SequenceRolloutBuffer)
from test_history_rollout import make_env

CONFIG = HistoryPolicyConfig(width=32, layers=1, heads=4)
LOSS_KEYS = ("policy_total", "value_total", "policy_loss", "value_loss", "entropy",
             "approx_kl", "clip_fraction", "ratio_deviation")


def small_config(**overrides) -> HistoryPPOConfig:
    values = dict(width=32, layers=1, heads=4, num_envs=4, steps_per_update=90, seed=3,
                  updates=1, epochs=1, minibatch_matches=2)
    values.update(overrides)
    return HistoryPPOConfig(**values)


# ---- pre-change code, copied verbatim (modulo self -> actor/trainer) ----------------

@torch.no_grad()
def pre_change_act(actor, inputs, generator=None, greedy=False, *, encoded=None,
                   max_candidates=None):
    log_probs = actor.candidate_log_probs(inputs, encoded=encoded)
    counts = inputs.counts
    offsets = inputs.offsets
    longest = max_candidates if max_candidates is not None else (int(counts.max()) if len(counts) else 0)
    table = torch.full((inputs.decisions, max(longest, 1)), float("-inf"),
                       device=log_probs.device, dtype=log_probs.dtype)
    local = torch.arange(len(log_probs), device=log_probs.device) - offsets[:-1][inputs.rows]
    table[inputs.rows, local] = log_probs
    if greedy:
        choice = table.argmax(1)
    else:
        uniform = torch.rand(table.shape, generator=generator, device=table.device,
                             dtype=table.dtype).clamp_min(1e-12)
        gumbel = -(-uniform.log()).log()
        choice = (table + gumbel).argmax(1)
    chosen = table.gather(1, choice[:, None])[:, 0]
    return choice, chosen


def pre_change_minibatch_loss(trainer, rows):
    cfg = trainer.config
    buffer = trainer.buffer
    log_prob, entropy = trainer.recompute_log_probs(rows)
    old = torch.as_tensor(buffer.compact()["logp"][rows], device=trainer.device)
    advantage = torch.as_tensor(buffer.advantage[rows], device=trainer.device)
    returns = torch.as_tensor(buffer.returns[rows], device=trainer.device)
    log_ratio = log_prob - old
    ratio = log_ratio.exp()
    if cfg.normalize_advantages and len(advantage) > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    clipped = torch.minimum(ratio * advantage,
                            ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * advantage)
    surrogate = -clipped.mean()
    policy_total = surrogate - cfg.entropy * entropy.mean()
    data = buffer.compact()
    obs = torch.as_tensor(data["obs"][rows], device=trainer.device)
    hidden = torch.as_tensor(data["hidden"][rows], device=trainer.device)
    value_loss = F.mse_loss(trainer.critic(obs, hidden), returns)
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean()
        clip_fraction = ((ratio - 1).abs() > cfg.clip).float().mean()
        ratio_deviation = (ratio - 1).abs().max()
    return {"policy_total": policy_total, "value_total": cfg.value_coef * value_loss,
            "policy_loss": surrogate, "value_loss": value_loss, "entropy": entropy.mean(),
            "approx_kl": approx_kl, "clip_fraction": clip_fraction,
            "ratio_deviation": ratio_deviation}


# ---- fixtures -------------------------------------------------------------------

def collect(steps=60, temperature=1.0, epsilon=0.0, seed=7, **kwargs):
    actor, _ = fresh_player(CONFIG, 5)
    store, buffer = MatchEventStore(), SequenceRolloutBuffer()
    collector = HistoryCollector(make_env(), actor, store, buffer,
                                 torch.Generator().manual_seed(seed), record_choices=True,
                                 temperature=temperature, epsilon=epsilon, **kwargs)
    stats = collector.collect(steps)
    return actor, store, buffer, collector, stats


@pytest.fixture(scope="module")
def real_inputs():
    actor, store, buffer, _, _ = collect(40)
    inputs, _ = buffer.decision_inputs(np.arange(min(len(buffer), 64)), store, "cpu")
    assert int(inputs.counts.max()) > 8
    return actor, inputs


def independent_log_probs(actor, inputs, temperature, epsilon):
    """Per decision float64 (target, behaviour) log-probability vectors."""
    with torch.no_grad():
        state = actor.decision_states(None, inputs)
        logits = actor.candidate_logits(state, inputs.cand, inputs.offsets).double()
    offsets = inputs.offsets.tolist()
    result = []
    for i in range(inputs.decisions):
        l = logits[offsets[i]:offsets[i + 1]]
        n = len(l)
        target = torch.log_softmax(l, 0)
        behaviour = torch.log((1 - epsilon) * torch.softmax(l / temperature, 0) + epsilon / n)
        result.append((target, behaviour))
    return result


# ---- 1. defaults are bitwise identical ----------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_default_act_and_explore_match_pre_change_sampler(real_inputs, seed):
    actor, inputs = real_inputs
    generators = [torch.Generator().manual_seed(seed) for _ in range(3)]
    before_choice, before_logp = pre_change_act(actor, inputs, generators[0])
    choice, logp = actor.act(inputs, generators[1])
    sample = actor.explore(inputs, generators[2])
    assert isinstance(sample, ExplorationSample)
    for c, l in ((choice, logp), (sample.choice, sample.logp)):
        assert torch.equal(c, before_choice) and torch.equal(l, before_logp)
    assert torch.equal(sample.behaviour_logp, sample.logp)
    assert not sample.uniform_pick.any()
    # Same random-stream consumption: one Gumbel table, nothing more.
    assert torch.equal(generators[0].get_state(), generators[1].get_state())
    assert torch.equal(generators[0].get_state(), generators[2].get_state())
    greedy = actor.act(inputs, greedy=True)
    assert all(torch.equal(a, b) for a, b in zip(greedy, pre_change_act(actor, inputs, greedy=True)))


def test_default_collector_matches_pre_change_sampler(monkeypatch):
    _, _, new_buffer, new, _ = collect(80)

    def patched(self, inputs, generator=None, *, temperature=1.0, epsilon=0.0, encoded=None,
                max_candidates=None):
        choice, logp = pre_change_act(self, inputs, generator, encoded=encoded,
                                      max_candidates=max_candidates)
        zeros = torch.zeros(len(choice))
        return ExplorationSample(choice, logp, logp, zeros.bool(), zeros)

    monkeypatch.setattr(HistoryActor, "explore", patched)
    _, _, old_buffer, old, _ = collect(80)
    assert len(new.choice_log) == len(old.choice_log)
    for a, b in zip(new.choice_log, old.choice_log):
        np.testing.assert_array_equal(a, b)
    a, b = new_buffer.compact(), old_buffer.compact()
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    assert np.array_equal(a["behaviour_logp"], a["logp"])


@pytest.fixture
def single_thread():
    # Multithreaded CPU backward reductions are not bitwise reproducible even
    # for the same loss twice; one thread makes the gradient comparison exact.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def test_default_learner_loss_is_bitwise_pre_change(tmp_path, single_thread):
    trainer = HistoryTrainer(small_config(), tmp_path)
    trainer.collect()
    data = trainer.buffer.compact()
    assert np.array_equal(data["behaviour_logp"], data["logp"])
    samples = trainer.buffer.finalize(trainer.refresh_values(), trainer.config.gamma,
                                      trainer.config.gae_lambda)
    assert samples > 0
    for rows in trainer.minibatches():
        new = trainer.minibatch_loss(rows)
        old = pre_change_minibatch_loss(trainer, rows)
        for key in LOSS_KEYS:
            assert torch.equal(new[key], old[key]), key
        assert float(new["behaviour_weight_mean"]) == 1.0
        assert float(new["behaviour_weight_cap_fraction"]) == 0.0
        grads = []
        for terms in (new, old):
            trainer.actor.zero_grad(set_to_none=True)
            terms["policy_total"].backward()
            grads.append([p.grad.clone() for p in trainer.actor.parameters()])
        assert all(torch.equal(x, y) for x, y in zip(*grads))
    line = trainer.learn()
    assert line["behaviour_weight_mean"] == 1.0 and line["behaviour_weight_cap_fraction"] == 0.0


# ---- 2. behaviour log-probabilities -------------------------------------------------

@pytest.mark.parametrize("temperature,epsilon", [(0.7, 0.2), (1.8, 0.05), (1.0, 0.3),
                                                 (2.5, 0.0), (1.0, 1.0)])
def test_explore_behaviour_log_prob_matches_float64_mixture(real_inputs, temperature, epsilon):
    actor, inputs = real_inputs
    sample = actor.explore(inputs, torch.Generator().manual_seed(4), temperature=temperature,
                           epsilon=epsilon)
    expected = independent_log_probs(actor, inputs, temperature, epsilon)
    counts = inputs.counts
    assert ((sample.choice >= 0) & (sample.choice < counts)).all()
    for i, (target, behaviour) in enumerate(expected):
        c = int(sample.choice[i])
        assert abs(float(sample.logp[i]) - float(target[c])) < 1e-5
        assert abs(float(sample.behaviour_logp[i]) - float(behaviour[c])) < 1e-5
        entropy = -float((behaviour.exp() * behaviour).sum())
        assert abs(float(sample.behaviour_entropy[i]) - entropy) < 1e-4
    if epsilon == 1.0:
        assert sample.uniform_pick.all()


def test_collector_stores_target_and_behaviour_log_probs():
    temperature, epsilon = 1.6, 0.25
    actor, store, buffer, _, stats = collect(60, temperature, epsilon)
    data = buffer.compact()
    rows = np.arange(len(buffer))
    inputs, _ = buffer.decision_inputs(rows, store, "cpu")
    expected = independent_log_probs(actor, inputs, temperature, epsilon)
    target = np.asarray([float(t[c]) for (t, _), c in zip(expected, data["chosen"])])
    behaviour = np.asarray([float(b[c]) for (_, b), c in zip(expected, data["chosen"])])
    np.testing.assert_allclose(data["logp"], target, rtol=0, atol=1e-5)
    np.testing.assert_allclose(data["behaviour_logp"], behaviour, rtol=0, atol=1e-5)
    assert not np.array_equal(data["logp"], data["behaviour_logp"])
    # Roughly epsilon of the learner rows took the uniform branch.
    fraction = stats.epsilon_picks / stats.learner_rows
    sigma = math.sqrt(epsilon * (1 - epsilon) / stats.learner_rows)
    assert abs(fraction - epsilon) < 5 * sigma
    assert stats.behaviour_entropy_sum > 0


# ---- 3. sampling frequencies ----------------------------------------------------------

CHI2_999 = {2: 13.82, 3: 16.27}   # 99.9% quantiles of chi-square with k - 1 dof


@pytest.mark.parametrize("temperature,epsilon", [(1.0, 0.0), (0.5, 0.1), (2.0, 0.3)])
@pytest.mark.parametrize("logits", [[2.0, 0.5, -1.0], [1.5, 0.0, -0.5, -3.0]])
def test_sampling_frequencies_follow_the_mixture(temperature, epsilon, logits):
    actor = HistoryActor(CONFIG)
    k, draws = len(logits), 40_000
    fixed = torch.log_softmax(torch.tensor(logits), 0)
    actor.candidate_log_probs = lambda inputs, encoded=None: fixed.repeat(draws)
    inputs = DecisionInputs(streams=None, match_index=torch.zeros(draws, dtype=torch.long),
                            prefix=torch.zeros(draws, dtype=torch.long),
                            obs=torch.zeros((draws, CONFIG.obs_dim), dtype=torch.uint8),
                            seat=torch.zeros(draws, dtype=torch.long),
                            cand=torch.zeros((draws * k, CONFIG.act_dim), dtype=torch.uint8),
                            offsets=torch.arange(0, draws * k + 1, k))
    sample = actor.explore(inputs, torch.Generator().manual_seed(11), temperature=temperature,
                           epsilon=epsilon)
    observed = np.bincount(sample.choice.numpy(), minlength=k)
    assert observed.size == k
    l = np.asarray(logits, np.float64) / temperature
    p = np.exp(l - l.max())
    p = (1 - epsilon) * p / p.sum() + epsilon / k
    expected = draws * p
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    assert chi2 < CHI2_999[k - 1], (observed, expected)
    np.testing.assert_allclose(sample.behaviour_logp.double().exp().numpy(),
                               p[sample.choice.numpy()], rtol=1e-5)
    picks = float(sample.uniform_pick.double().mean())
    assert abs(picks - epsilon) < 5 * math.sqrt(max(epsilon * (1 - epsilon), 1e-12) / draws)


# ---- 4. importance weights ------------------------------------------------------------

@pytest.mark.parametrize("cap", [1.0, 2.0])
def test_learner_weights_are_truncated_importance_ratios(tmp_path, cap):
    trainer = HistoryTrainer(small_config(rollout_temperature=2.0, rollout_epsilon=0.2,
                                          behaviour_weight_cap=cap), tmp_path)
    trainer.collect()
    trainer.buffer.finalize(trainer.refresh_values(), trainer.config.gamma,
                            trainer.config.gae_lambda)
    data = trainer.buffer.compact()
    raw = np.exp(data["logp"].astype(np.float64) - data["behaviour_logp"])
    assert (raw != 1).any()
    fractions = []
    for rows in trainer.minibatches():
        terms = trainer.minibatch_loss(rows)
        weight = torch.as_tensor(data["logp"][rows] - data["behaviour_logp"][rows]).exp()
        capped = weight > cap
        weight = weight.clamp(max=cap)
        np.testing.assert_allclose(weight.numpy(), np.minimum(cap, raw[rows]), rtol=1e-6)
        assert float(terms["behaviour_weight_mean"]) == pytest.approx(float(weight.mean()), rel=1e-6)
        assert float(terms["behaviour_weight_cap_fraction"]) == pytest.approx(
            float(capped.float().mean()))
        fractions.append(float(capped.float().mean()))
        # The surrogate uses the normalized advantage times the weight.
        with torch.no_grad():
            log_prob, _ = trainer.recompute_log_probs(rows)
        ratio = (log_prob - torch.as_tensor(data["logp"][rows])).exp()
        advantage = torch.as_tensor(trainer.buffer.advantage[rows])
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) * weight
        clip = trainer.config.clip
        surrogate = -torch.minimum(ratio * advantage,
                                   ratio.clamp(1 - clip, 1 + clip) * advantage).mean()
        assert float(terms["policy_loss"].detach()) == pytest.approx(float(surrogate), abs=1e-6)
    if cap == 1.0:
        assert 0 < max(fractions) < 1


def test_update_line_reports_exploration_metrics(tmp_path):
    trainer = HistoryTrainer(small_config(rollout_temperature=1.5, rollout_epsilon=0.1), tmp_path)
    line = trainer.update()
    assert 0 < line["behaviour_weight_mean"] <= 1.0
    assert 0 <= line["behaviour_weight_cap_fraction"] <= 1
    assert 0 < line["rollout_epsilon_pick_fraction"] < 0.3
    assert line["rollout_behaviour_entropy"] > 0
    for key in HistoryTrainer.STAT_KEYS:
        assert key in line
    plain = HistoryTrainer(small_config(), tmp_path / "plain").update()
    assert plain["behaviour_weight_mean"] == 1.0 and plain["behaviour_weight_cap_fraction"] == 0.0
    assert plain["rollout_epsilon_pick_fraction"] == 0.0


# ---- 5. frozen snapshot seats ----------------------------------------------------------

def test_snapshot_seats_sample_exactly_as_before(monkeypatch):
    actor, _ = fresh_player(CONFIG, 5)
    frozen, _ = fresh_player(CONFIG, 6)
    calls = {"act": 0}
    real_act = frozen.act

    def spy_act(inputs, generator=None, greedy=False, **kwargs):
        assert not greedy and set(kwargs) <= {"encoded", "max_candidates"}
        calls["act"] += 1
        return real_act(inputs, generator, greedy, **kwargs)

    def no_explore(*args, **kwargs):
        raise AssertionError("snapshot seats must not use the exploration floor")

    monkeypatch.setattr(frozen, "act", spy_act)
    monkeypatch.setattr(frozen, "explore", no_explore)
    learner_calls = []
    real_explore = actor.explore

    def spy_explore(*args, **kwargs):
        learner_calls.append((kwargs["temperature"], kwargs["epsilon"]))
        return real_explore(*args, **kwargs)

    monkeypatch.setattr(actor, "explore", spy_explore)
    store, buffer = MatchEventStore(), SequenceRolloutBuffer()
    collector = HistoryCollector(make_env(), actor, store, buffer, torch.Generator().manual_seed(2),
                                 seat_policy=lambda env, match: [0, 1, 1, 1],
                                 resolve_policy={1: frozen}.__getitem__,
                                 temperature=3.0, epsilon=0.5)
    collector.collect(60)
    assert calls["act"] > 0 and learner_calls
    assert set(learner_calls) == {(3.0, 0.5)}
    assert collector.policy_decisions[1] > 0
    assert (buffer.compact()["seat"] == 0).all()


def test_snapshot_seat_choices_do_not_depend_on_learner_settings():
    """With every seat frozen nothing is explored and the settings are inert."""
    runs = []
    for temperature, epsilon in ((1.0, 0.0), (3.0, 0.5)):
        actor, _ = fresh_player(CONFIG, 5)
        frozen, _ = fresh_player(CONFIG, 6)
        collector = HistoryCollector(make_env(), actor, MatchEventStore(), SequenceRolloutBuffer(),
                                     torch.Generator().manual_seed(2), record_choices=True,
                                     seat_policy=lambda env, match: [1, 1, 1, 1],
                                     resolve_policy={1: frozen}.__getitem__,
                                     temperature=temperature, epsilon=epsilon)
        collector.collect(40)
        runs.append(collector.choice_log)
    for a, b in zip(*runs):
        np.testing.assert_array_equal(a, b)


# ---- 6. configuration, CLI and checkpoints ----------------------------------------------

def test_cli_flags_and_validation(tmp_path):
    parser = build_parser()
    defaults = config_from_args(parser.parse_args(["--output", str(tmp_path)]))
    assert (defaults.rollout_temperature, defaults.rollout_epsilon,
            defaults.behaviour_weight_cap) == (1.0, 0.0, 1.0)
    assert defaults == HistoryPPOConfig()
    args = parser.parse_args(["--output", str(tmp_path), "--rollout-temperature", "1.5",
                              "--rollout-epsilon", "0.05", "--behaviour-weight-cap", "2.0"])
    config = config_from_args(args)
    assert (config.rollout_temperature, config.rollout_epsilon,
            config.behaviour_weight_cap) == (1.5, 0.05, 2.0)
    for bad in (dict(rollout_temperature=0.0), dict(rollout_temperature=math.inf),
                dict(rollout_epsilon=-0.1), dict(rollout_epsilon=1.5),
                dict(behaviour_weight_cap=0.0), dict(rollout_temperature=math.nan)):
        with pytest.raises(ValueError):
            HistoryPPOConfig(**bad)


def test_settings_survive_checkpoint_and_resume(tmp_path):
    config = small_config(steps_per_update=40, rollout_temperature=1.25, rollout_epsilon=0.07,
                          behaviour_weight_cap=1.5)
    trainer = HistoryTrainer(config, tmp_path / "run")
    trainer.update()
    path = trainer.save()
    payload = torch.load(path, weights_only=False)
    assert payload["config"]["rollout_temperature"] == 1.25
    assert payload["config"]["rollout_epsilon"] == 0.07
    assert payload["config"]["behaviour_weight_cap"] == 1.5
    resumed = HistoryTrainer(small_config(updates=2), tmp_path / "resumed", resume=path)
    assert resumed.config == HistoryPPOConfig.from_payload(payload["config"], updates=2)
    assert (resumed.config.rollout_temperature, resumed.config.rollout_epsilon,
            resumed.config.behaviour_weight_cap) == (1.25, 0.07, 1.5)
    assert (resumed.collector.temperature, resumed.collector.epsilon) == (1.25, 0.07)
    # Checkpoints written before the exploration floor resume with the defaults.
    old = {k: v for k, v in payload["config"].items()
           if k not in ("rollout_temperature", "rollout_epsilon", "behaviour_weight_cap")}
    legacy = HistoryPPOConfig.from_payload(old)
    assert (legacy.rollout_temperature, legacy.rollout_epsilon,
            legacy.behaviour_weight_cap) == (1.0, 0.0, 1.0)


# ---- 7. buffer layout -----------------------------------------------------------------

def test_buffer_fields_keep_their_dtypes_and_carry_behaviour_logp():
    _, _, buffer, _, _ = collect(90, temperature=1.4, epsilon=0.2)
    data = buffer.compact()
    n = len(buffer)
    for name in ROW_FIELDS:
        dtype = np.float32 if name in ("logp", "behaviour_logp") else np.int64
        assert data[name].dtype == dtype and data[name].shape == (n,), name
    assert data["obs"].dtype == np.uint8 and data["obs"].shape == (n, int(gd.OBS_DIM))
    assert data["hidden"].dtype == np.uint8 and data["hidden"].shape == (n, HIDDEN_DIM)
    assert data["cand"].dtype == np.uint8
    assert data["cand"].shape == (int(data["cand_count"].sum()), int(gd.ACT_DIM))
    empty = SequenceRolloutBuffer().compact()
    for name in ROW_FIELDS:
        dtype = np.float32 if name in ("logp", "behaviour_logp") else np.int64
        assert empty[name].dtype == dtype and empty[name].shape == (0,), name
    # Carried rows keep their behaviour log-probability across the update boundary.
    before = {(int(e), int(m), int(s), int(p)): float(b) for e, m, s, p, b in
              zip(data["env"], data["match"], data["seat"], data["prefix"], data["behaviour_logp"])}
    buffer.finalize(np.zeros(n, np.float32))
    buffer.next_iteration()
    carried = buffer.compact()
    assert len(buffer) > 0
    for e, m, s, p, b in zip(carried["env"], carried["match"], carried["seat"],
                             carried["prefix"], carried["behaviour_logp"]):
        assert before[(int(e), int(m), int(s), int(p))] == float(b)
