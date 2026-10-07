"""Q-boosting estimator: reduction to GAE, zero action-sampling variance with an
exact Q, unbiasedness of the score-function term for an arbitrary Q at lambda 1,
and the masking conventions shared with compute_gae."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from train.history_rollout import compute_gae
from train.q_boost import policy_expectation, q_boost_advantages


def test_policy_expectation() -> None:
    assert policy_expectation([0.25, 0.75], [4.0, 0.0]) == 1.0
    with pytest.raises(ValueError):
        policy_expectation([0.5, 0.5], [1.0])


def test_reduces_to_gae_when_q_ignores_the_action() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(5, 9))
    rewards = rng.normal(size=(5, 9))
    dones = np.zeros((5, 9), bool)
    dones[:, -1] = True
    dones[2, 4] = True
    for gamma, lam in ((1.0, 0.95), (0.99, 0.9), (1.0, 0.0), (1.0, 1.0)):
        gae_adv, gae_ret = compute_gae(values, rewards, dones, gamma, lam)
        adv, target = q_boost_advantages(values, values, rewards, dones, gamma, lam)
        np.testing.assert_allclose(adv, gae_adv, atol=1e-5)
        np.testing.assert_allclose(target, gae_ret, atol=1e-5)


def test_lambda_zero_is_the_expected_one_step_residual() -> None:
    q = np.array([[1.0, 2.0, -1.0]])
    v = np.array([[0.5, 1.0, 0.0]])
    rewards = np.array([[0.0, 0.5, 3.0]])
    dones = np.array([[False, False, True]])
    adv, target = q_boost_advantages(q, v, rewards, dones, 1.0, 0.0)
    np.testing.assert_allclose(adv, [[0.0 + 1.0 - 0.5, 0.5 + 0.0 - 1.0, 3.0 - 0.0]], atol=1e-6)
    np.testing.assert_allclose(target, q + (adv - (q - v)), atol=1e-6)


def two_step_game():
    """Step 0: a0 ~ pi0, no reward. Step 1 (state set by a0): a1 ~ uniform, reward r1."""
    pi0 = np.array([0.3, 0.7])
    pi1 = np.array([0.5, 0.5])
    r1 = np.array([[1.0, -1.0], [-3.0, 5.0]])           # r1[a0, a1]
    q1_true = r1                                         # terminal after step 1
    v1_true = q1_true @ pi1
    q0_true = v1_true                                    # Q(s0, a0), no reward at step 0
    return pi0, pi1, r1, q0_true, q1_true, v1_true


def run(q0, q1, pi0, pi1, r1, a0, a1, lam):
    v0 = float(pi0 @ q0)
    v1 = float(pi1 @ q1[a0])
    q = np.array([[q0[a0], q1[a0][a1]]])
    v = np.array([[v0, v1]])
    rewards = np.array([[0.0, r1[a0, a1]]])
    dones = np.array([[False, True]])
    return q_boost_advantages(q, v, rewards, dones, 1.0, lam)[0][0]


@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_exact_q_has_no_future_action_noise(lam: float) -> None:
    pi0, pi1, r1, q0, q1, v1 = two_step_game()
    for a0 in range(2):
        advantages = [run(q0, q1, pi0, pi1, r1, a0, a1, lam) for a1 in range(2)]
        # step 0's advantage does not depend on the sampled step-1 action
        assert advantages[0][0] == pytest.approx(advantages[1][0], abs=1e-6)
        assert advantages[0][0] == pytest.approx(q0[a0] - float(pi0 @ q0), abs=1e-6)
    # a V-critic GAE trace, by contrast, carries the sampled a1
    v0_true = float(pi0 @ q0)
    gae = {}
    for a0, a1 in itertools.product(range(2), range(2)):
        values = np.array([[v0_true, v1[a0]]])
        adv, _ = compute_gae(values, np.array([[0.0, r1[a0, a1]]]), np.array([[False, True]]), 1.0, 1.0)
        gae[a0, a1] = adv[0, 0]
    assert gae[0, 0] != pytest.approx(gae[0, 1])


def test_lambda_one_score_function_term_is_unbiased_for_any_q() -> None:
    pi0, pi1, r1, q0_true, q1_true, v1_true = two_step_game()
    rng = np.random.default_rng(3)
    q0 = q0_true + rng.normal(scale=2.0, size=2)         # arbitrary wrong critic
    q1 = q1_true + rng.normal(scale=2.0, size=(2, 2))
    v0 = float(pi0 @ q0)
    for a0 in range(2):
        expected = sum(pi1[a1] * run(q0, q1, pi0, pi1, r1, a0, a1, 1.0)[0] for a1 in range(2))
        # E[A_hat | s0, a0] = Q^pi(s0, a0) - V_critic(s0): a baseline that is a function of s0 only
        assert expected == pytest.approx(q0_true[a0] - v0, abs=1e-6)


def test_padding_and_terminals_match_an_unpadded_trajectory() -> None:
    rng = np.random.default_rng(1)
    q = rng.normal(size=(1, 5))
    v = rng.normal(size=(1, 5))
    rewards = rng.normal(size=(1, 5))
    dones = np.zeros((1, 5), bool)
    dones[0, -1] = True
    adv, target = q_boost_advantages(q, v, rewards, dones, 1.0, 0.95)
    pad = 3
    padded = [np.concatenate([x, np.full((1, pad), 7.0)], axis=1) for x in (q, v, rewards)]
    padded_dones = np.concatenate([dones, np.zeros((1, pad), bool)], axis=1)
    mask = np.concatenate([np.ones((1, 5), bool), np.zeros((1, pad), bool)], axis=1)
    adv_padded, target_padded = q_boost_advantages(*padded, padded_dones, 1.0, 0.95, mask=mask)
    np.testing.assert_allclose(adv_padded[:, :5], adv, atol=1e-6)
    np.testing.assert_allclose(target_padded[:, :5], target, atol=1e-6)
    assert not adv_padded[:, 5:].any() and not target_padded[:, 5:].any()
    # a terminal mid-trajectory cuts the trace
    dones[0, 2] = True
    cut_adv, _ = q_boost_advantages(q, v, rewards, dones, 1.0, 0.95)
    first, _ = q_boost_advantages(q[:, :3], v[:, :3], rewards[:, :3], dones[:, :3], 1.0, 0.95)
    np.testing.assert_allclose(cut_adv[:, :3], first, atol=1e-6)


def test_shape_checks() -> None:
    with pytest.raises(ValueError):
        q_boost_advantages(np.zeros((1, 3)), np.zeros((1, 2)), np.zeros((1, 3)), np.zeros((1, 3), bool))
    with pytest.raises(ValueError):
        q_boost_advantages(np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3), bool),
                           mask=np.ones((1, 2), bool))


def test_synthetic_game_exact_q_beats_exact_v_gae() -> None:
    from eval.q_boost_synthetic import compare

    row = compare(depth=4, branching=3, trajectories=2000, sigma=0.0, lam=0.95,
                  policy_scale=1.0, seed=0)
    assert row["q_boost_mse"] < 1e-12
    assert row["gae_exact_v_mse"] > 0.05


def test_anova_components_recover_known_variances() -> None:
    from eval.critic_noise_split import anova_components

    rng = np.random.default_rng(0)
    own_sd, opp_sd, inter_sd = 2.0, 1.0, 0.5
    totals = []
    for _ in range(400):
        grid = (rng.normal(scale=own_sd, size=(8, 1)) + rng.normal(scale=opp_sd, size=(1, 8))
                + rng.normal(scale=inter_sd, size=(8, 8)))
        totals.append(list(anova_components(grid).values()))
    own, opp, inter = np.mean(totals, axis=0)
    assert own == pytest.approx(own_sd ** 2, rel=0.15)
    assert opp == pytest.approx(opp_sd ** 2, rel=0.15)
    assert inter == pytest.approx(inter_sd ** 2, rel=0.15)
