"""Q-boosting advantage estimation (Fan and Farina, arXiv 2605.19235, eq. 3.2-3.3).

``compute_gae`` backs up sampled future actions through a V critic. Here the
critic is an action-value ``Q(s, a)`` and every backup step takes the policy's
expectation over the legal candidates, so a future action's sampling noise
never enters the trace. The inputs are two numbers per step:

* ``q``: the critic's value of the action actually taken, ``Q(s_t, a_t)``;
* ``v``: the policy expectation ``V(s_t) = sum_a pi(a | s_t) Q(s_t, a)``.

With ``delta+_t = r_t + gamma * V(s_{t+1}) - Q(s_t, a_t)`` the advantage is
``Q(s_t, a_t) - V(s_t) + sum_{t' >= t} (lambda*gamma)^(t'-t) delta+_t'`` and the
critic target is ``Q(s_t, a_t) + sum (lambda*gamma)^(t'-t) delta+_t'``.

Shapes and masking follow ``train.history_rollout.compute_gae``: arrays are
``[..., T]``, ``dones[t]`` is a terminal step with nothing bootstrapped across
it, padding (``mask`` false) contributes nothing and the value past the last
column is zero. If ``Q`` does not depend on the action (``q == v``) the result
is exactly GAE.
"""
from __future__ import annotations

import numpy as np


def policy_expectation(probs: np.ndarray, q_values: np.ndarray) -> float:
    """``sum_a pi(a) Q(a)`` over one decision's candidates."""
    probs = np.asarray(probs, np.float64)
    q_values = np.asarray(q_values, np.float64)
    if probs.shape != q_values.shape or probs.ndim != 1:
        raise ValueError("probs and q_values must be 1-D with one entry per candidate")
    return float(np.dot(probs, q_values))


def q_boost_advantages(q: np.ndarray, v: np.ndarray, rewards: np.ndarray, dones: np.ndarray,
                       gamma: float = 1.0, lam: float = 0.95,
                       mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Q-boosting advantages and critic targets along the last axis, ``[..., T]``.

    Returns float32 ``(advantages, targets)``; ``targets`` is the regression
    target for ``Q(s_t, a_t)``.
    """
    q = np.asarray(q, np.float64)
    v = np.asarray(v, np.float64)
    rewards = np.asarray(rewards, np.float64)
    dones = np.asarray(dones, bool)
    if not (q.shape == v.shape == rewards.shape == dones.shape):
        raise ValueError("q, v, rewards and dones must share a shape")
    valid = np.ones(q.shape, bool) if mask is None else np.asarray(mask, bool)
    if valid.shape != q.shape:
        raise ValueError("mask must match q")
    batch_shape = q.shape[:-1]
    trace = np.zeros(q.shape, np.float64)
    next_trace = np.zeros(batch_shape, np.float64)
    next_value = np.zeros(batch_shape, np.float64)
    for t in range(q.shape[-1] - 1, -1, -1):
        live = 1.0 - dones[..., t]
        delta = rewards[..., t] + gamma * next_value * live - q[..., t]
        accumulated = delta + gamma * lam * live * next_trace
        keep = valid[..., t]
        accumulated = np.where(keep, accumulated, 0.0)
        trace[..., t] = accumulated
        next_trace = accumulated
        next_value = np.where(keep, v[..., t], 0.0)
    advantages = np.where(valid, q - v + trace, 0.0)
    targets = np.where(valid, q + trace, 0.0)
    return advantages.astype(np.float32), targets.astype(np.float32)
