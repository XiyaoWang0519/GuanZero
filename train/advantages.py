"""NumPy advantage estimation shared by the PPO rollout buffers.

Round-based history training uses the default zero bootstrap. Callers with
truncated trajectories may explicitly provide a final value.
"""
from __future__ import annotations

import numpy as np


def compute_gae(values: np.ndarray, rewards: np.ndarray, dones: np.ndarray,
                gamma: float = 1.0, lam: float = 0.95,
                mask: np.ndarray | None = None,
                last_value: np.ndarray | float | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Generalized advantage estimation along the last axis (time).

    Arrays are `[..., T]`. `dones[t]` marks a terminal step: nothing is
    bootstrapped across it. `mask[t]` false marks padding; padded positions get
    zero advantage and return and do not feed into earlier steps. `last_value`
    bootstraps past the final column (default 0, i.e. treat it as terminal).
    Returns `(advantages, returns)` as float32 with `returns = adv + values`.
    """
    values = np.asarray(values, np.float64)
    rewards = np.asarray(rewards, np.float64)
    dones = np.asarray(dones, bool)
    if not (values.shape == rewards.shape == dones.shape):
        raise ValueError("values, rewards and dones must share a shape")
    valid = np.ones(values.shape, bool) if mask is None else np.asarray(mask, bool)
    if valid.shape != values.shape:
        raise ValueError("mask must match values")
    batch_shape = values.shape[:-1]
    advantages = np.zeros(values.shape, np.float64)
    next_adv = np.zeros(batch_shape, np.float64)
    next_value = np.broadcast_to(
        np.asarray(0.0 if last_value is None else last_value, np.float64),
        batch_shape).copy()
    for t in range(values.shape[-1] - 1, -1, -1):
        live = 1.0 - dones[..., t]
        delta = rewards[..., t] + gamma * next_value * live - values[..., t]
        adv = delta + gamma * lam * live * next_adv
        keep = valid[..., t]
        adv = np.where(keep, adv, 0.0)
        advantages[..., t] = adv
        next_adv = adv
        next_value = np.where(keep, values[..., t], 0.0)
    returns = np.where(valid, advantages + values, 0.0)
    return advantages.astype(np.float32), returns.astype(np.float32)

