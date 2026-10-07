"""Break-even critic error: GAE with a V critic vs Q-boosting with a Q critic.

A synthetic zero-reward-until-the-end game with a stochastic policy, small
enough to solve exactly: ``depth`` decisions, ``branching`` candidates each,
state = the action path, one terminal payoff per full path, the policy a fixed
random softmax at every state. ``Q^pi`` and ``V^pi`` come from backward
induction, so the true advantage ``A(s, a) = Q^pi(s, a) - V^pi(s)`` is known.

Critic errors are iid Gaussian: ``Q = Q^pi + sigma * xi`` per state-action,
``V = V^pi + sigma * eta`` per state, with the same ``sigma``. Trajectories are
sampled on-policy and each estimator's mean squared error against the true
advantage is averaged over all decisions. ``sigma = 0`` isolates the
action-sampling noise that exact-V GAE keeps and Q-boosting removes; the
column where Q-boosting stops beating exact-V GAE is the Q accuracy the real
critic must reach. This says nothing about Guandan's actual Q error.

    python -m eval.q_boost_synthetic --depth 7 --branching 4 --trajectories 20000
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from train.history_rollout import compute_gae
from train.q_boost import q_boost_advantages


def build_game(depth: int, branching: int, rng: np.random.Generator, policy_scale: float):
    """Exact ``policy[t]`` (K^t, K), ``q[t]`` (K^t, K), ``v[t]`` (K^t,)."""
    payoffs = rng.normal(size=(branching,) * depth)
    policy = []
    for t in range(depth):
        logits = policy_scale * rng.normal(size=(branching ** t, branching))
        e = np.exp(logits - logits.max(-1, keepdims=True))
        policy.append(e / e.sum(-1, keepdims=True))
    q = [None] * depth
    v = [None] * depth
    q[depth - 1] = payoffs.reshape(branching ** (depth - 1), branching)
    for t in range(depth - 1, -1, -1):
        v[t] = (policy[t] * q[t]).sum(-1)
        if t:
            q[t - 1] = v[t].reshape(branching ** (t - 1), branching)
    return policy, q, v


def compare(depth: int, branching: int, trajectories: int, sigma: float, lam: float,
            policy_scale: float, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    policy, q, v = build_game(depth, branching, rng, policy_scale)
    q_noisy = [x + sigma * rng.normal(size=x.shape) for x in q]
    v_noisy = [x + sigma * rng.normal(size=x.shape) for x in v]
    index = np.zeros(trajectories, np.int64)               # path index at each depth
    paths, chosen = [], []
    for t in range(depth):
        paths.append(index.copy())
        cumulative = policy[t][index].cumsum(-1)
        action = (rng.random(trajectories)[:, None] > cumulative).sum(-1).clip(max=branching - 1)
        chosen.append(action)
        index = index * branching + action
    rows = np.arange(trajectories)
    # Q-critic quantities along each trajectory
    q_taken = np.stack([q_noisy[t][paths[t], chosen[t]] for t in range(depth)], -1)
    v_expected = np.stack([(policy[t][paths[t]] * q_noisy[t][paths[t]]).sum(-1)
                           for t in range(depth)], -1)
    v_gae = np.stack([v_noisy[t][paths[t]] for t in range(depth)], -1)
    v_exact = np.stack([v[t][paths[t]] for t in range(depth)], -1)
    truth = np.stack([q[t][paths[t], chosen[t]] - v[t][paths[t]] for t in range(depth)], -1)
    payoff = np.zeros((trajectories, depth))
    payoff[:, -1] = q[depth - 1][paths[depth - 1], chosen[depth - 1]]
    dones = np.zeros((trajectories, depth), bool)
    dones[:, -1] = True
    boost, _ = q_boost_advantages(q_taken, v_expected, payoff, dones, 1.0, lam)
    gae, _ = compute_gae(v_gae, payoff, dones, 1.0, lam)
    gae_exact, _ = compute_gae(v_exact, payoff, dones, 1.0, lam)
    del rows
    mse = lambda estimate: float(np.mean((estimate - truth) ** 2))
    return {"sigma": sigma, "q_boost_mse": mse(boost), "gae_noisy_v_mse": mse(gae),
            "gae_exact_v_mse": mse(gae_exact), "advantage_variance": float(np.mean(truth ** 2))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--branching", type=int, default=4)
    parser.add_argument("--trajectories", type=int, default=20000)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--policy-scale", type=float, default=1.0)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.5, 0.8])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    rows = []
    for sigma in args.sigmas:
        runs = [compare(args.depth, args.branching, args.trajectories, sigma, args.lam,
                        args.policy_scale, seed) for seed in range(args.seeds)]
        row = {key: float(np.mean([r[key] for r in runs])) for key in runs[0]}
        rows.append(row)
        print(f"sigma {sigma:4.2f}  q-boost {row['q_boost_mse']:.4f}  "
              f"gae(noisy V) {row['gae_noisy_v_mse']:.4f}  gae(exact V) {row['gae_exact_v_mse']:.4f}  "
              f"E[A^2] {row['advantage_variance']:.4f}")
    if args.output:
        with open(args.output, "w") as handle:
            json.dump({"config": vars(args), "rows": rows}, handle, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
