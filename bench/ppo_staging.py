"""Host storage cost of PPO rollout features, with real engine batches.

    .venv/bin/python bench/ppo_staging.py --num-envs 2048 --repeats 8

Compares the previous float32 observation double gather, the CPU indexed
gather, and gathering from an existing uint8 staging copy as CUDA now does.
The uint8 arrays stand in for the already populated pinned upload buffers;
their creation is outside timing because uploading already performs it.
This measures buffer storage only, not CUDA or end-to-end training speed.
Candidate lists use the first top_k + 1 candidates as a pruning stand-in.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gd  # noqa: E402
from train.rollout_buffer import RolloutBuffer, RolloutBufferConfig, _ragged_index  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=32)
    args = parser.parse_args()
    if min(vars(args).values()) <= 0:
        parser.error("all arguments must be positive")
    env = gd.VecEnv(args.num_envs, num_threads=1, seed=19)
    env.reset()
    learner_team = np.arange(args.num_envs) % 2
    capacity = RolloutBufferConfig(
        num_envs=args.num_envs, obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM,
        max_steps=args.num_envs * 2,
        max_candidates=args.num_envs * 2 * (args.top_k + 1),
        max_trajectories=args.num_envs * 2)
    names = ("legacy_float32", "indexed_float32", "indexed_uint8")
    buffers = {name: RolloutBuffer(capacity) for name in names}
    timings: dict[str, list[float]] = {name: [] for name in names}
    rows_seen, candidates_seen = [], []
    for iteration in range(64 + args.batches):
        batch = env.pending()
        env.drain_finished_rounds()
        if iteration >= 64:
            env_id, seat, phase = map(np.asarray, (batch.env_id, batch.seat, batch.phase))
            rows = np.flatnonzero((seat % 2 == learner_team[env_id])
                                  & (phase == int(gd.Phase.Play)))
            offsets = np.asarray(batch.offsets, np.int64)
            counts = np.minimum(offsets[rows + 1] - offsets[rows], args.top_k + 1)
            local = np.concatenate(([0], np.cumsum(counts)))
            src = _ragged_index(offsets[rows], counts, int(local[-1]))
            obs, cand = np.asarray(batch.obs), np.asarray(batch.cand)
            staged_obs, staged_cand = obs.astype(np.uint8), cand.astype(np.uint8)
            common = dict(learner=np.ones(rows.size, bool), env_id=env_id[rows],
                          match_id=np.asarray(batch.match_id)[rows],
                          round_index=np.asarray(batch.round_index)[rows], seat=seat[rows],
                          phase=phase[rows], offsets=local, cand_index=src,
                          chosen=np.zeros(rows.size, np.int64), logp=np.zeros(rows.size, np.float32))
            for repeat in range(args.repeats + 1):
                # Alternate order to limit cache/order bias; first repeat warms storage.
                for name in names if repeat % 2 else names[::-1]:
                    buffer = buffers[name]
                    buffer.clear()
                    start = time.perf_counter()
                    buffer.add_batch(
                        **common, hidden_counts=np.asarray(batch.hidden_counts)[rows],
                        obs=obs[rows] if name == "legacy_float32" else (
                            staged_obs if name == "indexed_uint8" else obs),
                        obs_index=None if name == "legacy_float32" else rows,
                        cand=staged_cand if name == "indexed_uint8" else cand)
                    elapsed = time.perf_counter() - start
                    if repeat:
                        timings[name].append(elapsed)
            baseline = buffers[names[0]].storage()
            for name in names[1:]:
                for key, value in buffers[name].storage().items():
                    np.testing.assert_array_equal(value, baseline[key], err_msg=f"{name}: {key}")
            rows_seen.append(rows.size)
            candidates_seen.append(src.size)
        env.step(np.asarray(batch.greedy_choice).copy())
    medians = {name: float(np.median(values) * 1000) for name, values in timings.items()}
    print(json.dumps({"config": vars(args), "scope": "host buffer storage only; CUDA staging simulated",
                      "equivalence": "all buffer arrays identical", "median_ms": medians,
                      "speedup": {name: medians[names[0]] / medians[name] for name in names[1:]},
                      "mean_learner_rows": float(np.mean(rows_seen)),
                      "mean_kept_candidates": float(np.mean(candidates_seen))}, indent=2))


if __name__ == "__main__":
    main()
