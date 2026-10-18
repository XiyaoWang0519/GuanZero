"""Time response targets on one frozen CPU rollout; no optimizer or training.

``--baseline`` loads a saved history_response.py for exact target comparison.
Run with ``PYTHONPATH=python:oracle:. python -m bench.history_response_host``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import time

import gd
import numpy as np
import torch

from train.history_model import HistoryPolicyConfig, fresh_player
from train.history_response import opponent_response_labels
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(1)
    actor, _ = fresh_player(HistoryPolicyConfig(width=16, layers=1, heads=2), 29)
    env = gd.VecEnv(num_envs=args.envs, num_threads=1, seed=41,
                    log_public_actions=True, log_env_limit=args.envs)
    store, buffer = MatchEventStore(), SequenceRolloutBuffer()
    collector = HistoryCollector(env, actor, store, buffer, torch.Generator().manual_seed(53),
                                 kv_cache=True)
    collector.collect(args.steps)
    buffer.finalize(np.zeros(len(buffer), np.float32))
    implementations = {"current": opponent_response_labels}
    if args.baseline:
        spec = importlib.util.spec_from_file_location("response_baseline", args.baseline)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        implementations["baseline"] = module.opponent_response_labels
    results = {}
    for count in sorted({1, 16, 128, len(buffer.samples)}):
        rows = buffer.samples[:count]
        expected = opponent_response_labels(buffer, store, rows)
        for function in implementations.values():
            np.testing.assert_array_equal(expected, function(buffer, store, rows))
        timings = {name: [] for name in implementations}
        for repeat in range(args.repeats):
            names = list(implementations)
            if repeat % 2:
                names.reverse()
            for name in names:
                function = implementations[name]
                start = time.perf_counter()
                for _ in range(args.calls):
                    function(buffer, store, rows)
                timings[name].append((time.perf_counter() - start) * 1000 / args.calls)
        results[str(len(rows))] = {
            name: dict(median_ms=statistics.median(samples), samples_ms=samples)
            for name, samples in timings.items()}
    print(json.dumps(dict(claim="CPU response-label timing only", envs=args.envs,
                          steps=args.steps, completed_rows=len(buffer.samples),
                          target_parity="passed" if args.baseline else "not compared",
                          timings=results), indent=2))


if __name__ == "__main__":
    main()
