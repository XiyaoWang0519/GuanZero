"""Bounded host batching benchmark; no actor execution or training.

Run with ``PYTHONPATH=python:oracle:. python -m bench.history_batch_host``.
``--baseline /path/to/history_rollout.py`` compares the saved original methods
on identical data and checks every returned tensor before measuring.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import gd
import numpy as np
import torch

from train.history_model import HIDDEN_DIM
from train.history_rollout import MatchEventStore, SequenceRolloutBuffer
from train.logs import TOKEN_DIM


def fixture(matches=64, decisions=128, tokens=600, candidates=12):
    """Interleaved decisions, distinct canonical candidate rows and full history."""
    rng = np.random.default_rng(7)
    store, buffer = MatchEventStore(), SequenceRolloutBuffer()
    env = np.arange(matches, dtype=np.int64)
    match = env % 3
    counts = rng.integers(max(1, candidates // 2), candidates + 1, size=matches)
    offsets = np.r_[0, np.cumsum(counts)]
    for decision in range(decisions):
        buffer.add_step(
            keep=np.ones(matches, bool), env_id=env, match_id=match,
            round_index=np.zeros(matches, np.int64), seat=(env + decision) % 4,
            phase=np.full(matches, 2),
            obs=rng.integers(0, 2, (matches, int(gd.OBS_DIM)), dtype=np.uint8),
            hidden=rng.integers(0, 5, (matches, HIDDEN_DIM), dtype=np.uint8),
            cand=rng.integers(0, 2, (int(offsets[-1]), int(gd.ACT_DIM)), dtype=np.uint8),
            offsets=offsets, chosen=decision % counts, logp=np.full(matches, -1., np.float32),
            prefix=np.full(matches, decision * tokens // decisions), version=1)
    for e, m in zip(env.tolist(), match.tolist()):
        stream = store.stream(e, m)
        for i in range(tokens):
            token = np.zeros(TOKEN_DIM, np.uint8)
            token[i % 4] = token[4 + i % 146] = token[158 + i % 28] = 1
            stream.append_token(token, i // 80, 2)
        buffer.finish_round(e, m, 0, (1., -1., 1., -1.))
    buffer.finalize(np.zeros(len(buffer), np.float32))
    return store, buffer


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif is_dataclass(left):
        for field in fields(left):
            assert_equal(getattr(left, field.name), getattr(right, field.name))
    elif isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_equal(a, b)
    else:
        assert left == right


def measure(call, calls, repeats):
    call()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(calls):
            call()
        samples.append((time.perf_counter() - start) * 1000 / calls)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--matches", type=int, default=64)
    parser.add_argument("--decisions", type=int, default=128)
    parser.add_argument("--batch-matches", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=600)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--calls", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(1)
    store, buffer = fixture(args.matches, args.decisions, args.tokens, args.candidates)
    rows = next(buffer.minibatches(args.batch_matches, np.random.default_rng(11)))
    implementations = {"current": SequenceRolloutBuffer}
    if args.baseline:
        spec = importlib.util.spec_from_file_location("history_rollout_baseline", args.baseline)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        implementations["baseline"] = module.SequenceRolloutBuffer
        for groups in (0, 3):
            assert_equal(SequenceRolloutBuffer.training_batch(
                buffer, rows, store, "cpu", length_groups=groups, width=128),
                module.SequenceRolloutBuffer.training_batch(
                    buffer, rows, store, "cpu", length_groups=groups, width=128))
        assert_equal(buffer.match_groups(), module.SequenceRolloutBuffer.match_groups(buffer))
    calls = {}
    for name, cls in implementations.items():
        calls[f"{name}_training_batch"] = lambda cls=cls: cls.training_batch(buffer, rows, store, "cpu")
        calls[f"{name}_match_groups"] = lambda cls=cls: cls.match_groups(buffer)
    # Alternate operation order each round to limit thermal/order bias.
    timings = {name: [] for name in calls}
    for round_index in range(args.repeats):
        names = list(calls)
        if round_index % 2:
            names.reverse()
        for name in names:
            timings[name].append(measure(calls[name], args.calls, 1)["median_ms"])
    print(json.dumps({"rows": len(buffer), "batch_rows": len(rows), "matches": args.matches,
                      "batch_matches": args.batch_matches, "tokens": args.tokens,
                      "calls_per_repeat": args.calls,
                      "tensor_parity": "passed" if args.baseline else "not compared",
                      "timings": {name: {"median_ms": statistics.median(values),
                                          "samples_ms": values}
                                  for name, values in timings.items()}}, indent=2))


if __name__ == "__main__":
    main()
