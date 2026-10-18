"""Bounded CPU benchmark of merged snapshot host layout construction.

Pass a saved original ``history_snapshot_batch.py`` (or just its
``merged_layout`` function) to ``--reference-source`` for alternating before
and after timings. Every layout field is checked before any timing. This
measures host preparation only, not complete PPO or CUDA throughput.

PYTHONPATH=python:oracle:. python -m bench.history_snapshot_layout \
    --reference-source /tmp/history_snapshot_batch_before.py
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import numpy as np

from train import history_snapshot_batch as snapshot


def fixture(group_count: int, rows_per_group: int, shared: bool) -> tuple:
    rng = np.random.default_rng(41)
    identities = rng.permutation(np.repeat(np.arange(group_count), rows_per_group))
    n = len(identities)
    counts = rng.integers(1, 91, size=n, dtype=np.int64)
    counts[0] = 1
    offsets = np.concatenate(([0], np.cumsum(counts)))
    env_id = np.empty(n, np.int64)
    match_id = np.full(n, 3, np.int64)
    prefix = rng.integers(500, 1501, size=n, dtype=np.int64)
    groups = []
    for identity in range(group_count):
        rows = np.flatnonzero(identities == identity)
        owners = np.arange(len(rows)) // (3 if shared else 1)
        env_id[rows] = identity * rows_per_group + owners
        keys = list(zip(env_id[rows].tolist(), match_id[rows].tolist()))
        unique = list(dict.fromkeys(keys))
        groups.append(SimpleNamespace(
            rows=rows, keys=unique,
            streams=[SimpleNamespace(prefix=1500) for _ in unique],
            max_candidates=int(counts[rows].max()),
            one_decision_per_stream=len(keys) == len(unique)))
    obs = rng.integers(0, 2, size=(n, 1849), dtype=np.uint8)
    seat = rng.integers(0, 4, size=n, dtype=np.int64)
    cand = rng.integers(0, 2, size=(offsets[-1], 154), dtype=np.uint8)
    return (groups, list(range(group_count)), prefix, obs, seat, cand,
            offsets, counts, env_id, match_id)


def assert_equal(actual, expected) -> None:
    for field in fields(actual):
        value, wanted = getattr(actual, field.name), getattr(expected, field.name)
        if field.name == "arrays":
            for name, array, reference in zip(snapshot.LAYOUT_FIELDS, value, wanted):
                np.testing.assert_array_equal(array, reference, err_msg=name)
                assert array.dtype == reference.dtype
        elif field.name == "rows":
            np.testing.assert_array_equal(value, wanted)
        else:
            assert value == wanted, field.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--calls", type=int, default=200)
    args = parser.parse_args()
    if args.repeats < 1 or args.calls < 1:
        parser.error("repeats and calls must be positive")
    implementations = {"after": snapshot.merged_layout}
    if args.reference_source:
        # Execute the explicitly supplied repository source in a separate
        # namespace so its helpers and globals remain from that source.
        namespace = dict(vars(snapshot))
        exec(compile(args.reference_source.read_text(), str(args.reference_source), "exec"),
             namespace)
        implementations = {"before": namespace["merged_layout"], **implementations}
    results = []
    for groups, rows, shared in [(1, 1, False), (1, 16, False), (2, 4, False),
                                (12, 8, False), (16, 16, False), (16, 16, True)]:
        inputs = fixture(groups, rows, shared)
        if "before" in implementations:
            assert_equal(implementations["after"](*inputs), implementations["before"](*inputs))
        samples = {name: [] for name in implementations}
        for repeat in range(args.repeats):
            order = list(implementations.items())
            if repeat % 2:
                order.reverse()
            for name, implementation in order:
                for _ in range(20):
                    implementation(*inputs)
                started = time.perf_counter()
                for _ in range(args.calls):
                    implementation(*inputs)
                samples[name].append((time.perf_counter() - started) / args.calls * 1e6)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        result = dict(groups=groups, rows=groups * rows, shared_streams=shared,
                      median_us=medians, raw_us=samples)
        if "before" in medians:
            result["speedup"] = medians["before"] / medians["after"]
        results.append(result)
    payload = dict(scope="CPU merged snapshot layout only", calls=args.calls,
                   repeats=args.repeats, results=results)
    text = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
