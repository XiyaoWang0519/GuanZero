"""Bounded scalar/batched public-event drain and ingest comparison; no training.

Uses identical engine seeds and greedy decisions in alternating ABBA trials.
Only public event draining and stream ingestion are timed; engine stepping,
hashing and host/source inspection are outside those timings. This measures a
host component, not GPU throughput, complete PPO updates or playing strength.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import gd
import numpy as np

from infra.cpu_budget import host_facts
from train.history_rollout import MatchEventStore


def stream_digest(store: MatchEventStore) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(sorted(store.current.items())).encode())
    for key, stream in sorted(store.streams.items()):
        digest.update(json.dumps([key, stream.match_id, stream.generation]).encode())
        for array in stream.arrays():
            digest.update(json.dumps([array.dtype.str, array.shape]).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def run_trial(args, mode: str) -> dict:
    env = gd.VecEnv(num_envs=args.envs, num_threads=args.engine_threads, seed=args.seed,
                    log_public_actions=True, log_env_limit=args.envs)
    env.reset()
    store = MatchEventStore()
    seconds, events = 0.0, 0
    for step in range(args.warmup + args.steps):
        pending = env.pending()
        started = time.perf_counter()
        if mode == "batched":
            count = store.ingest_batch(env.drain_public_batch())
        else:
            count = store.ingest(env.drain_public_actions())
        elapsed = time.perf_counter() - started
        if step >= args.warmup:
            seconds += elapsed
            events += count
        env.drain_finished_rounds()
        env.step(np.array(pending.greedy_choice, np.int32))
    return dict(mode=mode, event_seconds=seconds, measured_events=events,
                events_per_second=events / seconds, stream_sha256=stream_digest(store),
                total_tokens=store.tokens, matches=len(store),
                max_prefix=max((stream.prefix for stream in store.streams.values()), default=0))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--engine-threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if min(args.envs, args.engine_threads, args.steps, args.blocks) < 1 or args.warmup < 0:
        parser.error("positive sizes and nonnegative warmup are required")
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a fresh file")
    host = host_facts()
    root = Path(__file__).resolve().parents[1]
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in ("python/bindings.cpp", "train/history_model.py",
                            "train/history_rollout.py", "bench/history_public_events.py")}
    rows, ratios = [], []
    for block in range(args.blocks):
        trials = []
        for mode in ("scalar", "batched", "batched", "scalar"):
            row = run_trial(args, mode)
            row["block"] = block
            trials.append(row)
        if len({(r["stream_sha256"], r["measured_events"]) for r in trials}) != 1:
            raise AssertionError("scalar/batched event stream parity failed")
        scalar = sum(r["event_seconds"] for r in trials if r["mode"] == "scalar")
        batched = sum(r["event_seconds"] for r in trials if r["mode"] == "batched")
        ratios.append(scalar / batched)
        rows.extend(trials)
    summary = {mode: dict(median_events_per_second=statistics.median(
        r["events_per_second"] for r in rows if r["mode"] == mode))
        for mode in ("scalar", "batched")}
    summary.update(median_component_speedup=statistics.median(ratios),
                   block_component_speedups=ratios)
    result = dict(schema=1, settings={k: str(v) if isinstance(v, Path) else v
                                     for k, v in vars(args).items()},
                  host=host, source_sha256=sources, trials=rows, summary=summary,
                  claim="CPU public-event component only; no PPO/GPU/strength claim")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
