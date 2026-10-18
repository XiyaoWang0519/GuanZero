"""CPU wall-time and Python-profile harness for the production-shaped history trainer.

Runs ``HistoryTrainer`` (or ``HistoryDDPTrainer`` over local gloo ranks) on the
CPU with the main lineage's architecture and option set, scaled down in table
count. It measures host work: on a GPU host the same Python, engine, collation
and collective code runs per rank, while the model arithmetic moves to the GPU.
Use ``--tiny`` (width 16, one layer, two heads) to shrink the model arithmetic
so the host share dominates; that is the CPU-measurable part of a GPU update.

Nothing here is a GPU throughput or playing-strength measurement.

    PYTHONPATH=python:. python -m bench.cpu_profile --output /tmp/prof \\
        --envs 32 --warmup 3 --updates 3 [--tiny] [--cprofile out.prof]
"""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import statistics
import time

import torch

from train.history_ppo import HistoryPPOConfig, HistoryTrainer

MAIN_OPTIONS = dict(
    width=128, layers=4, heads=8, response_mode="auxiliary", aux_heads="next,belief,outcome",
    aux_coef=0.1, entropy=0.03, epochs=2, lr=3e-4, rollout_epsilon=0.0,
    snapshot_updates=2, population_recent=4, population_archive_every=0,
    causal_sdpa=True, rollout_kv_cache=True, rollout_batched_attention=True,
    rollout_wide_projection=True, rollout_paged_cache=True, rollout_page_span=True,
    batch_snapshot_policies=True, batch_snapshot_encoder=True,
    rollout_prefill_learner_cache=True, learner_length_groups=2, learner_batched_attention=True,
)


def config_for(args) -> HistoryPPOConfig:
    options = dict(MAIN_OPTIONS)
    if args.tiny:
        options.update(width=16, layers=1, heads=2)
    return HistoryPPOConfig(**options, num_envs=args.envs, steps_per_update=args.steps,
                            minibatch_matches=args.minibatch_matches,
                            updates=args.warmup + args.updates, seed=args.seed,
                            torch_threads=args.torch_threads, num_threads=1,
                            checkpoint_updates=10**9)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--minibatch-matches", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026101801)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--cprofile", default="", help="pstats file for the timed updates")
    parser.add_argument("--json", default="", help="write the timing summary here")
    args = parser.parse_args(argv)
    torch.manual_seed(0)
    trainer = HistoryTrainer(config_for(args), args.output, device="cpu")
    for _ in range(args.warmup):
        trainer.update()
    profiler = cProfile.Profile() if args.cprofile else None
    rows = []
    for _ in range(args.updates):
        if profiler is not None:
            profiler.enable()
        begin = time.perf_counter()
        line = trainer.update()
        wall = time.perf_counter() - begin
        if profiler is not None:
            profiler.disable()
        rows.append(dict(wall=wall, collect=line["collect_seconds"], learn=line["learn_seconds"],
                         decisions=line["step_decisions"], samples=line["update_samples"],
                         mean_prefix=line["mean_prefix"], minibatches=line["minibatches"]))
    if profiler is not None:
        profiler.dump_stats(args.cprofile)
    summary = {key: statistics.mean(r[key] for r in rows) for key in rows[0]}
    summary["rows"] = rows
    print(json.dumps(summary, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
