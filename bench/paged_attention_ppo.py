"""One isolated single-process PPO trial with an explicit rollout backend.

Use separate subprocesses in ABBA order. This starts a NEW randomly initialized
player with cloned initial snapshots; it does not resume, modify, or certify a
production lineage. It measures complete trainer.update() wall time, not just
collection, but excludes checkpoint saving. Spawned multi-rank/MPS production
layouts require separate validation. CUDA gates must pass before timing.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import torch

from bench.paged_attention import environment, source_facts, strict_fp32
from train.history_paged_attention import experimental_paged_attention, require_triton


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--backend", choices=("torch", "triton"), required=True)
    p.add_argument("--case", choices=("paged", "snapshots"), default="snapshots")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=20261018)
    p.add_argument("--updates", type=int, default=40)
    p.add_argument("--warmup", type=int, default=24)
    p.add_argument("--envs", type=int, default=128)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--minibatch-matches", type=int, default=8)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--aux-heads", default="next,belief,outcome")
    return p


def run(args: argparse.Namespace) -> dict:
    # Lazy engine imports let --help and CPU-only CLI checks run without a build.
    from train.history_ppo import HistoryPPOConfig, HistoryTrainer
    from train.history_paged_cache import PagedHistoryCache

    device = torch.device(args.device)
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    merged = args.case == "snapshots"
    config = HistoryPPOConfig(
        width=args.width, layers=args.layers, heads=args.heads,
        num_envs=args.envs, num_threads=args.threads, torch_threads=args.threads,
        steps_per_update=args.steps, updates=args.updates, epochs=args.epochs,
        minibatch_matches=args.minibatch_matches, seed=args.seed,
        aux_heads=args.aux_heads, snapshot_updates=0,
        causal_sdpa=True, rollout_kv_cache=True, rollout_paged_cache=True,
        rollout_page_span=True, rollout_batched_attention=True,
        learner_batched_attention=True, batch_snapshot_policies=merged,
        batch_snapshot_encoder=merged)
    result = dict(schema=1, status="started", backend=args.backend, case=args.case,
                  environment=environment(device), source=source_facts(), config=asdict(config),
                  scope="single-process randomly initialized CUDA PPO; not production or strength evidence",
                  warmup_updates=args.warmup, updates=[],
                  excluded="checkpoint saves; queue time; multi-process synchronization",
                  gate_requirement="run the prototype and integration CUDA tests first",
                  timing_limit="late shape-specialization compilation, if any, is included in wall time")
    with (args.output / "result.json").open("x") as report:
        try:
            with torch.cuda.device(device), strict_fp32(), experimental_paged_attention(args.backend):
                trainer = HistoryTrainer(config, args.output / "run", device=args.device)
                result["run_identity"] = trainer.run_identity
                if merged:
                    for _ in range(2):
                        trainer.population.snapshot(0)
                seen_kernel_calls = 0
                for index in range(args.updates):
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    line = trainer.update()
                    torch.cuda.synchronize(device)
                    wall = time.perf_counter() - started
                    # Post-timing checks reject a disconnected or fallback-only candidate.
                    caches = list(trainer.collector.caches.values())
                    if not caches or any(not isinstance(c, PagedHistoryCache)
                                         or c.attention_backend != args.backend for c in caches):
                        raise RuntimeError("requested backend is not wired to every live rollout cache")
                    kernel_calls = sum(c.attention_kernel_calls for c in caches)
                    seen_kernel_calls = max(seen_kernel_calls, kernel_calls)
                    if args.backend == "triton" and not seen_kernel_calls:
                        raise RuntimeError("candidate did not execute the custom kernel")
                    if args.backend == "torch" and kernel_calls:
                        raise RuntimeError("baseline unexpectedly executed a custom kernel")
                    line = dict(line, wall_seconds=wall, warmup=index < args.warmup,
                                live_cache_kernel_calls=kernel_calls)
                    result["updates"].append(line)
                    print(json.dumps(dict(backend=args.backend, index=index,
                                          wall_seconds=wall, learned_rows=line["update_samples"])), flush=True)
                lines = result["updates"][args.warmup:]
                wall = sum(line["wall_seconds"] for line in lines)
                decisions = sum(line["step_decisions"] for line in lines)
                samples = sum(line["update_samples"] for line in lines)
                if samples <= 0 or decisions <= 0:
                    raise RuntimeError("no measured PPO learning or decisions; increase the run length")
                result["summary"] = dict(
                    updates=len(lines), wall_seconds=wall, decisions=decisions, learned_rows=samples,
                    decisions_per_wall_second=decisions / wall,
                    learned_rows_per_wall_second=samples / wall,
                    collect_seconds=sum(line["collect_seconds"] for line in lines),
                    learn_seconds=sum(line["learn_seconds"] for line in lines),
                    max_prefix=max(line["max_prefix"] for line in lines),
                    min_mean_prefix=min(line["mean_prefix"] for line in lines),
                    max_mean_prefix=max(line["mean_prefix"] for line in lines),
                    observed_custom_kernel_calls=seen_kernel_calls)
                result["status"] = "passed"
        except Exception as error:
            result["status"] = "failed"
            result["error_type"] = type(error).__name__
            raise
        finally:
            report.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    p = parser()
    args = p.parse_args()
    if (min(args.width, args.heads, args.layers, args.envs, args.steps,
            args.threads, args.epochs, args.minibatch_matches) < 1
            or args.width % args.heads or not 0 <= args.warmup < args.updates):
        p.error("positive dimensions, width divisible by heads, and 0 <= warmup < updates required")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        p.error("this benchmark needs a CUDA compute allocation")
    # Require the same CUDA/Triton-capable environment for BOTH trial arms.
    require_triton(device, torch.float32, args.width // args.heads)
    if args.output.exists():
        p.error("choose a fresh output directory, never a production run directory")
    args.output.mkdir(parents=True)
    print(json.dumps(run(args)["summary"], indent=2))


if __name__ == "__main__":
    main()
