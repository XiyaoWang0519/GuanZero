"""Bounded dense/causal-SDPA/KV history-stack comparisons on one host.

Alternate case order, keep FP32/model/seeds/population/learning fixed, preserve
every result. CPU mode permits frozen collection only; CUDA is required to
measure optimizer work or accept a training-speed claim. Phase profiling is
synchronized and must be kept separate from ordinary throughput measurements.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from infra.cpu_budget import host_facts
from infra.history_artifacts import source_identity
from train.history_config import HistoryPPOConfig
from train.history_ppo import HistoryTrainer


def summarize(lines: list[dict]) -> dict:
    if not lines:
        raise ValueError("no measured updates")
    collect = sum(x["collect_seconds"] for x in lines)
    learn = sum(x["learn_seconds"] for x in lines)
    samples = sum(x["update_samples"] for x in lines)
    bands = {}
    for lo, hi in ((0, 256), (256, 768), (768, 1536), (1536, float("inf"))):
        group = [x for x in lines if lo <= x["mean_prefix"] < hi]
        if group:
            bands[f"{lo}-{hi}"] = dict(
                updates=len(group), mean_prefix=sum(x["mean_prefix"] for x in group)/len(group),
                collect_dps=sum(x["step_decisions"] for x in group)/sum(x["collect_seconds"] for x in group))
    phases = {}
    for row in lines:
        for name, seconds in row.get("collection_phase_seconds", {}).items():
            phases[name] = phases.get(name, 0.0) + seconds
    return dict(measured_updates=len(lines), collect_seconds=collect, learn_seconds=learn,
                collect_dps=sum(x["step_decisions"] for x in lines)/collect,
                unique_learned_rows=samples,
                unique_rows_per_collect_learn_second=samples/(collect+learn) if samples else None,
                max_prefix=max(x["max_prefix"] for x in lines), prefix_bands=bands,
                peak_allocated_bytes=max(x["cuda_peak_allocated_bytes"] for x in lines),
                peak_reserved_bytes=max(x["cuda_peak_reserved_bytes"] for x in lines),
                max_collection_cache_bytes=max(x["collection_cache"]["bytes"] for x in lines),
                synchronized_phase_seconds=phases)


def run_case(args) -> dict:
    if args.device == "cpu" and not args.collect_only:
        raise ValueError("CPU diagnostics require --collect-only; training stays on CUDA")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA acceptance requires an available CUDA device")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = HistoryPPOConfig(width=args.width, layers=args.layers, heads=4,
                           num_envs=args.envs, num_threads=2, torch_threads=2,
                           steps_per_update=args.steps, updates=args.updates, epochs=1,
                           minibatch_matches=2, seed=2026092602,
                           snapshot_updates=2 if args.population else 0,
                           causal_sdpa=args.case != "dense", rollout_kv_cache=args.case == "kv",
                           profile_collection=args.profile)
    trainer = HistoryTrainer(cfg, args.output, device=args.device)
    if args.collect_only and args.population:
        # Exercise multiple policy groups without doing any local optimization.
        for _ in range(4):
            trainer.population.snapshot(0)
    lines = []
    for i in range(args.updates):
        if args.collect_only:
            # Model weights stay frozen, but charge the same learner-cache
            # invalidation/rebuild frequency as PPO (one per collection chunk).
            if not args.keep_learner_cache:
                trainer.collector.invalidate_learner_cache()
            if args.device == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            stats = trainer.collect()
            if args.device == "cuda":
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            line = dict(update=i+1, collect_seconds=seconds, learn_seconds=0.0, update_samples=0,
                        step_decisions=stats.decisions, mean_prefix=stats.mean_prefix,
                        max_prefix=stats.prefix_max, collection_phase_seconds=stats.phase_seconds,
                        collection_cache=trainer.collector.cache_metrics(),
                        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == "cuda" else 0,
                        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device == "cuda" else 0)
            with (args.output / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(line)+"\n")
            # Keep the diagnostic bounded without computing a learner update.
            trainer.buffer.chunks.clear()
            trainer.buffer = type(trainer.buffer)()
            trainer.collector.buffer = trainer.buffer
            trainer.store.prune(set())
        else:
            line = trainer.update()
        lines.append(line)
    result = dict(variant=args.case, population=args.population, config=asdict(cfg),
                  device=args.device, gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                  torch=torch.__version__, cuda=torch.version.cuda, host=host_facts(),
                  source_sha256=source_identity()["source_sha256"],
                  collect_only=args.collect_only, synchronized_profile=args.profile,
                  learner_cache_rebuild_each_chunk=args.collect_only and not args.keep_learner_cache,
                  diagnostic_snapshots="identical initial weights" if args.collect_only and args.population else None,
                  summary=summarize(lines[args.warmup:]),
                  claim="CPU frozen-policy diagnostic only" if args.device == "cpu" else "same-host engineering throughput only")
    (args.output / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--keep-learner-cache", action="store_true",
                        help="frozen-policy diagnostic only; omit to account for per-update rebuild")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--population", type=int, choices=[0, 1], default=1)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--updates", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--case-seconds", type=int, default=240)
    parser.add_argument("--case", choices=["dense", "sdpa", "kv"])
    parser.add_argument("--variants", nargs="+", choices=["dense", "sdpa", "kv"], default=["dense", "sdpa", "kv"])
    args = parser.parse_args(argv)
    if args.updates <= args.warmup or args.warmup < 0 or args.repeats < 1:
        parser.error("positive repeats and updates greater than nonnegative warmup required")
    if args.device == "cpu" and not args.collect_only:
        parser.error("CPU permits --collect-only diagnostics, never training")
    if args.keep_learner_cache and not args.collect_only:
        parser.error("--keep-learner-cache only applies to frozen collection")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case:
        if (args.output / "metrics.jsonl").exists():
            parser.error("case output already contains metrics; choose a fresh directory")
        run_case(args)
        return 0
    results = []
    for repeat in range(args.repeats):
        order = args.variants if repeat % 2 == 0 else list(reversed(args.variants))
        for variant in order:
            path = args.output / f"repeat-{repeat}-{variant}"
            path.mkdir(exist_ok=True)
            command = [sys.executable, "-m", "bench.history_stack", "--case", variant, "--output", str(path)]
            for name in ("device", "population", "envs", "width", "layers", "steps", "updates", "warmup"):
                command += ["--"+name, str(getattr(args, name))]
            command += ["--"+name.replace("_", "-") for name in ("collect_only", "profile", "keep_learner_cache") if getattr(args, name)]
            started = time.perf_counter()
            with (path / "console.log").open("w") as log:
                try:
                    code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                          timeout=args.case_seconds).returncode
                except subprocess.TimeoutExpired:
                    code = 124
            file = path / "result.json"
            result = json.loads(file.read_text()) if code == 0 and file.exists() else dict(error=f"exit {code}")
            result.update(repeat=repeat, variant=variant, wall_seconds=time.perf_counter()-started)
            results.append(result)
            print(json.dumps({k: result.get(k) for k in ("variant", "repeat", "error", "summary")}), flush=True)
    (args.output / "comparison.json").write_text(json.dumps(dict(cases=results,
        selection="none; inspect parity, long-prefix coverage, memory and repeated CUDA timings"), indent=2)+"\n")
    return int(any("error" in result for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
