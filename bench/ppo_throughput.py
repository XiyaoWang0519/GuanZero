"""Stage B PPO throughput: decisions/s and updates/s for a config and device.

Runs real `PPOTrainer` updates (rollout plus learner, no checkpoints) in a
temporary run directory, after warm-up updates that fill the carried-over
rounds and let CUDA pick its kernels. Examples:

    # the B6 arm config on a GPU pod, B5 path versus the B5b fast path
    .venv/bin/python bench/ppo_throughput.py --config train/configs/ppo.json \\
        --device cuda --compare --updates 8

    # the same with a per-phase time breakdown (synchronises the device at
    # every phase boundary, so the totals are a little slower)
    .venv/bin/python bench/ppo_throughput.py --config train/configs/ppo.json \\
        --device cuda --profile --updates 4

    # overhead only: a 16-wide network, so the time left is Python, copies,
    # launches and the engine, which is what a fast GPU leaves over
    .venv/bin/python bench/ppo_throughput.py --tiny --num-envs 1024 --compare

    # B5 path, then the fast path in-process and with 4 and 8 actor processes
    .venv/bin/python bench/ppo_throughput.py --config train/configs/ppo.json \\
        --device cuda --compare --actors 0 4 8 --updates 8

    # the B8 league config (league:train/configs/league-b8.json), in-process
    # and with actors; snapshot every 2 updates so learner snapshots are in play
    .venv/bin/python bench/ppo_throughput.py --config train/configs/ppo-league.json \\
        --device cuda --actors 0 4 8 --league-snapshot-every 2 --updates 8

Prints one JSON object per measured variant; `--out` also writes them.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

import gd  # noqa: E402
from train.ckpt import save_checkpoint  # noqa: E402
from train.model import GuandanModel, ModelConfig  # noqa: E402
from train.ppo import PPOConfig, PPOTrainer, load_config  # noqa: E402

TINY = ModelConfig(obs_dim=gd.OBS_DIM, act_dim=gd.ACT_DIM, state_width=16, state_layers=1,
                   action_width=16, action_layers=1, fusion_width=16, fusion_layers=1)


def tiny_overrides(directory: Path) -> dict:
    torch.manual_seed(5)
    path = directory / "tiny-init.pt"
    save_checkpoint(path, {"model_config": asdict(TINY), "model": GuandanModel(TINY).state_dict(),
                           "optimizer": {}, "config": {"action_mode": "canonical", "seed": 1},
                           "progress": {"updates": 0}, "rng": {}})
    return {"init_checkpoint": str(path), "critic_init": "", "critic_width": 16,
            "critic_layers": 1}


def measure(config: PPOConfig, device: str, warmup: int, updates: int, seconds: float,
            profile: bool, directory: Path) -> dict:
    trainer = PPOTrainer(config, directory, device)
    try:
        return _measure(trainer, config, warmup, updates, seconds, profile)
    finally:
        trainer.close()


def _measure(trainer: PPOTrainer, config: PPOConfig, warmup: int, updates: int, seconds: float,
             profile: bool) -> dict:
    sync = torch.cuda.synchronize if trainer.device.type == "cuda" else (lambda: None)
    quiet = contextlib.redirect_stdout(io.StringIO())
    with quiet:
        for _ in range(warmup):
            trainer.update()
    if profile:
        trainer.timers = {}
    if trainer.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(trainer.device)
    sync()
    decisions0, learner0 = trainer.progress["decisions"], trainer.progress["learner_decisions"]
    samples0, steps0 = trainer.progress["samples"], trainer.progress["optimizer_steps"]
    collect = learn = 0.0
    done = 0
    start = time.perf_counter()
    with quiet:
        while done < updates and time.perf_counter() - start < seconds:
            record = trainer.update()
            collect += record["collect_seconds"]
            learn += record["learn_seconds"]
            done += 1
    sync()
    wall = time.perf_counter() - start
    decisions = trainer.progress["decisions"] - decisions0
    result = {
        "fast_rollout": config.fast_rollout, "fused_opponent": trainer.fused_opponent,
        "league_fused": trainer.league_fused,
        "actor_processes": config.actor_processes,
        "device": str(trainer.device),
        "gpu": torch.cuda.get_device_name(trainer.device) if trainer.device.type == "cuda" else None,
        "num_envs": config.num_envs, "num_threads": config.num_threads,
        "torch_threads": config.torch_threads, "rollout_steps": config.rollout_steps,
        "epochs": config.epochs, "minibatch_size": config.minibatch_size,
        "candidate_chunk": config.candidate_chunk, "opponent": config.opponent,
        "updates": done, "wall_seconds": wall,
        "decisions_per_second": decisions / wall,
        "updates_per_second": done / wall,
        "learner_decisions_per_second": (trainer.progress["learner_decisions"] - learner0) / wall,
        "samples_per_second": (trainer.progress["samples"] - samples0) / wall,
        "optimizer_steps": trainer.progress["optimizer_steps"] - steps0,
        "collect_seconds": collect, "learn_seconds": learn,
        "collect_decisions_per_second": decisions / collect if collect else None,
    }
    if trainer.device.type == "cuda":
        result["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated(trainer.device) / 2**20
    if trainer.league is not None:
        # Pool-level league counters at the end (per-entry rows are in metrics).
        result["league"] = {k.removeprefix("league/"): v
                            for k, v in trainer.league_stats().items() if k.count("/") == 1}
        result["league_snapshot_every"] = trainer.league.config.snapshot_every
    if profile:
        result["phases_seconds"] = dict(sorted(trainer.timers.items(), key=lambda kv: -kv[1]))
        result["phases_percent_of_wall"] = {k: round(100 * v / wall, 1)
                                            for k, v in result["phases_seconds"].items()}
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="train/configs/ppo.json")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--rollout-steps", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--minibatch-size", type=int)
    parser.add_argument("--candidate-chunk", type=int)
    parser.add_argument("--opponent", help="frozen, frozen:<path>, greedy or league:<pool.json>")
    parser.add_argument("--league-snapshot-every", type=int,
                        help="league only: snapshot the learner every N updates (measures a "
                        "pool that already holds learner snapshots)")
    parser.add_argument("--actors", type=int, nargs="+",
                        help="actor_processes values to measure (fast path), e.g. 0 4 8")
    parser.add_argument("--warmup", type=int, default=2, help="unmeasured updates first")
    parser.add_argument("--updates", type=int, default=6, help="measured updates")
    parser.add_argument("--seconds", type=float, default=600, help="measurement time cap")
    parser.add_argument("--compare", action="store_true",
                        help="measure the B5 path (fast_rollout=false) and the fast path")
    parser.add_argument("--legacy", action="store_true", help="measure only the B5 path")
    parser.add_argument("--profile", action="store_true", help="per-phase time breakdown")
    parser.add_argument("--tiny", action="store_true",
                        help="16-wide network: measures everything but model compute")
    parser.add_argument("--out", help="also write the results as a JSON list")
    args = parser.parse_args(argv)
    config = load_config(ROOT / args.config if not Path(args.config).is_absolute()
                         else args.config)
    overrides = {"num_envs": args.num_envs, "num_threads": args.num_threads,
                 "torch_threads": args.torch_threads, "rollout_steps": args.rollout_steps,
                 "epochs": args.epochs, "minibatch_size": args.minibatch_size,
                 "candidate_chunk": args.candidate_chunk, "opponent": args.opponent,
                 "league_snapshot_every": args.league_snapshot_every}
    config = replace(config, **{k: v for k, v in overrides.items() if v is not None},
                     tensorboard=False, max_updates=10**9, max_seconds=10**9,
                     checkpoint_seconds=600)
    variants = [(False, 0), (True, 0)] if args.compare else [(not args.legacy, 0)]
    if args.actors:
        variants = ([(False, 0)] if args.compare else []) + [(True, w) for w in args.actors]
    results = []
    with tempfile.TemporaryDirectory(prefix="ppo-bench-") as scratch:
        scratch = Path(scratch)
        if args.tiny:
            config = replace(config, **tiny_overrides(scratch))
        for fast, actors in variants:
            result = measure(replace(config, fast_rollout=fast, actor_processes=actors),
                             args.device, args.warmup, args.updates, args.seconds, args.profile,
                             scratch / f"run-{int(fast)}-{actors}")
            result["tiny"] = args.tiny
            print(json.dumps(result), flush=True)
            results.append(result)
    if len(results) > 1:
        base = results[0]["decisions_per_second"]
        print(json.dumps({"speedup_vs_first": [
            {"fast_rollout": r["fast_rollout"], "actor_processes": r["actor_processes"],
             "speedup": r["decisions_per_second"] / base} for r in results[1:]]}))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
