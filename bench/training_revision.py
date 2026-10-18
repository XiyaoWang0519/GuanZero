"""Bounded complete-PPO comparison across source revisions.

Run this same script with each checkout's ``python:oracle:.`` on PYTHONPATH,
in alternating order and with identical arguments. ``--verify`` instead
captures exact trajectory/state fingerprints, including one checkpoint-resume
update; its instrumented timings are deliberately not reported as throughput.
CPU results are local engineering diagnostics, not CUDA or playing-strength
evidence. Initial snapshot clones exercise batching, not population diversity.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch

from infra.cpu_budget import host_facts
from train.ckpt import rng_state
from train.history_ppo import HistoryPPOConfig, HistoryTrainer


def fingerprint(value: Any) -> str:
    """Hash typed values and array bytes without pickle/storage identities."""
    digest = hashlib.sha256()

    def add(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            digest.update(b"tensor")
            item = item.detach().cpu().contiguous().numpy()
        if isinstance(item, np.ndarray):
            digest.update(b"array")
            add((str(item.dtype), tuple(item.shape)))
            digest.update(item.tobytes())
        elif is_dataclass(item):
            digest.update(("dataclass:" + type(item).__qualname__).encode())
            add(asdict(item))
        elif isinstance(item, dict):
            digest.update(b"dict")
            add(len(item))
            for key in sorted(item, key=repr):
                add(key)
                add(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(type(item).__name__.encode())
            add(len(item))
            for element in item:
                add(element)
        else:
            digest.update((type(item).__name__ + ":" + repr(item) + "\0").encode())

    add(value)
    return digest.hexdigest()


def state_fingerprints(trainer: HistoryTrainer) -> dict[str, str]:
    population = trainer.population.state_dict()
    population.pop("lineage")
    for snapshot in population["snapshots"].values():
        snapshot["metadata"] = {key: value for key, value in snapshot["metadata"].items()
                                if key != "lineage"}
    rng = rng_state(trainer.rng)
    rng["sampler"] = trainer.generator.get_state()
    return {name: fingerprint(value) for name, value in {
        "actor": trainer.actor.state_dict(),
        "critic": trainer.critic.state_dict(),
        "actor_optimizer": trainer.actor_optimizer.state_dict(),
        "critic_optimizer": trainer.critic_optimizer.state_dict(),
        "rng": rng,
        "population": population,
        "assignments": trainer.collector.assignments,
        "buffer": trainer.buffer.compact(),
        "trajectories": trainer.buffer.trajectories,
        "streams": {key: stream.arrays() for key, stream in trainer.store.streams.items()},
        "progress": {key: value for key, value in trainer.progress.items()
                     if key != "elapsed_seconds"},
    }.items()}


def verify_update(trainer: HistoryTrainer) -> dict[str, Any]:
    """Capture all collected rows before completed trajectories are discarded."""
    collect = trainer.collect
    collected = {}

    def capture():
        result = collect()
        collected["rows"] = fingerprint(trainer.buffer.compact())
        collected["trajectories"] = fingerprint(trainer.buffer.trajectories)
        return result

    trainer.collect = capture
    try:
        line = trainer.update()
    finally:
        trainer.collect = collect
    losses = {key: line[key] for key in trainer.STAT_KEYS}
    return dict(update=line["update"], collected=collected,
                state=state_fingerprints(trainer), losses=fingerprint(losses),
                update_samples=line["update_samples"], max_prefix=line["max_prefix"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA is unavailable")
    random.seed(args.seed)
    torch.set_num_threads(args.threads)
    # Exact verification must not confuse nondeterministic CPU encoder backward
    # with a source regression. Timed runs retain the usual backend settings.
    if args.verify:
        torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cached = args.case != "dense"
    config = HistoryPPOConfig(
        width=args.width, layers=args.layers, heads=args.heads,
        num_envs=args.envs, num_threads=args.threads, torch_threads=args.threads,
        steps_per_update=args.steps, updates=args.updates, epochs=args.epochs,
        minibatch_matches=args.minibatch_matches, seed=args.seed,
        aux_heads=args.aux_heads, snapshot_updates=0,
        causal_sdpa=cached, rollout_kv_cache=cached, rollout_paged_cache=cached,
        rollout_page_span=cached, rollout_batched_attention=cached,
        learner_batched_attention=cached,
        batch_snapshot_policies=args.case == "snapshots",
        batch_snapshot_encoder=args.case == "snapshots")
    trainer = HistoryTrainer(config, args.output / "run", device=args.device)
    if args.case == "snapshots":
        for _ in range(2):
            trainer.population.snapshot(0)
    result = dict(case=args.case, config=asdict(config), device=str(trainer.device),
                  torch=torch.__version__, cuda=torch.version.cuda, host=host_facts(),
                  source=trainer.run_identity, verification=args.verify,
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  scope="bounded local CPU diagnostic" if trainer.device.type == "cpu"
                        else "bounded single-process CUDA engineering measurement")
    if args.verify:
        result["updates"] = [verify_update(trainer) for _ in range(args.updates)]
        if not any(line["update_samples"] for line in result["updates"]):
            raise RuntimeError("verification has no completed PPO rows; increase --steps or --updates")
        path = trainer.save()
        resumed = HistoryTrainer(HistoryPPOConfig.from_payload(asdict(config),
                                  updates=config.updates + 1), args.output / "resumed",
                                 device=args.device, resume=path)
        result["resumed_initial_state"] = state_fingerprints(resumed)
        result["resumed_update"] = verify_update(resumed)
        if not result["resumed_update"]["update_samples"]:
            raise RuntimeError("resumed verification has no completed PPO rows; increase --steps")
    else:
        lines = []
        for i in range(args.updates):
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            start = time.perf_counter()
            line = trainer.update()
            if trainer.device.type == "cuda":
                torch.cuda.synchronize(trainer.device)
            line["wall_seconds"] = time.perf_counter() - start
            line["warmup"] = i < args.warmup
            lines.append(line)
        measured = lines[args.warmup:]
        wall = sum(line["wall_seconds"] for line in measured)
        collect = sum(line["collect_seconds"] for line in measured)
        learn = sum(line["learn_seconds"] for line in measured)
        decisions = sum(line["step_decisions"] for line in measured)
        samples = sum(line["update_samples"] for line in measured)
        if not samples:
            raise RuntimeError("no completed PPO rows; increase --steps or --updates")
        result["summary"] = dict(
            updates=len(measured), decisions=decisions, learned_rows=samples,
            wall_seconds=wall, collect_seconds=collect, learn_seconds=learn,
            decisions_per_wall_second=decisions / wall,
            learned_rows_per_wall_second=samples / wall,
            collection_decisions_per_second=decisions / collect,
            learned_rows_per_learn_second=samples / learn,
            min_mean_prefix=min(line["mean_prefix"] for line in measured),
            max_prefix=max(line["max_prefix"] for line in measured),
            peak_reserved_bytes=max(line["cuda_peak_reserved_bytes"] for line in measured))
        result["updates"] = lines
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--case", choices=("dense", "paged", "snapshots"), default="paged")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--minibatch-matches", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--aux-heads", default="next,belief,outcome")
    args = parser.parse_args()
    if not 0 <= args.warmup < args.updates or args.threads < 1:
        parser.error("updates must exceed nonnegative warmup, and threads must be positive")
    if args.output.exists():
        parser.error("choose a fresh output directory")
    args.output.mkdir(parents=True)
    result = run(args)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result.get("summary", {"verification": True,
                                           "updates": len(result["updates"])}), indent=2))


if __name__ == "__main__":
    main()
