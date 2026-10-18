"""In-run speed A/B of history-trainer options that a resume may change.

Resumes a ``history_ppo`` checkpoint into a scratch directory (never the
lineage's own directory; nothing is saved), then alternates blocks of updates
between named arms INSIDE the running processes, so every arm sees the same
host, histories and snapshot population. A resume restarts environments;
switching arms in place does not. Arms are ``--resume-set`` style overrides
(``train.history_ppo.RESUME_OVERRIDES``); on a switch every public KV cache,
private graph and stacked snapshot copy is dropped and rebuilt, and the first
update of each block is excluded from the timings, and so is an update that
ran under the kernel profiler (``GUANZERO_TORCH_PROFILE_UPDATES``). Arms that
switch ``profile_collection`` / ``profile_learn`` on report their synchronized
phase times; those arms are for the breakdown, not for the speed ratio.

Only speed is measured. Tier 2 options change float order, so the arms'
trajectories diverge; no strength claim follows from this tool.

Rank -> GPU layout: by default every rank of a bare ``--device cuda`` runs on
the current device (GPU 0). ``--spread-gpus`` gives rank r visible GPU
r % device_count, as ``train.history_ddp --spread-gpus`` does, so an A/B of
allocator, memory or contention-sensitive options runs in the production
layout (e.g. 8 ranks over 3 GPUs: 3/3/2 ranks per GPU). An explicit
``cuda:N`` keeps every rank on N. The ranks are launched, monitored and
stopped together by ``train.history_ddp.launch_local``.

Example (four actor ranks on one GPU, as the main lineage runs)::

    python -m bench.history_arms --resume latest.pt --output /tmp/arms --device cuda \\
        --world-size 4 --warmup 25 --blocks 6 --block-updates 6 \\
        --arm base= \\
        --arm fast=rollout_triton_cache=false,rollout_private_graphs=false,rollout_paged_cache=true,rollout_page_span=true,batch_snapshot_encoder=true,learner_length_groups=4
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch.distributed as dist

from train.history_ppo import (RESUME_OVERRIDES, TORCH_PROFILE_ENV, TORCH_PROFILE_UPDATES_ENV,
                               parse_resume_overrides)


# Options ``switch`` applies to a live trainer. Other resume overrides (batch
# layout, threads, cadence) are fixed when the trainer is built and must agree.
SWITCHABLE = frozenset({"rollout_paged_cache", "rollout_page_span", "rollout_triton_cache",
                        "rollout_triton_min_batch", "rollout_private_graphs",
                        "rollout_graph_budget_mb", "rollout_graph_policy_budget_mb",
                        "batch_snapshot_encoder", "batch_snapshot_policies",
                        "learner_length_groups", "learner_chosen_response",
                        "learner_compact_next",
                        "rollout_trim_cuda_cache", "rollout_trim_slack_mb",
                        "rollout_prefill_learner_cache",
                        "rollout_span_groups", "rollout_private_kv_pages",
                        "rollout_causal_prefill",
                        "profile_collection", "profile_learn", "ddp_reduce", "adam_fused"})


def check_arms(arms: dict[str, dict]) -> None:
    fixed = [{k: v for k, v in settings.items() if k not in SWITCHABLE}
             for settings in arms.values()]
    if any(f != fixed[0] for f in fixed):
        raise ValueError(f"arms may differ only in {sorted(SWITCHABLE)}")


def parse_arm(text: str) -> tuple[str, dict]:
    name, _, body = text.partition("=")
    if not name:
        raise ValueError("--arm takes NAME=KEY=VALUE,KEY=VALUE (the body may be empty)")
    items = [item for item in body.split(",") if item]
    return name, parse_resume_overrides(items)


def switch(trainer, origin, settings: dict) -> None:
    """Apply one arm's settings, relative to the resumed ``origin`` config, to a
    live trainer without restarting environments. Options read from
    ``trainer.config`` at every update (the trim and its slack, learner and
    profile options) need nothing beyond the config swap."""
    config = replace(origin, **settings)              # validated by __post_init__
    collector = trainer.collector
    if ((config.rollout_private_graphs or config.rollout_triton_cache)
            and collector.device.type != "cuda"):
        raise ValueError("private graphs and Triton cache copies need a CUDA rollout device")
    trainer.config = config
    # The gradient transport is read at every reduce; the shared mapping is
    # created and self-checked now (collective: every rank switches at the same
    # update). The Adam implementation is a group setting of the live optimizers.
    if getattr(trainer, "world_size", 1) > 1 and config.ddp_reduce == "shm":
        trainer.shm_reducer()
    trainer.apply_adam_policy()
    for cache in collector.caches.values():
        cache.clear()
    collector.caches.clear()
    collector.kv_pool = collector.learner_pool = None
    for graph in collector.decision_graphs.values():
        graph.clear()
    collector.decision_graphs.clear()
    collector.snapshot_heads = None
    collector.paged_cache = config.rollout_paged_cache
    collector.page_span = config.rollout_page_span
    collector.triton_cache = config.rollout_triton_cache
    collector.triton_min_batch = config.rollout_triton_min_batch
    collector.private_graphs = config.rollout_private_graphs
    collector.private_graph_budget_bytes = config.rollout_graph_budget_mb << 20
    collector.private_graph_policy_budget_bytes = config.rollout_graph_policy_budget_mb << 20
    collector.batch_snapshot_encoder = config.batch_snapshot_encoder
    collector.prefill_learner_cache = config.rollout_prefill_learner_cache
    collector.span_groups = config.rollout_span_groups
    collector.private_kv_pages = config.rollout_private_kv_pages
    collector.causal_prefill = config.rollout_causal_prefill
    if trainer.device.type == "cuda":
        torch.cuda.synchronize(trainer.device)
        torch.cuda.empty_cache()


def schedule(arms: list[str], warmup: int, blocks: int, block_updates: int) -> list[str]:
    """Warmup on the first arm, then ABAB... blocks (ABBA order within each pair of rounds)."""
    order = [arms[0]] * warmup
    for block in range(blocks):
        names = arms if block % 2 == 0 else list(reversed(arms))
        for name in names:
            order += [name] * block_updates
    return order


def measure(trainer, plan: list[str], arms: dict[str, dict], rank: int, log: Path | None
            ) -> list[dict]:
    rows, current, origin = [], None, trainer.config
    for settings in arms.values():
        replace(origin, **settings)    # every arm is valid on this lineage before warmup
    profiled = ({int(item) for item in os.environ.get(TORCH_PROFILE_UPDATES_ENV, "").split(",")
                 if item.strip()} if os.environ.get(TORCH_PROFILE_ENV) else set())
    appends = 0
    for index, name in enumerate(plan):
        first = name != current
        if first:
            switch(trainer, origin, arms[name])
            current = name
            appends = 0                 # the caches and their counters are new
        started = time.perf_counter()
        line = trainer.update()
        wall = time.perf_counter() - started
        passes = line["collection_cache"].get("appends", 0)
        row = dict(update=index, arm=name, first_of_block=first, wall_seconds=wall,
                   profiled=index in profiled,
                   # Encode passes of this update. Caches of identities pruned during
                   # the update take their counts with them, so this is a lower bound.
                   encode_passes=passes - appends,
                   resident_snapshots=line["population"].get("resident_snapshots"),
                   collect_phase_seconds=line["collection_phase_seconds"],
                   collect_group_phase_seconds=line["collection_group_phase_seconds"],
                   learn_phase_seconds=line.get("learn_phase_seconds", {}),
                   trim_seconds=line["cuda_trim_seconds"],
                   collect_seconds=line.get("global_collect_seconds", line["collect_seconds"]),
                   learn_seconds=line.get("global_learn_seconds", line["learn_seconds"]),
                   decisions=line.get("global_step_decisions", line["step_decisions"]),
                   mean_prefix=line["mean_prefix"], entropy=line["entropy"],
                   approx_kl=line["approx_kl"],
                   peak_reserved_bytes=line["cuda_peak_reserved_bytes"],
                   peak_allocated_bytes=line["cuda_peak_allocated_bytes"],
                   cache=line["collection_cache"])
        appends = passes
        rows.append(row)
        if log is not None:
            with log.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("update", "arm", "wall_seconds",
                                                   "collect_seconds", "learn_seconds",
                                                   "mean_prefix")}), flush=True)
    return rows


def summarize(rows: list[dict], warmup: int) -> dict:
    result = {}
    for name in dict.fromkeys(r["arm"] for r in rows[warmup:]):
        settled = [r for r in rows[warmup:] if r["arm"] == name and not r["first_of_block"]
                   and not r.get("profiled")]
        if not settled:
            continue
        pick = lambda key: [r[key] for r in settled]
        phases = lambda key: {phase: statistics.mean(r[key].get(phase, 0.0) for r in settled)
                              for phase in dict.fromkeys(p for r in settled for p in r[key])}
        result[name] = dict(
            settled_updates=len(settled),
            wall_seconds=pick("wall_seconds"),
            encode_passes_median=statistics.median(pick("encode_passes")),
            collect_phase_seconds=phases("collect_phase_seconds"),
            learn_phase_seconds=phases("learn_phase_seconds"),
            wall_median=statistics.median(pick("wall_seconds")), wall_min=min(pick("wall_seconds")),
            collect_median=statistics.median(pick("collect_seconds")),
            learn_median=statistics.median(pick("learn_seconds")),
            decisions_per_wall_second=sum(pick("decisions")) / sum(pick("wall_seconds")),
            mean_prefix=statistics.mean(pick("mean_prefix")),
            peak_reserved_bytes=max(pick("peak_reserved_bytes")),
            peak_allocated_bytes=max(pick("peak_allocated_bytes")))
    names = list(result)
    if len(names) >= 2:
        base = result[names[0]]
        for name in names[1:]:
            result[name]["wall_speedup_vs_" + names[0]] = (
                base["wall_median"] / result[name]["wall_median"])
    return result


def _rank(rank: int, world: int, port: int, args) -> None:
    from train.history_ddp import HistoryDDPTrainer, rank_output, select_rank_device
    if world > 1:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        select_rank_device(args.device, rank, args.spread_gpus)
        arms = dict(parse_arm(text) for text in args.arm)
        # Build with the settings every arm shares; switch() applies the rest.
        fixed = {k: v for k, v in next(iter(arms.values())).items() if k not in SWITCHABLE}
        from train.history_ppo import HistoryPPOConfig
        output = rank_output(args.output, rank)
        trainer = HistoryDDPTrainer(HistoryPPOConfig(updates=10**9), output, device=args.device,
                                    rank=rank, world_size=world, resume=args.resume,
                                    allow_source_change=True, resume_overrides=fixed)
        plan = schedule(list(arms), args.warmup, args.blocks, args.block_updates)
        log = Path(args.output) / "arms.jsonl" if rank == 0 else None
        rows = measure(trainer, plan, arms, rank, log)
        if rank == 0:
            summary = dict(arms={k: {kk: str(vv) for kk, vv in v.items()} for k, v in arms.items()},
                           device=args.device, spread_gpus=args.spread_gpus,
                           world_size=world, warmup=args.warmup,
                           blocks=args.blocks, block_updates=args.block_updates,
                           resume=str(args.resume), results=summarize(rows, args.warmup))
            (Path(args.output) / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps(summary["results"], indent=2), flush=True)
    finally:
        if world > 1:
            dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--resume", required=True, help="history_ppo checkpoint (read only)")
    parser.add_argument("--output", required=True, help="scratch directory; not the lineage")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--spread-gpus", action="store_true",
                        help="with --device cuda, rank r uses visible GPU r %% device_count "
                             "(the train.history_ddp --spread-gpus layout)")
    parser.add_argument("--arm", action="append", required=True,
                        help=f"NAME=KEY=VALUE,... with KEY in {sorted(RESUME_OVERRIDES)}")
    parser.add_argument("--warmup", type=int, default=25,
                        help="updates on the first arm before measuring (histories refill)")
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--block-updates", type=int, default=6)
    args = parser.parse_args(argv)
    names = [parse_arm(text)[0] for text in args.arm]
    if len(set(names)) != len(names) or len(names) < 2:
        parser.error("compare at least two arms with distinct names")
    try:
        check_arms(dict(parse_arm(text) for text in args.arm))
    except ValueError as error:
        parser.error(str(error))
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        parser.error("--output must be a new or empty scratch directory")
    if Path(args.resume).resolve().parent == output.resolve():
        parser.error("--output must not be the checkpoint's own directory")
    output.mkdir(parents=True, exist_ok=True)
    if args.world_size == 1:
        _rank(0, 1, 0, args)
        return 0
    # If any rank fails (an out-of-memory arm, say), the launcher stops them all;
    # arms.jsonl keeps the updates measured so far.
    from train.history_ddp import launch_local
    return launch_local(_rank, args.world_size, args, name="history-arms-rank")


if __name__ == "__main__":
    sys.exit(main())
