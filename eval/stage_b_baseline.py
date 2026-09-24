"""Stage B baseline (STAGE_B_TODO B0): the M1 final against the fixed opponent suite.

Plays the reference checkpoint against greedy, four fixed styled bots and
itself, in duplicate deals (eval/duplicate.py, paired bootstrap interval) and
full matches (eval/arena.py, Wilson interval). Every opponent sees the same
deal set and the same match seeds. Also times the current DMC loop
(train/dmc.py) on this host. Writes one JSON report.

Self-play in duplicate deals is exactly zero by construction: both legs are
played by the same deterministic policy. It is kept as a harness check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from .batched import (EvalConfig, add_eval_arguments, config_from_args,
                      play_duplicate_batch, play_matches_batch)
from .arena import MATCH_COUNTERS, play_matches, summarize_matches
from .duplicate import generate_deals, play_duplicate, summarize_duplicates
from .policies import Policy, load_policy

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = Path(".work/runpod/artifacts/pilot/final.pt")
DEFAULT_OUTPUT = Path("docs/reports/stage-b-baseline.json")
DEFAULT_THROUGHPUT_CONFIG = Path("train/configs/m1.json")
OPPONENTS: tuple[str, ...] = (
    "greedy", "styled:bomb-happy", "styled:bomb-shy", "styled:high-lead",
    "styled:low-lead", "self",
)
TIME_BUDGET_SECONDS = 3600.0


def resolve_path(path: Path) -> Path:
    """Resolve against the cwd, this checkout, then the main checkout.

    A git worktree under `<main>/.claude/worktrees/<name>` has no `.work`, so
    a relative artifact path falls back to the main repository.
    """
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, ROOT / path]
    if not path.is_absolute() and ROOT.parent.name == "worktrees" and ROOT.parents[1].name == ".claude":
        candidates.append(ROOT.parents[2] / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"{path} not found; tried {[str(c) for c in candidates]}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def host_info() -> dict:
    import torch

    cpu = platform.processor()
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, check=True).stdout.strip() or cpu
        except (OSError, subprocess.CalledProcessError):
            pass
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "platform": platform.platform(), "machine": platform.machine(), "cpu": cpu,
        "logical_cpus": os.cpu_count(), "python": platform.python_version(),
        "torch": torch.__version__, "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(), "git_commit": commit,
    }


def opponent_description(spec: str, policy: Policy) -> dict:
    if spec.startswith("styled:"):
        from train.styles import FIXED_STYLES

        name = spec[len("styled:"):]
        return {"spec": spec, "name": policy.name, "kind": "styled",
                "style_changes_from_neutral": FIXED_STYLES[name],
                "style_vector": [float(x) for x in policy.style]}
    if spec == "self":
        return {"spec": spec, "name": policy.name, "kind": "checkpoint"}
    return {"spec": spec, "name": policy.name, "kind": spec}


def style_divergence(rounds: int = 300, seed: int = 0) -> dict:
    """How often each fixed style picks a different play than greedy.

    Measured on the play decisions of greedy self-play first rounds, so a style
    that collapses onto greedy shows a zero rate.
    """
    import random

    import gd
    from train.styles import FIXED_STYLES

    engine = gd.Engine()
    greedy = load_policy("greedy")
    styles = {name: load_policy(f"styled:{name}") for name in FIXED_STYLES}
    differs = dict.fromkeys(styles, 0)
    decisions = 0
    for index in range(rounds):
        state = gd.MatchState()
        engine.new_match(state, seed + index)
        while state.phase != gd.Phase.RoundEnd:
            actions = engine.legal_actions(state)
            choice = greedy.select(engine, state, actions, random.Random(0))
            decisions += 1
            for name, policy in styles.items():
                differs[name] += policy.select(engine, state, actions, random.Random(0)) != choice
            engine.apply(state, actions[choice])
    return {"rounds": rounds, "seed": seed, "decisions": decisions,
            "definition": "fraction of greedy self-play decisions where the style picks another action",
            "styles": {name: {"differs": count, "rate": count / decisions}
                       for name, count in differs.items()}}


# Per-process caches so that pool workers load each policy and the deal set once.
_POLICIES: dict[tuple[str, str], Policy] = {}
_DEALS: dict[tuple[int, int], list] = {}


def _init_worker(threads: int) -> None:
    import torch

    torch.set_num_threads(threads)


def _policy(spec: str, device: str) -> Policy:
    key = (spec, device)
    if key not in _POLICIES:
        _POLICIES[key] = load_policy(spec, device)
    return _POLICIES[key]


def _run_chunk(task: tuple) -> tuple:
    """One contiguous slice of deals or matches for one pair.

    Deal `i` is played with seed `seed + i` and match `m` with seed `seed + m`,
    as in the serial reference. Batched stochastic policies use a different
    draw order and depend on the chunk and wave configuration.
    """
    kind, checkpoint, opponent_spec, device, deals, seed, start, stop, *options = task
    config = options[0] if options else EvalConfig()
    agent = _policy(checkpoint, device)
    opponent = agent if opponent_spec == "self" else _policy(opponent_spec, device)
    if kind == "duplicate":
        if (deals, seed) not in _DEALS:
            _DEALS[(deals, seed)] = generate_deals(deals, seed)
        deal_set = _DEALS[(deals, seed)]
        if config.backend == "batched":
            return kind, start, play_duplicate_batch(deal_set[start:stop], (agent, agent),
                                                      (opponent, opponent), seed + start, config)
        return kind, start, [play_duplicate(deal_set[i], agent, opponent, seed + i)
                             for i in range(start, stop)]
    if config.backend == "batched":
        return kind, start, play_matches_batch(agent, opponent, range(start, stop), seed, config=config)
    return kind, start, play_matches(agent, opponent, range(start, stop), seed)


def _chunks(total: int, workers: int) -> list[tuple[int, int]]:
    # Balance long runs with roughly four chunks per worker, but avoid
    # shrinking inference batches below 256 while workers can still stay busy.
    size = max(1, -(-total // max(1, workers * 4)),
               min(256, -(-total // max(1, workers))))
    return [(start, min(start + size, total)) for start in range(0, total, size)]


def evaluate_pair(checkpoint: str, opponent_spec: str, deals: int, matches: int, seed: int,
                  bootstrap_samples: int, device: str = "cpu",
                  pool: ProcessPoolExecutor | None = None, workers: int = 1,
                  eval_config: EvalConfig = EvalConfig()) -> dict:
    tasks = [("duplicate", checkpoint, opponent_spec, device, deals, seed, a, b, eval_config)
             for a, b in _chunks(deals, workers)]
    if matches:
        tasks += [("match", checkpoint, opponent_spec, device, deals, seed, a, b, eval_config)
                  for a, b in _chunks(matches, workers)]
    started = time.monotonic()
    results = list(pool.map(_run_chunk, tasks)) if pool else [_run_chunk(task) for task in tasks]
    seconds = time.monotonic() - started
    duplicate_parts = sorted(((start, part) for kind, start, part in results if kind == "duplicate"),
                             key=lambda item: item[0])
    scores = [score for _, part in duplicate_parts for score in part]
    duplicate = summarize_duplicates(scores, seed, bootstrap_samples)
    # Per-deal records are large (10,000 per pair) and reproducible from seeds.
    duplicate.pop("pair_scores")
    duplicate.pop("results")
    report = {"duplicate": duplicate, "seconds": seconds, "evaluation": eval_config.metadata()}
    if matches:
        parts = sorted(((start, part) for kind, start, part in results if kind == "match"),
                       key=lambda item: item[0])
        totals = {name: sum(part[name] for _, part in parts) for name in MATCH_COUNTERS}
        totals["records"] = [record for _, part in parts for record in part["records"]]
        match = summarize_matches(totals)
        match.pop("results")
        report["match"] = match
    return report


def measure_dmc_throughput(config_path: Path, seconds: float, torch_threads: int | None = None) -> dict:
    """Run the real DMC trainer from scratch for a short wall-clock budget.

    Throughput depends on model width, not on the weights, so a fresh model
    with the M1 config measures the loop Stage B will be compared against.
    """
    from train.dmc import Trainer, TrainConfig

    data = json.loads(config_path.read_text())
    data.update({"max_seconds": seconds, "tensorboard": False, "log_envs": 0,
                 "log_max_rounds": 0, "checkpoint_seconds": 600})
    if torch_threads is not None:
        data["torch_threads"] = torch_threads
    config = TrainConfig(**data)
    with tempfile.TemporaryDirectory(prefix="stage-b-throughput-") as run_dir:
        trainer = Trainer(config, run_dir, "cpu")
        started = time.monotonic()
        progress = dict(trainer.run())
        wall = time.monotonic() - started
        lines = (Path(run_dir) / "metrics.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    collect = sum(r["collect_seconds"] for r in records)
    learn = sum(r["learn_seconds"] for r in records)
    return {
        "config": str(config_path), "device": "cpu",
        "num_envs": config.num_envs, "num_threads": config.num_threads,
        "torch_threads": config.torch_threads, "rollout_steps": config.rollout_steps,
        "learn_steps": config.learn_steps, "batch_size": config.batch_size,
        "requested_seconds": seconds, "wall_seconds": wall,
        "iterations": len(records), "updates": progress["updates"],
        "decisions": progress["decisions"], "rounds": progress["rounds"],
        "samples": progress["samples"],
        "collect_seconds": collect, "learn_seconds": learn,
        "decisions_per_second": progress["decisions"] / max(wall, 1e-9),
        "rollout_decisions_per_second": progress["decisions"] / max(collect, 1e-9),
        "updates_per_second": progress["updates"] / max(wall, 1e-9),
        "learner_samples_per_second": progress["trained_samples"] / max(learn, 1e-9),
        "note": "fresh weights with the M1 architecture; includes env setup and the initial checkpoint save",
    }


def run_baseline(checkpoint: Path, deals: int, matches: int, seed: int,
                 bootstrap_samples: int = 2000, opponents: tuple[str, ...] = OPPONENTS,
                 throughput_config: Path | None = None, throughput_seconds: float = 0.0,
                 device: str = "cpu", threads: int = 1, workers: int = 1, log=print,
                 eval_config: EvalConfig = EvalConfig()) -> dict:
    """Evaluate every pair. `workers` processes each use `threads` torch threads."""
    import torch

    if deals < 1 or matches < 0 or bootstrap_samples < 1 or threads < 1 or workers < 1:
        raise ValueError("require deals, bootstrap samples, threads and workers > 0 and matches >= 0")
    unknown = [spec for spec in opponents if spec not in OPPONENTS]
    if unknown:
        raise ValueError(f"unknown opponents {unknown}; choose from {OPPONENTS}")
    started = time.monotonic()
    torch.set_num_threads(threads)
    agent = load_policy(str(checkpoint), device)
    report = {
        "schema_version": 1, "task": "STAGE_B_TODO B0", "protocol": "internal-house",
        "rules": "gd.RuleConfig.house()",
        "checkpoint": {"path": str(checkpoint), "name": agent.name,
                       "model_digest": agent.checkpoint_id, "file_sha256": file_sha256(checkpoint),
                       "training_seed": agent.training_seed, "action_mode": agent.action_mode,
                       "tribute": "heuristic" if agent.heuristic_tribute else "learned",
                       "selection": "argmax Q, margin 0"},
        "seeds": {"deal_seed": seed, "duplicate_play_seed": seed,
                  "duplicate_play_seed_rule": ("deal i uses seed + i" if eval_config.backend == "scalar"
                                               else "chunk/wave starts at seed + first deal index; actor streams are seeded by policy index"),
                  "match_seed": seed, "match_seed_rule": "match m uses seed + m; agent team is m % 2",
                  "bootstrap_seed": seed},
        "requested": {"deals": deals, "matches": matches, "bootstrap_samples": bootstrap_samples},
        "host": host_info(),
        "interval_definitions": {
            "duplicate": "95% percentile bootstrap over whole deals of mean net levels per round (agent minus opponent)",
            "match": "95% Wilson score interval of agent match win rate",
        },
        "style_divergence_from_greedy": style_divergence(seed=seed),
        "pairs": {},
    }
    pool = None
    if workers > 1:
        pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_init_worker, initargs=(threads,))
    try:
        for spec in opponents:
            opponent = agent if spec == "self" else load_policy(spec, device)
            log(f"[stage-b-baseline] {agent.name} vs {spec}: {deals} deals, {matches} matches")
            result = evaluate_pair(str(checkpoint), spec, deals, matches, seed, bootstrap_samples,
                                   device, pool, workers, eval_config)
            result["opponent"] = opponent_description(spec, opponent)
            report["pairs"][spec] = result
            dup = result["duplicate"]
            line = (f"  duplicate {dup['mean_net_levels_per_round']:+.4f} "
                    f"CI [{dup['bootstrap_95_ci'][0]:+.4f}, {dup['bootstrap_95_ci'][1]:+.4f}]")
            if "match" in result:
                match = result["match"]
                line += (f"; match {match['win_rate']:.3f} Wilson [{match['wilson_95_ci'][0]:.3f}, "
                         f"{match['wilson_95_ci'][1]:.3f}]")
            log(f"{line} ({result['seconds']:.0f}s)")
    finally:
        if pool is not None:
            pool.shutdown()
    evaluation_seconds = time.monotonic() - started
    if throughput_config is not None and throughput_seconds > 0:
        log(f"[stage-b-baseline] DMC throughput, {throughput_seconds:g}s with {throughput_config}")
        report["dmc_throughput"] = measure_dmc_throughput(throughput_config, throughput_seconds)
        torch.set_num_threads(threads)
    total = time.monotonic() - started
    report["runtime"] = {"evaluation_seconds": evaluation_seconds, "total_seconds": total,
                         "budget_seconds": TIME_BUDGET_SECONDS,
                         "under_budget": total <= TIME_BUDGET_SECONDS,
                         "workers": workers, "torch_threads_per_worker": threads}
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                        help="M1 final checkpoint; relative paths also try the main checkout")
    parser.add_argument("--deals", type=int, default=10000)
    parser.add_argument("--matches", type=int, default=1000, help="0 skips full matches")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--opponents", nargs="+", default=list(OPPONENTS), choices=OPPONENTS)
    parser.add_argument("--throughput-config", type=Path, default=DEFAULT_THROUGHPUT_CONFIG)
    parser.add_argument("--throughput-seconds", type=float, default=120.0,
                        help="wall-clock budget for the DMC timing run; 0 skips it")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="evaluation processes (default: all logical CPUs); deterministic policies preserve results")
    parser.add_argument("--threads", type=int, default=1,
                        help="PyTorch CPU threads per evaluation worker")
    parser.add_argument("--out", "--output", dest="output", type=Path, default=DEFAULT_OUTPUT)
    add_eval_arguments(parser)
    args = parser.parse_args(argv)
    if (args.deals < 1 or args.matches < 0 or args.bootstrap_samples < 1
            or args.threads < 1 or args.workers < 1):
        parser.error("require deals, bootstrap samples, threads and workers > 0 and matches >= 0")
    if args.throughput_seconds < 0:
        parser.error("throughput seconds must be nonnegative")
    checkpoint = resolve_path(args.checkpoint)
    throughput_config = resolve_path(args.throughput_config) if args.throughput_seconds else None
    report = run_baseline(checkpoint, args.deals, args.matches, args.seed, args.bootstrap_samples,
                          tuple(args.opponents), throughput_config, args.throughput_seconds,
                          args.device, args.threads, args.workers, eval_config=config_from_args(args))
    report["command"] = ["python", "-m", "eval.stage_b_baseline",
                         *(argv if argv is not None else sys.argv[1:])]
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Baseline written to {output}")


if __name__ == "__main__":
    main()
