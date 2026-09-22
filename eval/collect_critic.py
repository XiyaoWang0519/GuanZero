"""Collect M1 self-play decisions for the Stage B perfect-information critic.

Stage B task B1 (``docs/STAGE_B_TODO.md``). The frozen M1 checkpoint drives
all four seats with argmax play and the C++ heuristic tribute, exactly as the
self-play belief collections do. Every decision, tribute and back-tribute
included, records:

- the actor observation (bit-packed, ``gd.OBS_DIM`` bits),
- the three ``hidden_counts`` rows ``[3, 54]`` for seats +1, +2, +3,
- phase, seat, stage bin (early/middle/late, the belief probe definition),
- the M1 play head's ``Q`` of the chosen action and max ``Q`` over the
  candidates (NaN for tribute rows, where the play head does not act),
- the Monte Carlo round return for the actor's team (DESIGN 8.2 item 1),
  which is the engine's ``RoundResult.seat_return`` of the acting seat.

Rounds are grouped whole and each match is assigned to exactly one of
train/val/test by a hash of ``(seed, env_id, match_id)``, so the split is
match-disjoint and deterministic under the seed. Output is sharded per split
(``<split>/shard-NNNNN.npz``) with a ``manifest.json`` that carries the
schema version, provenance and running counts; it is rewritten atomically
after every shard, so a stopped run still leaves a consistent dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

import gd
import numpy as np
import torch

from eval.policies import ModelPolicy, load_policy
from train.ckpt import load_checkpoint
from train.model import select_actions
from train.tribute_data import engine_source_digest

SCHEMA_VERSION = 1
DATASET = "stage_b_critic"
SPLITS = ("train", "val", "test")
STAGES = ("early", "middle", "late")
# Stage from the public number of cards played, as in train/belief_probe.py:
# the four 108-plane played-card blocks of the observation, cut at 36 and 72.
PLAYED_SLICE = slice(216, 648)
STAGE_EDGES = (36, 72)
RETURN_VALUES = tuple(range(-3, 4))
M1_RELATIVE = Path(".work/runpod/artifacts/pilot/final.pt")
ROOT = Path(__file__).resolve().parents[1]


def default_checkpoint() -> Path:
    """The M1 final. Worktrees lack ``.work``, so search the enclosing repos."""
    for base in (ROOT, *ROOT.parents):
        candidate = base / M1_RELATIVE
        if candidate.is_file():
            return candidate
    return ROOT / M1_RELATIVE


def stage_bins(obs: np.ndarray) -> np.ndarray:
    """Stage bin 0/1/2 (early/middle/late) of unpacked observations."""
    played = np.asarray(obs)[:, PLAYED_SLICE].sum(-1)
    return np.digitize(played, STAGE_EDGES).astype(np.int8)


def split_of(seed: int, env_id: int, match_id: int,
             fractions: tuple[float, float, float]) -> str:
    """Deterministic match-level split from a hash of the match identity."""
    digest = hashlib.sha256(f"{seed}:{env_id}:{match_id}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2.0**64
    if u < fractions[0]:
        return SPLITS[0]
    return SPLITS[1] if u < fractions[0] + fractions[1] else SPLITS[2]


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


class ShardWriter:
    """Buffers whole rounds for one split and writes fixed-size shards."""

    def __init__(self, directory: Path, split: str, shard_rounds: int, seed: int) -> None:
        self.directory, self.split = directory / split, split
        self.shard_rounds, self.seed = shard_rounds, seed
        self.rounds: list[dict] = []
        self.shards: list[dict] = []

    def add(self, record: dict) -> dict | None:
        self.rounds.append(record)
        return self.flush() if len(self.rounds) >= self.shard_rounds else None

    def flush(self) -> dict | None:
        if not self.rounds:
            return None
        rounds, self.rounds = self.rounds, []
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"shard-{len(self.shards):05d}.npz"
        lengths = np.asarray([len(r["seat"]) for r in rounds], dtype=np.int64)
        per_round = {key: np.asarray([r[key] for r in rounds])
                     for key in ("env_id", "match_id", "round_index", "round_level",
                                 "winning_team", "gain", "match_winner", "order",
                                 "seat_return")}
        payload = {
            "schema_version": np.asarray(SCHEMA_VERSION), "dataset": np.asarray(DATASET),
            "split": np.asarray(self.split), "seed": np.asarray(self.seed, dtype=np.uint64),
            "obs_dim": np.asarray(gd.OBS_DIM),
            "round_offsets": np.concatenate(([0], np.cumsum(lengths))).astype(np.int64),
            "obs_bits": np.concatenate([r["obs_bits"] for r in rounds]),
            "hidden": np.concatenate([r["hidden"] for r in rounds]),
            "phase": np.concatenate([r["phase"] for r in rounds]),
            "seat": np.concatenate([r["seat"] for r in rounds]),
            "stage": np.concatenate([r["stage"] for r in rounds]),
            "q_chosen": np.concatenate([r["q_chosen"] for r in rounds]),
            "q_max": np.concatenate([r["q_max"] for r in rounds]),
            "num_candidates": np.concatenate([r["num_candidates"] for r in rounds]),
            "team_return": np.concatenate([r["team_return"] for r in rounds]),
        }
        payload.update({key: value.astype(np.int64 if key == "match_id" else np.int32)
                        for key, value in per_round.items()})
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **payload)
        os.replace(temporary, path)
        stages = np.bincount(payload["stage"], minlength=3)
        returns = np.bincount(payload["team_return"].astype(np.int64) + 3, minlength=7)
        entry = {"path": str(path.relative_to(self.directory.parent)),
                 "rounds": len(rounds), "decisions": int(lengths.sum()),
                 "play_decisions": int((payload["phase"] == int(gd.Phase.Play)).sum()),
                 "stage_decisions": {name: int(stages[i]) for i, name in enumerate(STAGES)},
                 "decision_return_counts": {str(v): int(returns[v + 3]) for v in RETURN_VALUES},
                 "round_return_team0_counts": {
                     str(v): int((per_round["seat_return"][:, 0] == v).sum())
                     for v in RETURN_VALUES},
                 # Globally unique match identity, also across parallel workers.
                 "matches": sorted({f"{self.seed}:{e}:{m}" for e, m in zip(
                     per_round["env_id"].tolist(), per_round["match_id"].tolist())}),
                 "sha256": file_sha256(path)}
        self.shards.append(entry)
        return entry


def load_shard(path: Path, unpack: bool = True) -> dict[str, np.ndarray]:
    """Read one shard; ``obs`` is unpacked to ``uint8`` 0/1 when requested."""
    with np.load(path, allow_pickle=False) as data:
        if int(data["schema_version"]) != SCHEMA_VERSION or str(data["dataset"]) != DATASET:
            raise ValueError(f"unsupported critic shard schema: {path}")
        record = {key: data[key].copy() for key in data.files}
    if unpack:
        record["obs"] = np.unpackbits(record["obs_bits"], axis=1,
                                      count=int(record["obs_dim"]))
    return record


def summarize(output: Path) -> dict:
    """Coverage summary of a collection directory, from its manifest."""
    return summarize_shards(json.loads((Path(output) / "manifest.json").read_text())["shards"])


def summarize_shards(shards: list[dict]) -> dict:
    """Rounds, decisions, matches, stage bins and returns, per split and total."""
    splits = {}
    for split in SPLITS:
        entries = [s for s in shards if f"/{split}/" in f"/{s['path']}"]
        splits[split] = {
            "shards": len(entries),
            "rounds": sum(s["rounds"] for s in entries),
            "decisions": sum(s["decisions"] for s in entries),
            "play_decisions": sum(s["play_decisions"] for s in entries),
            "matches": len({m for s in entries for m in s["matches"]}),
            "stage_decisions": {k: sum(s["stage_decisions"][k] for s in entries) for k in STAGES},
            "decision_return_counts": {str(v): sum(s["decision_return_counts"][str(v)]
                                                   for s in entries) for v in RETURN_VALUES},
            "round_return_team0_counts": {str(v): sum(s["round_return_team0_counts"][str(v)]
                                                      for s in entries) for v in RETURN_VALUES},
        }
    total = {key: sum(splits[s][key] for s in SPLITS)
             for key in ("rounds", "decisions", "play_decisions", "matches")}
    total["stage_decisions"] = {k: sum(splits[s]["stage_decisions"][k] for s in SPLITS)
                                for k in STAGES}
    for key in ("decision_return_counts", "round_return_team0_counts"):
        total[key] = {str(v): sum(splits[s][key][str(v)] for s in SPLITS) for v in RETURN_VALUES}
    return {"total": total, "splits": splits}


def collect_critic(checkpoint: str | Path, output: Path, *, rounds: int = 1000,
                   num_envs: int = 256, seed: int = 20260923, max_seconds: float = 600,
                   device: str = "cpu", threads: int = 1, torch_threads: int | None = None,
                   shard_rounds: int = 2000,
                   split_fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
                   min_updates: int = 0) -> dict:
    """Collect ``rounds`` whole self-play rounds, or as many as fit in the time.

    Reaching ``max_seconds`` is not an error: rounds still in flight are
    discarded, every buffered round is flushed, and the manifest says
    ``time_limited``. Any other failure marks it ``incomplete`` and re-raises.
    """
    started = time.monotonic()
    if rounds < 1 or num_envs < 1 or threads < 1 or shard_rounds < 1:
        raise ValueError("rounds, environments, threads and shard size must be positive")
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("maximum seconds must be positive and finite")
    if not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative 63-bit integer")
    if (len(split_fractions) != 3 or min(split_fractions) <= 0
            or abs(sum(split_fractions) - 1) > 1e-9):
        raise ValueError("split fractions must be three positive numbers summing to one")
    if output.exists():
        raise FileExistsError(f"critic output already exists: {output}")
    checkpoint = Path(checkpoint)
    extension = Path(importlib.import_module("gd._gd_core").__file__)
    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset": DATASET, "status": "collecting",
        "checkpoint_source": str(checkpoint), "checkpoint_file_sha256": file_sha256(checkpoint),
        "seed": seed, "requested_rounds": rounds, "num_envs": num_envs,
        "threads": threads, "torch_threads": torch_threads or threads, "device": device,
        "max_seconds": max_seconds, "shard_rounds": shard_rounds,
        "split_fractions": dict(zip(SPLITS, split_fractions)),
        "split_rule": "sha256(f'{seed}:{env_id}:{match_id}')[:8] / 2**64 against cumulative fractions",
        "stage_rule": {"played_slice": [PLAYED_SLICE.start, PLAYED_SLICE.stop],
                       "edges": list(STAGE_EDGES), "names": list(STAGES)},
        "return_rule": "RoundResult.seat_return[seat]: +gain for the winning team, -gain for "
                       "the other, 0 for both in a zeroed level-A round (DESIGN 8.2 item 1)",
        "play_mode": "fp32_argmax self-play, all four seats", "tribute_policy": "heuristic",
        "q_rule": "play head score_candidates; NaN on tribute and back-tribute rows",
        "rules": "house", "obs_dim": gd.OBS_DIM, "act_dim": gd.ACT_DIM,
        "fields": {
            "obs_bits": "[N, ceil(obs_dim/8)] uint8, np.packbits of the 0/1 actor observation",
            "hidden": "[N, 3, 54] uint8 hidden_counts, seats +1, +2, +3 (privileged)",
            "phase": "[N] int8 gd.Phase", "seat": "[N] int8", "stage": "[N] int8 0/1/2",
            "q_chosen": "[N] float32", "q_max": "[N] float32",
            "num_candidates": "[N] int32", "team_return": "[N] int8",
            "round_offsets": "[R+1] int64 decision offsets of whole rounds",
            "env_id/match_id/round_index": "[R] round identity; match = (seed, env_id, match_id)",
            "round_level/winning_team/gain/match_winner/order/seat_return": "[R] engine RoundResult",
        },
        "collected_rounds": 0, "collected_decisions": 0, "discarded_inflight_rounds": 0,
        "learner_updates": 0, "shards": [],
        "torch_version": torch.__version__,
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "engine_source_sha256": engine_source_digest(), "engine_sha256": file_sha256(extension),
    }
    output.mkdir(parents=True, exist_ok=False)
    writers = {split: ShardWriter(output, split, shard_rounds, seed) for split in SPLITS}
    pending: dict[tuple[int, int, int], dict[str, list]] = {}
    deadline = started + max_seconds

    def write_manifest() -> None:
        manifest["elapsed_seconds"] = time.monotonic() - started
        manifest["shards"] = [s for split in SPLITS for s in writers[split].shards]
        write_json(output / "manifest.json", manifest)

    write_manifest()
    try:
        torch.set_num_threads(torch_threads or threads)
        policy = load_policy(str(checkpoint), device=device)
        if not isinstance(policy, ModelPolicy):
            raise ValueError("critic collection requires a checkpoint")
        if (policy.stage != "dmc" or not policy.heuristic_tribute
                or policy.action_mode != "canonical" or policy.margin != 0):
            raise ValueError("critic collection requires a canonical DMC checkpoint with "
                             "heuristic tribute and argmax play")
        if policy.training_seed is None or seed == policy.training_seed:
            raise ValueError("collection seed must differ from a known training seed")
        progress = load_checkpoint(checkpoint, device="cpu").get("progress", {})
        updates = int(progress.get("updates", 0))
        if updates < min_updates:
            raise ValueError(f"checkpoint has {updates} updates, fewer than {min_updates}; "
                             "is this the smoke model instead of the M1 final?")
        manifest.update(checkpoint=policy.name, checkpoint_id=policy.checkpoint_id,
                        training_seed=policy.training_seed, checkpoint_updates=updates)
        env = gd.VecEnv(num_envs, num_threads=threads, seed=seed,
                        rules=gd.RuleConfig.house(), actions=gd.ActionConfig())
        env.reset()
        play_code = int(gd.Phase.Play)
        done = False
        while not done:
            if time.monotonic() >= deadline:
                manifest["status"] = "time_limited"
                break
            batch = env.pending()
            for result in env.drain_finished_rounds():
                key = (int(result.env_id), int(result.match_id), int(result.round_index))
                rows = pending.pop(key, None)
                if rows is None:
                    continue
                seat_return = np.asarray(result.seat_return, dtype=np.int8)
                seats = np.asarray(rows["seat"], dtype=np.int8)
                obs = np.stack(rows["obs"])
                record = {
                    "obs_bits": np.packbits(obs, axis=1),
                    "hidden": np.stack(rows["hidden"]),
                    "phase": np.asarray(rows["phase"], dtype=np.int8), "seat": seats,
                    "stage": stage_bins(obs),
                    "q_chosen": np.asarray(rows["q_chosen"], dtype=np.float32),
                    "q_max": np.asarray(rows["q_max"], dtype=np.float32),
                    "num_candidates": np.asarray(rows["num_candidates"], dtype=np.int32),
                    "team_return": seat_return[seats],
                    "env_id": key[0], "match_id": key[1], "round_index": key[2],
                    "round_level": int(result.round_level),
                    "winning_team": int(result.winning_team), "gain": int(result.gain),
                    "match_winner": int(result.match_winner),
                    "order": list(result.order), "seat_return": seat_return.tolist(),
                }
                split = split_of(seed, key[0], key[1], split_fractions)
                if writers[split].add(record) is not None:
                    write_manifest()
                manifest["collected_rounds"] += 1
                manifest["collected_decisions"] += len(seats)
                if manifest["collected_rounds"] >= rounds:
                    done = True
                    break
            if done:
                break
            phase = np.asarray(batch.phase)
            offsets_np = np.asarray(batch.offsets)
            choices = np.array(batch.greedy_choice, dtype=np.int32, copy=True)
            play = phase == play_code
            q_chosen = np.full(batch.rows, np.nan, dtype=np.float32)
            q_max = np.full(batch.rows, np.nan, dtype=np.float32)
            if play.any():
                with torch.inference_mode():
                    offsets = torch.as_tensor(np.array(offsets_np, copy=True),
                                              device=device, dtype=torch.long)
                    values = policy.model.score_candidates(
                        torch.as_tensor(np.array(batch.obs, copy=True), device=device),
                        torch.as_tensor(np.array(batch.cand, copy=True), device=device),
                        offsets, torch.as_tensor(np.array(phase, copy=True), device=device,
                                                 dtype=torch.long), phase_code=play_code)
                    if not bool(torch.isfinite(values).all()):
                        raise FloatingPointError("non-finite frozen policy values")
                    selected = select_actions(values.float(), offsets).cpu().numpy()
                    scores = values.float().cpu().numpy()
                choices[play] = selected[play]
                starts = offsets_np[:-1]
                q_max_all = np.maximum.reduceat(scores, starts) if len(scores) else scores
                q_chosen[play] = scores[starts + selected][play]
                q_max[play] = q_max_all[play]
            obs_all = np.asarray(batch.obs)
            hidden_all = np.asarray(batch.hidden_counts)
            for row in range(batch.rows):
                key = (int(batch.env_id[row]), int(batch.match_id[row]),
                       int(batch.round_index[row]))
                rows = pending.setdefault(key, {name: [] for name in (
                    "obs", "hidden", "phase", "seat", "q_chosen", "q_max", "num_candidates")})
                rows["obs"].append(obs_all[row].astype(np.uint8))
                rows["hidden"].append(np.array(hidden_all[row], dtype=np.uint8, copy=True))
                rows["phase"].append(int(phase[row]))
                rows["seat"].append(int(batch.seat[row]))
                rows["q_chosen"].append(q_chosen[row])
                rows["q_max"].append(q_max[row])
                rows["num_candidates"].append(int(offsets_np[row + 1] - offsets_np[row]))
            env.step(choices)
        if manifest["status"] == "collecting":
            manifest["status"] = "complete"
        manifest["discarded_inflight_rounds"] = len(pending)
        for writer in writers.values():
            writer.flush()
    except BaseException as error:
        manifest["status"] = "incomplete"
        manifest["error_type"] = type(error).__name__
        for writer in writers.values():
            writer.flush()
        raise
    finally:
        manifest["summary"] = summarize_shards([s for split in SPLITS
                                                for s in writers[split].shards])
        write_manifest()
    return manifest


def _worker(arguments: tuple[str, str, dict]) -> dict:
    checkpoint, output, options = arguments
    return collect_critic(checkpoint, Path(output), **options)


def collect_parallel(checkpoint: str | Path, output: Path, *, workers: int,
                     rounds: int, seed: int, **options) -> dict:
    """Run ``workers`` independent collector processes and merge their manifests.

    Worker ``i`` writes ``output/worker-NN`` with seed ``seed + i`` and an
    even share of the rounds. The split hash covers the seed, so match
    identities, and therefore splits, stay disjoint across workers. The
    top-level ``manifest.json`` lists every shard relative to ``output``.
    """
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing

    if workers < 2:
        raise ValueError("parallel collection needs at least two workers")
    if rounds < workers:
        raise ValueError("every worker needs at least one round")
    if output.exists():
        raise FileExistsError(f"critic output already exists: {output}")
    output.mkdir(parents=True)
    started = time.monotonic()
    quotas = [rounds // workers + int(i < rounds % workers) for i in range(workers)]
    jobs = [(str(checkpoint), str(output / f"worker-{i:02d}"),
             dict(options, rounds=quotas[i], seed=seed + i)) for i in range(workers)]
    errors = []
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for future in futures:
            try:
                future.result()
            except Exception as error:  # every worker's manifest records its own failure
                errors.append(error)
    reports = []
    for i in range(workers):
        path = output / f"worker-{i:02d}" / "manifest.json"
        reports.append(json.loads(path.read_text()) if path.exists() else {"status": "missing"})
    shards = [dict(s, path=f"worker-{i:02d}/{s['path']}")
              for i, report in enumerate(reports) for s in report.get("shards", [])]
    statuses = {report["status"] for report in reports}
    first = next((r for r in reports if "checkpoint_id" in r), {})
    manifest = {
        "schema_version": SCHEMA_VERSION, "dataset": DATASET, "layout": "workers",
        "status": statuses.pop() if len(statuses) == 1 else "mixed:" + ",".join(sorted(statuses)),
        "requested_rounds": rounds, "seed": seed, "workers": workers,
        "worker_seeds": [seed + i for i in range(workers)], "worker_quotas": quotas,
        "collected_rounds": sum(r.get("collected_rounds", 0) for r in reports),
        "collected_decisions": sum(r.get("collected_decisions", 0) for r in reports),
        "elapsed_seconds": time.monotonic() - started,
        "worker_elapsed_seconds": [r.get("elapsed_seconds") for r in reports],
        **{key: first.get(key) for key in (
            "checkpoint_source", "checkpoint_file_sha256", "checkpoint", "checkpoint_id",
            "checkpoint_updates", "training_seed", "num_envs", "threads", "torch_threads",
            "split_fractions", "split_rule", "stage_rule", "return_rule", "play_mode",
            "tribute_policy", "q_rule", "fields", "obs_dim", "act_dim", "rules",
            "collector_sha256", "engine_source_sha256", "engine_sha256", "torch_version")},
        "learner_updates": 0, "shards": shards, "summary": summarize_shards(shards),
    }
    write_json(output / "manifest.json", manifest)
    if errors:
        raise errors[0]
    return manifest


def shard_paths(output: Path) -> list[Path]:
    """Every shard of a single-process or multi-worker collection."""
    manifest = json.loads((Path(output) / "manifest.json").read_text())
    return [Path(output) / s["path"] for s in manifest["shards"]]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help=f"default: {M1_RELATIVE} in this repo or an enclosing one")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=200_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--max-seconds", type=float, default=3 * 3600)
    parser.add_argument("--threads", type=int, default=4, help="VecEnv threads")
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--shard-rounds", type=int, default=2000)
    parser.add_argument("--split", type=float, nargs=3, default=(0.8, 0.1, 0.1),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--min-updates", type=int, default=30_000,
                        help="refuse checkpoints with fewer learner updates (the smoke model has 6)")
    parser.add_argument("--workers", type=int, default=1,
                        help="independent collector processes, each with --threads VecEnv "
                             "threads and --torch-threads inference threads")
    parser.add_argument("--summary", type=Path, default=None,
                        help="also write the coverage summary JSON here")
    args = parser.parse_args(argv)
    options = dict(num_envs=args.num_envs, max_seconds=args.max_seconds, device=args.device,
                   threads=args.threads, torch_threads=args.torch_threads,
                   shard_rounds=args.shard_rounds, split_fractions=tuple(args.split),
                   min_updates=args.min_updates)
    checkpoint = args.checkpoint or default_checkpoint()
    if args.workers > 1:
        report = collect_parallel(checkpoint, args.output, workers=args.workers,
                                  rounds=args.rounds, seed=args.seed, **options)
    else:
        report = collect_critic(checkpoint, args.output, rounds=args.rounds, seed=args.seed,
                                **options)
    report.pop("shards")
    if args.summary is not None:
        keys = ("status", "schema_version", "dataset", "checkpoint_source",
                "checkpoint_file_sha256", "checkpoint_id", "checkpoint_updates", "seed",
                "workers", "worker_seeds", "num_envs", "threads", "torch_threads",
                "requested_rounds", "collected_rounds", "collected_decisions",
                "elapsed_seconds", "split_fractions", "split_rule", "stage_rule",
                "return_rule", "play_mode", "q_rule", "collector_sha256",
                "engine_source_sha256", "engine_sha256", "fields")
        summary = {key: report[key] for key in keys if key in report}
        summary["output"] = str(args.output)
        summary["rounds_per_second"] = report["collected_rounds"] / report["elapsed_seconds"]
        summary["coverage"] = report["summary"]
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.summary, summary)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
