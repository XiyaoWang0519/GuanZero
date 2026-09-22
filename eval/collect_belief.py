"""Collect independent frozen-checkpoint games for hidden-hand experiments.

No optimizer, learner update or training checkpoint is created. Output uses
the normal round-log schema. Architecture probes explicitly identify their
intended downstream supervised training; the default remains evaluation-only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time

import gd
import numpy as np
import torch

from eval.policies import ModelPolicy, load_policy
from train.buffer import Decision
from train.logs import public_token, save_round
from train.model import select_actions
from train.tribute_data import engine_source_digest


def collect_belief(checkpoint: str | Path, output: Path, *, rounds: int = 100,
                   num_envs: int = 4, seed: int = 20260922, max_seconds: float = 300,
                   device: str = "cpu", threads: int = 1,
                   purpose: str = "evaluation_only") -> dict:
    """Collect balanced per-environment round quotas under a fixed policy.

    At least two environments provide distinct match groups even for a
    two-round smoke test. An incomplete run is clearly marked and raises.
    """
    started = time.monotonic()
    if rounds < 2 or num_envs < 2 or threads < 1:
        raise ValueError("at least two rounds/environments and positive threads are required")
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("maximum seconds must be positive and finite")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if purpose not in ("evaluation_only", "architecture_probe"):
        raise ValueError("purpose must be evaluation_only or architecture_probe")
    if output.exists():
        raise FileExistsError(f"belief output already exists: {output}")
    num_envs = min(num_envs, rounds)
    quotas = [rounds // num_envs + int(i < rounds % num_envs) for i in range(num_envs)]
    completed = [0] * num_envs
    pending: dict[tuple[int, int, int], list[Decision]] = {}
    tokens: dict[tuple[int, int, int], list[np.ndarray]] = {}
    groups: set[str] = set()
    provenance = {
        "schema_version": 1, "purpose": purpose, "status": "collecting",
        "checkpoint_source": str(checkpoint), "seed": seed,
        "requested_rounds": rounds,
        "num_envs": num_envs, "collected_rounds": 0, "learner_updates": 0,
        "collected_decisions": 0, "match_groups": 0,
        "round_quotas": quotas, "completed_per_env": completed,
        "device": device, "threads": threads, "max_seconds": max_seconds,
        "rules": "house", "obs_dim": gd.OBS_DIM, "act_dim": gd.ACT_DIM,
        "torch_version": torch.__version__,
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "engine_source_sha256": engine_source_digest(),
    }
    output.mkdir(parents=True, exist_ok=False)
    deadline = started + max_seconds

    def check_deadline() -> None:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"belief collection exceeded {max_seconds} seconds")

    def write_provenance() -> None:
        provenance["elapsed_seconds"] = time.monotonic() - started
        temporary = output / "provenance.tmp"
        temporary.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "provenance.json")

    write_provenance()
    try:
        check_deadline()
        torch.set_num_threads(threads)
        policy = load_policy(str(checkpoint), device=device)
        check_deadline()
        if not isinstance(policy, ModelPolicy):
            raise ValueError("independent belief collection requires a checkpoint")
        if (policy.stage != "dmc" or not policy.heuristic_tribute
                or policy.action_mode != "canonical" or policy.margin != 0):
            raise ValueError("belief collection requires a canonical DMC checkpoint with "
                             "heuristic tribute and argmax play")
        if seed == policy.training_seed:
            raise ValueError("collection seed must differ from the checkpoint training seed")
        if policy.training_seed is None:
            raise ValueError("checkpoint training seed must be known to verify independence")
        extension = Path(importlib.import_module("gd._gd_core").__file__)
        with extension.open("rb") as stream:
            engine_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        provenance.update(checkpoint=policy.name, checkpoint_id=policy.checkpoint_id,
                          training_seed=policy.training_seed, stage=policy.stage,
                          action_mode=policy.action_mode, tribute_policy="heuristic",
                          sampling_margin=policy.margin, play_mode="fp32_argmax",
                          engine_sha256=engine_sha256)
        check_deadline()
        env = gd.VecEnv(num_envs, num_threads=threads, seed=seed,
                        rules=gd.RuleConfig.house(), actions=gd.ActionConfig(),
                        log_public_actions=True, log_env_limit=num_envs)
        env.reset()
        check_deadline()
        while sum(completed) < rounds:
            check_deadline()
            batch = env.pending()
            for event in env.drain_public_actions():
                key = (int(event.env_id), int(event.match_id), int(event.round_index))
                if completed[key[0]] < quotas[key[0]]:
                    tokens.setdefault(key, []).append(public_token(event))
            for result in env.drain_finished_rounds():
                key = (int(result.env_id), int(result.match_id), int(result.round_index))
                decisions = pending.pop(key, [])
                history = tokens.pop(key, [])
                if decisions and completed[key[0]] < quotas[key[0]]:
                    # The action at each prefix is the action just about to be
                    # chosen. Consumers must read only tokens[:prefix].
                    if any(not 0 <= d.prefix < len(history)
                           or history[d.prefix][:4].argmax() != d.seat for d in decisions):
                        raise RuntimeError("public history is not aligned with decision prefixes")
                    index = sum(completed)
                    group_prefix = "evaluation" if purpose == "evaluation_only" else purpose
                    group = f"{group_prefix}:{seed}:{key[0]}:{key[1]}"
                    save_round(output / f"round-{index:08d}.npz", decisions, history,
                               group=group)
                    completed[key[0]] += 1
                    provenance["collected_rounds"] = sum(completed)
                    provenance["collected_decisions"] += len(decisions)
                    groups.add(group)
                    provenance["match_groups"] = len(groups)
            if sum(completed) == rounds:
                break
            choices = np.array(batch.greedy_choice, dtype=np.int32, copy=True)
            network = np.asarray(batch.phase) == int(gd.Phase.Play)
            if network.any():
                with torch.inference_mode():
                    offsets = torch.as_tensor(np.array(batch.offsets, copy=True),
                                              device=device, dtype=torch.long)
                    values = policy.model.score_candidates(
                        torch.as_tensor(np.array(batch.obs, copy=True), device=device),
                        torch.as_tensor(np.array(batch.cand, copy=True), device=device),
                        offsets, torch.as_tensor(np.array(batch.phase, copy=True),
                                                  device=device, dtype=torch.long), phase_code=3)
                    if not bool(torch.isfinite(values).all()):
                        raise FloatingPointError("non-finite frozen policy values")
                    selected = select_actions(values.float(), offsets).cpu().numpy()
                check_deadline()
                choices[network] = selected[network]
            for row in np.flatnonzero(network):
                key = (int(batch.env_id[row]), int(batch.match_id[row]), int(batch.round_index[row]))
                if completed[key[0]] >= quotas[key[0]]:
                    continue
                index = int(batch.offsets[row]) + int(choices[row])
                pending.setdefault(key, []).append(Decision(
                    obs=np.array(batch.obs[row], dtype=np.uint8, copy=True),
                    action=np.array(batch.cand[index], dtype=np.uint8, copy=True),
                    hidden=np.array(batch.hidden_counts[row], dtype=np.uint8, copy=True),
                    seat=int(batch.seat[row]), phase=3, prefix=len(tokens.get(key, [])),
                ))
            env.step(choices)
        check_deadline()
        provenance["status"] = "complete"
    except BaseException as error:
        provenance["status"] = "incomplete"
        provenance["error_type"] = type(error).__name__
        raise
    finally:
        write_provenance()
    return provenance


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--max-seconds", type=float, default=300)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--purpose", default="evaluation_only",
                        choices=("evaluation_only", "architecture_probe"))
    args = parser.parse_args(argv)
    report = collect_belief(args.checkpoint, args.output, rounds=args.rounds,
                            num_envs=args.num_envs, seed=args.seed, max_seconds=args.max_seconds,
                            device=args.device, threads=args.threads, purpose=args.purpose)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
