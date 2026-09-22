"""Stage A2: label exchanges by cloning a decision and finishing each branch.

Only acting-seat observations/candidates and scalar returns are saved. Native
engine snapshots stay in memory; they are not a portable dataset format.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import time

import gd
import numpy as np
import torch

from eval.policies import ModelPolicy, Policy, choose_action, load_policy


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("tribute collection exceeded its walltime limit")


def engine_source_digest() -> str:
    """Portable engine/binding identity; compiled-binary identity is also logged."""
    root = Path(__file__).resolve().parents[1]
    paths = sorted([*root.glob("cpp/include/gd/*.h"), *root.glob("cpp/src/*.cpp"),
                    root / "python/bindings.cpp"])
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def branch_returns(engine: gd.Engine, state: gd.MatchState, policy: Policy, *,
                   seed: int = 0, max_decisions: int = 2000,
                   deadline: float | None = None) -> np.ndarray:
    """Return every candidate's terminal return for the original acting seat.

    Batch play inference across branches. Other exchange decisions use the
    frozen policy's heuristic. Per-seat RNG streams start identically in each
    branch; the deployed base policy uses deterministic FP32 argmax.
    """
    if state.phase not in (gd.Phase.Tribute, gd.Phase.BackTribute):
        raise ValueError("counterfactual labels require an exchange decision")
    if max_decisions < 1:
        raise ValueError("max_decisions must be positive")
    if isinstance(policy, ModelPolicy) and (policy.margin or not policy.heuristic_tribute):
        raise ValueError("branch policy must use argmax play and heuristic tribute")
    source = state.serialize()
    actor = state.to_move
    actions = engine.legal_actions(state)
    if not actions:
        raise ValueError("exchange decision has no candidates")
    states = [gd.MatchState.deserialize(source) for _ in actions]
    rngs = [[random.Random(seed + seat * 0x9E3779B9) for seat in range(4)]
            for _ in actions]
    counts = [1] * len(actions)
    for branch, action in zip(states, actions):
        engine.apply(branch, action)
    returns = np.empty(len(actions), dtype=np.float32)
    active = list(range(len(states)))
    while active:
        _check_deadline(deadline)
        playing: list[int] = []
        candidates: list[list] = []
        remaining = []
        for index in active:
            branch = states[index]
            if branch.phase == gd.Phase.RoundEnd:
                returns[index] = engine.end_round(branch).seat_return[actor]
                continue
            if counts[index] >= max_decisions:
                raise RuntimeError("counterfactual branch exceeded decision bound")
            remaining.append(index)
            if isinstance(policy, ModelPolicy) and branch.phase == gd.Phase.Play:
                playing.append(index)
                legal = engine.legal_actions(branch)
                if not legal:
                    raise RuntimeError("branch has no legal play")
                candidates.append(legal)
            else:
                engine.apply(branch, choose_action(policy, engine, branch,
                                                    rngs[index][branch.to_move]))
                counts[index] += 1
        if playing:
            obs = np.stack([states[i].observation(states[i].to_move) for i in playing])
            cand = np.stack([engine.encode_action(a, states[i], states[i].to_move)
                             for i, legal in zip(playing, candidates) for a in legal])
            offsets = np.cumsum([0] + [len(a) for a in candidates])
            with torch.inference_mode():
                values = policy.model.score_candidates(
                    torch.as_tensor(obs, device=policy.device),
                    torch.as_tensor(cand, device=policy.device),
                    torch.as_tensor(offsets, device=policy.device),
                    torch.full((len(playing),), 3, device=policy.device), phase_code=3)
                if not bool(torch.isfinite(values).all()):
                    raise FloatingPointError("non-finite counterfactual play values")
                # One host copy; candidate order gives deterministic argmax ties.
                values = values.cpu().numpy()
            for row, (index, legal) in enumerate(zip(playing, candidates)):
                choice = int(values[offsets[row]:offsets[row + 1]].argmax())
                engine.apply(states[index], legal[choice])
                counts[index] += 1
        active = remaining
    if state.serialize() != source:
        raise RuntimeError("counterfactual labeling changed the source trajectory")
    return returns


def label_position(engine: gd.Engine, state: gd.MatchState, policy: Policy, *,
                   seed: int = 0, group: str = "", deadline: float | None = None) -> dict:
    """Capture public actor input before revealing outcomes in cloned branches."""
    if state.phase not in (gd.Phase.Tribute, gd.Phase.BackTribute):
        raise ValueError("expected tribute or back-tribute")
    actor = state.to_move
    actions = engine.legal_actions(state)
    obs = np.array(state.observation(actor), dtype=np.uint8, copy=True)
    cand = np.stack([engine.encode_action(a, state, actor) for a in actions]).astype(np.uint8)
    heuristic = engine.greedy(state)
    returns = branch_returns(engine, state, policy, seed=seed, deadline=deadline)
    return {"schema_version": np.asarray(1), "group": np.asarray(group),
            "obs": obs, "cand": cand, "phase": np.asarray(int(state.phase)),
            "seat": np.asarray(actor), "returns": returns,
            "advantages": returns - returns.mean(),
            "heuristic_index": np.asarray(heuristic),
            "candidate_cards": np.asarray([a.cards[0] for a in actions], dtype=np.int16)}


def collect(checkpoint: Path, output: Path, *, positions: int = 256,
            sample_fraction: float = 0.02, seed: int = 20260923,
            max_seconds: float = 600, device: str = "cpu", threads: int = 1) -> dict:
    if positions < 2 or threads < 1 or not 0 < sample_fraction <= 1:
        raise ValueError("need at least two positions, positive threads and fraction in (0,1]")
    if not math.isfinite(max_seconds) or max_seconds <= 0 or not 0 <= seed < 2**64:
        raise ValueError("invalid walltime or unsigned 64-bit seed")
    if output.exists():
        raise FileExistsError(f"dataset already exists: {output}")
    torch.set_num_threads(threads)
    policy = load_policy(str(checkpoint), device=device)
    if not isinstance(policy, ModelPolicy) or not policy.heuristic_tribute:
        raise ValueError("collection needs a frozen Stage A checkpoint")
    if seed == policy.training_seed:
        raise ValueError("collection seed must differ from the base training seed")
    action_mode = getattr(policy, "action_mode", "canonical")
    engine = gd.Engine(actions=gd.ActionConfig.full() if action_mode == "full" else gd.ActionConfig())
    deal_rng = random.Random(seed)
    sample_rng = random.Random(seed ^ 0xA2A2A2A2)
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + max_seconds
    extension = Path(importlib.import_module("gd._gd_core").__file__)
    with extension.open("rb") as stream:
        engine_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"schema_version": 1, "purpose": "tribute_counterfactual_training",
              "status": "collecting", "base_checkpoint_id": policy.checkpoint_id,
              "base_training_seed": policy.training_seed, "seed": seed,
              "requested_positions": positions, "positions": 0, "branches": 0,
              "sample_fraction": sample_fraction, "sampling_unit": "round",
              "rounds": 0, "sampled_rounds": 0, "matches_started": 0,
              "phase_positions": {"1": 0, "2": 0}, "nonconstant_positions": 0,
              "play_mode": "fp32_argmax", "other_tribute": "heuristic",
              "action_mode": action_mode, "device": device, "rules": "house",
              "obs_dim": gd.OBS_DIM, "act_dim": gd.ACT_DIM,
              "max_decisions_per_round": 2000,
              "engine_sha256": engine_sha256,
              "engine_source_sha256": engine_source_digest(),
              "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch_version": torch.__version__,
              "seed_streams": "independent deal, round-sampling, seat-policy and branch streams"}

    def write_report() -> None:
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = output / "provenance.tmp"
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "provenance.json")

    try:
        write_report()
        while report["positions"] < positions:
            _check_deadline(deadline)
            state = gd.MatchState()
            match = report["matches_started"]
            engine.new_match(state, deal_rng.getrandbits(64))
            policy_rngs = [random.Random(seed + match * 1000003 + seat)
                           for seat in range(4)]
            report["matches_started"] += 1
            while state.winner < 0 and report["positions"] < positions:
                sampled = sample_rng.random() < sample_fraction
                report["rounds"] += 1
                report["sampled_rounds"] += int(sampled)
                decisions = 0
                while state.phase != gd.Phase.RoundEnd:
                    _check_deadline(deadline)
                    if decisions >= 2000:
                        raise RuntimeError("source round exceeded decision bound")
                    if sampled and state.phase in (gd.Phase.Tribute, gd.Phase.BackTribute):
                        if len(engine.legal_actions(state)) > 1:
                            case = label_position(engine, state, policy,
                                seed=seed + match * 1000003 + state.round_index * 2003 + decisions,
                                group=f"{seed}:{match}", deadline=deadline)
                            index = report["positions"]
                            temporary = output / f"position-{index:08d}.tmp"
                            with temporary.open("wb") as stream:
                                np.savez_compressed(stream, **case)
                            os.replace(temporary, temporary.with_suffix(".npz"))
                            report["positions"] += 1
                            report["branches"] += len(case["returns"])
                            report["phase_positions"][str(int(case["phase"]))] += 1
                            report["nonconstant_positions"] += int(np.ptp(case["returns"]) > 0)
                            write_report()
                            if report["positions"] == positions:
                                break
                    engine.apply(state, choose_action(policy, engine, state,
                                                       policy_rngs[state.to_move]))
                    decisions += 1
                if report["positions"] == positions:
                    break
                engine.end_round(state)
                if state.winner < 0:
                    engine.begin_round(state)
        report["status"] = "complete"
    except BaseException:
        report["status"] = "incomplete"
        raise
    finally:
        write_report()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positions", type=int, default=256)
    parser.add_argument("--sample-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(collect(args.checkpoint, args.output, positions=args.positions,
        sample_fraction=args.sample_fraction, seed=args.seed, max_seconds=args.max_seconds,
        device=args.device, threads=args.threads), allow_nan=False))


if __name__ == "__main__":
    main()
