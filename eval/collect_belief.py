"""Collect independent frozen-checkpoint games for hidden-hand experiments.

No optimizer, learner update or training checkpoint is created. Output uses
the normal round-log schema. Architecture probes explicitly identify their
intended downstream supervised training; the default remains evaluation-only.

With ``--styled`` the checkpoint policy drives one team and the continuous
style-parameterised bot drives the other two seats. Styles are sampled once per
match per seat from a configurable style space (``train/styles.py``), with a
held-out style region reserved for test matches. Every logged decision records
its match, round, seat, driver and the four seats' style vectors.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time
import warnings

import gd
import numpy as np
import torch

from eval.policies import ModelPolicy, load_policy
from train import styles as style_lib
from train.buffer import Decision
from train.logs import DRIVER_BOT, DRIVER_POLICY, public_token, save_round
from train.model import select_actions
from train.tribute_data import engine_source_digest


class StyleAssignment:
    """Per-match seat drivers and style vectors for one environment."""

    def __init__(self, space: style_lib.StyleSpace, env_id: int, match_id: int,
                 style_seed: int, region: str, heldout_fraction: float,
                 policy_team: str, styled: bool, available: bool) -> None:
        rng = style_lib.match_style_rng(style_seed, env_id, match_id)
        team = int(rng.integers(2)) if policy_team == "random" else int(policy_team)
        self.env_id, self.match_id, self.policy_team = env_id, match_id, team
        # Without styled opponents the checkpoint drives all four seats, exactly
        # as in the published self-play belief collections.
        self.seat_driver = [DRIVER_POLICY if not styled or seat % 2 == team
                            else DRIVER_BOT for seat in range(4)]
        self.styles = np.zeros((4, space.dim), dtype=np.float32)
        # Resolve "mixed" once per match so both bot seats share one region and
        # every match is a clean training or held-out sample.
        if region == "mixed":
            region = "heldout" if rng.random() < heldout_fraction else "train"
        for seat in range(4):
            if self.seat_driver[seat] == DRIVER_POLICY:
                self.styles[seat] = style_lib.neutral(space.dim)
            elif available:
                self.styles[seat] = space.sample(rng, region, heldout_fraction)
            else:
                # The styled bot is missing; the seat plays greedily and the
                # style is recorded as unknown rather than invented.
                self.styles[seat] = style_lib.unknown_style(space.dim)
        bots = [self.styles[s] for s in range(4) if self.seat_driver[s] == DRIVER_BOT]
        labels = {style_lib.region_of(space, v) for v in bots}
        self.region = labels.pop() if len(labels) == 1 else "mixed" if labels else "none"

    def to_json(self) -> dict:
        return {
            "env_id": self.env_id, "match_id": self.match_id,
            "policy_team": self.policy_team, "seat_driver": list(self.seat_driver),
            "region": self.region,
            "styles": [[None if not np.isfinite(x) else float(x) for x in row]
                       for row in self.styles],
        }


def collect_belief(checkpoint: str | Path, output: Path, *, rounds: int = 100,
                   num_envs: int = 4, seed: int = 20260922, max_seconds: float = 300,
                   device: str = "cpu", threads: int = 1,
                   purpose: str = "evaluation_only", styled: bool = False,
                   style_region: str = "train", style_seed: int | None = None,
                   heldout_fraction: float = 0.5, policy_team: str = "random") -> dict:
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
    if style_region not in style_lib.REGIONS:
        raise ValueError(f"style region must be one of {style_lib.REGIONS}")
    if policy_team not in ("random", "0", "1"):
        raise ValueError("policy team must be random, 0 or 1")
    if output.exists():
        raise FileExistsError(f"belief output already exists: {output}")
    style_seed = seed if style_seed is None else int(style_seed)
    if not 0 <= style_seed < 2**63:
        raise ValueError("style seed must be a nonnegative 63-bit integer")
    space = style_lib.StyleSpace.default()
    available = style_lib.styled_bot_available()
    if styled and not available:
        warnings.warn(
            "gd.STYLE_DIM is absent: the styled bot binding is not built. Opponent "
            "seats fall back to the greedy bot and their style vectors are recorded "
            "as NaN. This collection cannot show cross-round opponent modelling.",
            RuntimeWarning, stacklevel=2)
    num_envs = min(num_envs, rounds)
    quotas = [rounds // num_envs + int(i < rounds % num_envs) for i in range(num_envs)]
    completed = [0] * num_envs
    pending: dict[tuple[int, int, int], list[Decision]] = {}
    tokens: dict[tuple[int, int, int], list[np.ndarray]] = {}
    groups: set[str] = set()
    assignments: dict[tuple[int, int], StyleAssignment] = {}
    behaviour: dict[tuple[int, int], style_lib.BehaviourAccumulator] = {}
    match_rounds: dict[tuple[int, int], int] = {}
    provenance = {
        "schema_version": 2, "purpose": purpose, "status": "collecting",
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
        "styled": bool(styled), "styled_bot_unavailable": bool(styled and not available),
        "style_region": style_region, "style_seed": style_seed,
        "heldout_fraction": heldout_fraction, "policy_team": policy_team,
        "policy_seats": "seats team and team+2, team drawn per match and recorded",
        "style_space": space.describe(),
        "styles_sha256": hashlib.sha256(Path(style_lib.__file__).read_bytes()).hexdigest(),
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

    def write_matches() -> None:
        payload = [dict(assignments[key].to_json(),
                        behaviour=behaviour[key].as_dict() if key in behaviour else {},
                        rounds=match_rounds.get(key, 0))
                   for key in sorted(assignments)]
        temporary = output / "matches.tmp"
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        temporary.replace(output / "matches.json")

    def assignment_for(env_id: int, match_id: int) -> StyleAssignment:
        key = (env_id, match_id)
        entry = assignments.get(key)
        if entry is None:
            entry = StyleAssignment(space, env_id, match_id, style_seed, style_region,
                                    heldout_fraction, policy_team, styled,
                                    available and styled)
            assignments[key] = entry
        return entry

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
        use_styled_bot = bool(styled and available)
        if use_styled_bot:
            for env_id in range(num_envs):
                assignment_for(env_id, 0)
            env.set_styles(np.stack([assignments[(i, 0)].styles for i in range(num_envs)]))
        while sum(completed) < rounds:
            check_deadline()
            batch = env.pending()
            if use_styled_bot:
                # A match can restart inside pending(). Resample that env's
                # styles and refresh the batch so styled_choice is never stale;
                # pending() is idempotent while every environment is waiting.
                fresh = {(int(e), int(m)) for e, m in zip(batch.env_id, batch.match_id)}
                if any(key not in assignments for key in fresh):
                    for env_id, match_id in sorted(fresh):
                        assignment_for(env_id, match_id)
                    latest = {}
                    for env_id, match_id in sorted(assignments):
                        latest[env_id] = assignments[(env_id, match_id)].styles
                    env.set_styles(np.stack([latest[i] for i in range(num_envs)]))
                    batch = env.pending()
            for event in env.drain_public_actions():
                key = (int(event.env_id), int(event.match_id), int(event.round_index))
                if completed[key[0]] < quotas[key[0]]:
                    tokens.setdefault(key, []).append(public_token(event))
                match = (key[0], key[1])
                behaviour.setdefault(match, style_lib.BehaviourAccumulator()).add_event(event)
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
                    entry = assignment_for(key[0], key[1])
                    save_round(output / f"round-{index:08d}.npz", decisions, history,
                               group=group,
                               meta={"match_id": key[1], "round_index": key[2],
                                     "env_id": key[0], "styles": entry.styles,
                                     "seat_driver": entry.seat_driver,
                                     "style_region": entry.region, "styled": styled})
                    completed[key[0]] += 1
                    match_rounds[(key[0], key[1])] = match_rounds.get((key[0], key[1]), 0) + 1
                    provenance["collected_rounds"] = sum(completed)
                    provenance["collected_decisions"] += len(decisions)
                    groups.add(group)
                    provenance["match_groups"] = len(groups)
            if sum(completed) == rounds:
                break
            choices = np.array(batch.greedy_choice, dtype=np.int32, copy=True)
            play = np.asarray(batch.phase) == int(gd.Phase.Play)
            drivers = np.array([assignment_for(int(e), int(m)).seat_driver[int(s)]
                                for e, m, s in zip(batch.env_id, batch.match_id, batch.seat)],
                               dtype=np.int64)
            bot = play & (drivers == DRIVER_BOT)
            network = play & (drivers == DRIVER_POLICY)
            if use_styled_bot and bot.any():
                styled_choice = np.array(batch.styled_choice, dtype=np.int32, copy=True)
                choices[bot] = styled_choice[bot]
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
            for row in np.flatnonzero(play):
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
        # Behaviour is accumulated per match, so a coverage report can bin it by
        # style slot; the collection-wide total is recorded alongside.
        parts = [accumulator.as_dict() for accumulator in behaviour.values()]
        provenance["behaviour_total"] = style_lib.merge_behaviour(parts)
        provenance["matches_sampled"] = len(assignments)
        write_matches()
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
    parser.add_argument("--styled", action=argparse.BooleanOptionalAction, default=False,
                        help="drive the opponent team with the styled heuristic bot")
    parser.add_argument("--style-region", default="train", choices=style_lib.REGIONS)
    parser.add_argument("--style-seed", type=int, default=None)
    parser.add_argument("--heldout-fraction", type=float, default=0.5,
                        help="probability of a held-out style under --style-region mixed")
    parser.add_argument("--policy-team", default="random", choices=("random", "0", "1"))
    args = parser.parse_args(argv)
    report = collect_belief(args.checkpoint, args.output, rounds=args.rounds,
                            num_envs=args.num_envs, seed=args.seed, max_seconds=args.max_seconds,
                            device=args.device, threads=args.threads, purpose=args.purpose,
                            styled=args.styled, style_region=args.style_region,
                            style_seed=args.style_seed, heldout_fraction=args.heldout_fraction,
                            policy_team=args.policy_team)
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
