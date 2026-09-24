"""Evaluate local policies with duplicate rounds, full matches, and probes.

This is an internal house-rules arena. It does not claim to implement the
unavailable OpenGuanDan Rule One--Four agents or their benchmark protocol.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Iterable

import gd

from .duplicate import evaluate_duplicates, generate_deals, play_round
from .policies import ModelPolicy, Policy, load_policy
from .probes import evaluate_probes


def wilson_interval(wins: int, games: int, confidence: float = 0.95) -> tuple[float, float]:
    if games < 1 or not 0 <= wins <= games or not 0 < confidence < 1:
        raise ValueError("require 0 <= wins <= games, games > 0, confidence in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    rate = wins / games
    scale = 1 + z * z / games
    center = (rate + z * z / (2 * games)) / scale
    radius = z * math.sqrt(rate * (1 - rate) / games + z * z / (4 * games * games)) / scale
    return max(0.0, center - radius), min(1.0, center + radius)


def evaluate_matches(agent: Policy, opponent: Policy, count: int = 100,
                     seed: int = 0, max_rounds: int = 1000) -> dict:
    """Independent full matches with alternating agent team assignment.

    Match seeds differ, so the Wilson interval treats matches as independent.
    These are not duplicate matches (which would require clustered intervals).
    A runaway match raises rather than silently counting a truncation as loss.
    """
    if count < 1:
        raise ValueError("match count and round bound must be positive")
    return summarize_matches(play_matches(agent, opponent, range(count), seed, max_rounds))


MATCH_COUNTERS = ("matches", "wins", "rounds", "bankers", "double_wins",
                  "opponent_doubles", "awarded", "net")


def play_matches(agent: Policy, opponent: Policy, indices: Iterable[int],
                 seed: int = 0, max_rounds: int = 1000) -> dict:
    """Raw counters for the given match indices; summable across workers.

    Match `m` uses seed `seed + m` and seats the agent on team `m % 2`, so a
    split of `range(count)` into chunks reproduces the serial run exactly.
    """
    if max_rounds < 1:
        raise ValueError("match count and round bound must be positive")
    engine = gd.Engine()
    matches = wins = rounds = bankers = double_wins = opponent_doubles = awarded = net = 0
    records = []
    for match in indices:
        matches += 1
        team = match % 2
        policies = (agent, opponent) if team == 0 else (opponent, agent)
        state = gd.MatchState()
        engine.new_match(state, seed + match)
        match_rounds = 0
        while state.winner < 0:
            if match_rounds >= max_rounds:
                raise RuntimeError(f"match {match} exceeded {max_rounds} rounds")
            score = play_round(engine, state, policies, seed + match * 1000003 + match_rounds)
            rounds += 1
            match_rounds += 1
            bankers += score.winning_team == team
            double_wins += score.double_win(team)
            opponent_doubles += score.double_win(1 - team)
            awarded += score.gain if score.winning_team == team else 0
            net += score.net_gain(team)
            if state.winner < 0:
                engine.begin_round(state)
        won = state.winner == team
        wins += won
        records.append({"seed": seed + match, "agent_team": team,
                        "winner": state.winner, "agent_won": won, "rounds": match_rounds})
    return {"matches": matches, "wins": wins, "rounds": rounds, "bankers": bankers,
            "double_wins": double_wins, "opponent_doubles": opponent_doubles,
            "awarded": awarded, "net": net, "records": records}


def summarize_matches(totals: dict) -> dict:
    """Match report from (possibly merged) play_matches counters."""
    count, wins, rounds = totals["matches"], totals["wins"], totals["rounds"]
    if count < 1:
        raise ValueError("match count and round bound must be positive")
    return {"matches": count, "wins": wins, "win_rate": wins / count,
            "wilson_95_ci": list(wilson_interval(wins, count)), "rounds": rounds,
            "banker_rate": totals["bankers"] / rounds, "double_win_rate": totals["double_wins"] / rounds,
            "opponent_double_win_rate": totals["opponent_doubles"] / rounds,
            "mean_finish_order_award_per_round": totals["awarded"] / rounds,
            "net_level_definition": "signed DMC return; an owned A Banker/Dweller result returns zero to both teams",
            "mean_net_levels_per_round": totals["net"] / rounds, "results": totals["records"]}


def evaluate_checkpoint_belief(policy: Policy, directory: Path, batch_size: int = 64,
                               max_rounds: int = 10000) -> dict:
    """Measure a checkpoint's existing hidden-hand head without fitting weights.

    The same deterministic match split as the architecture probe is used.
    This split cannot retroactively hold data out of checkpoint training: use
    independent evaluation logs for claims about unseen matches.
    """
    if not isinstance(policy, ModelPolicy):
        raise ValueError("belief metrics require a checkpoint policy")
    if batch_size < 1 or max_rounds < 1:
        raise ValueError("belief batch size and maximum rounds must be positive")
    from torch import nn
    from train.belief_probe import evaluate, examples, load_rounds

    class CheckpointBelief(nn.Module):
        def __init__(self, model: object) -> None:
            super().__init__()
            self.model = model

        def forward(self, obs, tokens, lengths, seat):
            state = self.model.state_tower(obs)
            return self.model.hidden_head(state).reshape(-1, 3, 54, 3)

    excluded, selected = load_rounds(directory, max_rounds)
    data = examples(selected)
    if not data:
        raise ValueError("belief evaluation split contains no decisions")
    metrics = evaluate(CheckpointBelief(policy.model), data, str(policy.device), batch_size)
    report = {
        "checkpoint": policy.name, "logs": str(directory),
        "split": "deterministic match-hash holdout: 20%, at least one match",
        "checkpoint_training_overlap": "not_verified",
        "split_note": "Use independent evaluation logs for unseen-data claims; training logs provide diagnostics only.",
        "evaluated_matches": len({record["group"] for record in selected}),
        "excluded_matches": len({record["group"] for record in excluded}),
        "evaluated_rounds": len(selected), "evaluated_decisions": len(data),
        **metrics,
    }
    provenance_path = directory / "provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        report["collection_provenance"] = provenance
        if (provenance.get("purpose") == "evaluation_only"
                and provenance.get("status") == "complete"
                and provenance.get("learner_updates") == 0
                and provenance.get("checkpoint_id") == getattr(policy, "checkpoint_id", None)
                and provenance.get("seed") != provenance.get("training_seed")):
            report["checkpoint_training_overlap"] = "independent_frozen_checkpoint_collection"
            report["split_note"] = "Collected from this frozen checkpoint with an independent seed and no learner updates."
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="greedy", help="random, greedy, or checkpoint path")
    parser.add_argument("--opponent", default="random", help="random, greedy, or checkpoint path")
    parser.add_argument("--deals", type=int, default=100)
    parser.add_argument("--matches", type=int, default=0, help="0 skips full matches")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--level", type=int, default=None, help="0=2 through 12=A; default samples all levels")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--probe-repeats", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads for checkpoint inference")
    parser.add_argument("--margin", type=float, default=0.0,
                        help="uniform near-best Q margin for both checkpoint policies; 0 uses argmax")
    parser.add_argument("--belief-logs", type=Path,
                        help="evaluate agent checkpoint's hidden-hand head on the match holdout from these logs")
    parser.add_argument("--belief-batch-size", type=int, default=64)
    parser.add_argument("--belief-max-rounds", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    from .batched import (add_eval_arguments, config_from_args, play_duplicate_batch,
                          play_matches_batch)
    from .duplicate import summarize_duplicates
    add_eval_arguments(parser)
    args = parser.parse_args(argv)
    eval_config = config_from_args(args)
    if args.deals < 1 or args.matches < 0 or args.bootstrap_samples < 1 or args.probe_repeats < 1:
        parser.error("require deals > 0, matches >= 0, bootstrap samples > 0 and probe repeats > 0")
    if not math.isfinite(args.margin) or args.margin < 0:
        parser.error("margin must be nonnegative and finite")
    if min(args.threads, args.belief_batch_size, args.belief_max_rounds) < 1:
        parser.error("threads, belief batch size and belief maximum rounds must be positive")
    if args.belief_logs and args.agent in ("random", "greedy"):
        parser.error("--belief-logs requires --agent to be a checkpoint")
    if args.agent not in ("random", "greedy") or args.opponent not in ("random", "greedy"):
        import torch
        torch.set_num_threads(args.threads)
    agent = load_policy(args.agent, args.device, args.margin)
    opponent = load_policy(args.opponent, args.device, args.margin)
    belief = (evaluate_checkpoint_belief(agent, args.belief_logs, args.belief_batch_size,
                                         args.belief_max_rounds) if args.belief_logs else None)
    deals = generate_deals(args.deals, args.seed, args.level)
    duplicate = (summarize_duplicates(play_duplicate_batch(
        deals, (agent, agent), (opponent, opponent), args.seed, eval_config),
        args.seed, args.bootstrap_samples) if eval_config.backend == "batched" else
        evaluate_duplicates(agent, opponent, deals, args.seed, args.bootstrap_samples))
    report = {
        "evaluation": eval_config.metadata(),
        "schema_version": 1, "protocol": "internal-house", "agent": agent.name,
        "opponent": opponent.name, "seed": args.seed,
        "agent_source": args.agent, "opponent_source": args.opponent,
        "checkpoint_sampling_margin": args.margin,
        "baseline_note": "random and greedy are local sanity baselines, not OpenGuanDan Rule One--Four",
        "duplicate": duplicate,
        "probes": evaluate_probes(agent, args.seed, args.probe_repeats),
    }
    if args.matches:
        report["match"] = (summarize_matches(play_matches_batch(
            agent, opponent, range(args.matches), args.seed, config=eval_config))
            if eval_config.backend == "batched" else
            evaluate_matches(agent, opponent, args.matches, args.seed))
    if belief is not None:
        report["belief"] = belief
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
        print(f"Evaluation written to {args.output}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
