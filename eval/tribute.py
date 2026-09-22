"""Measure Stage A2 tribute payoff with a frozen play model on paired deals.

The same explicit hands and previous finishing order are played twice, swapping
learned-tribute and heuristic-tribute teams. This synthetic uniform-deal test is
independent of training matches; it is not an external benchmark score.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
from typing import Iterable

import gd
import numpy as np

from .duplicate import DuplicateScore, RoundScore, bootstrap_interval
from .policies import ModelPolicy, choose_action, load_policy


def generate_tribute_deals(count: int, seed: int) -> list[gd.DealSpec]:
    """Sample complete double decks, levels and previous finishing permutations.

    Anti-tribute deals remain in the sample and are explicitly counted. Rejecting
    them would change the estimand from all tribute-eligible rounds.
    """
    if count < 1:
        raise ValueError("deal count must be positive")
    rng = random.Random(seed)
    deals = []
    for _ in range(count):
        deck = [card for card in range(54) for _ in range(2)]
        rng.shuffle(deck)
        previous = list(range(4))
        rng.shuffle(previous)
        deal = gd.DealSpec()
        deal.hands = [sorted(deck[seat * 27:(seat + 1) * 27]) for seat in range(4)]
        deal.level = rng.randrange(13)
        deal.team_levels = [deal.level, deal.level]
        # Match ownership is intentionally absent, like ordinary duplicate
        # rounds, to avoid the asymmetric owner-at-A zero-return exception.
        deal.owner = -1
        deal.leader = -1
        deal.prev_order = previous
        deals.append(deal)
    return deals


def assert_same_play_model(agent: ModelPolicy, opponent: ModelPolicy) -> None:
    """Reject any change outside the two permitted tribute fusion heads."""
    import torch

    if not isinstance(agent, ModelPolicy) or not isinstance(opponent, ModelPolicy):
        raise ValueError("tribute evaluation requires two checkpoint model policies")
    if agent.heuristic_tribute or not opponent.heuristic_tribute:
        raise ValueError("require learned tribute agent and heuristic tribute opponent")
    if agent.margin or opponent.margin:
        raise ValueError("tribute evaluation requires deterministic argmax play")
    if agent.action_mode != opponent.action_mode:
        raise ValueError("checkpoint action modes differ")
    base_id = getattr(agent, "base_checkpoint_id", None)
    if base_id is not None and base_id != getattr(opponent, "checkpoint_id", None):
        raise ValueError("opponent does not match the Stage A2 base checkpoint identity")
    left, right = agent.model.state_dict(), opponent.model.state_dict()
    if left.keys() != right.keys():
        raise ValueError("checkpoint model tensor keys differ")
    for name in left:
        if name.startswith(("phase_heads.1.", "phase_heads.2.")):
            continue
        a, b = left[name].detach().cpu(), right[name].detach().cpu()
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise ValueError(f"play model is not frozen: {name}")


def _phase_counts() -> dict:
    return {who: {phase: {"decisions": 0, "multiple_candidates": 0,
                         "single_candidate": 0, "different_from_heuristic": 0}
                  for phase in ("tribute", "back_tribute")}
            for who in ("agent", "opponent")}


def play_tribute_leg(engine: gd.Engine, state: gd.MatchState,
                     agent: ModelPolicy, opponent: ModelPolicy, agent_team: int,
                     seed: int, max_decisions: int = 2000) -> tuple[RoundScore, dict]:
    """Play one leg, measuring only choices actually made on its trajectory."""
    rngs = [random.Random(seed + seat * 0x9E3779B9) for seat in range(4)]
    counts = _phase_counts()
    choices = []
    decisions = 0
    while state.phase != gd.Phase.RoundEnd:
        if state.phase == gd.Phase.MatchEnd:
            raise ValueError("cannot play an already completed match")
        if decisions >= max_decisions:
            raise RuntimeError(f"round exceeded {max_decisions} decisions")
        seat = state.to_move
        who = "agent" if seat % 2 == agent_team else "opponent"
        policy = agent if who == "agent" else opponent
        if state.phase in (gd.Phase.Tribute, gd.Phase.BackTribute):
            phase = "tribute" if state.phase == gd.Phase.Tribute else "back_tribute"
            actions = engine.legal_actions(state)
            heuristic = actions[engine.greedy(state)]
            action = choose_action(policy, engine, state, rngs[seat])
            changed = action.cards != heuristic.cards
            count = counts[who][phase]
            count["decisions"] += 1
            count["multiple_candidates" if len(actions) > 1 else "single_candidate"] += 1
            count["different_from_heuristic"] += int(changed)
            choices.append({"phase": phase, "seat": seat, "policy": who,
                            "candidates": len(actions), "selected_card": action.cards[0],
                            "heuristic_card": heuristic.cards[0],
                            "different_from_heuristic": changed})
        else:
            action = choose_action(policy, engine, state, rngs[seat])
        engine.apply(state, action)
        decisions += 1
    result = engine.end_round(state)
    score = RoundScore(tuple(result.order), result.winning_team, result.gain,
                       tuple(result.seat_return), decisions)
    return score, {"phase_counts": counts, "choices": choices}


def _score_summary(values: list[float], seed: int, samples: int) -> dict:
    return {"deals": len(values),
            "mean_net_levels_per_round": float(np.mean(values)) if values else None,
            "bootstrap_95_ci": list(bootstrap_interval(values, seed, samples)) if values else None}


def evaluate_tribute_duplicates(agent: ModelPolicy, opponent: ModelPolicy,
                                deals: Iterable[gd.DealSpec], seed: int = 0,
                                bootstrap_samples: int = 2000) -> dict:
    assert_same_play_model(agent, opponent)
    if bootstrap_samples < 1:
        raise ValueError("bootstrap count must be positive")
    for policy in (agent, opponent):
        if seed in (getattr(policy, "training_seed", None),
                    getattr(policy, "base_training_seed", None),
                    getattr(policy, "collection_seed", None)):
            raise ValueError("evaluation seed must differ from all training seeds and collection seeds")
    action_config = gd.ActionConfig.full() if agent.action_mode == "full" else gd.ActionConfig()
    engine = gd.Engine(gd.RuleConfig.house(), action_config)
    counts = _phase_counts()
    records = []
    groups = {name: [] for name in ("single_opponent", "single_partner", "double",
                                  "anti_tribute", "with_actual_choices", "without_actual_choices")}
    for index, deal in enumerate(deals):
        order = list(deal.prev_order)
        if deal.leader != -1 or sorted(order) != list(range(4)):
            raise ValueError("tribute deals require leader=-1 and a previous finishing permutation")
        if deal.owner != -1:
            raise ValueError("independent tribute duplicate deals require owner=-1")
        category = ("double" if order[0] % 2 == order[1] % 2 else
                    "single_partner" if order[0] % 2 == order[3] % 2 else "single_opponent")
        legs, diagnostics = [], []
        anti_tribute = None
        for agent_team in (0, 1):
            state = gd.MatchState()
            engine.set_deal(state, deal)
            is_anti = state.phase == gd.Phase.Play
            if anti_tribute is not None and is_anti != anti_tribute:
                raise RuntimeError("duplicate legs disagree about anti-tribute")
            anti_tribute = is_anti
            score, leg = play_tribute_leg(engine, state, agent, opponent, agent_team, seed + index)
            legs.append(score)
            diagnostics.append(leg)
            for who in counts:
                for phase in counts[who]:
                    for key in counts[who][phase]:
                        counts[who][phase][key] += leg["phase_counts"][who][phase][key]
        paired = DuplicateScore(*legs)
        actual_choices = any(choice["candidates"] > 1
                             for leg in diagnostics for choice in leg["choices"])
        changed_choices = any(choice["different_from_heuristic"]
                              for leg in diagnostics for choice in leg["choices"])
        if anti_tribute and paired.pair_difference != 0:
            raise RuntimeError("identical frozen play must cancel on anti-tribute deals")
        groups[category].append(paired.levels_per_round)
        if anti_tribute:
            groups["anti_tribute"].append(paired.levels_per_round)
        groups["with_actual_choices" if actual_choices else "without_actual_choices"].append(
            paired.levels_per_round)
        records.append({"deal_index": index, "policy_seed": seed + index,
                        "hands": deal.hands, "level": deal.level, "prev_order": order,
                        "category": category, "anti_tribute": anti_tribute,
                        "has_actual_choices": actual_choices, "has_changed_choices": changed_choices,
                        "pair_difference": paired.pair_difference,
                        "net_levels_per_round": paired.levels_per_round,
                        "first": asdict(paired.first), "swapped": asdict(paired.swapped),
                        "diagnostics": diagnostics})
    if not records:
        raise ValueError("at least one tribute deal is required")
    values = [record["net_levels_per_round"] for record in records]
    summary = _score_summary(values, seed, bootstrap_samples)
    eligible = len(groups["with_actual_choices"])
    changed = sum(record["has_changed_choices"] for record in records)
    supported = eligible >= 30 and changed > 0 and summary["bootstrap_95_ci"][0] > 0
    if not eligible:
        conclusion = "no_multi_candidate_tribute_decisions"
    elif eligible < 30:
        conclusion = "insufficient_independent_deals"
    elif supported:
        conclusion = "positive_payoff_on_this_evaluation_distribution"
    else:
        conclusion = "positive_payoff_not_established"
    return {**summary, "rounds": len(records) * 2,
            "score_definition": "(team-0 net return in learned-first leg minus team-0 net return in swapped leg) / 2",
            "ci_sampling_unit": "whole paired deal; 95% percentile bootstrap",
            "anti_tribute_deals": len(groups["anti_tribute"]),
            "deals_with_actual_choices": eligible, "deals_with_changed_choices": changed,
            "phase_counts": counts,
            "breakdown": {name: _score_summary(data, seed, bootstrap_samples)
                          for name, data in groups.items()},
            "pair_scores": values, "results": records,
            "payoff_supported": supported, "conclusion": conclusion,
            "deployment_recommendation": "candidate_for_confirmation" if supported else "retain_heuristic",
            "inference_note": "Tiny samples are diagnostics only; anti-tribute and single-candidate decisions do not demonstrate learned choice quality."}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, help="learned Stage A2 checkpoint")
    parser.add_argument("--opponent", required=True, help="original frozen DMC checkpoint")
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if min(args.deals, args.threads, args.bootstrap_samples) < 1:
        parser.error("deals, threads and bootstrap samples must be positive")
    import torch
    torch.set_num_threads(args.threads)
    agent, opponent = load_policy(args.agent, args.device), load_policy(args.opponent, args.device)
    if not isinstance(agent, ModelPolicy) or not isinstance(opponent, ModelPolicy):
        parser.error("agent and opponent must be checkpoint paths")
    if agent.stage != "a2" or opponent.stage != "dmc":
        parser.error("require Stage A2 agent and original DMC opponent")
    if agent.base_checkpoint_id != opponent.checkpoint_id:
        parser.error("opponent does not match the Stage A2 base checkpoint identity")
    result = evaluate_tribute_duplicates(agent, opponent,
        generate_tribute_deals(args.deals, args.seed), args.seed, args.bootstrap_samples)
    report = {"schema_version": 1, "protocol": "internal-house-tribute-duplicate",
              "agent": agent.name, "opponent": opponent.name,
              "agent_source": args.agent, "opponent_source": args.opponent,
              "agent_checkpoint_id": agent.checkpoint_id,
              "base_checkpoint_id": opponent.checkpoint_id,
              "agent_training_seed": agent.training_seed,
              "agent_collection_seed": agent.collection_seed,
              "base_training_seed": opponent.training_seed,
              "seed": args.seed, "action_mode": agent.action_mode,
              "play_model_exactly_equal": True,
              "distribution": "uniform shuffled double deck, uniform level and previous finishing permutation; owner=-1; anti-tribute retained",
              "scope": "Independent synthetic rounds; not the natural full-match distribution or OpenGuanDan benchmark.",
              "duplicate": result}
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Tribute evaluation written to {args.output}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
