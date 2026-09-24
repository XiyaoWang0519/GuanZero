"""Bounded residual-shuffle search for late play decisions (experimental C3 v0)."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
import random
import time
from typing import Sequence

import gd

from .policies import (GreedyPolicy, ModelPolicy, Policy, PrunedPolicy,
                       RandomPolicy, StyledPolicy)


@dataclass(frozen=True)
class SearchConfig:
    unseen_threshold: int = 12
    time_ms: float = 500.0
    max_actions: int = 4
    max_worlds: int = 4
    min_worlds: int = 2
    max_rollout_steps: int = 120
    prior_blueprint: float = 0.70
    kl_temperature: float = 1.0
    value_margin: float = 0.5

    def __post_init__(self) -> None:
        if not (0 <= self.unseen_threshold <= 108
                and math.isfinite(self.time_ms) and self.time_ms > 0
                and self.max_actions >= 2 and self.max_worlds >= self.min_worlds >= 1
                and self.max_rollout_steps >= 1 and 0 < self.prior_blueprint < 1
                and math.isfinite(self.prior_blueprint)
                and math.isfinite(self.kl_temperature) and self.kl_temperature > 0
                and math.isfinite(self.value_margin) and self.value_margin >= 0):
            raise ValueError("invalid search configuration")


class SearchPolicy:
    """Evaluate root candidates over sampled legal hidden worlds.

    The C++ sampler is the only code that may inspect the live state before
    rollouts. It excludes true opponent card identities. Every rollout seat
    acts through the unchanged blueprint's own-seat observation path.
    """

    def __init__(self, blueprint: Policy, config: SearchConfig | None = None) -> None:
        # The root call receives a full engine state from scalar evaluation.
        # Accept only the audited built-ins, whose choices use their acting
        # seat's hand/observation and public counts. An arbitrary Policy could
        # inspect other live hands before determinization.
        if type(blueprint) not in (GreedyPolicy, RandomPolicy, StyledPolicy,
                                   ModelPolicy, PrunedPolicy):
            raise TypeError("search blueprint must be an audited built-in policy")
        self.blueprint = blueprint
        self.config = config or SearchConfig()
        self.name = f"search-residual:{blueprint.name}"
        self.checkpoint_id = getattr(blueprint, "checkpoint_id", None)
        self.calls = self.triggered = self.completed = self.fallbacks = self.overrides = 0
        self.elapsed_ms: list[float] = []
        self.worlds_completed: list[int] = []

    def select(self, engine: gd.Engine, state: gd.MatchState,
               actions: Sequence[gd.Action], rng: random.Random) -> int:
        self.calls += 1
        baseline = self.blueprint.select(engine, state, actions, rng)
        if state.phase != gd.Phase.Play or len(actions) < 2:
            return baseline
        # Played cards and own cards are public/private-to-actor respectively.
        seat = state.to_move
        unseen = 108 - len(state.hand(seat)) - sum(len(state.played(s)) for s in range(4))
        if unseen > self.config.unseen_threshold:
            return baseline

        self.triggered += 1
        start = time.perf_counter()
        deadline = start + self.config.time_ms / 1000.0
        others = [i for i in range(len(actions)) if i != baseline]
        candidates = [baseline] + rng.sample(others, min(len(others), self.config.max_actions - 1))
        totals = [0.0] * len(candidates)
        worlds = 0
        for _ in range(self.config.max_worlds):
            if time.perf_counter() >= deadline:
                break
            # This seed comes solely from the evaluation RNG, never state.hash.
            sample = state.determinize_uniform(seat, rng.getrandbits(64))
            values: list[float] = []
            for action_index in candidates:
                if time.perf_counter() >= deadline:
                    break
                branch = gd.MatchState.deserialize(sample.serialize())
                engine.apply(branch, actions[action_index])
                steps = 0
                while branch.phase == gd.Phase.Play and steps < self.config.max_rollout_steps:
                    if time.perf_counter() >= deadline:
                        break
                    legal = engine.legal_actions(branch)
                    # Blueprint implementations consume observation(branch.to_move)
                    # and encode actions for that seat only. Its own RNG is local.
                    choice = self.blueprint.select(engine, branch, legal, rng)
                    engine.apply(branch, legal[choice])
                    steps += 1
                if branch.phase != gd.Phase.RoundEnd:
                    break
                result = engine.end_round(branch)
                values.append(float(result.seat_return[seat]))
            if len(values) != len(candidates):
                break  # partial worlds cannot bias an action's mean
            for i, value in enumerate(values):
                totals[i] += value
            worlds += 1

        elapsed = (time.perf_counter() - start) * 1000
        self.elapsed_ms.append(elapsed)
        self.worlds_completed.append(worlds)
        if worlds < self.config.min_worlds:
            self.fallbacks += 1
            return baseline
        self.completed += 1
        means = [value / worlds for value in totals]
        prior_other = (1 - self.config.prior_blueprint) / (len(candidates) - 1)
        log_post = [math.log(self.config.prior_blueprint) + means[0] / self.config.kl_temperature]
        log_post += [math.log(prior_other) + value / self.config.kl_temperature
                     for value in means[1:]]
        winner = max(range(len(candidates)), key=log_post.__getitem__)
        if winner and means[winner] >= means[0] + self.config.value_margin:
            self.overrides += 1
            return candidates[winner]
        return baseline


def main() -> None:
    """Small scalar duplicate comparison; never starts training or a GPU run."""
    from dataclasses import asdict
    from .duplicate import evaluate_duplicates, generate_deals
    from .policies import load_policy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--deals", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--time-ms", type=float, default=500.0)
    parser.add_argument("--unseen-threshold", type=int, default=12)
    parser.add_argument("--max-actions", type=int, default=4)
    parser.add_argument("--max-worlds", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.deals < 1 or args.threads < 1:
        parser.error("deals and threads must be positive")
    import torch

    torch.set_num_threads(args.threads)
    config = SearchConfig(unseen_threshold=args.unseen_threshold, time_ms=args.time_ms,
                          max_actions=args.max_actions, max_worlds=args.max_worlds)
    blueprint = load_policy(args.checkpoint)
    policy = SearchPolicy(blueprint, config)
    deals = generate_deals(args.deals, args.seed)
    start = time.perf_counter()
    score = evaluate_duplicates(policy, blueprint, deals, args.seed, bootstrap_samples=500)
    latencies = sorted(policy.elapsed_ms)

    def percentile(p: float) -> float | None:
        if not latencies:
            return None
        return latencies[math.ceil(p * len(latencies)) - 1]

    report = {
        "status": "bounded_uniform_search_probe_not_G9",
        "checkpoint_id": policy.checkpoint_id,
        "config": asdict(config),
        "deals": args.deals,
        "seed": args.seed,
        "duplicate": score,
        "wall_seconds": time.perf_counter() - start,
        "search": {
            "calls": policy.calls,
            "triggered": policy.triggered,
            "completed": policy.completed,
            "fallbacks": policy.fallbacks,
            "overrides": policy.overrides,
            "complete_worlds": sum(policy.worlds_completed),
            "extra_search_latency_ms": {
                "p50": percentile(0.5), "p95": percentile(0.95),
                "max": max(latencies) if latencies else None,
            },
        },
    }
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        from pathlib import Path

        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)
        print(f"Search probe written to {path}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
