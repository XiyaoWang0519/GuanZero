"""Bounded residual-shuffle search for late play decisions (experimental C3 v0)."""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
import random
import time
import copy
from typing import Sequence

import gd

from .policies import (GreedyPolicy, ModelPolicy, Policy, PrunedPolicy,
                       RandomPolicy, StyledPolicy)
from .history_policy import (HistoryPolicy, apply_and_observe, explicit_passes,
                             resolve_forced_passes)


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
    rollout_batch_size: int = 0  # 0 retains the scalar reference
    selection: str = "prior"    # "mean" ranks sampled returns without the artificial prior
    # When to search. "unseen": the other three hands hold <= unseen_threshold cards
    # in total. "hand": some seat (the actor included) holds <= hand_threshold cards,
    # the public count a player must announce at ten. "unsure" (history blueprints):
    # anywhere in the round, when the blueprint gives its own choice a probability
    # below confidence_threshold. "hand_or_unsure": either of the two.
    trigger: str = "unseen"
    hand_threshold: int = 10
    confidence_threshold: float = 0.6
    # History blueprints only: root candidates are the blueprint's choice plus its
    # next most probable actions, top_actions in all (0: the max_actions rule).
    top_actions: int = 0
    # Hidden-hand worlds: "uniform" shuffles the unseen copies (determinize_uniform);
    # "belief" (history blueprints with a belief head) deals each unseen copy to a
    # seat in proportion to the actor's per-card seat probabilities, among the
    # seats with room (MatchState.determinize_weighted). belief_floor mixes a
    # uniform share into those probabilities so no seat is ever ruled out.
    sampler: str = "uniform"
    belief_floor: float = 0.0

    def __post_init__(self) -> None:
        if not (0 <= self.unseen_threshold <= 108
                and math.isfinite(self.time_ms) and self.time_ms > 0
                and (self.max_actions == 0 or self.max_actions >= 2)
                and self.max_worlds >= self.min_worlds >= 1
                and self.max_rollout_steps >= 0 and 0 < self.prior_blueprint < 1
                and math.isfinite(self.prior_blueprint)
                and math.isfinite(self.kl_temperature) and self.kl_temperature > 0
                and math.isfinite(self.value_margin) and self.value_margin >= 0
                and self.rollout_batch_size >= 0 and self.selection in ("prior", "mean")
                and self.trigger in ("unseen", "hand", "unsure", "hand_or_unsure")
                and 0 <= self.hand_threshold <= 27 and 0 < self.confidence_threshold <= 1
                and (self.top_actions == 0 or self.top_actions >= 2)
                and self.sampler in ("uniform", "belief")
                and math.isfinite(self.belief_floor) and 0 <= self.belief_floor <= 1):
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
                                   ModelPolicy, PrunedPolicy, HistoryPolicy):
            raise TypeError("search blueprint must be an audited built-in policy")
        self.blueprint = blueprint
        self.config = config or SearchConfig()
        self.name = f"search-residual:{blueprint.name}"
        self.checkpoint_id = getattr(blueprint, "checkpoint_id", None)
        self.needs_history = isinstance(blueprint, HistoryPolicy)
        if self.config.sampler == "belief" and not (
                self.needs_history and "belief" in getattr(blueprint.actor, "aux_head_names", ())):
            raise TypeError("the belief sampler needs a history blueprint with a belief head")
        self.action_mode = getattr(blueprint, "action_mode", "canonical")
        self._world_weights: list[float] | None = None
        self.calls = self.triggered = self.completed = self.fallbacks = self.overrides = 0
        self.elapsed_ms: list[float] = []
        self.worlds_completed: list[int] = []
        self.decisions: list[dict] = []

    def start_match(self, match_id: int = -1) -> None:
        if self.needs_history:
            self.blueprint.start_match(match_id)

    def observe(self, event) -> None:
        if self.needs_history:
            self.blueprint.observe(event)

    @property
    def events_seen(self) -> int:
        return self.blueprint.events_seen if self.needs_history else 0

    def _rollout(self, engine, branch, action, seat, rng, deadline):
        # Share frozen weights only. Each candidate/world owns its public
        # prefix, so speculative events cannot enter the live match or siblings.
        policy = self.blueprint
        listeners = []
        if self.needs_history:
            policy = copy.copy(policy)
            policy.stream = copy.deepcopy(policy.stream)
            listeners = [policy]
        with explicit_passes(engine, listeners):
            if listeners:
                apply_and_observe(engine, branch, action, listeners)
            else:
                engine.apply(branch, action)
            steps = 0
            while branch.phase == gd.Phase.Play and (not self.config.max_rollout_steps
                                                     or steps < self.config.max_rollout_steps):
                if time.perf_counter() >= deadline:
                    return None
                if listeners:
                    resolve_forced_passes(engine, branch, listeners)
                    if branch.phase == gd.Phase.RoundEnd:
                        break
                legal = engine.legal_actions(branch)
                choice = policy.select(engine, branch, legal, rng)
                if listeners:
                    apply_and_observe(engine, branch, legal[choice], listeners)
                else:
                    engine.apply(branch, legal[choice])
                steps += 1
            if branch.phase != gd.Phase.RoundEnd or time.perf_counter() >= deadline:
                return None
            return float(engine.end_round(branch).seat_return[seat])

    def _root_log_probs(self, engine, state, actions):
        """The history blueprint's log-probability of every root candidate; with the
        belief sampler, also this decision's per-card seat weights (one actor pass)."""
        import torch
        from train.history_model import segment_log_softmax
        actor = self.blueprint.actor
        with torch.inference_mode():
            inputs = self.blueprint.decision_inputs(engine, state, actions)
            decision = actor.decision_states(None, inputs)
            logits = actor.candidate_logits(decision, inputs.cand, inputs.offsets, rows=inputs.rows)
            log_probs = segment_log_softmax(logits, inputs.rows, inputs.decisions)
            if self.config.sampler == "belief":
                # [54, 3] per-card probabilities over relative seats +1, +2, +3.
                probs = torch.softmax(actor.belief_logits(decision)[0].float(), dim=-1)
                floor = self.config.belief_floor
                probs = (1.0 - floor) * probs + floor / 3.0
                # determinize_weighted wants relative-seat-major: [(r - 1) * 54 + card].
                self._world_weights = probs.transpose(0, 1).reshape(-1).cpu().tolist()
        return log_probs.float().cpu().numpy()

    def sample_world(self, state: gd.MatchState, seat: int, rng: random.Random) -> gd.MatchState:
        """One legal hidden-hand world for the root decision: uniform, or weighted by
        the belief head's seat probabilities computed for this decision."""
        seed = rng.getrandbits(64)
        if self.config.sampler == "belief":
            if self._world_weights is None:
                raise RuntimeError("belief weights are computed at the root before sampling")
            return state.determinize_weighted(seat, seed, self._world_weights)
        return state.determinize_uniform(seat, seed)

    def select(self, engine: gd.Engine, state: gd.MatchState,
               actions: Sequence[gd.Action], rng: random.Random) -> int:
        self.calls += 1
        baseline = self.blueprint.select(engine, state, actions, rng)
        if state.phase != gd.Phase.Play or len(actions) < 2:
            return baseline
        # Played cards and own cards are public/private-to-actor respectively.
        seat = state.to_move
        unseen = 108 - len(state.hand(seat)) - sum(len(state.played(s)) for s in range(4))
        # Card counts are public (every play token carries the actor's count).
        counts = [len(state.hand(s)) for s in range(4)]
        log_probs = None
        self._world_weights = None
        if self.config.trigger in ("unsure", "hand_or_unsure"):
            if not self.needs_history:
                raise TypeError("the unsure trigger needs a history blueprint")
            log_probs = self._root_log_probs(engine, state, actions)
            sure = math.exp(log_probs[baseline]) >= self.config.confidence_threshold
            late = (self.config.trigger == "hand_or_unsure"
                    and min(counts) <= self.config.hand_threshold)
            if sure and not late:
                return baseline
        elif self.config.trigger == "hand":
            if min(counts) > self.config.hand_threshold:
                return baseline
        elif unseen > self.config.unseen_threshold:
            return baseline

        self.triggered += 1
        start = time.perf_counter()
        deadline = start + self.config.time_ms / 1000.0
        others = [i for i in range(len(actions)) if i != baseline]
        baseline_probability = None
        if self.needs_history:
            # The blueprint's own distribution over the root candidates: recorded for
            # every search, the ranking behind ``top_actions`` and, with the belief
            # sampler, this decision's hidden-hand weights.
            if log_probs is None:
                log_probs = self._root_log_probs(engine, state, actions)
            baseline_probability = float(math.exp(log_probs[baseline]))
            if self.config.top_actions:
                others = sorted(others, key=lambda i: -log_probs[i])[:self.config.top_actions - 1]
        if self.config.top_actions and self.needs_history:
            candidates = [baseline] + others
        else:
            candidates = [baseline] + (others if self.config.max_actions == 0 else
                                      rng.sample(others, min(len(others), self.config.max_actions - 1)))
        totals = [0.0] * len(candidates)
        worlds = 0
        if self.config.rollout_batch_size:
            from .search_rollout import batched_world_values
            totals, worlds = batched_world_values(self, engine, state, actions, candidates,
                                                   seat, rng, deadline)
        for _ in range(0 if self.config.rollout_batch_size else self.config.max_worlds):
            if time.perf_counter() >= deadline:
                break
            # The seed comes solely from the evaluation RNG, never state.hash.
            sample = self.sample_world(state, seat, rng)
            values: list[float] = []
            for action_index in candidates:
                if time.perf_counter() >= deadline:
                    break
                branch = gd.MatchState.deserialize(sample.serialize())
                value = self._rollout(engine, branch, actions[action_index], seat, rng, deadline)
                if value is None:
                    break
                values.append(value)
            if len(values) != len(candidates):
                break  # partial worlds cannot bias an action's mean
            for i, value in enumerate(values):
                totals[i] += value
            worlds += 1

        elapsed = (time.perf_counter() - start) * 1000
        self.elapsed_ms.append(elapsed)
        self.worlds_completed.append(worlds)
        record = {"unseen": unseen, "own_cards": counts[seat], "min_cards": min(counts),
                  "sampler": self.config.sampler,
                  "baseline_probability": baseline_probability,
                  "legal_actions": len(actions), "candidates": len(candidates),
                  "worlds": worlds, "elapsed_ms": elapsed, "baseline": baseline,
                  "choice": baseline, "means": ([v / worlds for v in totals] if worlds else []),
                  "budget_exhausted": time.perf_counter() >= deadline}
        self.decisions.append(record)
        if worlds < self.config.min_worlds:
            self.fallbacks += 1
            return baseline
        self.completed += 1
        means = [value / worlds for value in totals]
        prior_other = (1 - self.config.prior_blueprint) / (len(candidates) - 1)
        log_post = [math.log(self.config.prior_blueprint) + means[0] / self.config.kl_temperature]
        log_post += [math.log(prior_other) + value / self.config.kl_temperature
                     for value in means[1:]]
        scores = means if self.config.selection == "mean" else log_post
        winner = max(range(len(candidates)), key=scores.__getitem__)
        if winner and means[winner] >= means[0] + self.config.value_margin:
            self.overrides += 1
            record["choice"] = candidates[winner]
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
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--min-worlds", type=int, default=2)
    parser.add_argument("--max-rollout-steps", type=int, default=120)
    parser.add_argument("--rollout-batch-size", type=int, default=0)
    parser.add_argument("--selection", choices=("prior", "mean"), default="prior")
    parser.add_argument("--value-margin", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.deals < 1 or args.threads < 1:
        parser.error("deals and threads must be positive")
    import torch

    torch.set_num_threads(args.threads)
    config = SearchConfig(unseen_threshold=args.unseen_threshold, time_ms=args.time_ms,
                          max_actions=args.max_actions, max_worlds=args.max_worlds,
                          min_worlds=args.min_worlds, max_rollout_steps=args.max_rollout_steps,
                          rollout_batch_size=args.rollout_batch_size, selection=args.selection,
                          value_margin=args.value_margin)
    blueprint = load_policy(args.checkpoint, device=args.device)
    policy = SearchPolicy(blueprint, config)
    deals = generate_deals(args.deals, args.seed)
    start = time.perf_counter()
    # History-bearing sides must never share the same live stream object.
    opponent = load_policy(args.checkpoint, device=args.device)
    score = evaluate_duplicates(policy, opponent, deals, args.seed, bootstrap_samples=500)
    latencies = sorted(policy.elapsed_ms)

    def percentile(p: float) -> float | None:
        if not latencies:
            return None
        return latencies[math.ceil(p * len(latencies)) - 1]

    report = {
        "status": "bounded_uniform_search_probe_not_G9",
        "checkpoint_id": policy.checkpoint_id,
        "config": asdict(config),
        "device": args.device,
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
            "decisions": policy.decisions,
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
