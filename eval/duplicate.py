"""Fixed-hand, fixed-leader duplicate rounds and paired bootstrap statistics."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Iterable, Sequence

import gd
import numpy as np

from .history_policy import (apply_and_observe, explicit_passes, history_listeners,
                             resolve_forced_passes)
from .policies import Policy, choose_action


@dataclass(frozen=True)
class RoundScore:
    order: tuple[int, ...]
    winning_team: int
    gain: int
    seat_return: tuple[int, ...]
    decisions: int

    def net_gain(self, team: int) -> int:
        """Objective-aligned return, including the owner-at-A zero-return case."""
        if team not in (0, 1):
            raise ValueError("team must be 0 or 1")
        return self.seat_return[team]

    def double_win(self, team: int) -> bool:
        return self.order[0] % 2 == team and self.order[1] % 2 == team


@dataclass(frozen=True)
class DuplicateScore:
    first: RoundScore
    swapped: RoundScore

    @property
    def pair_difference(self) -> int:
        # Team 0 keeps exactly the same hand allocation in both legs. Only
        # policy assignment changes. Positive values favour agent A.
        return self.first.net_gain(0) - self.swapped.net_gain(0)

    @property
    def levels_per_round(self) -> float:
        return self.pair_difference / 2


def generate_deals(count: int, seed: int = 0, level: int | None = None) -> list[gd.DealSpec]:
    """Generate a reproducible evaluation set, independently of policy RNGs.

    Each double-deck card appears exactly twice. Levels and leader are sampled
    uniformly unless a level is supplied. No tribute history is assumed.
    """
    if count < 1 or (level is not None and not 0 <= level <= 12):
        raise ValueError("positive deal count and level in [0, 12] required")
    rng = random.Random(seed)
    deals = []
    for _ in range(count):
        deck = [card for card in range(54) for _ in range(2)]
        rng.shuffle(deck)
        deal = gd.DealSpec()
        deal.hands = [sorted(deck[seat * 27:(seat + 1) * 27]) for seat in range(4)]
        deal.level = rng.randrange(13) if level is None else level
        deal.team_levels = [deal.level, deal.level]
        # These independent rounds have no owner, so the A-owner exception
        # does not asymmetrically zero one team's training return.
        deal.owner = -1
        deal.leader = rng.randrange(4)
        deals.append(deal)
    return deals


def play_round(engine: gd.Engine, state: gd.MatchState,
               policies: tuple[Policy, Policy], seed: int = 0,
               max_decisions: int = 2000) -> RoundScore:
    """Play the current round and close it exactly once, updating match state.

    `policies[t]` plays both seats of team `t`.
    """
    return play_round_seats(engine, state, (policies[0], policies[1], policies[0], policies[1]),
                            seed, max_decisions)


def play_round_seats(engine: gd.Engine, state: gd.MatchState,
                     policies: Sequence[Policy], seed: int = 0,
                     max_decisions: int = 2000) -> RoundScore:
    """Like `play_round`, but `policies[seat]` plays each of the four seats.

    Seat RNGs are the same as in `play_round`, so seating one policy at both
    seats of a team replays `play_round` exactly.

    History policies (``needs_history``) receive every applied action of every
    seat, engine-resolved passes included, through ``observe``; the engine's
    auto-pass is off for such a round so those passes become events. The
    caller opens the match with ``start_match()``: every leg of a duplicate
    is one, and a full match keeps its stream across rounds.
    """
    if len(policies) != 4:
        raise ValueError("one policy per seat required")
    rngs = [random.Random(seed + seat * 0x9E3779B9) for seat in range(4)]
    listeners = history_listeners(policies)
    decisions = 0
    with explicit_passes(engine, listeners):
        while state.phase != gd.Phase.RoundEnd:
            if state.phase == gd.Phase.MatchEnd:
                raise ValueError("cannot play an already completed match")
            if listeners:
                resolve_forced_passes(engine, state, listeners)
                if state.phase == gd.Phase.RoundEnd:
                    break
            if decisions >= max_decisions:
                raise RuntimeError(f"round exceeded {max_decisions} decisions")
            seat = state.to_move
            action = choose_action(policies[seat], engine, state, rngs[seat])
            if listeners:
                apply_and_observe(engine, state, action, listeners)
            else:
                engine.apply(state, action)
            decisions += 1
    result = engine.end_round(state)
    return RoundScore(tuple(result.order), result.winning_team, result.gain,
                      tuple(result.seat_return), decisions)


def play_duplicate(deal: gd.DealSpec, agent: Policy, opponent: Policy,
                   seed: int = 0, rules: gd.RuleConfig | None = None) -> DuplicateScore:
    engine = gd.Engine(rules or gd.RuleConfig.house())
    scores = []
    for policies in ((agent, opponent), (opponent, agent)):
        state = gd.MatchState()
        engine.set_deal(state, deal)
        for listener in history_listeners(policies):
            listener.start_match()   # each leg is an independent single-round match
        scores.append(play_round(engine, state, policies, seed))
    return DuplicateScore(*scores)


def play_duplicate_teams(deal: gd.DealSpec, team: tuple[Policy, Policy],
                         opponents: tuple[Policy, Policy], seed: int = 0,
                         rules: gd.RuleConfig | None = None) -> DuplicateScore:
    """Duplicate deal with per-seat teams.

    A team `(a, b)` sits at seats `(s, s + 2)`: `s = 0` in the first leg and
    `s = 1` in the swapped leg, so `a` always holds the lower seat of its team
    and `b` is its partner two seats on. The opponents likewise.
    """
    engine = gd.Engine(rules or gd.RuleConfig.house())
    scores = []
    for seats in ((team[0], opponents[0], team[1], opponents[1]),
                  (opponents[0], team[0], opponents[1], team[1])):
        state = gd.MatchState()
        engine.set_deal(state, deal)
        for listener in history_listeners(seats):
            listener.start_match()
        scores.append(play_round_seats(engine, state, seats, seed))
    return DuplicateScore(*scores)


def bootstrap_interval(values: Sequence[float], seed: int = 0,
                       samples: int = 2000, confidence: float = 0.95) -> tuple[float, float]:
    """Percentile CI resampling whole paired deals, never individual legs."""
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or not len(data) or not np.isfinite(data).all():
        raise ValueError("bootstrap requires a nonempty finite one-dimensional sample")
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("positive bootstrap count and confidence in (0, 1) required")
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    # Limit scratch memory for the normal 10,000-deal evaluation set.
    block = max(1, min(256, 1_000_000 // len(data)))
    for start in range(0, samples, block):
        stop = min(start + block, samples)
        indices = rng.integers(len(data), size=(stop - start, len(data)))
        means[start:stop] = data[indices].mean(axis=1)
    tail = (1 - confidence) / 2
    lo, hi = np.quantile(means, [tail, 1 - tail])
    return float(lo), float(hi)


def evaluate_duplicates(agent: Policy, opponent: Policy, deals: Iterable[gd.DealSpec],
                        seed: int = 0, bootstrap_samples: int = 2000) -> dict:
    scores = [play_duplicate(deal, agent, opponent, seed + index)
              for index, deal in enumerate(deals)]
    return summarize_duplicates(scores, seed, bootstrap_samples)


def summarize_duplicates(scores: Sequence[DuplicateScore], seed: int = 0,
                         bootstrap_samples: int = 2000) -> dict:
    """Report for already played deals, in deal order; shared by parallel runners."""
    if not scores:
        raise ValueError("at least one duplicate deal is required")
    per_round = [score.levels_per_round for score in scores]
    legs = [(score.first, 0) for score in scores] + [(score.swapped, 1) for score in scores]
    return {
        "deals": len(scores),
        "rounds": len(legs),
        "score_definition": "(A net level gain in leg 1 + A net level gain in leg 2) / 2",
        "mean_net_levels_per_round": float(np.mean(per_round)),
        "bootstrap_95_ci": list(bootstrap_interval(per_round, seed, bootstrap_samples)),
        "mean_pair_difference": float(np.mean([score.pair_difference for score in scores])),
        "banker_rate": sum(r.winning_team == team for r, team in legs) / len(legs),
        "double_win_rate": sum(r.double_win(team) for r, team in legs) / len(legs),
        "opponent_double_win_rate": sum(r.double_win(1 - team) for r, team in legs) / len(legs),
        "mean_finish_order_award_per_round": sum(r.gain for r, team in legs if r.winning_team == team) / len(legs),
        "pair_scores": per_round,
        "results": [asdict(score) for score in scores],
    }
