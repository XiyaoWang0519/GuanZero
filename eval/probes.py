"""Small legal behavioural scenarios, not a substitute for strength evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

import gd

from .policies import ModelPolicy, Policy, choose_action


@dataclass
class Probe:
    name: str
    group: int
    description: str
    engine: gd.Engine
    state: gd.MatchState
    good: Callable[[gd.Action], bool]


def _position(hands: list[str], leader: int = 0, level: int = 5,
              previous: list[int] | None = None) -> tuple[gd.Engine, gd.MatchState]:
    engine = gd.Engine()
    # Script passes explicitly, so every prefix is independently auditable.
    engine.auto_pass = False
    deal = gd.DealSpec()
    deal.hands = [gd.cards(hand) for hand in hands]
    deal.level = level
    deal.team_levels = [level, level]
    deal.owner = 0
    deal.leader = leader
    if previous is not None:
        deal.prev_order = previous
    state = gd.MatchState()
    engine.set_deal(state, deal)
    return engine, state


def _play(engine: gd.Engine, state: gd.MatchState, seat: int,
          cards: str = "", kind: str | None = None) -> None:
    assert state.to_move == seat, (state.to_move, seat)
    expected = sorted(gd.cards(cards))
    candidates = [a for a in engine.legal_actions(state)
                  if a.cards == expected and (kind is None or a.type == kind)]
    if len(candidates) != 1:
        raise ValueError(f"ambiguous or illegal probe prefix: seat {seat}, {cards!r}, {kind}")
    engine.apply(state, candidates[0])


def build_probes() -> list[Probe]:
    """Construct fresh states using only set_deal and legal Engine.apply calls.

    Short endgame hands are intentional. The scripted setup can inspect all
    hands, but ModelPolicy sees only the acting seat's normal observation.
    """
    probes = []
    e, s = _position(["S3 S9 D9", "ST SJ SQ", "S8", "D4 D5 D6"])
    low = gd.cards("S3")
    probes.append(Probe("feed_partner_single", 1, "Lead a low single to a partner with one card.",
                        e, s, lambda a: a.type == "Single" and a.cards == low))

    e, s = _position(["S3 D3 S8", "SK", "S5 S6", "D7 D8"])
    probes.append(Probe("opponent_one_avoid_single", 2, "Lead a pair when an opponent holds one card.",
                        e, s, lambda a: a.type == "Pair"))

    e, s = _position(["S5 SK SA", "S3 S9", "D6 D7 D8", "D4 D5 D9"], leader=1)
    _play(e, s, 1, "S3")
    _play(e, s, 2)
    _play(e, s, 3)
    highest = gd.cards("SA")
    probes.append(Probe("opponent_one_cover_high", 2, "Cover a threatened single with the highest card.",
                        e, s, lambda a: a.type == "Single" and a.cards == highest))

    e, s = _position(["S4 SQ D9", "S3 S8", "SJ C9", "S5 S7"], leader=1)
    for seat, cards in [(1, "S3"), (2, ""), (3, ""), (0, "S4"),
                        (1, ""), (2, "SJ"), (3, "")]:
        _play(e, s, seat, cards)
    probes.append(Probe("let_partner_hold", 3,
                        "Pass on partner's top after both opponents have passed in this trick.",
                        e, s, lambda a: a.is_pass))

    e, s = _position(["S3 D3 C3 H3 S6 D6", "SK DK S2", "S4 D4 S8", "S5 D5 S9"], leader=1)
    _play(e, s, 1, "SK DK", "Pair")
    _play(e, s, 2)
    _play(e, s, 3)
    probes.append(Probe("bomb_imminent_finish", 4, "Bomb an opponent's pair before their last card.",
                        e, s, lambda a: a.type == "Bomb"))

    e, s = _position(["S3 D3 C3 H3 H7 S8", "D2 D4 D6", "C2 C4 C6", "S4 S9 D9 C9 H9"], leader=3)
    _play(e, s, 3, "S4")
    wild = gd.card_id("H7")
    probes.append(Probe("save_bomb_and_wild", 5, "Use a natural single when nobody is near finishing.",
                        e, s, lambda a: a.type == "Single" and wild not in a.cards))

    e, s = _position(["S3 D3 C3", "S5 D5 S8", "S2", "S6 D6 S9"], leader=2)
    _play(e, s, 2, "S2")
    for seat in (3, 0, 1):
        _play(e, s, seat)
    assert s.to_move == 0 and s.top_is_open and s.order[0] == 2
    finish_cards = sorted(s.hand(0))
    probes.append(Probe("inherited_lead_double_win", 6,
                        "After partner finishes first, use the inherited lead to finish second.",
                        e, s, lambda a: a.cards == finish_cards))

    e, s = _position(["S3 D3 S8", "S4 D4 S9", "S5 D5 S6", "ST SJ SQ SK SA DA"],
                     leader=-1, previous=[0, 1, 2, 3])
    spare_ace = gd.cards("DA")
    assert s.to_move == 3 and s.phase == gd.Phase.Tribute
    probes.append(Probe("tribute_preserve_straight_flush", 7,
                        "Give the diamond ace and preserve the spade straight flush.",
                        e, s, lambda a: a.cards == spare_ace))
    return probes


def evaluate_probes(policy: Policy, seed: int = 0, repeats: int = 1) -> dict:
    if repeats < 1:
        raise ValueError("probe repeats must be positive")
    results = []
    for index, probe in enumerate(build_probes()):
        selected = []
        passed = 0
        for repeat in range(repeats):
            rng = random.Random(seed + index * 1009 + repeat)
            action = choose_action(policy, probe.engine, probe.state, rng)
            passed += bool(probe.good(action))
            selected.append(str(action))
        results.append({"name": probe.name, "group": probe.group,
                        "description": probe.description, "passed": passed,
                        "attempts": repeats, "pass_rate": passed / repeats,
                        "selected": selected})
    groups = []
    for group in range(1, 8):
        items = [item for item in results if item["group"] == group]
        attempts = sum(item["attempts"] for item in items)
        passed = sum(item["passed"] for item in items)
        groups.append({"group": group, "passed": passed, "attempts": attempts,
                       "pass_rate": passed / attempts})
    learned_tribute = isinstance(policy, ModelPolicy) and not policy.heuristic_tribute
    return {"cases": results, "groups": groups,
            "mean_group_pass_rate": sum(g["pass_rate"] for g in groups) / len(groups),
            "tribute_note": ("This checkpoint uses learned tribute heads; these scenarios are diagnostics, not a measured payoff."
                             if learned_tribute else "This policy uses heuristic tribute.")}
