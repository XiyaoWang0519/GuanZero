"""Game-review recorder: a history-aware copy of eval/record_games.py's round loop.

Run with PYTHONPATH=<export b8c0c54>/python:<export>/oracle:<export> so every
other module (engine, policies, history plumbing) comes from the export.

Differences from eval/record_games.py (export b8c0c54):
* per-seat lineups (four policies, not two teams);
* history policies (needs_history) get start_match() once per match and every
  public action through eval.history_policy.apply_and_observe, including
  tribute moves and forced passes -- the same event source eval.duplicate's
  play_round_seats uses;
* a per-decision analysis record (hand, legal-set facts, trick owner, and for
  history policies the top candidate probabilities) is kept alongside.
Forced passes are still resolved without asking the policy, exactly as
resolve_forced_passes does in the evaluator.
"""
from __future__ import annotations

import math
import random
from typing import Sequence

import gd
import numpy as np

from eval.history_policy import apply_and_observe, history_listeners, needs_history
from eval.policies import choose_action
from eval.record_games import (RANKS, card_json, hands_of, play_order, sorted_cards,
                               tribute_moves, type_zh)

BOMBS = ("Bomb", "StraightFlush", "JokerBomb")


def cards_str(cards, level):
    out = []
    for c in sorted_cards(list(cards), level):
        j = card_json(c, level)
        if j["r"] in ("BJ", "RJ"):
            out.append("小王" if j["r"] == "BJ" else "大王")
        else:
            out.append({"S": "♠", "H": "♥", "C": "♣", "D": "♦"}[j["s"]] + j["r"] + ("*" if j["w"] else ""))
    return " ".join(out)


def action_desc(a, level):
    if a.is_pass:
        return "过"
    return f"{type_zh(a)} {cards_str(a.cards, level)}"


def instrument_history(policy):
    """Record the candidate log-prob table of each history decision (no effect on choice)."""
    actor = policy.actor
    if getattr(actor, "_review_wrapped", False):
        return
    orig = actor._log_prob_table

    def wrapped(*args, **kwargs):
        table = orig(*args, **kwargs)
        actor._review_last = table.detach().cpu().numpy().copy()
        return table
    actor._log_prob_table = wrapped
    actor._review_wrapped = True


def play_round_recorded(engine, state, seats: Sequence, seed: int, index: int,
                        prev_order, labels: Sequence[str], decisions: list,
                        trace: list | None = None, max_decisions: int = 4000,
                        match_tag: str = "") -> dict:
    """Play one round; all seats' actions go to history listeners."""
    listeners = history_listeners(seats)
    level = state.level
    owner = state.owner
    levels_before = list(state.levels)
    rngs = [random.Random(seed + seat * 0x9E3779B9) for seat in range(4)]
    tribute = []
    saw_tribute = False
    while state.phase in (gd.Phase.Tribute, gd.Phase.BackTribute):
        saw_tribute = True
        kind = "tribute" if state.phase == gd.Phase.Tribute else "back"
        before = hands_of(state)
        seat = state.to_move
        action = choose_action(seats[seat], engine, state, rngs[seat])
        apply_and_observe(engine, state, action, listeners)
        tribute += tribute_moves(before, hands_of(state), kind, level)
    if prev_order is not None and not saw_tribute:
        banker, follower, third, dweller = prev_order
        payers = [third, dweller] if follower == (banker + 2) % 4 else [dweller]
        tribute += [{"kind": "anti", "from": s, "to": None, "card": None} for s in payers]
    if state.phase != gd.Phase.Play:
        raise RuntimeError(f"round {index} did not reach play: {state.phase}")

    hands = hands_of(state)
    leader = state.to_move
    left = [len(h) for h in hands]
    steps = []
    top_seat = None
    while state.phase != gd.Phase.RoundEnd:
        if len(steps) >= max_decisions:
            raise RuntimeError("too many decisions")
        seat = state.to_move
        new_trick = bool(state.top_is_open)
        if new_trick:
            top_seat = None
        actions = engine.legal_actions(state)
        forced = len(actions) == 1 and actions[0].is_pass
        hand_before = list(state.hand(seat))
        probs = None
        if forced:
            action = actions[0]
        else:
            policy = seats[seat]
            prefix = policy.events_seen if needs_history(policy) else None
            action = choose_action(policy, engine, state, rngs[seat])
            if needs_history(policy):
                table = policy.actor._review_last[0]
                lp = table[: len(actions)]
                order = np.argsort(-lp)[:4]
                probs = [(action_desc(actions[i], level), float(math.exp(lp[i]))) for i in order]
            if trace is not None:
                trace.append((int(state.round_index), seat, bool(action.is_pass),
                              tuple(sorted(action.cards)), prefix))
        nonpass = [a for a in actions if not a.is_pass]
        decisions.append({
            "match": match_tag, "round": index, "step": len(steps) + 1, "seat": seat,
            "label": labels[seat], "level": level, "forced": forced, "lead": new_trick,
            "hand": hand_before, "left": list(left),
            "top_seat": top_seat, "top_rel": (None if top_seat is None else
                                              "partner" if top_seat % 2 == seat % 2 else "opponent"),
            "n_legal": len(actions), "has_nonpass": bool(nonpass),
            "has_nonbomb": any(a.type not in BOMBS for a in nonpass),
            "pass": bool(action.is_pass), "type": None if action.is_pass else action.type,
            "cards": list(action.cards), "bomb_size": (int(action.bomb_size) if not action.is_pass and action.type == "Bomb" else None),
            "probs": probs,
        })
        apply_and_observe(engine, state, action, listeners, forced=forced)
        if not action.is_pass:
            top_seat = seat
        cards = list(action.cards)
        left[seat] -= len(cards)
        steps.append({
            "seat": seat, "pass": action.is_pass,
            "cards": [card_json(c, level) for c in play_order(action, level)],
            "type": None if action.is_pass else action.type,
            "type_zh": None if action.is_pass else type_zh(action),
            "new_trick": new_trick, "left": list(left),
        })
    result = engine.end_round(state)
    return {
        "index": index, "level_rank": RANKS[level],
        "owner_team": owner if owner >= 0 else None,
        "levels_before": [RANKS[x] for x in levels_before],
        "hands": [[card_json(c, level) for c in sorted_cards(h, level)] for h in hands],
        "tribute": tribute, "leader": leader, "steps": steps,
        "finish_order": list(result.order), "winning_team": result.winning_team,
        "gain": result.gain, "levels_after": [RANKS[x] for x in result.levels],
        "match_over": result.match_winner >= 0,
        "seat_return": list(result.seat_return),
        "_raw_hands": hands,
    }


def record_match(title, seed, team_names, seats, labels, decisions, max_rounds=40,
                 trace=None, match_tag=""):
    engine = gd.Engine(gd.RuleConfig.house())
    engine.auto_pass = False
    state = gd.MatchState()
    engine.new_match(state, seed)
    for listener in history_listeners(seats):
        listener.start_match()
    rounds, prev_order = [], None
    while True:
        rnd = play_round_recorded(engine, state, seats, seed * 1000003 + len(rounds),
                                  len(rounds), prev_order, labels, decisions, trace,
                                  match_tag=match_tag)
        rounds.append(rnd)
        prev_order = rnd["finish_order"]
        if state.winner >= 0 or len(rounds) >= max_rounds:
            break
        engine.begin_round(state)
    return {"title": title, "seed": seed, "team_names": list(team_names),
            "seat_labels": list(labels),
            "winner_team": state.winner if state.winner >= 0 else None, "rounds": rounds}


def play_deal_recorded(deal, seats, labels, decisions, seed=0, trace=None, match_tag=""):
    """One duplicate leg: a fresh single-round match from a DealSpec."""
    engine = gd.Engine(gd.RuleConfig.house())
    engine.auto_pass = False
    state = gd.MatchState()
    engine.set_deal(state, deal)
    for listener in history_listeners(seats):
        listener.start_match()
    return play_round_recorded(engine, state, seats, seed, 0, None, labels, decisions,
                               trace, match_tag=match_tag)
