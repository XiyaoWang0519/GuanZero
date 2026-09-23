"""Record full matches between policies as a JSON game log for the viewer.

Every decision is recorded, including forced passes: the engine's auto-pass is
switched off and a seat whose only legal action is pass passes without asking
its policy, which is exactly what auto-pass does. Tribute moves are recovered
from hand differences around each tribute decision, since the engine does not
expose its tribute bookkeeping to Python.

Usage:
    PYTHONPATH=python:. .venv/bin/python -m eval.record_games --out .work/demo/games.json
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Sequence

import gd

from .policies import Policy, choose_action, load_policy

RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
SUITS = ("S", "H", "C", "D")
TYPE_ZH = {
    "Single": "单张", "Pair": "对子", "Triple": "三张", "FullHouse": "三带二",
    "Straight": "顺子", "Tube": "三连对", "Plate": "钢板", "Bomb": "炸弹",
    "StraightFlush": "同花顺", "JokerBomb": "天王炸",
}
LEAGUE = ".work/runpod-b8/results/runs/league/run/latest.pt"
M1 = ".work/runpod/artifacts/pilot/final.pt"
FROZEN = ".work/runpod-b8/results/runs/frozen/run/latest.pt"
DEFAULT_GAMES = (
    ("轮换陪练模型 vs M1", 101, ("轮换陪练模型", "M1"), (LEAGUE, M1)),
    ("轮换陪练模型 vs 专练M1模型", 102, ("轮换陪练模型", "专练M1模型"), (LEAGUE, FROZEN)),
    ("轮换陪练模型 vs 贪心规则机器人", 103, ("轮换陪练模型", "贪心规则机器人"), (LEAGUE, "greedy")),
)


def card_json(card: int, level: int) -> dict:
    if card == 52:
        return {"r": "BJ", "s": "", "w": False}
    if card == 53:
        return {"r": "RJ", "s": "", "w": False}
    rank, suit = divmod(card, 4)
    return {"r": RANKS[rank], "s": SUITS[suit], "w": rank == level and suit == 1}


def power_key(card: int, level: int) -> tuple[int, int]:
    rank = 13 if card == 52 else 14 if card == 53 else card // 4
    return gd.power(rank, level), card


def sorted_cards(cards: Sequence[int], level: int) -> list[int]:
    return sorted(cards, key=lambda c: power_key(c, level))


SEQUENCES = ("Straight", "Tube", "Plate", "StraightFlush")


def play_order(action: gd.Action, level: int) -> list[int]:
    """Display order of a play: sequence order for sequences, else power order."""
    cards = list(action.cards)
    if action.type not in SEQUENCES:
        return sorted_cards(cards, level)
    ace_low = int(action.key) == 0      # window 0 is always the ace-low window
    return sorted(cards, key=lambda c: (-1 if ace_low and c // 4 == 12 else c // 4, c))


def type_zh(action: gd.Action) -> str:
    name = TYPE_ZH[action.type]
    if action.type == "Bomb":
        name += f"({int(action.bomb_size)}张)"
    return name


def hands_of(state: gd.MatchState) -> list[list[int]]:
    return [list(state.hand(seat)) for seat in range(4)]


def tribute_moves(before: list[list[int]], after: list[list[int]], kind: str,
                  level: int) -> list[dict]:
    """Card transfers between two hand snapshots, matched by card id."""
    lost = [Counter(b) - Counter(a) for b, a in zip(before, after)]
    gained = [Counter(a) - Counter(b) for b, a in zip(before, after)]
    moves = []
    for src in range(4):
        for card, n in lost[src].items():
            for _ in range(n):
                dst = next(s for s in range(4) if gained[s][card] > 0)
                gained[dst][card] -= 1
                moves.append({"kind": kind, "from": src, "to": dst,
                              "card": card_json(card, level)})
    return moves


def play_recorded_round(engine: gd.Engine, state: gd.MatchState, policies: Sequence[Policy],
                        seed: int, index: int, prev_order: list[int] | None,
                        max_decisions: int = 4000) -> dict:
    level = state.level
    owner = state.owner
    levels_before = list(state.levels)
    rngs = [random.Random(seed + seat * 0x9E3779B9) for seat in range(4)]
    tribute: list[dict] = []
    saw_tribute = False
    while state.phase in (gd.Phase.Tribute, gd.Phase.BackTribute):
        saw_tribute = True
        kind = "tribute" if state.phase == gd.Phase.Tribute else "back"
        before = hands_of(state)
        seat = state.to_move
        engine.apply(state, choose_action(policies[seat], engine, state, rngs[seat]))
        tribute += tribute_moves(before, hands_of(state), kind, level)
    if prev_order is not None and not saw_tribute:
        # Anti-tribute: the losing seats that were due to pay.
        banker, follower, third, dweller = prev_order
        payers = [third, dweller] if follower == (banker + 2) % 4 else [dweller]
        tribute += [{"kind": "anti", "from": s, "to": None, "card": None} for s in payers]
    if state.phase != gd.Phase.Play:
        raise RuntimeError(f"round {index} did not reach play: {state.phase}")

    hands = hands_of(state)
    leader = state.to_move
    left = [len(h) for h in hands]
    steps = []
    while state.phase != gd.Phase.RoundEnd:
        if len(steps) >= max_decisions:
            raise RuntimeError(f"round {index} exceeded {max_decisions} decisions")
        seat = state.to_move
        new_trick = state.top_is_open
        actions = engine.legal_actions(state)
        if len(actions) == 1 and actions[0].is_pass:
            action = actions[0]           # forced pass, as auto-pass would do
        else:
            action = choose_action(policies[seat], engine, state, rngs[seat])
        engine.apply(state, action)
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
        "index": index,
        "level_rank": RANKS[level],
        "owner_team": owner if owner >= 0 else None,
        "levels_before": [RANKS[x] for x in levels_before],
        "hands": [[card_json(c, level) for c in sorted_cards(h, level)] for h in hands],
        "tribute": tribute,
        "leader": leader,
        "steps": steps,
        "finish_order": list(result.order),
        "winning_team": result.winning_team,
        "gain": result.gain,
        "levels_after": [RANKS[x] for x in result.levels],
        "match_over": result.match_winner >= 0,
        "_raw_hands": hands,            # stripped before writing; used by validate
    }


def record_match(title: str, seed: int, team_names: Sequence[str],
                 team_policies: Sequence[Policy], max_rounds: int = 20,
                 rules: gd.RuleConfig | None = None) -> dict:
    engine = gd.Engine(rules or gd.RuleConfig.house())
    engine.auto_pass = False
    state = gd.MatchState()
    engine.new_match(state, seed)
    seats = [team_policies[seat % 2] for seat in range(4)]
    rounds = []
    prev_order = None
    while True:
        rnd = play_recorded_round(engine, state, seats, seed * 1000003 + len(rounds),
                                  len(rounds), prev_order)
        rounds.append(rnd)
        prev_order = rnd["finish_order"]
        if state.winner >= 0 or len(rounds) >= max_rounds:
            break
        engine.begin_round(state)
    return {"title": title, "seed": seed, "team_names": list(team_names),
            "winner_team": state.winner if state.winner >= 0 else None, "rounds": rounds}


def validate_round(rnd: dict) -> None:
    hands = [Counter(h) for h in rnd["_raw_hands"]]
    if [sum(h.values()) for h in hands] != [27] * 4:
        raise AssertionError(f"round {rnd['index']}: hands are not 27 cards each")
    total = sum(hands, Counter())
    if total != Counter({c: 2 for c in range(54)}):
        raise AssertionError(f"round {rnd['index']}: hands are not the 108-card deck")
    level = RANKS.index(rnd["level_rank"])
    by_json = {json.dumps(card_json(c, level), sort_keys=True): c for c in range(54)}
    went_out = []
    for step in rnd["steps"]:
        seat = step["seat"]
        cards = Counter(by_json[json.dumps(c, sort_keys=True)] for c in step["cards"])
        if step["pass"] != (not cards):
            raise AssertionError("pass flag disagrees with cards")
        if cards - hands[seat]:
            raise AssertionError(f"seat {seat} played cards it does not hold")
        hands[seat] -= cards
        if [sum(h.values()) for h in hands] != step["left"]:
            raise AssertionError("left counts disagree with replay")
        if cards and not hands[seat]:
            went_out.append(seat)
    order = rnd["finish_order"]
    if sorted(order) != [0, 1, 2, 3] or order[:len(went_out)] != went_out:
        raise AssertionError(f"finish order {order} disagrees with replay {went_out}")
    if any(not hands[s] for s in order[len(went_out):]):
        raise AssertionError("a seat outside the replayed order has an empty hand")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=".work/demo/games.json")
    parser.add_argument("--max-rounds", type=int, default=20)
    args = parser.parse_args(argv)
    cache: dict[str, Policy] = {}
    games = []
    for title, seed, names, specs in DEFAULT_GAMES:
        policies = [cache.setdefault(s, load_policy(s)) for s in specs]
        game = record_match(title, seed, names, policies, args.max_rounds)
        for rnd in game["rounds"]:
            validate_round(rnd)
            del rnd["_raw_hands"]
        steps = sum(len(r["steps"]) for r in game["rounds"])
        print(f"{title}: rounds={len(game['rounds'])} winner={game['winner_team']} steps={steps}")
        games.append(game)
    doc = {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "rules": "house", "games": games}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text.encode('utf-8')) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
