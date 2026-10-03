"""The browser game of eval/play_vs_model.py, driven without a browser."""
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from eval.history_policy import HistoryPolicy
from eval.play_vs_model import HUMAN, Game, Server
from train.history_model import HistoryPolicyConfig, fresh_player

TINY = HistoryPolicyConfig(width=16, layers=1, heads=2, action_width=16, fusion_width=16,
                           critic_width=16, critic_layers=1)


def tiny_policy() -> HistoryPolicy:
    return HistoryPolicy(fresh_player(TINY, 0)[0])


def play_rounds(game: Game, rng: random.Random, rounds: int) -> int:
    played = 0
    while played < rounds:
        view = game.view()
        if view["result"]:
            played += 1
            if played == rounds or view["result"]["match_over"]:
                break
            game.next_round()
            continue
        assert view["your_turn"] and view["to_move"] == HUMAN
        action = rng.choice(game.human_actions())
        out = game.play(list(action.cards))
        if "options" in out:
            out = game.play(list(action.cards), rng.randrange(len(out["options"])))
        assert "events" in out, out
    return played


def test_rounds_keep_the_stream_and_the_deck_consistent():
    game = Game(tiny_policy(), seed=3)
    game.advance()
    assert play_rounds(game, random.Random(0), 3) == 3
    view = game.view()
    assert len(view["history"]) == 3
    # The model saw every action: its stream check passes at the next decision.
    if not view["result"]["match_over"]:
        game.next_round()
        game.policy.verify_stream(game.state)
    assert "hands" not in view and "hands" not in view["result"]


def test_tracker_counts_every_unseen_card():
    game = Game(tiny_policy(), seed=4)
    game.advance()
    rng = random.Random(2)
    for _ in range(6):
        view = game.view()
        if view["result"]:
            break
        mine = len(view["hand"])
        played = sum(game.played.values())
        assert sum(u["left"] for u in view["unseen"]) == 108 - mine - played
        assert all(u["left"] >= 0 for u in view["unseen"])
        action = rng.choice(game.human_actions())
        out = game.play(list(action.cards))
        if "options" in out:
            game.play(list(action.cards), 0)


def test_rejects_cards_not_held_and_illegal_plays():
    game = Game(tiny_policy(), seed=5)
    game.advance()
    view = game.view()
    held = {c["id"] for c in view["hand"]}
    missing = next(c for c in range(54) if c not in held)
    with pytest.raises(ValueError):
        game.play([missing])
    if view["phase"] == "play" and not view["can_pass"]:
        assert "error" in game.play([])


def test_advice_is_shown_on_your_turn_only():
    game = Game(tiny_policy(), seed=11)
    game.advance()
    view = game.view()
    assert view["your_turn"] and view["advice"] == game.advice()
    hints = view["advice"]["hints"]
    assert hints and all(0.0 <= h["p"] <= 1.0 for h in hints)
    assert [h["p"] for h in hints] == sorted((h["p"] for h in hints), reverse=True)
    out = game.play(hints[0]["ids"])
    assert "events" in out or "options" in out


def test_server_routes():
    server = Server(tiny_policy())
    assert server.handle("/api/state", {}) == {"view": None}
    out = server.handle("/api/new", {"seed": 9})
    assert out["view"]["seed"] == 9 and out["view"]["your_turn"]
    assert out["view"]["advice"]["hints"]
    with pytest.raises(KeyError):
        server.handle("/api/nope", {})
