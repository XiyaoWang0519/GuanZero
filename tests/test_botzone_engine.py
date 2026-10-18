"""The Botzone bot's pure-Python engine against ``gd``, step by step.

Rounds are played in lockstep in both engines from random deals (with and
without tribute, anti-tribute forced in some). At every step the legal
actions (canonical, and full on a subset, in gd's order), every seat's
observation, the encoding of every candidate and the tribute heuristic must
agree exactly.
"""
import os
import random

import gd
import numpy as np
import pytest

from eval.botzone import pyengine as pe


def key_of(action):
    return (action.type, int(action.key), int(action.bomb_size), int(action.wilds),
            tuple(sorted(int(c) for c in action.cards)))


def port_key(action):
    return (action.type_name, action.key, action.bomb_size, action.wilds, tuple(action.cards))


def make_deal(rng, kind):
    deck = [c for c in range(54) for _ in range(2)]
    rng.shuffle(deck)
    hands = [sorted(deck[27 * s:27 * (s + 1)]) for s in range(4)]
    level = rng.randrange(13)
    order = None
    if kind != "none":
        order = list(range(4))
        rng.shuffle(order)
        if kind == "anti":
            # Hand both big jokers to the payers.
            payers = [order[3]] if (order[0] - order[1]) % 2 else [order[2], order[3]]
            while sum(hands[p].count(53) for p in payers) < 2:
                holder = next(s for s in range(4) if s not in payers and 53 in hands[s])
                target = min(payers, key=lambda p: hands[p].count(53))
                swap = next(c for c in hands[target] if c != 53)
                hands[holder].remove(53)
                hands[target].remove(swap)
                hands[holder].append(swap)
                hands[target].append(53)
            hands = [sorted(h) for h in hands]
    deal = gd.DealSpec()
    deal.hands = hands
    deal.level = level
    deal.team_levels = [level, level]
    deal.owner = -1
    leader = rng.randrange(4)
    if order is not None:
        deal.prev_order = order
    else:
        deal.leader = leader
    port = pe.State(hands, level, [level, level], leader, order)
    return deal, port


def compare_step(engine, full, state, port, check_full):
    assert int(state.phase) == port.phase
    assert int(state.to_move) == port.to_move
    for seat in range(4):
        assert sorted(state.hand(seat)) == port.hand(seat)
        np.testing.assert_array_equal(np.asarray(state.observation(seat)),
                                      pe.encode_observation(port, seat), err_msg=f"seat {seat}")
    ours = port.legal_actions(canonical=True)
    theirs = engine.legal_actions(state)
    assert [port_key(a) for a in ours] == [key_of(a) for a in theirs]
    seat = int(state.to_move)
    for a, b in zip(ours, theirs):
        np.testing.assert_array_equal(np.asarray(engine.encode_action(b, state, seat)),
                                      pe.encode_action(a, port, seat))
    if check_full and port.phase == pe.PHASE_PLAY:
        assert [port_key(a) for a in port.legal_actions(canonical=False)] == \
            [key_of(a) for a in full.legal_actions(state)]
    if port.phase in (pe.PHASE_TRIBUTE, pe.PHASE_BACK_TRIBUTE):
        assert pe.tribute_choice(port, ours) == engine.greedy(state)
    return ours, theirs


@pytest.mark.parametrize("kind", ["none", "single", "double", "anti"])
def test_rounds_match_gd(kind):
    rng = random.Random({"none": 1, "single": 2, "double": 3, "anti": 4}[kind])
    engine = gd.Engine(gd.RuleConfig.house(), gd.ActionConfig())
    full = gd.Engine(gd.RuleConfig.house(), gd.ActionConfig.full())
    engine.auto_pass = full.auto_pass = False
    steps = 0
    for game in range(int(os.environ.get("GZ_ENGINE_GAMES", "6"))):
        deal, port = make_deal(rng, kind)
        state = gd.MatchState()
        engine.set_deal(state, deal)
        if kind == "anti":
            assert port.anti_tribute and int(state.phase) == int(gd.Phase.Play)
        while int(state.phase) in (1, 2, 3):
            ours, theirs = compare_step(engine, full, state, port, check_full=game % 3 == 0)
            if port.phase == pe.PHASE_PLAY and rng.random() < 0.5:
                pick = engine.greedy(state)
            else:
                pick = rng.randrange(len(theirs))
            engine.apply(state, theirs[pick])
            port.apply(ours[pick])
            steps += 1
        assert port.phase == pe.PHASE_ROUND_END
        assert list(state.order) == port.order
    assert steps > 100


def test_move_generation_with_tops_and_wilds():
    """Random hands with many wild cards against random tops, both modes."""
    rng = random.Random(9)
    checked = 0
    for _ in range(int(os.environ.get("GZ_ENGINE_HANDS", "300"))):
        level = rng.randrange(13)
        wild = level * 4 + 1
        deck = [c for c in range(54) for _ in range(2) if c != wild]
        rng.shuffle(deck)
        size = rng.choice([5, 8, 12, 18, 27])
        hand = sorted(deck[:size - 2] + [wild, wild])
        top_hand = sorted(deck[size:size + 12])
        tops = [None] + [t for t in gd.legal_actions(top_hand, level, None, False)]
        top = rng.choice(tops)
        top = None if top is None else top[:2]
        for canonical in (True, False):
            if top is None:
                port_top = pe.PASS_ACTION
            else:
                kind, key = top
                bomb = key[0] if isinstance(key, tuple) else 0
                key = key[1] if isinstance(key, tuple) else key
                port_top = pe.Action(pe.TYPE_BY_NAME[kind], key, bomb)
            ours = {(a.type_name, a.key, a.bomb_size, tuple(a.cards))
                    for a in pe.generate_moves(pe.counts_of(hand), level, port_top, canonical)}
            theirs = set()
            for kind, key, cards in gd.legal_actions(hand, level, top, canonical):
                bomb = key[0] if isinstance(key, tuple) else 0
                key = key[1] if isinstance(key, tuple) else (key if key is not None else 0)
                theirs.add((kind, key, bomb, tuple(sorted(cards))))
            assert ours == theirs, (hand, level, top, canonical)
            checked += len(ours)
    assert checked > 10_000


def test_classify_reads_claims_like_gd():
    from eval.botzone.protocol import claim_faces
    rng = random.Random(12)
    for _ in range(40):
        deck = [c for c in range(54) for _ in range(2)]
        rng.shuffle(deck)
        hand, level = sorted(deck[:27]), rng.randrange(13)
        for a in pe.generate_moves(pe.counts_of(hand), level, pe.PASS_ACTION, canonical=True):
            declared = claim_faces(a.type_name, a.key, a.cards, level)
            if declared is None:
                continue
            got = pe.classify(declared, a.cards, level)
            assert (got.type, got.key, got.bomb_size, got.wilds, got.counts) == \
                (a.type, a.key, a.bomb_size, a.wilds, a.counts)
