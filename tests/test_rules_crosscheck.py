"""M0 task 3: the C++ reading of a card multiset must agree with the oracle.

10^5 random multisets of 1 to 10 cards at random levels, plus every ordering
and beats() pair the oracle can produce.
"""
import random

import gd
import gd_reference as g
import pytest

N_MULTISETS = int(10**5)
KINDS = [g.SINGLE, g.PAIR, g.TRIPLE, g.FULL_HOUSE, g.STRAIGHT, g.TUBE,
         g.PLATE, g.BOMB, g.STRAIGHT_FLUSH, g.JOKER_BOMB, g.PASS]


def deck():
    return [c for c in range(54) for _ in range(2)]


def test_power_table_matches_oracle():
    for level in range(13):
        for rank in range(15):
            assert gd.power(rank, level) == g.power(rank, level), (rank, level)


def test_card_text_round_trip():
    for c in range(54):
        s = gd.card_str(c)
        assert s == g.cstr(c)
        assert gd.card_id(s) == c
    assert gd.card_id("zz") == -1


def test_abstract_action_count():
    assert gd.abstract_action_count() == g.abstract_action_count() == 393


def _random_multiset(rng, size):
    pool = deck()
    rng.shuffle(pool)
    return sorted(pool[:size])


@pytest.mark.parametrize("seed", [0])
def test_interpret_matches_oracle(seed):
    rng = random.Random(seed)
    for i in range(N_MULTISETS):
        level = rng.randrange(13)
        size = rng.randint(1, 10)
        cs = _random_multiset(rng, size)
        assert gd.interpret(cs, level) == g.interpret(cs, level), (
            i, level, [g.cstr(c) for c in cs])


@pytest.mark.parametrize("seed", [1])
def test_best_readings_matches_oracle(seed):
    rng = random.Random(seed)
    for _ in range(N_MULTISETS // 10):
        level = rng.randrange(13)
        cs = _random_multiset(rng, rng.randint(1, 10))
        assert gd.best_readings(cs, level) == g.best_readings(cs, level), (
            level, [g.cstr(c) for c in cs])


def test_interpret_rejects_oversized():
    assert gd.interpret(list(range(11)), 5) == set()


def test_beats_matches_oracle_on_every_reading_pair():
    """Collect the readings the oracle produces on a sample, then cross every
    pair through both beats() implementations."""
    rng = random.Random(2)
    readings = set()
    for _ in range(4000):
        level = rng.randrange(13)
        cs = _random_multiset(rng, rng.randint(1, 10))
        readings |= g.interpret(cs, level)
    readings.add((g.PASS, 0))
    readings = sorted(readings, key=repr)
    for cand in readings:
        assert gd.beats(cand, None) == g.beats(cand, None), cand
        for top in readings:
            if top[0] == g.PASS:
                continue
            assert gd.beats(cand, top) == g.beats(cand, top), (cand, top)


def test_section_12_vectors_through_bindings():
    """Spot-check the RULES.md vectors that the C++ suite also covers, so a
    binding mistake cannot hide a working engine."""
    L7 = g.RANKS.index("7")
    assert gd.interpret(g.cards("H7 SA"), L7) == {(g.PAIR, 11)}
    assert gd.interpret(g.cards("H7 SB"), L7) == set()
    assert gd.interpret(g.cards("SB SB HR HR"), L7) == {(g.JOKER_BOMB, 0)}
    assert gd.interpret(g.cards("S5 D5 C5 SB SB"), L7) == {(g.FULL_HOUSE, 3)}
    assert gd.interpret(g.cards("SA S2 S3 S4 S5"), L7) == {(g.STRAIGHT_FLUSH, 0)}
    eight = g.cards("S7 S7 D7 D7 C7 C7 H7 H7")
    assert gd.interpret(eight, L7) == {(g.BOMB, (8, 12))}
