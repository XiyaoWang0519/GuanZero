"""M0 tasks 4 and 5: full mode equals the oracle, canonical mode is a sound
subset of it.
"""
import os
import random

import gd
import gd_reference as g

# The M0 gate is 10^4 hands. CI runs a smaller sample by default; set
# GD_XCHECK_HANDS=10000 for the full sweep.
N_SMALL_HANDS = int(os.environ.get("GD_XCHECK_HANDS", 1500))


def _random_hand(rng, size):
    pool = [c for c in range(54) for _ in range(2)]
    rng.shuffle(pool)
    return sorted(pool[:size])


def _random_top(rng, hand, level):
    """A top play the oracle can produce, or None for a lead."""
    if rng.random() < 0.25:
        return None
    for _ in range(20):
        cs = _random_hand(rng, rng.randint(1, 6))
        readings = sorted(g.interpret(cs, level), key=repr)
        if readings:
            return readings[rng.randrange(len(readings))]
    return None


def test_full_mode_equals_oracle_on_small_hands():
    rng = random.Random(7)
    for i in range(N_SMALL_HANDS):
        level = rng.randrange(13)
        hand = _random_hand(rng, rng.randint(1, 12))
        top = _random_top(rng, hand, level)
        got = gd.legal_actions(hand, level, top, canonical=False)
        want = g.legal_actions(hand, level, top, canonical=False)
        assert got == want, (i, level, [g.cstr(c) for c in hand], top,
                             sorted(got - want, key=repr)[:3],
                             sorted(want - got, key=repr)[:3])


def test_full_mode_sound_on_full_hands():
    """27-card hands are far too big for the oracle's brute force, so check
    soundness only: every generated action is a real reading that beats top."""
    rng = random.Random(8)
    for _ in range(20):
        level = rng.randrange(13)
        hand = _random_hand(rng, 27)
        top = _random_top(rng, hand, level)
        acts = gd.legal_actions(hand, level, top, canonical=False)
        assert acts
        counts = {}
        for c in hand:
            counts[c] = counts.get(c, 0) + 1
        for kind, key, cs in acts:
            if kind == g.PASS:
                assert top is not None
                continue
            used = {}
            for c in cs:
                used[c] = used.get(c, 0) + 1
            for c, n in used.items():
                assert counts.get(c, 0) >= n, (kind, key, cs)
            assert (kind, key) in g.interpret(list(cs), level)
            assert g.beats((kind, key), top)


def _reading_order(kind, key):
    """A comparable strength for 'a stronger reading of the same cards'."""
    if kind == g.BOMB:
        return g.strength(kind, key)
    if kind in (g.STRAIGHT_FLUSH, g.JOKER_BOMB):
        return g.strength(kind, key)
    return (0, key)


def test_canonical_is_a_sound_subset():
    """RULES.md 11.2 and 14.5, as amended in M0.

    Canonical mode is a subset of full mode; it never loses a type and never
    lowers the best key of a type; and whenever it drops a full-mode action it
    keeps either the same reading or a strictly stronger reading of the same
    type. What it may drop is a deliberately weak declaration: with wild cards
    in hand every multiset that reads as a low full house also reads as a
    higher one, so prune_dominated_readings removes the low key outright.
    """
    rng = random.Random(9)
    for _ in range(2000):
        level = rng.randrange(13)
        hand = _random_hand(rng, rng.randint(1, 14))
        top = _random_top(rng, hand, level)
        full = gd.legal_actions(hand, level, top, canonical=False)
        canon = gd.legal_actions(hand, level, top, canonical=True)

        assert canon <= full
        assert (top is None) == all(a[0] != g.PASS for a in full)

        full_best, canon_best = {}, {}
        for kind, key, _ in full:
            o = _reading_order(kind, key)
            full_best[kind] = max(o, full_best.get(kind, o))
        for kind, key, _ in canon:
            o = _reading_order(kind, key)
            canon_best[kind] = max(o, canon_best.get(kind, o))
        ctx = (level, [g.cstr(c) for c in hand], top)
        assert set(canon_best) == set(full_best), ctx
        for kind, best in full_best.items():
            assert canon_best[kind] == best, (ctx, kind, canon_best[kind], best)

        readings = {(kind, key) for kind, key, _ in canon}
        for kind, key, cs in full:
            if (kind, key) in readings:
                continue
            assert canon_best[kind] > _reading_order(kind, key), (ctx, kind, key, cs)


def test_lead_and_follow_shape():
    rng = random.Random(10)
    for _ in range(500):
        level = rng.randrange(13)
        hand = _random_hand(rng, rng.randint(1, 20))
        lead = gd.legal_actions(hand, level, None, canonical=True)
        assert lead and all(a[0] != g.PASS for a in lead)
        follow = gd.legal_actions(hand, level, (g.SINGLE, 0), canonical=True)
        assert (g.PASS, 0, ()) in follow


def test_tribute_choices_match_oracle():
    rng = random.Random(11)
    for _ in range(5000):
        level = rng.randrange(13)
        # A real payer holds 27 cards; a hand of nothing but wild cards is not
        # reachable and the oracle does not define it.
        hand = _random_hand(rng, rng.randint(4, 27))
        if all(c == g.wild_id(level) for c in hand):
            continue
        assert gd.tribute_choices(hand, level) == g.tribute_choices(hand, level)
        assert gd.back_tribute_choices(hand, level) == g.back_tribute_choices(hand, level)
