"""M0 tasks 4 and 5: full mode equals the oracle, canonical mode is a sound
subset of it.
"""
import random

import gd
import gd_reference as g

N_SMALL_HANDS = 10**4


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


def test_canonical_is_a_sound_subset():
    """RULES.md 11.2 and 14.5: canonical is a subset of full, and every full
    action is represented either by its abstract action or by a stronger
    reading of the same cards."""
    rng = random.Random(9)
    for _ in range(2000):
        level = rng.randrange(13)
        hand = _random_hand(rng, rng.randint(1, 14))
        top = _random_top(rng, hand, level)
        full = gd.legal_actions(hand, level, top, canonical=False)
        canon = gd.legal_actions(hand, level, top, canonical=True)
        assert canon <= full
        assert (top is None) == all(a[0] != g.PASS for a in full)
        canon_abstract = {gd.abstract_id(a) for a in canon}
        by_cards = {}
        for kind, key, cs in canon:
            by_cards.setdefault(cs, []).append((kind, key))
        for a in full:
            kind, key, cs = a
            if gd.abstract_id(a) in canon_abstract:
                continue
            same = by_cards.get(cs, [])
            assert any(k == kind and kk >= key for k, kk in same), (
                level, [g.cstr(c) for c in hand], top, a, same)


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
        hand = _random_hand(rng, rng.randint(1, 27))
        assert gd.tribute_choices(hand, level) == g.tribute_choices(hand, level)
        assert gd.back_tribute_choices(hand, level) == g.back_tribute_choices(hand, level)
