"""Executable form of the test vectors in RULES.md section 12.

Run: python3 -m pytest -q test_gd_reference.py   (or: python3 test_gd_reference.py)
"""
import gd_reference as g
from gd_reference import (BOMB, FULL_HOUSE, JOKER_BOMB, PAIR, PASS, PLATE,
                          SINGLE, STRAIGHT, STRAIGHT_FLUSH, TRIPLE, TUBE,
                          beats, best_readings, cards, interpret)

L2, L5, L7, LA = (g.RANKS.index(x) for x in "257A")


def P(rank_char, level):
    return g.power(g.RANKS.index(rank_char), level)


def test_ord_singles():
    assert beats((SINGLE, P("7", L7)), (SINGLE, P("A", L7)))          # T-ORD-01
    assert interpret(cards("SB"), L7) == {(SINGLE, 13)}               # T-ORD-02
    assert interpret(cards("HR"), L7) == {(SINGLE, 14)}
    assert beats((SINGLE, 13), (SINGLE, P("7", L7)))
    assert interpret(cards("H7"), L7) == interpret(cards("S7"), L7)   # T-ORD-03
    assert not beats((SINGLE, 12), (SINGLE, 12))
    assert P("8", L7) == 5 and P("6", L7) == 4                        # T-ORD-04
    assert P("A", LA) == 12 and P("K", LA) == 11                      # T-ORD-05
    assert [g.power(r, L7) for r in range(15)] == \
        [0, 1, 2, 3, 4, 12, 5, 6, 7, 8, 9, 10, 11, 13, 14]


def test_pairs():
    assert interpret(cards("H7 SA"), L7) == {(PAIR, P("A", L7))}      # T-PAIR-01
    assert interpret(cards("H7 SB"), L7) == set()                     # T-PAIR-02
    assert interpret(cards("SB HR"), L7) == set()                     # T-PAIR-03
    assert interpret(cards("SB SB"), L7) == {(PAIR, 13)}              # T-PAIR-04
    assert interpret(cards("HR HR"), L7) == {(PAIR, 14)}
    assert interpret(cards("H7 H7"), L7) == {(PAIR, 12)}              # T-PAIR-05
    assert interpret(cards("H7 S7"), L7) == {(PAIR, 12)}              # T-PAIR-06
    assert interpret(cards("S3 D4"), L7) == set()


def test_triples():
    assert interpret(cards("H7 H7 S9"), L7) == {(TRIPLE, P("9", L7))}  # T-TRI-01
    assert interpret(cards("SB SB H7"), L7) == set()                   # T-TRI-02
    assert interpret(cards("S9 D9 C9"), L7) == {(TRIPLE, 6)}


def test_full_house():
    assert interpret(cards("S5 D5 C5 SB SB"), L7) == {(FULL_HOUSE, P("5", L7))}  # T-FH-01
    assert interpret(cards("S5 D5 C5 SB H7"), L7) == set()                        # T-FH-02
    r = interpret(cards("S5 D5 H7 S9 D9"), L7)                                    # T-FH-03
    assert r == {(FULL_HOUSE, P("5", L7)), (FULL_HOUSE, P("9", L7))}
    assert best_readings(cards("S5 D5 H7 S9 D9"), L7) == {(FULL_HOUSE, P("9", L7))}
    r = interpret(cards("H7 H7 S9 D9 C9"), L7)                                    # T-FH-04
    assert r == {(BOMB, (5, P("9", L7))), (FULL_HOUSE, P("9", L7))}
    lvl = (FULL_HOUSE, 12)                                                        # T-FH-05
    assert interpret(cards("S7 D7 C7 S2 D2"), L7) == {lvl}
    assert beats(lvl, (FULL_HOUSE, P("A", L7)))
    assert interpret(cards("SB SB HR HR S3"), L7) == set()


def test_straights():
    assert interpret(cards("SA D2 C3 S4 H5"), L7) == {(STRAIGHT, 0)}   # T-STR-01
    assert interpret(cards("SK DA C2 S3 H4"), L7) == set()             # T-STR-02
    assert interpret(cards("ST DJ CQ SK HA"), L7) == {(STRAIGHT, 9)}   # T-STR-03
    assert interpret(cards("S5 D6 S7 C8 D9"), L7) == {(STRAIGHT, 4)}   # T-STR-04
    assert interpret(cards("SA S2 S3 S4 S5"), L7) == {(STRAIGHT_FLUSH, 0)}  # T-STR-05
    r = interpret(cards("H2 S6 D7 C8 D9"), L2)                         # T-STR-06
    assert r == {(STRAIGHT, 4), (STRAIGHT, 5)}
    assert best_readings(cards("H2 S6 D7 C8 D9"), L2) == {(STRAIGHT, 5)}
    assert interpret(cards("S3 S4 S5 S6 S7 S8"), L2) == set()          # T-STR-07
    assert interpret(cards("ST SJ SQ SK SB"), L2) == set()             # T-STR-08
    r = best_readings(cards("H7 S3 S4 S5 S6"), L7)                     # T-STR-09
    assert r == {(STRAIGHT, 2), (STRAIGHT_FLUSH, 2)}
    r = interpret(cards("HT HJ HQ HK HA"), LA)                         # T-STR-10
    assert r == {(STRAIGHT, 8), (STRAIGHT, 9), (STRAIGHT_FLUSH, 8), (STRAIGHT_FLUSH, 9)}
    assert (STRAIGHT_FLUSH, 9) in best_readings(cards("HT HJ HQ HK HA"), LA)


def test_tubes_and_plates():
    assert interpret(cards("SQ DQ SK DK SA DA"), L7) == {(TUBE, 11)}   # T-SEQ-01
    assert interpret(cards("SK DK SA DA S2 D2"), L7) == set()          # T-SEQ-02
    assert interpret(cards("SA DA S2 D2 S3 D3"), L7) == {(TUBE, 0)}    # T-SEQ-03
    assert interpret(cards("SK DK CK SA DA CA"), L7) == {(PLATE, 12)}  # T-SEQ-04
    assert interpret(cards("SA DA CA S2 D2 C2"), L7) == {(PLATE, 0)}
    r = interpret(cards("H7 H7 S3 D3 S4 D4"), L7)                      # T-SEQ-05
    assert r == {(TUBE, 1), (TUBE, 2), (PLATE, 2)}
    assert best_readings(cards("H7 H7 S3 D3 S4 D4"), L7) == {(TUBE, 2), (PLATE, 2)}
    assert interpret(cards("S6 D6 S7 D7 S8 D8"), L7) == {(TUBE, 5)}    # level at natural rank


def test_bombs():
    eight = cards("S7 S7 D7 D7 C7 C7 H7 H7")                           # T-BOMB-01
    assert interpret(eight, L7) == {(BOMB, (8, 12))}
    ten = cards("S9 S9 D9 D9 C9 C9 H9 H9 H7 H7")                       # T-BOMB-02
    assert interpret(ten, L7) == {(BOMB, (10, P("9", L7)))}
    assert interpret(ten + cards("S3"), L7) == set()
    assert beats((BOMB, (5, 0)), (BOMB, (4, 11)))                      # T-BOMB-03
    assert beats((STRAIGHT_FLUSH, 0), (BOMB, (5, 12)))                 # T-BOMB-04
    assert beats((BOMB, (6, 0)), (STRAIGHT_FLUSH, 9))
    assert not beats((STRAIGHT_FLUSH, 9), (BOMB, (6, 0)))
    assert beats((JOKER_BOMB, 0), (BOMB, (10, 11)))                    # T-BOMB-05
    assert not beats((BOMB, (10, 11)), (JOKER_BOMB, 0))
    assert interpret(cards("SB SB HR"), L7) == set()                   # T-BOMB-06
    assert interpret(cards("SB SB HR HR"), L7) == {(JOKER_BOMB, 0)}
    assert interpret(cards("SB SB HR H7"), L7) == set()
    assert beats((BOMB, (4, 0)), (FULL_HOUSE, 12))                     # T-BOMB-07
    assert not beats((STRAIGHT_FLUSH, 3), (STRAIGHT_FLUSH, 3))         # T-BOMB-08
    assert interpret(cards("H7 S4 D4 C4"), L7) == {(BOMB, (4, P("4", L7)))}


def test_type_mismatch():
    assert not beats((PAIR, 14), (SINGLE, 0))                          # T-MIS-01
    assert not beats((TUBE, 11), (PLATE, 0))
    assert not beats((STRAIGHT, 9), (FULL_HOUSE, 0))
    assert not beats((PASS, 0), None)                                  # cannot pass on lead
    assert beats((PASS, 0), (SINGLE, 14))


def test_legal_actions_small_hand():
    hand = cards("S3 D3 H7 SB")                                        # T-LEG-01
    got = g.legal_actions(hand, L7, (PAIR, 0))
    want = {(PASS, 0, ()),
            (PAIR, 1, tuple(sorted(cards("S3 D3")))),
            (PAIR, 1, tuple(sorted(cards("S3 H7")))),
            (PAIR, 1, tuple(sorted(cards("D3 H7"))))}
    assert got == want
    lead = g.legal_actions(hand, L7, None)                             # T-LEG-02
    assert all(a[0] != PASS for a in lead)
    kinds = {(a[0], a[1]) for a in lead}
    assert kinds == {(SINGLE, 1), (SINGLE, 12), (SINGLE, 13), (PAIR, 1), (TRIPLE, 1)}


def test_round_bookkeeping():
    assert g.level_gain([0, 2, 1, 3]) == (0, 3)                        # T-RND-01
    assert g.level_gain([0, 1, 2, 3]) == (0, 2)
    assert g.level_gain([0, 1, 3, 2]) == (0, 1)
    assert g.level_gain([1, 0, 3, 2]) == (1, 2)
    assert g.promote(11, 3) == 12 and g.promote(10, 3) == 12           # T-RND-02
    assert g.promote(12, 1) == 12 and g.promote(0, 3) == 3


def test_tribute():
    assert g.tribute_choices(cards("H7 S7 SA DA"), L7) == set(cards("S7"))      # T-TRB-01
    assert g.tribute_choices(cards("H7 SA DA SK"), L7) == set(cards("SA DA"))   # T-TRB-02
    assert g.tribute_choices(cards("HR SB SA"), L7) == set(cards("HR"))         # T-TRB-03
    assert g.back_tribute_choices(cards("S5 S3 DT SJ"), L5) == set(cards("S3 DT"))  # T-TRB-04
    assert g.back_tribute_choices(cards("SJ SQ S5"), L5) == set(cards("SJ"))    # fallback


def test_double_tribute_pairing():
    assert g.double_tribute(0, {1: 13, 3: 13}) == (3, 1, 3)            # T-TRB-05 tie: upstream
    assert g.double_tribute(0, {1: 14, 3: 13}) == (1, 3, 1)            # T-TRB-06
    assert g.double_tribute(0, {1: 11, 3: 12}) == (3, 1, 3)
    assert g.double_tribute(1, {2: 12, 0: 12}) == (0, 2, 0)            # banker 1, upstream is 0


def test_match_bookkeeping():
    A, K = 12, 11
    # T-MATCH-01: both at A, team 1 owns the round, team 0 double wins: no match win
    assert g.end_of_round([A, A], [0, 0], 1, A, [0, 2, 1, 3]) == ([A, A], [0, 1], 0, None)
    # T-MATCH-02: owner at A, Banker plus Third: match won
    assert g.end_of_round([A, 5], [0, 0], 0, A, [0, 1, 2, 3])[3] == 0
    # T-MATCH-03: owner at A, Banker plus Dweller: failure, stays A, keeps ownership
    assert g.end_of_round([A, 5], [0, 0], 0, A, [0, 1, 3, 2]) == ([A, 5], [1, 0], 0, None)
    # T-MATCH-04: third failure by losing the round: back to 2, opponents promote
    assert g.end_of_round([A, 5], [2, 0], 0, A, [1, 3, 0, 2]) == ([0, 8], [0, 0], 1, None)
    # T-MATCH-05: team at A wins a round it does not own and that is not at A
    assert g.end_of_round([A, K], [1, 0], 1, K, [0, 2, 1, 3]) == ([A, K], [1, 0], 0, None)
    # T-MATCH-06: third failure while winning the round with Banker plus Dweller
    assert g.end_of_round([A, 5], [2, 0], 0, A, [0, 1, 3, 2]) == ([0, 5], [0, 0], 0, None)
    # round 1 has no owner
    assert g.end_of_round([0, 0], [0, 0], None, 0, [2, 0, 1, 3]) == ([3, 0], [0, 0], 0, None)


def test_counts():
    assert g.abstract_action_count() == 393
    assert (len(g.STRAIGHT_WINDOWS), len(g.TUBE_WINDOWS), len(g.PLATE_WINDOWS)) == (10, 12, 13)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"{len(fns)} test groups passed")
