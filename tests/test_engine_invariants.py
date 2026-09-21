"""M0 tasks 6 and 7: the state machine invariants of RULES.md section 14,
exercised through the bindings. The heavy 10^7-round sweep lives in the C++
fuzzer (cpp/fuzz); this suite is the portable version.
"""
import random

import gd
import gd_reference as g

N_MATCHES = 200


def _conservation(m):
    counts = [0] * 54
    for seat in range(4):
        for c in m.hand(seat):
            counts[c] += 1
        for c in m.played(seat):
            counts[c] += 1
    assert counts == [2] * 54


def _play_match(engine, m, rng, check=True):
    steps = 0
    round_steps = 0
    prev_levels = list(m.levels)
    while m.winner < 0 and steps < 100000:
        if m.phase == gd.Phase.RoundEnd:
            res = engine.end_round(m)
            assert sorted(res.order[:res.num_finished_seats]) == \
                sorted(set(res.order[:res.num_finished_seats]))
            if check:
                for t in (0, 1):
                    assert m.levels[t] >= prev_levels[t] or m.fails[t] == 0
                prev_levels = list(m.levels)
            round_steps = 0
            if m.winner >= 0:
                break
            engine.begin_round(m)
            continue
        acts = engine.legal_actions(m)
        assert acts, (m.phase, m.to_move)
        if check and m.phase == gd.Phase.Play:
            leading = m.top_is_open
            has_pass = any(a.type == "Pass" for a in acts)
            assert has_pass != leading
            _conservation(m)
            round_steps += 1
            assert round_steps < 600, "round did not terminate"
        engine.apply(m, acts[rng.randrange(len(acts))])
        steps += 1
    return steps


def test_random_matches_hold_every_invariant():
    engine = gd.Engine()
    rng = random.Random(3)
    for seed in range(N_MATCHES):
        m = gd.MatchState()
        engine.new_match(m, seed)
        _play_match(engine, m, rng)
        assert 0 <= m.winner <= 1
        assert m.levels[m.winner] == 12


def test_determinism_and_serialization_round_trip():
    engine = gd.Engine()
    for seed in (0, 1, 2, 17, 999):
        hashes = []
        for _ in range(2):
            rng = random.Random(seed)
            m = gd.MatchState()
            engine.new_match(m, seed)
            trace = []
            while m.winner < 0:
                if m.phase == gd.Phase.RoundEnd:
                    engine.end_round(m)
                    if m.winner >= 0:
                        break
                    engine.begin_round(m)
                    continue
                acts = engine.legal_actions(m)
                a = acts[rng.randrange(len(acts))]
                engine.apply(m, a)
                trace.append(m.hash())
                blob = m.serialize()
                assert gd.MatchState.deserialize(blob).hash() == m.hash()
            hashes.append(trace)
        assert hashes[0] == hashes[1]


def test_deal_spec_round_trip():
    engine = gd.Engine()
    rng = random.Random(4)
    pool = [c for c in range(54) for _ in range(2)]
    for _ in range(50):
        rng.shuffle(pool)
        d = gd.DealSpec()
        d.hands = [sorted(pool[i * 27:(i + 1) * 27]) for i in range(4)]
        d.level = rng.randrange(13)
        d.team_levels = [d.level, rng.randrange(13)]
        d.owner = 0
        d.leader = rng.randrange(4)
        m = gd.MatchState()
        engine.set_deal(m, d)
        assert m.phase == gd.Phase.Play
        assert m.to_move == d.leader
        for seat in range(4):
            assert sorted(m.hand(seat)) == d.hands[seat]
        _conservation(m)


def test_deal_spec_with_previous_order_starts_at_tribute():
    engine = gd.Engine()
    rng = random.Random(5)
    pool = [c for c in range(54) for _ in range(2)]
    seen_tribute = False
    for _ in range(50):
        rng.shuffle(pool)
        d = gd.DealSpec()
        d.hands = [sorted(pool[i * 27:(i + 1) * 27]) for i in range(4)]
        d.level = 5
        d.team_levels = [5, 3]
        d.owner = 0
        d.prev_order = [0, 1, 2, 3]
        m = gd.MatchState()
        engine.set_deal(m, d)
        # Either a tribute decision is pending or anti-tribute skipped it.
        assert m.phase in (gd.Phase.Tribute, gd.Phase.Play)
        seen_tribute |= m.phase == gd.Phase.Tribute
    assert seen_tribute


def test_level_gain_matches_oracle():
    for order in ([0, 2, 1, 3], [0, 1, 2, 3], [0, 1, 3, 2], [1, 0, 3, 2],
                  [2, 3, 0, 1], [3, 1, 2, 0]):
        assert gd.level_gain(order) == g.level_gain(order)


def test_end_of_round_matches_oracle():
    rng = random.Random(6)
    seats = [0, 1, 2, 3]
    for _ in range(20000):
        levels = [rng.randrange(13), rng.randrange(13)]
        fails = [rng.randrange(4), rng.randrange(4)]
        owner = rng.choice([None, 0, 1])
        round_level = levels[owner] if owner is not None else rng.randrange(13)
        order = seats[:]
        rng.shuffle(order)
        got = gd.end_of_round(levels, fails, owner, round_level, order)
        want = g.end_of_round(levels, fails, owner, round_level, order)
        assert got == want, (levels, fails, owner, round_level, order, got, want)
