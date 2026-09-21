"""M0 task 7: the in-engine bots, and is_legal.

The bots bootstrap self-play out of random play (DESIGN.md 8.2 item 4) and
drive the fuzzer, so what matters here is that they always return a legal
choice and that the greedy one actually follows its stated policy.
"""
import random

import gd
import gd_reference as g


def _drive(engine, m, pick, limit=100000):
    steps = 0
    while m.winner < 0 and steps < limit:
        if m.phase == gd.Phase.RoundEnd:
            engine.end_round(m)
            if m.winner >= 0:
                break
            engine.begin_round(m)
            continue
        acts = engine.legal_actions(m)
        assert acts
        i = pick(m, acts)
        assert 0 <= i < len(acts), (i, len(acts))
        engine.apply(m, acts[i])
        steps += 1
    return steps


def test_greedy_bot_finishes_matches_and_stays_legal():
    engine = gd.Engine()
    for seed in range(20):
        m = gd.MatchState()
        engine.new_match(m, seed)
        _drive(engine, m, lambda st, acts: engine.greedy(st, 0))
        assert 0 <= m.winner <= 1
        assert m.levels[m.winner] == 12


def test_greedy_beats_random_over_many_matches():
    """Not a strength claim, just a sanity check that the heuristic is not
    worse than random: greedy seats should take more Banker finishes."""
    engine = gd.Engine()
    rng = random.Random(0)
    greedy_first = 0
    rounds = 0
    for seed in range(60):
        m = gd.MatchState()
        engine.new_match(m, seed)
        while m.winner < 0 and rounds < 4000:
            if m.phase == gd.Phase.RoundEnd:
                r = engine.end_round(m)
                rounds += 1
                if r.order[0] % 2 == 0:
                    greedy_first += 1
                if m.winner >= 0:
                    break
                engine.begin_round(m)
                continue
            acts = engine.legal_actions(m)
            # Team 0 plays greedily, team 1 at random.
            if m.to_move % 2 == 0:
                engine.apply(m, acts[engine.greedy(m, 0)])
            else:
                engine.apply(m, acts[rng.randrange(len(acts))])
    assert rounds > 200
    assert greedy_first / rounds > 0.5, greedy_first / rounds


def test_greedy_does_not_overtake_its_own_partner():
    """Partner holds the trick and nobody has to beat it: pass."""
    e = gd.Engine()
    d = gd.DealSpec()
    # Level A, so the heart ace is wild and the scripted hands avoid aces.
    d.hands = [g.cards("S3 D3 S7 S8"), g.cards("S4 D4 C7 C8"),
               g.cards("S6 D6 D7 D8"), g.cards("SK DK H7 H8")]
    d.level = 12
    d.team_levels = [12, 5]
    d.owner = 0
    d.leader = 0
    m = gd.MatchState()
    e.set_deal(m, d)
    # Seat 0 leads a pair of threes, seat 1 passes, seat 2 is seat 0's partner.
    acts = e.legal_actions(m)
    pair = next(a for a in acts if tuple(sorted(a.cards)) == tuple(sorted(g.cards("S3 D3"))))
    e.apply(m, pair)
    e.apply(m, next(a for a in e.legal_actions(m) if a.is_pass))
    assert m.to_move == 2
    choice = e.legal_actions(m)[e.greedy(m, 0)]
    assert choice.is_pass, "the partner holds the trick, so greedy lets it stand"


def test_greedy_hoards_bombs_when_nothing_is_at_stake():
    e = gd.Engine()
    d = gd.DealSpec()
    d.hands = [g.cards("S3 D3 C3 H3 S7 S8 S9"), g.cards("S4 D4 C7 C8 C9 CT CJ"),
               g.cards("S6 D6 D7 D8 D9 DT DJ"), g.cards("SK DK H7 H8 H9 HT HJ")]
    d.level = 12
    d.team_levels = [12, 5]
    d.owner = 0
    d.leader = 3
    m = gd.MatchState()
    e.set_deal(m, d)
    e.apply(m, next(a for a in e.legal_actions(m)
                    if tuple(sorted(a.cards)) == tuple(sorted(g.cards("SK DK")))))
    e.apply(m, next(a for a in e.legal_actions(m) if a.is_pass))
    assert m.to_move == 1 or m.to_move == 2
    # Whoever is to move, seat 0 holds a four-bomb of threes and nobody is
    # close to going out, so it must not be spent.
    while m.to_move != 0 and m.phase == gd.Phase.Play:
        e.apply(m, e.legal_actions(m)[e.greedy(m, 0)])
    if m.phase == gd.Phase.Play and m.to_move == 0:
        choice = e.legal_actions(m)[e.greedy(m, 0)]
        assert choice.type != "Bomb", "no threat on the table, so keep the bomb"


def test_random_bot_index_is_in_range():
    env = gd.VecEnv(num_envs=8, num_threads=1, seed=3)
    env.reset()
    for _ in range(200):
        b = env.pending()
        n = b.offsets.shape[0] - 1
        if n == 0:
            break
        import numpy as np
        widths = np.diff(b.offsets)
        env.step(np.zeros(n, dtype=np.int32))
        assert np.all(widths >= 1)


def test_is_legal_agrees_with_generation():
    rng = random.Random(12)
    pool = [c for c in range(54) for _ in range(2)]
    for _ in range(300):
        level = rng.randrange(13)
        rng.shuffle(pool)
        hand = sorted(pool[:rng.randint(1, 12)])
        top = None if rng.random() < 0.3 else (g.SINGLE, rng.randrange(15))
        acts = gd.legal_actions(hand, level, top, canonical=False)
        for a in acts:
            assert gd.is_legal(hand, level, top, a), a
        # A play of cards we do not hold is never legal.
        missing = [c for c in range(54) if c not in hand]
        if missing:
            bogus = (g.SINGLE, gd.power(missing[0] // 4 if missing[0] < 52
                                        else missing[0] - 39, level), (missing[0],))
            assert not gd.is_legal(hand, level, top, bogus)


def test_is_legal_rejects_a_pass_on_a_lead():
    hand = g.cards("S3 D3 S7")
    assert not gd.is_legal(hand, 5, None, (g.PASS, 0, ()))
    assert gd.is_legal(hand, 5, (g.SINGLE, 0), (g.PASS, 0, ()))
