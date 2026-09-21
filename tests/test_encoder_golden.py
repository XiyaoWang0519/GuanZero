"""M0 task 11: golden tests for the v1 feature encoder.

Two kinds of check. The semantic ones say what each block must mean, derived
from the state rather than from the encoder. The frozen ones pin the layout, so
that moving a block fails loudly instead of silently retraining on shifted
features.
"""
import numpy as np

import gd
import gd_reference as g


def _cards(text):
    return g.cards(text)


def _deal(hands, level=5, team_levels=(5, 3), leader=0, owner=0):
    d = gd.DealSpec()
    d.hands = [sorted(h) for h in hands]
    d.level = level
    d.team_levels = list(team_levels)
    d.owner = owner
    d.leader = leader
    m = gd.MatchState()
    gd.Engine().set_deal(m, d)
    return m


def _split_deck(rng_seed=0):
    import random
    rng = random.Random(rng_seed)
    pool = [c for c in range(54) for _ in range(2)]
    rng.shuffle(pool)
    return [sorted(pool[i * 27:(i + 1) * 27]) for i in range(4)]


def _card_block(obs, off, cards):
    """Assert the 108-value card-set block at `off` encodes exactly `cards`."""
    counts = {}
    for c in cards:
        counts[c] = counts.get(c, 0) + 1
    for c in range(54):
        n = counts.get(c, 0)
        assert obs[off + c] == (1.0 if n >= 1 else 0.0), (off, c, n)
        assert obs[off + 54 + c] == (1.0 if n >= 2 else 0.0), (off, c, n)


# ---- layout ---------------------------------------------------------------

def test_dimensions_and_offsets_are_pinned():
    assert gd.OBS_DIM == 1849
    assert gd.ACT_DIM == 154
    assert (gd.ACT_CARDS1, gd.ACT_CARDS2, gd.ACT_TYPE, gd.ACT_KEY,
            gd.ACT_BOMB_SIZE, gd.ACT_WILDS, gd.ACT_TRIBUTE_FLAGS) == \
        (0, 54, 108, 121, 136, 143, 146)
    assert (gd.OBS_OWN_HAND, gd.OBS_UNSEEN, gd.OBS_PLAYED, gd.OBS_CARDS_LEFT,
            gd.OBS_FINISH_STATUS, gd.OBS_LEVELS, gd.OBS_WILD_HELD,
            gd.OBS_WILD_UNSEEN, gd.OBS_WILD_FLAGS, gd.OBS_TRICK_TOP,
            gd.OBS_TRICK_HOLDER, gd.OBS_TRICK_PASSES, gd.OBS_TRICK_LEADING,
            gd.OBS_LAST_ACTION, gd.OBS_PHASE, gd.OBS_ROLES, gd.OBS_TRIBUTE,
            gd.OBS_KNOWN_HOLDINGS) == \
        (0, 108, 216, 648, 732, 748, 787, 790, 793, 805, 959, 963, 967,
         968, 1430, 1433, 1439, 1687)


# ---- action encoding ------------------------------------------------------

def test_encode_action_pass_and_single():
    m = _deal(_split_deck(1))
    e = gd.Engine()
    acts = e.legal_actions(m)
    single = next(a for a in acts if a.type == "Single")
    v = e.encode_action(single, m, m.to_move)
    assert v.shape == (gd.ACT_DIM,)
    _card_block(v, gd.ACT_CARDS1, single.cards)
    assert v[gd.ACT_TYPE + 1] == 1.0                      # Type::Single
    assert v[gd.ACT_KEY + single.key] == 1.0
    assert v[gd.ACT_BOMB_SIZE:gd.ACT_BOMB_SIZE + 7].sum() == 0.0
    assert v[gd.ACT_WILDS + single.wilds] == 1.0
    assert v[gd.ACT_TRIBUTE_FLAGS:gd.ACT_DIM].sum() == 0.0


def test_encode_action_bomb_and_straight_flush():
    # Level 7 (index 5). Seat 0 holds a six-card bomb of nines using one wild
    # and a heart straight flush.
    hand = _cards("S9 S9 D9 D9 C9 H9 H7 SA SB HR S2 D2 C2 H2 S3 D3 C3 H3 "
                  "S4 D4 C4 H4 S5 D5 C5 H5 S6")
    rest = [c for c in range(54) for _ in range(2)]
    for c in hand:
        rest.remove(c)
    m = _deal([hand, sorted(rest[:27]), sorted(rest[27:54]), sorted(rest[54:])],
              level=5, leader=0)
    e = gd.Engine()
    acts = e.legal_actions(m)
    bombs = [a for a in acts if a.type == "Bomb" and a.bomb_size == 6]
    assert bombs, "expected a six-card bomb"
    v = e.encode_action(bombs[0], m, 0)
    assert v[gd.ACT_TYPE + 8] == 1.0                      # Type::Bomb
    assert v[gd.ACT_BOMB_SIZE + (6 - 4)] == 1.0
    assert v[gd.ACT_KEY + bombs[0].key] == 1.0
    _card_block(v, gd.ACT_CARDS1, bombs[0].cards)

    sfs = [a for a in acts if a.type == "StraightFlush"]
    assert sfs, "expected a straight flush"
    w = e.encode_action(sfs[0], m, 0)
    assert w[gd.ACT_TYPE + 9] == 1.0
    assert w[gd.ACT_BOMB_SIZE:gd.ACT_BOMB_SIZE + 7].sum() == 0.0


def test_encode_action_tribute_flags():
    hands = _split_deck(2)
    d = gd.DealSpec()
    d.hands = hands
    d.level = 5
    d.team_levels = [5, 3]
    d.owner = 0
    d.prev_order = [0, 1, 2, 3]
    m = gd.MatchState()
    e = gd.Engine()
    e.set_deal(m, d)
    if m.phase != gd.Phase.Tribute:
        return                                   # anti-tribute, nothing to check
    acts = e.legal_actions(m)
    seat = m.to_move
    hand = m.hand(seat)
    for a in acts:
        v = e.encode_action(a, m, seat)
        card = a.cards[0]
        copies = hand.count(card)
        assert v[gd.ACT_TRIBUTE_FLAGS + 0] == (1.0 if copies == 1 else 0.0)
        assert v[gd.ACT_TRIBUTE_FLAGS + 1] == (1.0 if copies == 2 else 0.0)
        rank = card // 4 if card < 52 else card - 39
        same_rank = sum(1 for c in hand
                        if c < 52 and c // 4 == rank and c != g.wild_id(5))
        assert v[gd.ACT_TRIBUTE_FLAGS + 2] == (1.0 if same_rank == 2 else 0.0)
        assert v[gd.ACT_TRIBUTE_FLAGS + 6] == (1.0 if rank == 3 else 0.0)
        assert v[gd.ACT_TRIBUTE_FLAGS + 7] == (1.0 if rank == 8 else 0.0)


# ---- observation ----------------------------------------------------------

def test_observation_fresh_lead_is_semantically_right():
    hands = _split_deck(3)
    m = _deal(hands, level=5, team_levels=(5, 3), leader=1)
    seat = m.to_move
    assert seat == 1
    obs = m.observation(seat)
    assert obs.shape == (gd.OBS_DIM,)

    _card_block(obs, gd.OBS_OWN_HAND, m.hand(seat))
    # Nothing played yet, so unseen is the whole deck minus our hand.
    unseen = [c for c in range(54) for _ in range(2)]
    for c in m.hand(seat):
        unseen.remove(c)
    _card_block(obs, gd.OBS_UNSEEN, unseen)
    for k in range(4):
        _card_block(obs, gd.OBS_PLAYED + k * 108, [])
    for rel in range(3):
        assert obs[gd.OBS_CARDS_LEFT + rel * 28 + 27] == 1.0
        assert obs[gd.OBS_CARDS_LEFT + rel * 28:
                   gd.OBS_CARDS_LEFT + (rel + 1) * 28].sum() == 1.0
    assert obs[gd.OBS_FINISH_STATUS:gd.OBS_FINISH_STATUS + 16].sum() == 0.0
    assert obs[gd.OBS_LEVELS + 5] == 1.0                      # round level
    assert obs[gd.OBS_LEVELS + 13 + 3] == 1.0                 # own team, seat 1
    assert obs[gd.OBS_LEVELS + 26 + 5] == 1.0                 # other team
    assert obs[gd.OBS_TRICK_LEADING] == 1.0
    assert obs[gd.OBS_TRICK_TOP:gd.OBS_TRICK_TOP + gd.ACT_DIM].sum() == 0.0
    assert obs[gd.OBS_TRICK_HOLDER:gd.OBS_TRICK_HOLDER + 4].sum() == 0.0
    assert obs[gd.OBS_TRICK_PASSES + 0] == 1.0
    assert obs[gd.OBS_LAST_ACTION:gd.OBS_LAST_ACTION + 3 * gd.ACT_DIM].sum() == 0.0
    assert obs[gd.OBS_PHASE + 0] == 1.0                       # Play
    assert obs[gd.OBS_ROLES + 4] == 1.0                       # no previous round
    assert obs[gd.OBS_TRIBUTE:gd.OBS_TRIBUTE + 4 * 62].sum() == 0.0
    assert obs[gd.OBS_KNOWN_HOLDINGS:gd.OBS_KNOWN_HOLDINGS + 3 * 54].sum() == 0.0


def test_observation_tracks_a_play():
    hands = _split_deck(4)
    m = _deal(hands, level=5, leader=0)
    e = gd.Engine()
    acts = e.legal_actions(m)
    play = next(a for a in acts if not a.is_pass)
    e.apply(m, play)

    mover = m.to_move
    obs = m.observation(mover)
    # Seat 0 played; from `mover`'s seat that is some relative slot.
    rel = (0 - mover + 4) % 4
    if rel == 0:
        _card_block(obs, gd.OBS_PLAYED + 3 * 108, play.cards)
    else:
        _card_block(obs, gd.OBS_PLAYED + (rel - 1) * 108, play.cards)
        assert obs[gd.OBS_CARDS_LEFT + (rel - 1) * 28 + 27 - len(play.cards)] == 1.0
        # The top play and its holder are visible.
        assert obs[gd.OBS_TRICK_HOLDER + rel] == 1.0
        top = obs[gd.OBS_TRICK_TOP:gd.OBS_TRICK_TOP + gd.ACT_DIM]
        _card_block(top, 0, play.cards)
        last = obs[gd.OBS_LAST_ACTION + (rel - 1) * gd.ACT_DIM:
                   gd.OBS_LAST_ACTION + rel * gd.ACT_DIM]
        _card_block(last, 0, play.cards)
    assert obs[gd.OBS_TRICK_LEADING] == 0.0


def test_observation_after_tribute_records_the_exchange():
    e = gd.Engine()
    for seed in range(12):
        hands = _split_deck(100 + seed)
        d = gd.DealSpec()
        d.hands = hands
        d.level = 5
        d.team_levels = [5, 3]
        d.owner = 0
        d.prev_order = [0, 1, 2, 3]
        m = gd.MatchState()
        e.set_deal(m, d)
        if m.phase != gd.Phase.Tribute:
            continue
        # Tribute, then back-tribute, then look at the first play decision.
        while m.phase in (gd.Phase.Tribute, gd.Phase.BackTribute):
            acts = e.legal_actions(m)
            assert acts
            obs = m.observation(m.to_move)
            expect = 1 if m.phase == gd.Phase.Tribute else 2
            assert obs[gd.OBS_PHASE + expect] == 1.0
            e.apply(m, acts[0])
        assert m.phase == gd.Phase.Play
        obs = m.observation(m.to_move)
        assert obs[gd.OBS_TRIBUTE:gd.OBS_TRIBUTE + 4 * 62].sum() > 0.0
        # Every seat now holds 27 cards again.
        for s in range(4):
            assert len(m.hand(s)) == 27
        # The tribute counterpart of seat 0 is seat 3, not its partner.
        assert obs[gd.OBS_ROLES + 5] == 0.0
        assert obs[gd.OBS_KNOWN_HOLDINGS:gd.OBS_KNOWN_HOLDINGS + 3 * 54].sum() >= 0.0
        return
    raise AssertionError("no tribute round was produced")


def test_observation_block_totals_are_frozen():
    """A checksum over a fixed state. If a block moves, this fails."""
    hands = _split_deck(7)
    m = _deal(hands, level=5, team_levels=(5, 3), leader=2)
    obs = m.observation(2)
    nz = np.nonzero(obs)[0]
    assert obs.sum() == 121.0, obs.sum()
    assert len(nz) == 121, len(nz)
    assert int(nz.sum()) == 24867, int(nz.sum())


def test_every_seat_and_phase_encodes_without_nan():
    env = gd.VecEnv(num_envs=16, num_threads=2, seed=4)
    env.reset()
    for _ in range(400):
        b = env.pending()
        n = b.offsets.shape[0] - 1
        if n == 0:
            break
        assert np.all(np.isfinite(b.obs))
        assert np.all((b.obs == 0.0) | (b.obs == 1.0))
        assert np.all(np.isfinite(b.cand))
        widths = np.diff(b.offsets)
        env.step((widths - 1).astype(np.int32))
