"""Guandan rules oracle: slow, brute force, independent of the C++ engine.

Normative companion to RULES.md. Pure Python, no dependencies.
Use it only in tests: as the validator for every action the engine
generates, and as the completeness oracle on small hands.
"""
from __future__ import annotations

from itertools import combinations, product

RANKS = "23456789TJQKA"          # rank index 0..12
SUITS = "SHCD"                   # suit index 0..3, hearts = 1
BJ, RJ = 52, 53                  # black (small) joker, red (big) joker
HEARTS = 1

SINGLE, PAIR, TRIPLE, FULL_HOUSE, STRAIGHT, TUBE, PLATE = (
    "Single", "Pair", "Triple", "FullHouse", "Straight", "Tube", "Plate")
BOMB, STRAIGHT_FLUSH, JOKER_BOMB, PASS = (
    "Bomb", "StraightFlush", "JokerBomb", "Pass")
BOMB_CLASS = {BOMB, STRAIGHT_FLUSH, JOKER_BOMB}

# Sequence windows in natural order. Index 0 is always the ace-low window.
STRAIGHT_WINDOWS = [[12, 0, 1, 2, 3]] + [list(range(s, s + 5)) for s in range(9)]
TUBE_WINDOWS = [[12, 0, 1]] + [list(range(s, s + 3)) for s in range(11)]
PLATE_WINDOWS = [[12, 0]] + [list(range(s, s + 2)) for s in range(12)]


def cid(text: str) -> int:
    """'S2' -> 0, 'HT' -> hearts ten, 'SB' -> black joker, 'HR' -> red joker."""
    if text == "SB":
        return BJ
    if text == "HR":
        return RJ
    return RANKS.index(text[1]) * 4 + SUITS.index(text[0])


def cstr(card: int) -> str:
    if card == BJ:
        return "SB"
    if card == RJ:
        return "HR"
    return SUITS[card % 4] + RANKS[card // 4]


def cards(text: str) -> list[int]:
    return [cid(t) for t in text.split()]


def rank_of(card: int) -> int:
    """Natural rank index. Jokers map to 13 and 14."""
    return card // 4 if card < 52 else card - 39


def wild_id(level: int) -> int:
    return level * 4 + HEARTS


def power(rank: int, level: int) -> int:
    """Strength order for singles, pairs, triples, full houses and bombs."""
    if rank >= 13:
        return rank                      # BJ = 13, RJ = 14
    if rank == level:
        return 12
    return rank if rank < level else rank - 1


def _classify_plain(cs: list[int], level: int) -> set[tuple]:
    """Classify concrete cards with no wild semantics left."""
    n = len(cs)
    ranks = sorted(rank_of(c) for c in cs)
    distinct = sorted(set(ranks))
    counts = {r: ranks.count(r) for r in distinct}
    out: set[tuple] = set()
    same = len(distinct) == 1
    no_joker = all(c < 52 for c in cs)
    if n == 1:
        out.add((SINGLE, power(ranks[0], level)))
    elif n == 2 and same:
        out.add((PAIR, power(ranks[0], level)))
    elif n == 3 and same and no_joker:
        out.add((TRIPLE, power(ranks[0], level)))
    if n == 4 and sorted(cs) == [BJ, BJ, RJ, RJ]:
        out.add((JOKER_BOMB, 0))
    if n >= 4 and same and no_joker:
        out.add((BOMB, (n, power(ranks[0], level))))
    if n == 5 and sorted(counts.values()) == [2, 3]:
        triple = next(r for r, k in counts.items() if k == 3)
        if triple < 13:
            out.add((FULL_HOUSE, power(triple, level)))
    if n == 5 and no_joker and len(distinct) == 5:
        for idx, window in enumerate(STRAIGHT_WINDOWS):
            if sorted(window) == distinct:
                flush = len({c % 4 for c in cs}) == 1
                out.add((STRAIGHT_FLUSH if flush else STRAIGHT, idx))
    if n == 6 and no_joker:
        for idx, window in enumerate(TUBE_WINDOWS):
            if sorted(window) == distinct and all(k == 2 for k in counts.values()):
                out.add((TUBE, idx))
        for idx, window in enumerate(PLATE_WINDOWS):
            if sorted(window) == distinct and all(k == 3 for k in counts.values()):
                out.add((PLATE, idx))
    return out


def interpret(cs: list[int], level: int) -> set[tuple]:
    """All (type, key) readings of a card multiset at the given round level."""
    if not cs:
        return {(PASS, 0)}
    if len(cs) > 10:
        return set()
    wid = wild_id(level)
    naturals = [c for c in cs if c != wid]
    wilds = len(cs) - len(naturals)
    if not naturals:                     # only wild cards: read at level rank
        return _classify_plain(list(cs), level)
    out: set[tuple] = set()
    for sub in product(range(52), repeat=wilds):
        out |= _classify_plain(naturals + list(sub), level)
    return out


def strength(kind: str, key) -> tuple:
    """Total order inside the bomb class."""
    if kind == JOKER_BOMB:
        return (99, 0)
    if kind == STRAIGHT_FLUSH:
        return (5.5, key)
    return (key[0], key[1])


def beats(cand: tuple, top: tuple | None) -> bool:
    """True if cand may be played over top. top None means leading."""
    kind, key = cand
    if top is None:
        return kind != PASS
    if kind == PASS:
        return True
    tkind, tkey = top
    if kind in BOMB_CLASS:
        if tkind in BOMB_CLASS:
            return strength(kind, key) > strength(tkind, tkey)
        return True
    return kind == tkind and key > tkey


def best_readings(cs: list[int], level: int) -> set[tuple]:
    """Canonical mode: keep only the highest key per type."""
    best: dict[str, tuple] = {}
    for kind, key in interpret(cs, level):
        skey = strength(kind, key) if kind in BOMB_CLASS else key
        if kind not in best or skey > best[kind][0]:
            best[kind] = (skey, key)
    return {(kind, v[1]) for kind, v in best.items()}


def legal_actions(hand: list[int], level: int, top: tuple | None,
                  canonical: bool = False) -> set[tuple]:
    """Brute force over every sub-multiset. Only for hands of about 14 cards
    or fewer. Returns {(type, key, sorted card tuple)}."""
    out: set[tuple] = set()
    if top is not None:
        out.add((PASS, 0, ()))
    seen: set[tuple] = set()
    read = best_readings if canonical else interpret
    for n in range(1, min(10, len(hand)) + 1):
        for combo in combinations(sorted(hand), n):
            if combo in seen:
                continue
            seen.add(combo)
            for kind, key in read(list(combo), level):
                if beats((kind, key), top):
                    out.add((kind, key, combo))
    return out


# ---- round bookkeeping helpers -------------------------------------------

def level_gain(order: list[int]) -> tuple[int, int]:
    """order lists seats by finish position. Returns (winning team, gain)."""
    banker = order[0]
    partner = (banker + 2) % 4
    return banker % 2, {1: 3, 2: 2, 3: 1}[order.index(partner)]


def promote(level: int, gain: int) -> int:
    return min(level + gain, 12)         # ace cannot be skipped


def tribute_choices(hand: list[int], level: int) -> set[int]:
    """Distinct cards a payer may give: top power, wild cards excluded."""
    wid = wild_id(level)
    pool = [c for c in hand if c != wid]
    top = max(power(rank_of(c), level) for c in pool)
    return {c for c in pool if power(rank_of(c), level) == top}


def back_tribute_choices(hand: list[int], level: int) -> set[int]:
    """Default reading of 'rank not above ten': natural rank 2..10 and not a
    level card. Falls back to the weakest cards if nothing qualifies."""
    ok = {c for c in hand if c < 52 and rank_of(c) <= 8 and rank_of(c) != level}
    if ok:
        return ok
    low = min(power(rank_of(c), level) for c in hand)
    return {c for c in hand if power(rank_of(c), level) == low}


def double_tribute(banker: int, tribute_power: dict[int, int],
                   tie: str = "downstream") -> tuple[int, int, int]:
    """Double tribute pairing (RULES.md 9.2). tribute_power maps each losing
    seat to the power of the card it gives. Returns (payer to Banker, payer to
    Follower, leader). Higher card goes to the Banker. On a tie, tribute goes
    clockwise: each loser pays the seat that plays right before it, so the
    Banker's downstream seat (B + 1) % 4 pays the Banker. tie="upstream" is the
    optional older rule where (B + 3) % 4 pays the Banker. The payer to the
    Banker always leads."""
    if tie not in ("downstream", "upstream"):
        raise ValueError(tie)
    down, up = (banker + 1) % 4, (banker + 3) % 4
    if tribute_power[down] > tribute_power[up]:
        return down, up, down
    if tribute_power[up] > tribute_power[down] or tie == "upstream":
        return up, down, up
    return down, up, down                # tie, clockwise


def end_of_round(levels: list[int], fails: list[int], owner: int | None,
                 round_level: int, order: list[int],
                 a_fail_limit: int = 0) -> tuple[list[int], list[int], int, int | None]:
    """Match bookkeeping after a round. owner is the team whose level the round
    was played at (None in round 1). A team can pass A only in a round it owns.
    Every owned A round that does not pass counts as a failure. The house rule
    has no reset (a_fail_limit=0); with a_fail_limit=3, the optional variant,
    the owner drops to the deuce when its count reaches the limit. Returns
    (levels, fails, next owner, match winner or None)."""
    team, gain = level_gain(order)
    levels, fails = list(levels), list(fails)
    partner_is_dweller = order.index((order[0] + 2) % 4) == 3
    winner = None
    if owner is not None and round_level == 12 and levels[owner] == 12:
        if team == owner and not partner_is_dweller:
            winner = owner
        else:
            fails[owner] += 1
    if winner is None:
        levels[team] = promote(levels[team], gain)
        if owner is not None and a_fail_limit and fails[owner] >= a_fail_limit:
            levels[owner], fails[owner] = 0, 0
    return levels, fails, team, winner


def abstract_action_count() -> int:
    single, pair, triple = 15, 15, 13
    full_house = 13 * 14                 # triple rank x pair rank or joker pair
    bombs = 13 * 7                       # sizes 4..10
    flushes = len(STRAIGHT_WINDOWS) * 4
    return (1 + single + pair + triple + full_house + len(STRAIGHT_WINDOWS)
            + len(TUBE_WINDOWS) + len(PLATE_WINDOWS) + bombs + flushes + 1)
