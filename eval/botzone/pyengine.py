"""A pure-Python port of the parts of ``gd`` the Botzone bot needs.

Botzone runs Python 3.6 with NumPy and cannot load our C++ extension, so the
bot replays rounds with this module instead. It follows ``cpp/src`` line for
line, under the ``house`` rule profile with ``auto_pass`` off:

* ``make_view``, ``sf_relevant_mask``, the window tables (``cards.cpp``,
  ``movegen.cpp``);
* ``generate_moves`` in canonical and full mode, in gd's output order, plus
  ``generate_tribute`` and ``generate_back_tribute`` (``movegen.cpp``);
* the round state machine from ``set_deal`` through tribute, back-tribute and
  play (``state.cpp``);
* ``encode_action`` and ``encode_observation`` (``encoder.cpp``);
* the engine's tribute heuristic, ``tribute_bot`` (``bots.cpp``).

``tests/test_botzone_engine.py`` checks every piece against ``gd`` on random
deals and whole rounds. Hands are lists of 54 counts (gd card faces).
Python 3.6 compatible, NumPy only.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np

NUM_IDS = 54
BJ, RJ = 52, 53
HEARTS = 1

PASS, SINGLE, PAIR, TRIPLE, FULL_HOUSE, STRAIGHT, TUBE, PLATE, BOMB, STRAIGHT_FLUSH, \
    JOKER_BOMB, TRIBUTE, BACK_TRIBUTE = range(13)
NUM_TYPES = 13
TYPE_NAMES = ("Pass", "Single", "Pair", "Triple", "FullHouse", "Straight", "Tube", "Plate",
              "Bomb", "StraightFlush", "JokerBomb", "Tribute", "BackTribute")
TYPE_BY_NAME = {name: i for i, name in enumerate(TYPE_NAMES)}

PHASE_DEAL, PHASE_TRIBUTE, PHASE_BACK_TRIBUTE, PHASE_PLAY, PHASE_ROUND_END = range(5)

STRAIGHT_WINDOWS = [[12, 0, 1, 2, 3]] + [[w + i for i in range(5)] for w in range(9)]
TUBE_WINDOWS = [[12, 0, 1]] + [[w + i for i in range(3)] for w in range(11)]
PLATE_WINDOWS = [[12, 0]] + [[w + i for i in range(2)] for w in range(12)]


def _mask(ranks: Sequence[int]) -> int:
    m = 0
    for r in ranks:
        m |= 1 << r
    return m


STRAIGHT_MASKS = [_mask(w) for w in STRAIGHT_WINDOWS]
_SUIT_CARDS = [sum(1 << c for c in range(52) if c % 4 == s) for s in range(4)]

# Observation and action layout (gd/encoder.h).
ACT_TYPE, ACT_KEY, ACT_BOMB_SIZE, ACT_WILDS, ACT_TRIBUTE_FLAGS, ACT_DIM = 108, 121, 136, 143, 146, 154
OBS_UNSEEN, OBS_PLAYED, OBS_CARDS_LEFT, OBS_FINISH = 108, 216, 648, 732
OBS_LEVELS, OBS_WILD_HELD, OBS_WILD_UNSEEN, OBS_WILD_FLAGS = 748, 787, 790, 793
OBS_TRICK_TOP, OBS_TRICK_HOLDER, OBS_TRICK_PASSES, OBS_TRICK_LEADING = 805, 959, 963, 967
OBS_LAST_ACTION, OBS_PHASE, OBS_ROLES, OBS_TRIBUTE, OBS_KNOWN, OBS_DIM = \
    968, 1430, 1433, 1439, 1687, 1849


def popcount(x: int) -> int:
    return bin(x).count("1")


def rank_of(c: int) -> int:
    return c // 4 if c < 52 else c - 39


def suit_of(c: int) -> int:
    return c % 4 if c < 52 else -1


def wild_id(level: int) -> int:
    return level * 4 + HEARTS


def power(rank: int, level: int) -> int:
    if rank >= 13:
        return rank
    if rank == level:
        return 12
    return rank if rank < level else rank - 1


def rank_of_power(p: int, level: int) -> int:
    if p >= 13:
        return p
    if p == 12:
        return level
    return p if p < level else p + 1


def counts_of(faces: Sequence[int]) -> List[int]:
    counts = [0] * NUM_IDS
    for f in faces:
        counts[f] += 1
    return counts


def faces_of(counts: Sequence[int]) -> List[int]:
    out = []
    for c in range(NUM_IDS):
        out.extend([c] * counts[c])
    return out


def bits(counts: Sequence[int]) -> Tuple[int, int]:
    """gd's ``Hand`` words: (has1, has2)."""
    has1 = has2 = 0
    for c in range(NUM_IDS):
        if counts[c] >= 1:
            has1 |= 1 << c
        if counts[c] >= 2:
            has2 |= 1 << c
    return has1, has2


# ---- actions ---------------------------------------------------------------------

class Action(object):
    __slots__ = ("type", "key", "bomb_size", "wilds", "fh_pair", "counts", "_bits")

    def __init__(self, kind: int = PASS, key: int = 0, bomb_size: int = 0, wilds: int = 0,
                 fh_pair: int = -1, counts: Optional[Sequence[int]] = None) -> None:
        self.type = kind
        self.key = key
        self.bomb_size = bomb_size
        self.wilds = wilds
        self.fh_pair = fh_pair
        self.counts = tuple(counts) if counts is not None else (0,) * NUM_IDS
        self._bits = None

    @property
    def is_pass(self) -> bool:
        return self.type == PASS

    @property
    def type_name(self) -> str:
        return TYPE_NAMES[self.type]

    @property
    def cards(self) -> List[int]:
        return faces_of(self.counts)

    @property
    def size(self) -> int:
        return sum(self.counts)

    def bits(self) -> Tuple[int, int]:
        if self._bits is None:
            self._bits = bits(self.counts)
        return self._bits

    def __repr__(self) -> str:
        return "%s %d %r" % (TYPE_NAMES[self.type], self.key, self.cards)


PASS_ACTION = Action()


def is_bomb_class(kind: int) -> bool:
    return kind in (BOMB, STRAIGHT_FLUSH, JOKER_BOMB)


def strength(kind: int, key: int, bomb_size: int) -> Tuple[int, int]:
    if kind == JOKER_BOMB:
        return (999, 0)
    if kind == STRAIGHT_FLUSH:
        return (11, key)
    return (bomb_size * 2, key)


def beats_reading(ct: int, ckey: int, cbomb: int, tt: int, tkey: int, tbomb: int) -> bool:
    if tt == PASS:
        return ct != PASS
    if ct == PASS:
        return True
    cb, tb = is_bomb_class(ct), is_bomb_class(tt)
    if cb:
        if tb:
            return strength(tt, tkey, tbomb) < strength(ct, ckey, cbomb)
        return True
    if tb:
        return False
    return ct == tt and ckey > tkey


# ---- hand views ------------------------------------------------------------------

class HandView(object):
    __slots__ = ("rank_count", "card_count", "suit_rank_mask", "wilds", "size")

    def __init__(self, counts: Sequence[int], level: int) -> None:
        wid = wild_id(level)
        self.rank_count = [0] * 15
        self.card_count = list(counts)
        self.suit_rank_mask = [0] * 4
        self.wilds = 0
        for c in range(NUM_IDS):
            n = counts[c]
            if not n:
                continue
            if c == wid:
                self.wilds += n
            else:
                self.rank_count[rank_of(c)] += n
            if c < 52:
                self.suit_rank_mask[c % 4] |= 1 << rank_of(c)
        self.size = sum(counts)


def sf_relevant_mask(v: HandView) -> List[int]:
    out = []
    for s in range(4):
        nat = v.suit_rank_mask[s]
        acc = 0
        for win in STRAIGHT_MASKS:
            if 5 - popcount(nat & win) <= v.wilds:
                acc |= win
        out.append(acc & nat)
    return out


# ---- move generation (movegen.cpp) ---------------------------------------------------

class _RankCands(object):
    __slots__ = ("keep", "pool", "pool_total", "total")

    def __init__(self) -> None:
        self.keep = []       # type: List[Tuple[int, int]]
        self.pool = []       # type: List[Tuple[int, int]]
        self.pool_total = 0
        self.total = 0


class _Gen(object):
    def __init__(self, counts: Sequence[int], level: int, top: Action, canonical: bool,
                 full_house_joker_pair: bool = True) -> None:
        self.v = v = HandView(counts, level)
        self.level = level
        self.wid = wild_id(level)
        self.dedup = canonical
        self.minimal = canonical
        self.fh_joker_pair = full_house_joker_pair
        self.top = top
        self.leading = top.type == PASS
        self.out = []        # type: List[Action]
        self.nat_mask = list(v.suit_rank_mask)
        self.nat_mask[HEARTS] &= ~(1 << level)
        self.sf_mask = sf_relevant_mask(v)
        self.any_suit = [self._candidates_raw(r, -1) for r in range(15)]
        most = 0
        self.present = 0
        for r in range(13):
            most = max(most, v.rank_count[r])
            if v.rank_count[r]:
                self.present |= 1 << r
        self.max_same = most + v.wilds

    def want(self, kind: int, key: int, bomb_size: int) -> bool:
        t = self.top
        return beats_reading(kind, key, bomb_size, t.type, t.key, t.bomb_size)

    def _candidates_raw(self, rank: int, suit: int) -> _RankCands:
        c = _RankCands()
        v = self.v
        if rank >= 13:
            card = BJ if rank == 13 else RJ
            if v.card_count[card]:
                c.keep.append((card, v.card_count[card]))
                c.total = v.card_count[card]
            return c
        for s in range(4):
            if suit >= 0 and s != suit:
                continue
            card = rank * 4 + s
            if card == self.wid:
                continue
            n = v.card_count[card]
            if n == 0:
                continue
            c.total += n
            relevant = (self.sf_mask[s] >> rank) & 1
            if not self.dedup or relevant:
                c.keep.append((card, n))
            else:
                c.pool.append((card, n))
                c.pool_total += n
        return c

    def candidates(self, rank: int, suit: int) -> _RankCands:
        return self.any_suit[rank] if suit < 0 else self._candidates_raw(rank, suit)

    def natural(self, rank: int, suit: int) -> int:
        if suit < 0:
            return self.v.rank_count[rank]
        if (self.nat_mask[suit] >> rank) & 1:
            return self.v.card_count[rank * 4 + suit]
        return 0

    @staticmethod
    def _each_choice(c: _RankCands, take: int, acc: List[int], fn) -> None:
        nk = len(c.keep)
        pick = [0] * nk
        while True:
            rest = take - sum(pick)
            if 0 <= rest <= c.pool_total:
                for i in range(nk):
                    acc[c.keep[i][0]] += pick[i]
                left = rest
                taken = []
                for card, n in c.pool:
                    if left <= 0:
                        break
                    k = min(left, n)
                    acc[card] += k
                    taken.append((card, k))
                    left -= k
                fn()
                for card, k in taken:
                    acc[card] -= k
                for i in range(nk):
                    acc[c.keep[i][0]] -= pick[i]
            i = 0
            while i < nk:
                if pick[i] < c.keep[i][1]:
                    pick[i] += 1
                    break
                pick[i] = 0
                i += 1
            if i == nk:
                break

    def feasible(self, reqs: Sequence[Tuple[int, int]], suit: int) -> bool:
        short = 0
        for rank, need in reqs:
            have = self.natural(rank, suit)
            if have >= need:
                continue
            if rank >= 13:
                return False
            short += need - have
        return short <= self.v.wilds

    def expand(self, reqs, idx: int, wilds_left: int, wilds_used: int, suit: int,
               acc: List[int], emit) -> None:
        if idx == len(reqs):
            cards = list(acc)
            cards[self.wid] += wilds_used
            emit(cards, wilds_used)
            return
        rank, need = reqs[idx]
        c = self.candidates(rank, suit)
        hi = min(need, c.total)
        floor = need if rank >= 13 else max(0, need - wilds_left)
        lo = hi if self.minimal else floor
        take = hi
        while take >= lo:
            if take < floor:
                break
            spend = need - take
            if spend <= wilds_left:
                self._each_choice(c, take, acc, lambda spend=spend: self.expand(
                    reqs, idx + 1, wilds_left - spend, wilds_used + spend, suit, acc, emit))
            take -= 1

    def add(self, kind: int, key: int, bomb_size: int, fh_pair: int, cards: List[int],
            wilds: int) -> None:
        if sum(cards) == wilds and key != 12:
            return
        self.out.append(Action(kind, key, bomb_size, wilds, fh_pair, cards))

    def run(self) -> None:
        v, level = self.v, self.level
        for p in range(15):                                            # singles
            if not self.want(SINGLE, p, 0):
                continue
            req = [(rank_of_power(p, level), 1)]
            if self.feasible(req, -1):
                self.expand(req, 0, v.wilds, 0, -1, [0] * NUM_IDS,
                            lambda cards, w, p=p: self.add(SINGLE, p, 0, -1, cards, w))
        self._same_rank(PAIR, 2)
        self._same_rank(TRIPLE, 3)
        for n in range(4, 11):
            self._same_rank(BOMB, n)
        if self.want(JOKER_BOMB, 0, 0) and v.card_count[BJ] >= 2 and v.card_count[RJ] >= 2:
            cards = [0] * NUM_IDS
            cards[BJ] = cards[RJ] = 2
            self.add(JOKER_BOMB, 0, 0, -1, cards, 0)
        self._full_houses()
        self._straights()
        self._sequences(TUBE, 2, TUBE_WINDOWS)
        self._sequences(PLATE, 3, PLATE_WINDOWS)

    def _same_rank(self, kind: int, size: int) -> None:
        if kind != PAIR and size > self.max_same:
            return
        max_p = 15 if kind == PAIR else 13
        bomb = size if kind == BOMB else 0
        for p in range(max_p):
            if not self.want(kind, p, bomb):
                continue
            rank = rank_of_power(p, self.level)
            if kind != PAIR and rank >= 13:
                continue
            req = [(rank, size)]
            if self.feasible(req, -1):
                self.expand(req, 0, self.v.wilds, 0, -1, [0] * NUM_IDS,
                            lambda cards, w, p=p: self.add(kind, p, bomb, -1, cards, w))

    def _full_houses(self) -> None:
        level = self.level
        for tp in range(13):
            if not self.want(FULL_HOUSE, tp, 0):
                continue
            rt = rank_of_power(tp, level)
            if not self.feasible([(rt, 3)], -1):
                continue
            for pp in range(15):
                if pp == tp:
                    continue
                rp = rank_of_power(pp, level)
                if rp == rt or (rp >= 13 and not self.fh_joker_pair):
                    continue
                reqs = [(rt, 3), (rp, 2)]
                if self.feasible(reqs, -1):
                    self.expand(reqs, 0, self.v.wilds, 0, -1, [0] * NUM_IDS,
                                lambda cards, w, tp=tp, pp=pp: self.add(
                                    FULL_HOUSE, tp, 0, pp, cards, w))

    def _straights(self) -> None:
        top = self.top
        need_straight = self.want(STRAIGHT, 0, 0) or (not self.leading and top.type == STRAIGHT)
        need_sf = any(self.want(STRAIGHT_FLUSH, w, 0) for w in range(10))
        if not need_straight and not need_sf:
            return
        wid = self.wid
        for w in range(10):
            take_straight = self.want(STRAIGHT, w, 0)
            take_sf = self.want(STRAIGHT_FLUSH, w, 0)
            if not take_straight and not take_sf:
                continue
            reqs = [(r, 1) for r in STRAIGHT_WINDOWS[w]]
            win = STRAIGHT_MASKS[w]
            if take_straight and popcount(win & ~self.present) <= self.v.wilds:
                def emit(cards, wl, w=w, take_sf=take_sf):
                    nat = bits(cards)[0] & ~(1 << wid)
                    one_suit = nat == 0
                    for s in range(4):
                        if one_suit:
                            break
                        one_suit = (nat & ~_SUIT_CARDS[s]) == 0
                    if not one_suit or wl > 0:
                        self.add(STRAIGHT, w, 0, -1, cards, wl)
                    if one_suit and take_sf:
                        self.add(STRAIGHT_FLUSH, w, 0, -1, cards, wl)
                self.expand(reqs, 0, self.v.wilds, 0, -1, [0] * NUM_IDS, emit)
            if take_sf:
                for s in range(4):
                    if popcount(win & ~self.nat_mask[s]) > self.v.wilds:
                        continue
                    self.expand(reqs, 0, self.v.wilds, 0, s, [0] * NUM_IDS,
                                lambda cards, wl, w=w: self.add(
                                    STRAIGHT_FLUSH, w, 0, -1, cards, wl))

    def _sequences(self, kind: int, per_rank: int, windows) -> None:
        for w, ranks in enumerate(windows):
            if not self.want(kind, w, 0):
                continue
            reqs = [(r, per_rank) for r in ranks]
            if self.feasible(reqs, -1):
                self.expand(reqs, 0, self.v.wilds, 0, -1, [0] * NUM_IDS,
                            lambda cards, wl, w=w: self.add(kind, w, 0, -1, cards, wl))


def generate_moves(counts: Sequence[int], level: int, top: Action = PASS_ACTION,
                   canonical: bool = True) -> List[Action]:
    """gd's ``generate_moves`` (house rules), in gd's output order."""
    gen = _Gen(counts, level, top, canonical)
    gen.run()
    body = gen.out
    # drop_duplicates: sort by (cards, type, key, bomb size), keep the first.
    body.sort(key=lambda a: (a.bits(), a.type, a.key, a.bomb_size))
    unique = []
    for a in body:
        if unique:
            b = unique[-1]
            if (b.counts == a.counts and b.type == a.type and b.key == a.key
                    and b.bomb_size == a.bomb_size):
                continue
        unique.append(a)
    if canonical and len(unique) >= 2:
        def order(a: Action):
            if is_bomb_class(a.type):
                s = strength(a.type, a.key, a.bomb_size)
                return (a.bits(), a.type, -s[0], -s[1])
            return (a.bits(), a.type, 0, -a.key)
        unique.sort(key=order)
        pruned = [unique[0]]
        for a in unique[1:]:
            p = pruned[-1]
            if p.counts == a.counts and p.type == a.type:
                continue
            pruned.append(a)
        unique = pruned
    return ([] if top.type == PASS else [PASS_ACTION]) + unique


def generate_tribute(counts: Sequence[int], level: int) -> List[Action]:
    wid = wild_id(level)
    best = -1
    for c in range(NUM_IDS):
        if counts[c] and c != wid:
            best = max(best, power(rank_of(c), level))
    out = []
    for c in range(NUM_IDS):
        if counts[c] and c != wid and power(rank_of(c), level) == best:
            out.append(Action(TRIBUTE, best, counts=counts_of([c])))
    return out


def generate_back_tribute(counts: Sequence[int], level: int) -> List[Action]:
    out = []
    for c in range(52):
        if not counts[c]:
            continue
        rank = rank_of(c)
        if rank > 8 or rank == level:                 # house: no level cards
            continue
        out.append(Action(BACK_TRIBUTE, power(rank, level), counts=counts_of([c])))
    if out:
        return out
    low = min(power(rank_of(c), level) for c in range(NUM_IDS) if counts[c])
    return [Action(BACK_TRIBUTE, low, counts=counts_of([c]))
            for c in range(NUM_IDS) if counts[c] and power(rank_of(c), level) == low]


# ---- state machine (state.cpp, house rules, auto_pass off) ----------------------------

class TributeMove(object):
    __slots__ = ("payer", "receiver", "card", "back")

    def __init__(self, payer: int, receiver: int, card: int, back: bool) -> None:
        self.payer, self.receiver, self.card, self.back = payer, receiver, card, back


def partner(seat: int) -> int:
    return (seat + 2) % 4


class State(object):
    """``MatchState`` for one round started by ``set_deal``."""

    def __init__(self, hands: Sequence[Sequence[int]], level: int,
                 team_levels: Sequence[int] = None, leader: int = 0,
                 prev_order: Optional[Sequence[int]] = None) -> None:
        self.hands = [counts_of(h) for h in hands]
        self.played = [[0] * NUM_IDS for _ in range(4)]
        self.level = level
        self.levels = list(team_levels) if team_levels is not None else [level, level]
        self.to_move = 0
        self.holder = -1
        self.top = PASS_ACTION
        self.passes = 0
        self.finish_pos = [-1] * 4
        self.order = [-1] * 4
        self.num_finished = 0
        self.num_out = 0
        self.phase = PHASE_PLAY
        self.last_action = [PASS_ACTION] * 4
        self.has_acted = [False] * 4
        self.tribute_moves = []           # type: List[TributeMove]
        self.anti_tribute = False
        self.tribute_payers = [-1, -1]
        self.tribute_receivers = [-1, -1]
        self.tribute_cards = [-1, -1]
        self.tribute_step = 0
        self.first_leader = -1
        self.prev_order = list(prev_order) if prev_order is not None else [-1] * 4
        self.has_prev = prev_order is not None
        if self.has_prev:
            self.round_index = 1
            self._open_tribute()
        else:
            self.round_index = 0
            self.first_leader = self.to_move = leader if leader >= 0 else 0

    def active(self, seat: int) -> bool:
        return self.finish_pos[seat] < 0

    def hand(self, seat: int) -> List[int]:
        return faces_of(self.hands[seat])

    def _open_tribute(self) -> None:
        banker, follower, third, dweller = self.prev_order
        if partner(banker) == follower:
            if self.hands[third][RJ] + self.hands[dweller][RJ] >= 2:
                self._start_play(banker, anti=True)
                return
            self.tribute_payers = [third, dweller]
            self.tribute_receivers = [-1, -1]
        else:
            if self.hands[dweller][RJ] >= 2:
                self._start_play(banker, anti=True)
                return
            self.tribute_payers = [dweller, -1]
            self.tribute_receivers = [banker, -1]
        self.tribute_step = 0
        self.phase = PHASE_TRIBUTE
        self.to_move = self.tribute_payers[0]

    def _start_play(self, leader: int, anti: bool = False) -> None:
        self.anti_tribute = anti
        self.first_leader = self.to_move = leader
        self.phase = PHASE_PLAY

    def legal_actions(self, canonical: bool = True) -> List[Action]:
        hand = self.hands[self.to_move]
        if self.phase == PHASE_TRIBUTE:
            return generate_tribute(hand, self.level)
        if self.phase == PHASE_BACK_TRIBUTE:
            return generate_back_tribute(hand, self.level)
        if self.phase == PHASE_PLAY:
            return generate_moves(hand, self.level, self.top, canonical)
        return []

    def _take(self, seat: int, counts: Sequence[int]) -> None:
        hand = self.hands[seat]
        for c in range(NUM_IDS):
            if counts[c]:
                if hand[c] < counts[c]:
                    raise ValueError("seat %d does not hold card %d" % (seat, c))
                hand[c] -= counts[c]

    def apply(self, a: Action) -> None:
        if self.phase == PHASE_TRIBUTE:
            self._apply_tribute(a)
        elif self.phase == PHASE_BACK_TRIBUTE:
            self._apply_back_tribute(a)
        elif self.phase == PHASE_PLAY:
            self._apply_play(a)

    def _apply_tribute(self, a: Action) -> None:
        card = a.cards[0]
        self.tribute_cards[self.tribute_step] = card
        self.tribute_step += 1
        if self.tribute_step < 2 and self.tribute_payers[self.tribute_step] >= 0:
            self.to_move = self.tribute_payers[self.tribute_step]
            return
        banker, follower = self.prev_order[0], self.prev_order[1]
        if partner(banker) == follower:
            a_seat = self.tribute_payers[0]
            pa = power(rank_of(self.tribute_cards[0]), self.level)
            pb = power(rank_of(self.tribute_cards[1]), self.level)
            down, up = (banker + 1) % 4, (banker + 3) % 4
            p_down = pa if a_seat == down else pb
            p_up = pa if a_seat == up else pb
            if p_down > p_up:
                to_banker, leader = down, down
            elif p_up > p_down:
                to_banker, leader = up, up
            else:
                to_banker, leader = down, down          # TributeTie::Downstream
            for i in range(2):
                seat = self.tribute_payers[i]
                self.tribute_receivers[i] = banker if seat == to_banker else follower
        else:
            leader = self.tribute_payers[0]
        for i in range(2):
            if self.tribute_payers[i] < 0:
                continue
            src, dst, c = self.tribute_payers[i], self.tribute_receivers[i], self.tribute_cards[i]
            self.hands[src][c] -= 1
            self.hands[dst][c] += 1
            self.tribute_moves.append(TributeMove(src, dst, c, False))
        self.first_leader = leader
        self.tribute_step = 0
        self.phase = PHASE_BACK_TRIBUTE
        self.to_move = self.tribute_receivers[0]

    def _apply_back_tribute(self, a: Action) -> None:
        giver = self.to_move
        card = a.cards[0]
        receiver = self.tribute_payers[self.tribute_step]
        self.hands[giver][card] -= 1
        self.hands[receiver][card] += 1
        self.tribute_moves.append(TributeMove(giver, receiver, card, True))
        self.tribute_step += 1
        if self.tribute_step < 2 and self.tribute_payers[self.tribute_step] >= 0:
            self.to_move = self.tribute_receivers[self.tribute_step]
            return
        self.phase = PHASE_PLAY
        self.to_move = self.first_leader

    def _advance_seat(self) -> None:
        for i in range(1, 5):
            s = (self.to_move + i) % 4
            if self.active(s):
                self.to_move = s
                return

    def _close_trick_if_done(self) -> None:
        others = sum(1 for s in range(4) if self.active(s) and s != self.holder)
        if self.passes < others:
            return
        if self.holder >= 0 and self.active(self.holder):
            leader = self.holder
        elif self.holder >= 0 and self.active(partner(self.holder)):
            leader = partner(self.holder)
        else:
            leader = self.holder if self.holder >= 0 else self.to_move
            for i in range(1, 5):
                s = (leader + i) % 4
                if self.active(s):
                    leader = s
                    break
        self.top = PASS_ACTION
        self.holder = -1
        self.passes = 0
        self.to_move = leader

    def _round_over(self) -> bool:
        if self.num_finished >= 3:
            return True
        if self.num_finished >= 2:
            for t in range(2):
                if sum(1 for i in range(self.num_finished) if self.order[i] % 2 == t) == 2:
                    return True
        return False

    def _seal_order(self) -> None:
        if self.num_finished >= 4:
            return
        start = (self.order[1] + 1) % 4 if self.num_finished >= 2 else 0
        for i in range(4):
            if self.num_finished >= 4:
                break
            s = (start + i) % 4
            if self.active(s):
                self.finish_pos[s] = self.num_finished
                self.order[self.num_finished] = s
                self.num_finished += 1

    def _apply_play(self, a: Action) -> None:
        seat = self.to_move
        self.last_action[seat] = a
        self.has_acted[seat] = True
        if a.is_pass:
            self.passes += 1
            self._advance_seat()
            self._close_trick_if_done()
            return
        self._take(seat, a.counts)
        played = self.played[seat]
        for c in range(NUM_IDS):
            played[c] += a.counts[c]
        self.top = a
        self.holder = seat
        self.passes = 0
        if not any(self.hands[seat]):
            self.finish_pos[seat] = self.num_finished
            self.order[self.num_finished] = seat
            self.num_finished += 1
            self.num_out += 1
            if self._round_over():
                self._seal_order()
                self.phase = PHASE_ROUND_END
                return
        self._advance_seat()
        self._close_trick_if_done()


# ---- encoding (encoder.cpp) ------------------------------------------------------------

def _put_cards(counts: Sequence[int], dst: np.ndarray, off: int) -> None:
    for c in range(NUM_IDS):
        if counts[c] >= 1:
            dst[off + c] = 1.0
        if counts[c] >= 2:
            dst[off + NUM_IDS + c] = 1.0


def _one_hot(dst: np.ndarray, off: int, index: int, size: int) -> None:
    if 0 <= index < size:
        dst[off + index] = 1.0


def encode_action(a: Action, state: State, seat: int) -> np.ndarray:
    dst = np.zeros(ACT_DIM, dtype=np.float32)
    _put_cards(a.counts, dst, 0)
    _one_hot(dst, ACT_TYPE, a.type, NUM_TYPES)
    _one_hot(dst, ACT_KEY, a.key, 15)
    if a.type == BOMB:
        _one_hot(dst, ACT_BOMB_SIZE, a.bomb_size - 4, 7)
    _one_hot(dst, ACT_WILDS, min(max(a.wilds, 0), 2), 3)
    if a.type not in (TRIBUTE, BACK_TRIBUTE) or not any(a.counts):
        return dst
    c = a.cards[0]
    hand = state.hands[seat]
    v = HandView(hand, state.level)
    sf = sf_relevant_mask(v)
    rank, suit = rank_of(c), suit_of(c)
    copies = hand[c]
    rc = v.card_count[c] if rank >= 13 else v.rank_count[rank]
    f = ACT_TRIBUTE_FLAGS
    dst[f + 0] = copies == 1
    dst[f + 1] = copies == 2
    dst[f + 2] = rc == 2
    dst[f + 3] = rc == 3
    dst[f + 4] = rc >= 4
    dst[f + 5] = suit >= 0 and bool((sf[suit] >> rank) & 1)
    dst[f + 6] = rank == 3
    dst[f + 7] = rank == 8
    return dst


def _wild_flags(v: HandView, level: int, dst: np.ndarray, off: int) -> None:
    w, rc = v.wilds, v.rank_count

    def put(i: int) -> None:
        dst[off + i] = 1.0

    if w > 0:
        for r in range(13):
            if 1 <= rc[r] < 2 and rc[r] + w >= 2:
                put(0)
            if 1 <= rc[r] < 3 and rc[r] + w >= 3:
                put(1)
            if 1 <= rc[r] < 4 and rc[r] + w >= 4:
                put(6)
            for n in range(5, 11):
                if 1 <= rc[r] < n and rc[r] + w >= n:
                    put(7)
                    break
        for a in range(13):
            if dst[off + 2]:
                break
            for b in range(13):
                if a == b:
                    continue
                need = max(0, 3 - rc[a]) + max(0, 2 - rc[b])
                if 1 <= need <= w and rc[a] + rc[b] > 0:
                    put(2)
                    break
        for ranks in STRAIGHT_WINDOWS:
            missing = sum(0 if rc[r] else 1 for r in ranks)
            if 1 <= missing <= w:
                put(3)
        for ranks in TUBE_WINDOWS:
            need = sum(max(0, 2 - rc[r]) for r in ranks)
            if 1 <= need <= w:
                put(4)
        for ranks in PLATE_WINDOWS:
            need = sum(max(0, 3 - rc[r]) for r in ranks)
            if 1 <= need <= w:
                put(5)
        nat = [0] * 4
        wid = wild_id(level)
        for c in range(52):
            if v.card_count[c] and c != wid:
                nat[c % 4] |= 1 << rank_of(c)
        for s in range(4):
            for ranks in STRAIGHT_WINDOWS:
                missing = sum(0 if (nat[s] >> r) & 1 else 1 for r in ranks)
                if 1 <= missing <= w:
                    put(8)
                    break
    if rc[level] > 0:
        put(9)
    if w >= 2:
        put(10)
    if dst[off + 6] > 0 or dst[off + 7] > 0:
        put(11)


def encode_observation(state: State, seat: int) -> np.ndarray:
    dst = np.zeros(OBS_DIM, dtype=np.float32)
    own = state.hands[seat]
    level = state.level
    _put_cards(own, dst, 0)
    seen = list(own)
    for s in range(4):
        for c in range(NUM_IDS):
            seen[c] += state.played[s][c]
    _put_cards([max(0, 2 - n) for n in seen], dst, OBS_UNSEEN)
    for rel in range(1, 4):
        _put_cards(state.played[(seat + rel) % 4], dst, OBS_PLAYED + (rel - 1) * 108)
    _put_cards(state.played[seat], dst, OBS_PLAYED + 3 * 108)
    for rel in range(1, 4):
        _one_hot(dst, OBS_CARDS_LEFT + (rel - 1) * 28, sum(state.hands[(seat + rel) % 4]), 28)
    for k in range(4):
        s = (seat + k) % 4
        if state.finish_pos[s] >= 0:
            _one_hot(dst, OBS_FINISH + k * 4, state.finish_pos[s], 4)
    _one_hot(dst, OBS_LEVELS, level, 13)
    _one_hot(dst, OBS_LEVELS + 13, state.levels[seat % 2], 13)
    _one_hot(dst, OBS_LEVELS + 26, state.levels[1 - seat % 2], 13)
    v = HandView(own, level)
    _one_hot(dst, OBS_WILD_HELD, min(max(v.wilds, 0), 2), 3)
    wid = wild_id(level)
    wild_unseen = 2 - own[wid] - sum(state.played[s][wid] for s in range(4))
    _one_hot(dst, OBS_WILD_UNSEEN, min(max(wild_unseen, 0), 2), 3)
    _wild_flags(v, level, dst, OBS_WILD_FLAGS)
    if not state.top.is_pass:
        dst[OBS_TRICK_TOP:OBS_TRICK_TOP + ACT_DIM] = encode_action(state.top, state, seat)
        rel = (state.holder - seat) % 4 if state.holder >= 0 else 0
        _one_hot(dst, OBS_TRICK_HOLDER, rel, 4)
    else:
        dst[OBS_TRICK_LEADING] = 1.0
    _one_hot(dst, OBS_TRICK_PASSES, min(max(state.passes, 0), 3), 4)
    for rel in range(1, 4):
        s = (seat + rel) % 4
        if not state.has_acted[s]:
            continue
        off = OBS_LAST_ACTION + (rel - 1) * ACT_DIM
        dst[off:off + ACT_DIM] = encode_action(state.last_action[s], state, s)
        dst[off + ACT_TRIBUTE_FLAGS:off + ACT_DIM] = 0.0
    phase_index = {PHASE_PLAY: 0, PHASE_TRIBUTE: 1, PHASE_BACK_TRIBUTE: 2}.get(state.phase, -1)
    _one_hot(dst, OBS_PHASE, phase_index, 3)
    role = 4
    if state.has_prev:
        for i in range(4):
            if state.prev_order[i] == seat:
                role = i
    _one_hot(dst, OBS_ROLES, role, 5)
    for t in state.tribute_moves:
        if (t.payer == seat and t.receiver == partner(seat)) or \
                (t.receiver == seat and t.payer == partner(seat)):
            dst[OBS_ROLES + 5] = 1.0
    for i, t in enumerate(state.tribute_moves[:4]):
        off = OBS_TRIBUTE + i * 62
        _one_hot(dst, off, t.card, 54)
        _one_hot(dst, off + 54, (t.payer - seat) % 4, 4)
        _one_hot(dst, off + 58, (t.receiver - seat) % 4, 4)
    if not state.tribute_moves:
        return dst
    for rel in range(1, 4):
        s = (seat + rel) % 4
        known = [0] * NUM_IDS
        for t in state.tribute_moves:
            if t.payer == s:
                known[t.card] = max(0, known[t.card] - 1)
            if t.receiver == s:
                known[t.card] += 1
        for t in state.tribute_moves:
            if known[t.card] > state.played[s][t.card]:
                dst[OBS_KNOWN + (rel - 1) * 54 + t.card] = 1.0
    return dst


# ---- the tribute heuristic (bots.cpp tribute_bot) ------------------------------------------

def tribute_choice(state: State, cands: Sequence[Action]) -> int:
    """Index into ``cands`` (the phase's legal actions) the engine heuristic picks."""
    if len(cands) <= 1:
        return 0
    me = state.to_move
    v = HandView(state.hands[me], state.level)
    sf = sf_relevant_mask(v)
    if cands[0].type == TRIBUTE:
        best, best_sf = 0, True
        for i, a in enumerate(cands):
            c = a.cards[0]
            suit = suit_of(c)
            relevant = suit >= 0 and bool((sf[suit] >> rank_of(c)) & 1)
            if not relevant and best_sf:
                best, best_sf = i, False
        return best
    to_partner = False
    for t in state.tribute_moves:
        if not t.back and t.receiver == me:
            to_partner = t.payer == partner(me)
    best, best_score = 0, None
    for i, a in enumerate(cands):
        c = a.cards[0]
        rank, suit = rank_of(c), suit_of(c)
        copies = v.rank_count[rank]
        penalty = 0
        if copies == 2:
            penalty += 300
        if copies == 3:
            penalty += 600
        if copies >= 4:
            penalty += 5000
        if suit >= 0 and (sf[suit] >> rank) & 1:
            penalty += 400
        if rank in (3, 8):
            penalty += 200
        value = power(rank, state.level)
        score = (value * 10 if to_partner else -value * 10) - penalty
        if best_score is None or score > best_score:
            best, best_score = i, score
    return best


# ---- reading a Botzone claim -------------------------------------------------------------

def classify(claim_faces: Sequence[int], action_faces: Sequence[int], level: int) -> Action:
    """The gd action a Botzone play declares: the reading from the claim, the
    cards from the action. Raises ValueError for a claim that names no play."""
    counts = counts_of(action_faces)
    wilds = counts[wild_id(level)]
    if not claim_faces:
        return PASS_ACTION
    ranks = sorted(rank_of(f) for f in claim_faces)
    n = len(ranks)
    per = {}
    for r in ranks:
        per[r] = per.get(r, 0) + 1
    if n == 4 and per.get(13) == 2 and per.get(14) == 2:
        return Action(JOKER_BOMB, 0, 0, wilds, -1, counts)
    if len(per) == 1:
        p = power(ranks[0], level)
        kind = {1: SINGLE, 2: PAIR, 3: TRIPLE}.get(n, BOMB)
        if kind != PAIR and kind != SINGLE and ranks[0] >= 13:
            raise ValueError("jokers form no triple or bomb")
        return Action(kind, p, n if kind == BOMB else 0, wilds, -1, counts)
    if n == 5 and sorted(per.values()) == [2, 3]:
        triple = [r for r, k in per.items() if k == 3][0]
        pair = [r for r, k in per.items() if k == 2][0]
        return Action(FULL_HOUSE, power(triple, level), 0, wilds, power(pair, level), counts)
    distinct = set(per)
    if n == 5 and len(distinct) == 5:
        for w, window in enumerate(STRAIGHT_WINDOWS):
            if set(window) == distinct:
                one_suit = len({suit_of(f) for f in claim_faces}) == 1
                return Action(STRAIGHT_FLUSH if one_suit else STRAIGHT, w, 0, wilds, -1, counts)
    if n == 6 and set(per.values()) == {2}:
        for w, window in enumerate(TUBE_WINDOWS):
            if set(window) == distinct:
                return Action(TUBE, w, 0, wilds, -1, counts)
    if n == 6 and set(per.values()) == {3}:
        for w, window in enumerate(PLATE_WINDOWS):
            if set(window) == distinct:
                return Action(PLATE, w, 0, wilds, -1, counts)
    raise ValueError("unclassifiable claim %r" % (list(claim_faces),))
