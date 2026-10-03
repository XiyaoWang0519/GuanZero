"""Botzone GuanDan protocol: cards, levels, claims and the request log.

Botzone (https://wiki.botzone.org.cn/index.php?title=GuanDan) numbers the 108
physical cards ``0..107``: ``id % 54`` is ``rank * 4 + suit`` with ranks
``A, 2, ..., K = 0..12`` and suits ``h, d, s, c = 0..3``, then ``52`` small
joker and ``53`` big joker; ``54..107`` repeat the layout for the second deck.
``gd`` numbers the 54 card faces ``rank * 4 + suit`` with ranks
``2, ..., A = 0..12`` and suits ``S, H, C, D = 0..3`` (RULES.md section 3),
and the same jokers. A face has two physical copies.

A play is answered as ``[action, claim]``: the physical cards and the same
cards as declared, with each wild card (heart of the level rank) replaced by
the card it stands for. A pass is ``[[], []]``.

The judge asks every seat that has not finished, passes included, so the
public play sequence equals ``gd``'s with ``auto_pass`` off. A play request's
``history`` is positional on the real platform: four slots, slot ``i``
holding the latest move of seat ``(me + i) % 4`` since our own last move, an
empty list for a seat that did not move (observed by FableDan in platform
logs; the wiki shows a list of ``{"player", "response"}`` dicts instead, and
both are accepted here).

This module needs only the standard library and stays Python 3.6
compatible: Botzone runs Python 3.6.
"""
from typing import Dict, List, Optional, Sequence, Tuple

SMALL_JOKER = 52
BIG_JOKER = 53
LEVEL_NAMES = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A")
HEARTS = 1                       # gd suit index of hearts
_SUIT_BZ_TO_GD = (1, 3, 0, 2)    # h, d, s, c -> S=0, H=1, C=2, D=3
_SUIT_GD_TO_BZ = (2, 0, 3, 1)
# gd natural rank lists of the windows (ace-low window 0 starts at A).
_WINDOW_LENGTH = {"Straight": 5, "StraightFlush": 5, "Tube": 3, "Plate": 2}
_WINDOW_COPIES = {"Straight": 1, "StraightFlush": 1, "Tube": 2, "Plate": 3}


# ---- cards and levels -----------------------------------------------------------

def card_to_gd(card: int) -> int:
    """Botzone physical id -> gd card face."""
    base = int(card) % 54
    if base >= SMALL_JOKER:
        return base
    rank, suit = divmod(base, 4)
    return ((rank - 1) % 13) * 4 + _SUIT_BZ_TO_GD[suit]


def face_to_bz(face: int, copy: int = 0) -> int:
    """gd card face -> Botzone physical id of deck ``copy`` (0 or 1)."""
    if face >= SMALL_JOKER:
        base = int(face)
    else:
        rank, suit = divmod(int(face), 4)
        base = ((rank + 1) % 13) * 4 + _SUIT_GD_TO_BZ[suit]
    return base + 54 * copy


def level_to_gd(level: object) -> int:
    """Botzone level string ``"2".."10", "J", "Q", "K", "A"`` -> gd level 0..12."""
    text = str(level).strip().upper()
    if text == "1":
        text = "A"
    return LEVEL_NAMES.index(text)


def rank_of(face: int) -> int:
    """gd natural rank 0..12, 13 small joker, 14 big joker."""
    if face == SMALL_JOKER:
        return 13
    if face == BIG_JOKER:
        return 14
    return face // 4


def suit_of(face: int) -> int:
    return -1 if face >= SMALL_JOKER else face % 4


def power(rank: int, level: int) -> int:
    """RULES.md section 4 power order; ``rank`` 0..14."""
    if rank >= 13:
        return rank
    if rank == level:
        return 12
    return rank if rank < level else rank - 1


def rank_of_power(key: int, level: int) -> int:
    if key >= 13:
        return key
    if key == 12:
        return level
    return key if key < level else key + 1


def is_wild(face: int, level: int) -> bool:
    return face == level * 4 + HEARTS


def window_ranks(kind: str, window: int) -> List[int]:
    """Natural ranks of an ace-low window: window 0 starts at the ace."""
    length = _WINDOW_LENGTH[kind]
    return [12 if window + i == 0 else window + i - 1 for i in range(length)]


# ---- claims --------------------------------------------------------------------

def reading_ranks(kind: str, key: int, faces: Sequence[int], level: int) -> Optional[List[int]]:
    """The declared natural rank of every card of a gd reading, or None.

    None means Botzone cannot express the reading: a full house whose triple
    is the level rank with the two wild cards as its pair (T-FH-04), which a
    claim can only spell as a five-card bomb.
    """
    n = len(faces)
    if kind in ("Single", "Pair", "Triple", "Bomb"):
        return [rank_of_power(key, level)] * n
    if kind == "JokerBomb":
        return sorted(rank_of(f) for f in faces)
    if kind in _WINDOW_LENGTH:
        ranks = []
        for rank in window_ranks(kind, key):
            ranks.extend([rank] * _WINDOW_COPIES[kind])
        return ranks if len(ranks) == n else None
    if kind == "FullHouse":
        triple = rank_of_power(key, level)
        others = {rank_of(f) for f in faces if not is_wild(f, level) and rank_of(f) != triple}
        if len(others) > 1:
            return None
        if others:
            pair = others.pop()
        else:
            pair = level        # three naturals plus both wild cards as themselves
        if pair == triple:
            return None
        return [triple] * 3 + [pair] * 2
    return None


def claim_faces(kind: str, key: int, faces: Sequence[int], level: int) -> Optional[List[int]]:
    """Declared card faces aligned with ``faces`` (wilds substituted), or None.

    Natural cards declare themselves. A wild card declares itself when its
    rank is still needed, otherwise it takes a missing slot: in the flush's
    suit for a straight flush, in a suit that keeps a straight from looking
    like a flush, in spades otherwise.
    """
    if kind == "Pass":
        return []
    ranks = reading_ranks(kind, key, faces, level)
    if ranks is None:
        return None
    need = {}
    for rank in ranks:
        need[rank] = need.get(rank, 0) + 1
    out = [None] * len(faces)                     # type: List[Optional[int]]
    flush_suit = -1
    if kind == "StraightFlush":
        suits = {suit_of(f) for f in faces if not is_wild(f, level)}
        if len(suits) != 1:
            return None
        flush_suit = suits.pop()
    for i, face in enumerate(faces):
        if is_wild(face, level):
            continue
        rank = rank_of(face)
        if need.get(rank, 0) == 0:
            return None
        need[rank] -= 1
        out[i] = face
    for i, face in enumerate(faces):
        if not is_wild(face, level):
            continue
        if need.get(level, 0) and flush_suit in (-1, HEARTS):
            need[level] -= 1
            out[i] = face
    missing = [rank for rank, count in sorted(need.items()) for _ in range(count)]
    slots = [i for i, face in enumerate(out) if face is None]
    if len(missing) != len(slots) or any(rank >= 13 for rank in missing):
        return None
    if kind == "StraightFlush":
        suit = flush_suit
    elif kind == "Straight":
        natural = {suit_of(f) for f in out if f is not None}
        suit = 1 if natural == {0} else 0
    else:
        suit = 0
    for i, rank in zip(slots, missing):
        out[i] = rank * 4 + suit
    if kind == "Straight" and len({suit_of(f) for f in out}) == 1:
        return None
    return [int(f) for f in out]


def parse_claim(claim: Sequence[int]) -> Tuple[List[int], List[int]]:
    """Botzone claim ids -> (sorted declared ranks, declared suits)."""
    faces = [card_to_gd(c) for c in claim]
    return sorted(rank_of(f) for f in faces), [suit_of(f) for f in faces]


def claim_matches(kind: str, key: int, faces: Sequence[int], level: int,
                  claim: Sequence[int]) -> bool:
    """Whether the gd reading declares the same play as a Botzone claim."""
    ranks = reading_ranks(kind, key, faces, level)
    if ranks is None:
        return False
    declared, suits = parse_claim(claim)
    if sorted(ranks) != declared:
        return False
    if len(claim) == 5 and kind in ("Straight", "StraightFlush"):
        flush = len(set(suits)) == 1
        return flush == (kind == "StraightFlush")
    return True


def physical_claim(faces: Sequence[int], declared: Sequence[int], used: Sequence[int]
                   ) -> List[int]:
    """Botzone ids for declared faces: a card declaring itself keeps its id.

    A substituted card takes a free copy of its declared face. When both
    copies are already in the play (a wild completing a pair, triple or bomb
    of that exact card) it takes the same rank in another suit; that cannot
    happen in a sequence, whose ranks are distinct, so a flush stays a flush.
    """
    taken = set(int(c) for c in used)
    out = []                                      # type: List[int]
    for face, physical in zip(declared, used):
        if card_to_gd(physical) == face:
            out.append(int(physical))
            continue
        options = [face]
        if face < SMALL_JOKER:
            options += [(face // 4) * 4 + s for s in range(4) if s != face % 4]
        free = [face_to_bz(option, copy) for option in options for copy in (0, 1)
                if face_to_bz(option, copy) not in taken]
        if not free:
            raise ValueError("no free card id for a declared face")
        taken.add(free[0])
        out.append(free[0])
    return out


# ---- the request log ---------------------------------------------------------------

def _cards(value: object) -> List[int]:
    """A tribute or return entry: an int, a list of ints, ``-1`` for none."""
    if value is None:
        return []
    if isinstance(value, list):
        return [int(c) for c in value if c is not None and int(c) >= 0]
    return [int(value)] if int(value) >= 0 else []


class Move(object):
    """One public play-phase move: seat, physical cards and claim (empty = pass)."""
    __slots__ = ("seat", "action", "claim")

    def __init__(self, seat: int, action: Sequence[int], claim: Sequence[int]) -> None:
        self.seat = int(seat)
        self.action = [int(c) for c in action]
        self.claim = [int(c) for c in claim]

    @property
    def is_pass(self) -> bool:
        return not self.claim

    def __repr__(self) -> str:
        return "Move(%d, %r, %r)" % (self.seat, self.action, self.claim)


class RoundLog(object):
    """Everything one seat has been told in a round, in order.

    ``feed(request, response)`` takes each request with the response we gave
    to it (None for the pending one). The log keeps the deal, the tribute
    setup, every tribute and return card announced in ``global``, and the
    play-phase moves of every seat in table order, our own included.
    """

    def __init__(self) -> None:
        self.me = -1
        self.deliver = []        # type: List[int]
        self.level = 0
        self.tribute = 0
        self.first = -1
        self.last = -1
        self.resist = False
        self.tribute_cards = {}  # type: Dict[int, int]
        self.return_cards = {}   # type: Dict[int, int]
        self.moves = []          # type: List[Move]
        self.done = []           # type: List[int]
        self.my_tribute = None   # type: Optional[int]
        self.my_return = None    # type: Optional[int]
        self.stage = ""
        self.first_play_seen = False
        self.leader_hint = -1    # first mover of the round as the requests show it

    def _global(self, g: dict) -> None:
        if not g:
            return
        if g.get("level") is not None:
            self.level = level_to_gd(g["level"])
        if g.get("tribute") is not None:
            self.tribute = int(g["tribute"])
        if g.get("first") is not None:
            self.first = int(g["first"])
        if g.get("last") is not None:
            self.last = int(g["last"])
        if g.get("resist") is not None:
            self.resist = bool(g["resist"])
        for key, store in (("tribute_cards", self.tribute_cards),
                           ("return_cards", self.return_cards)):
            for seat, value in (g.get(key) or {}).items():
                cards = _cards(value)
                if cards:
                    store[int(seat)] = cards[0]

    def _history(self, history: list) -> None:
        if not history:
            if not self.first_play_seen:
                self.leader_hint = self.me
            return
        new = []                 # type: List[Move]
        if any(isinstance(h, dict) for h in history):
            start = 0
            for i, entry in enumerate(history):
                if isinstance(entry, dict) and int(entry.get("player", -1)) == self.me:
                    start = i + 1
            for entry in history[start:]:
                if isinstance(entry, dict):
                    action, claim = entry["response"]
                    new.append(Move(int(entry["player"]), action, claim))
        else:
            for i in range(1, len(history)):
                entry = history[i]
                if isinstance(entry, list) and len(entry) == 2:
                    new.append(Move((self.me + i) % 4, entry[0], entry[1]))
        if not self.first_play_seen:
            self.leader_hint = new[0].seat if new else self.me
        self.moves.extend(new)

    def feed(self, request: dict, response: object = None) -> None:
        stage = request.get("stage", "")
        self.stage = stage
        if "your_id" in request:
            self.me = int(request["your_id"])
        self._global(request.get("global") or {})
        if stage == "deal":
            self.deliver = [int(c) for c in request.get("deliver") or []]
        elif stage == "play":
            self.done = [int(s) for s in request.get("done") or []]
            self._history(request.get("history") or [])
            self.first_play_seen = True
        if response is None:
            return
        if stage == "tribute" and response:
            self.my_tribute = int(response[0])
            self.tribute_cards.setdefault(self.me, self.my_tribute)
        elif stage == "return" and response:
            self.my_return = int(response[0])
            self.return_cards.setdefault(self.me, self.my_return)
        elif stage == "play":
            action, claim = response
            self.moves.append(Move(self.me, action, claim))

    @staticmethod
    def from_turns(requests: Sequence[dict], responses: Sequence[object]) -> "RoundLog":
        """The log of the round the last request belongs to (a deal starts a round)."""
        start = 0
        for i, request in enumerate(requests):
            if request.get("stage") == "deal":
                start = i
        log = RoundLog()
        for i in range(start, len(requests)):
            log.feed(requests[i], responses[i] if i < len(responses) else None)
        return log
