"""Card, level and play conversions between DanLM's engine and ``gd``.

DanLM (``danzero.engine``) encodes a card as ``suit * 13 + rank`` with suits
``H=0, S=1, C=2, D=3`` and ranks ``2..A = 0..12``; jokers are ``52`` (small,
``SB``) and ``53`` (big, ``HR``). ``gd`` encodes ``rank * 4 + suit`` with
suits ``S=0, H=1, C=2, D=3`` (RULES.md section 3) and the same jokers.

A DanLM play is an 80-float vector: 54 card counts, an 11-way type one-hot
and a 15-way rank one-hot. The rank is the natural rank index of the play's
defining card: the card itself for singles, pairs, triples and bombs, the
triple for a full house, and the HIGHEST rank of the window for straights,
tubes and plates. ``gd`` keys are powers for the first group and window
indices (ace-low window 0) for sequences, so the two need a level to convert.

DanLM's level is the rank value ``2..14``; ``gd``'s is the rank index ``0..12``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

SMALL_JOKER = 52
BIG_JOKER = 53

# DanLM suit index -> gd suit index, and back.
_SUIT_THEIRS_TO_OURS = (1, 0, 2, 3)  # H, S, C, D  ->  S=0, H=1, C=2, D=3
_SUIT_OURS_TO_THEIRS = (1, 0, 2, 3)  # S, H, C, D  ->  H=0, S=1, C=2, D=3

TYPE_THEIRS_TO_OURS = {
    "pass": "Pass", "single": "Single", "double": "Pair", "triple": "Triple",
    "fullhouse": "FullHouse", "tube": "Tube", "plate": "Plate", "straight": "Straight",
    "normalbomb": "Bomb", "flushbomb": "StraightFlush", "jokerbomb": "JokerBomb",
}
TYPE_OURS_TO_THEIRS = {ours: theirs for theirs, ours in TYPE_THEIRS_TO_OURS.items()}
PLAY_TYPES = ("pass", "single", "double", "triple", "fullhouse", "tube", "plate",
              "straight", "normalbomb", "flushbomb", "jokerbomb")
DIM_CARDS, DIM_TYPE, DIM_RANK = 54, 11, 15
# Highest natural rank index of a window minus the gd window index.
_WINDOW_OFFSET = {"Straight": 3, "StraightFlush": 3, "Tube": 1, "Plate": 0}


def card_theirs_to_ours(card: int) -> int:
    if card >= SMALL_JOKER:
        return int(card)
    suit, rank = divmod(int(card), 13)
    return rank * 4 + _SUIT_THEIRS_TO_OURS[suit]


def card_ours_to_theirs(card: int) -> int:
    if card >= SMALL_JOKER:
        return int(card)
    rank, suit = divmod(int(card), 4)
    return _SUIT_OURS_TO_THEIRS[suit] * 13 + rank


def level_theirs_to_ours(level: int) -> int:
    if not 2 <= level <= 14:
        raise ValueError("DanLM level must be 2..14")
    return int(level) - 2


def level_ours_to_theirs(level: int) -> int:
    if not 0 <= level <= 12:
        raise ValueError("gd level must be 0..12")
    return int(level) + 2


def hand_ours_to_theirs(cards: Sequence[int]) -> np.ndarray:
    """gd card list -> DanLM 54-count vector (int8)."""
    vector = np.zeros(DIM_CARDS, np.int8)
    for card in cards:
        vector[card_ours_to_theirs(card)] += 1
    return vector


def hand_theirs_to_ours(vector: np.ndarray) -> list[int]:
    """DanLM 54-count vector -> sorted gd card list."""
    cards = [card_theirs_to_ours(i) for i in range(DIM_CARDS) for _ in range(int(vector[i]))]
    return sorted(cards)


def power(rank: int, level: int) -> int:
    """RULES.md section 4 power order; ``rank`` 0..14, ``level`` 0..12."""
    if rank >= 13:
        return rank
    if rank == level:
        return 12
    return rank if rank < level else rank - 1


def rank_of_power(key: int, level: int) -> int:
    """Inverse of ``power`` for a natural rank; jokers pass through."""
    if key >= 13:
        return key
    if key == 12:
        return level
    return key if key < level else key + 1


def key_from_their_rank(our_type: str, rank: int, level: int) -> int | None:
    """The gd key that DanLM's rank index denotes for a play of ``our_type``."""
    if our_type in ("Pass", "JokerBomb"):
        return None
    if our_type in _WINDOW_OFFSET:
        return rank - _WINDOW_OFFSET[our_type]
    return power(rank, level)


def their_rank_from_key(our_type: str, key: int, level: int) -> int:
    if our_type in ("Pass", "JokerBomb"):
        return 0
    if our_type in _WINDOW_OFFSET:
        return key + _WINDOW_OFFSET[our_type]
    return rank_of_power(key, level)


@dataclass(frozen=True)
class NormalizedPlay:
    """A play in gd terms: type name, sorted gd cards, gd key (None when typeless)."""
    type: str
    cards: tuple[int, ...]
    key: int | None

    @property
    def is_pass(self) -> bool:
        return self.type == "Pass"


def decode_play(play: np.ndarray, level: int) -> NormalizedPlay:
    """DanLM 80-vector -> NormalizedPlay under gd level ``level`` (0..12)."""
    play = np.asarray(play)
    if play.shape != (DIM_CARDS + DIM_TYPE + DIM_RANK,):
        raise ValueError("a DanLM play has 80 entries")
    type_index = int(play[DIM_CARDS:DIM_CARDS + DIM_TYPE].argmax())
    our_type = TYPE_THEIRS_TO_OURS[PLAY_TYPES[type_index]]
    rank = int(play[DIM_CARDS + DIM_TYPE:].argmax())
    cards = tuple(hand_theirs_to_ours(play[:DIM_CARDS]))
    return NormalizedPlay(our_type, cards, key_from_their_rank(our_type, rank, level))


def encode_play(normalized: NormalizedPlay, level: int) -> np.ndarray:
    """NormalizedPlay -> DanLM 80-vector (float32), inverse of ``decode_play``."""
    play = np.zeros(DIM_CARDS + DIM_TYPE + DIM_RANK, np.float32)
    play[:DIM_CARDS] = hand_ours_to_theirs(normalized.cards)
    play[DIM_CARDS + PLAY_TYPES.index(TYPE_OURS_TO_THEIRS[normalized.type])] = 1.0
    rank = 0 if normalized.key is None else their_rank_from_key(normalized.type, normalized.key, level)
    play[DIM_CARDS + DIM_TYPE + rank] = 1.0
    return play


def normalize_action(action) -> NormalizedPlay:
    """A ``gd.Action`` -> NormalizedPlay."""
    key = None if action.type in ("Pass", "JokerBomb") else int(action.key)
    return NormalizedPlay(action.type, tuple(sorted(int(c) for c in action.cards)), key)


class PlayIndex:
    """Lookup from NormalizedPlay to row index in a DanLM ``legal_plays`` array.

    ``by_cards`` groups rows by (type, cards) so that a play whose exact key
    is absent can still be matched to a differently declared reading of the
    same cards, which the caller reports as a divergence.
    """

    def __init__(self, legal_plays: np.ndarray, level: int, *,
                 cache: dict[tuple[int, str, bytes], NormalizedPlay] | None = None) -> None:
        # A DanLM seat only needs the play it chose. Delay decoding the full
        # legal set until our policy needs the reverse lookup (or diff mode
        # explicitly compares both sets). The optional cache belongs to a
        # single round, so repeated plays after passes cost only a lookup.
        self._rows = np.asarray(legal_plays)
        self._level = level
        self._cache = cache
        self._decoded: dict[int, NormalizedPlay] = {}
        self._exact: dict[NormalizedPlay, int] | None = None
        self._by_cards: dict[tuple[str, tuple[int, ...]], list[int]] = {}
        self._plays: list[NormalizedPlay] | None = None

    def __len__(self) -> int:
        return len(self._rows)

    def play(self, index: int) -> NormalizedPlay:
        """Decode one chosen row without building the reverse lookup."""
        if index not in self._decoded:
            row = self._rows[index]
            if self._cache is None:
                play = decode_play(row, self._level)
            else:
                key = (self._level, row.dtype.str, row.tobytes())
                play = self._cache.get(key)
                if play is None:
                    play = decode_play(row, self._level)
                    self._cache[key] = play
            self._decoded[index] = play
        return self._decoded[index]

    def _build(self) -> None:
        if self._exact is not None:
            return
        self._exact = {}
        self._plays = [self.play(index) for index in range(len(self))]
        for index, play in enumerate(self._plays):
            self._exact.setdefault(play, index)
            self._by_cards.setdefault((play.type, play.cards), []).append(index)

    @property
    def plays(self) -> list[NormalizedPlay]:
        self._build()
        return self._plays

    @property
    def exact(self) -> dict[NormalizedPlay, int]:
        self._build()
        return self._exact

    @property
    def by_cards(self) -> dict[tuple[str, tuple[int, ...]], list[int]]:
        self._build()
        return self._by_cards

    def find(self, play: NormalizedPlay) -> tuple[int | None, str]:
        """(row, how): how is 'exact', 'reading' (same cards, other key) or 'missing'."""
        row = self.exact.get(play)
        if row is not None:
            return row, "exact"
        rows = self.by_cards.get((play.type, play.cards))
        if rows:
            return rows[0], "reading"
        return None, "missing"
