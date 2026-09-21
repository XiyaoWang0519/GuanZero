"""Maps OpenGuanDan's raw action tuples onto our engine's `(type, key, cards)`.

Closes parity check O9 of docs/RULES.md section 13. The jar's action tuples
are `[type_str, rank_str, card_strings]`, e.g. `["Single", "5", ["D5"]]`,
`["ThreeWithTwo", "5", ["S5", "C5", "D5", "SB", "SB"]]` or
`["PASS", "PASS", "PASS"]`. Verified live against the jar (see
`probe_rules.py` and `docs/ogd_parity_probes.md`); the type-string set is
`Single Pair Trips ThreePair ThreeWithTwo TwoTrips Straight StraightFlush
Bomb FourKings tribute back PASS` (13 strings, matching README.md section 5.2
plus `FourKings`, which is the joker bomb -- absent from RULES.md O9's
"expected" list).

Our side uses the vocabulary of `docs/PY_API.md` / `oracle/gd_reference.py`:
types `Single Pair Triple FullHouse Straight Tube Plate Bomb StraightFlush
JokerBomb Pass Tribute BackTribute`, a `key` that is an `int` (or a
`(size, power)` tuple for `Bomb`), and card ids 0..53 as in docs/RULES.md
section 3 (`rank * 4 + suit` for 0..51, 52 = BJ, 53 = RJ).
"""

from __future__ import annotations

# ---- cards (docs/RULES.md section 3) --------------------------------------

_RANK_CHARS = "23456789TJQKA"  # index 0..12
_SUIT_CHARS = "SHCD"           # index 0..3

RANK_INDEX: dict[str, int] = {c: i for i, c in enumerate(_RANK_CHARS)}
RANK_INDEX["B"] = 13  # black/small joker "rank"
RANK_INDEX["R"] = 14  # red/big joker "rank"
SUIT_INDEX: dict[str, int] = {c: i for i, c in enumerate(_SUIT_CHARS)}


def card_id(text: str) -> int:
    """`"S2"` -> 0 .. `"DA"` -> 51, `"SB"` -> 52, `"HR"` -> 53."""
    if text == "SB":
        return 52
    if text == "HR":
        return 53
    suit, rank = text[0], text[1]
    return RANK_INDEX[rank] * 4 + SUIT_INDEX[suit]


def id_to_card(cid: int) -> str:
    if cid == 52:
        return "SB"
    if cid == 53:
        return "HR"
    rank, suit = divmod(cid, 4)
    return _SUIT_CHARS[suit] + _RANK_CHARS[rank]


# ---- power order (docs/RULES.md section 4) ---------------------------------


def power(rank: int, level: int) -> int:
    """Mirrors `gd::power` in cpp/include/gd/cards.h."""
    if rank >= 13:  # BJ, RJ
        return rank
    if rank == level:
        return 12
    return rank if rank < level else rank - 1


def rank_char_power(rank_char: str, level: int) -> int:
    return power(RANK_INDEX[rank_char], level)


# ---- sequence windows (docs/RULES.md section 4) ----------------------------
# Window index 0 is always the ace-low window. The jar labels a window by the
# rank character of its *lowest* card, except the ace-low window, which it
# labels 'A' (verified against the jar: T-STR-01's `SA D2 C3 S4 H5` reads back
# as `Straight "A"`, and T-SEQ-04's `SK DK CK SA DA CA` reads back as
# `TwoTrips "K"`, the highest plate window -- see probe_rules.py).

_STRAIGHT_LABELS = "A23456789T"      # 10 windows: A2345 .. TJQKA
_TUBE_LABELS = "A23456789TJQ"        # 12 windows: AA2233 .. QQKKAA
_PLATE_LABELS = "A23456789TJQK"      # 13 windows: AAA222 .. KKKAAA


def straight_window(label: str) -> int:
    return _STRAIGHT_LABELS.index(label)


def tube_window(label: str) -> int:
    return _TUBE_LABELS.index(label)


def plate_window(label: str) -> int:
    return _PLATE_LABELS.index(label)


# ---- type strings -----------------------------------------------------------

# OpenGuanDan type string -> ours. All 13 strings the jar is known to emit.
OGD_TYPE_TO_OURS: dict[str, str] = {
    "Single": "Single",
    "Pair": "Pair",
    "Trips": "Triple",
    "ThreeWithTwo": "FullHouse",
    "ThreePair": "Tube",
    "TwoTrips": "Plate",
    "Straight": "Straight",
    "StraightFlush": "StraightFlush",
    "Bomb": "Bomb",
    "FourKings": "JokerBomb",
    "tribute": "Tribute",
    "back": "BackTribute",
    "PASS": "Pass",
}

Key = int | tuple[int, int] | None
NormalizedAction = tuple[str, Key, tuple[int, ...]]


def normalize_action(raw: list, level_char: str) -> NormalizedAction:
    """Convert one raw OGD action tuple to our `(type, key, cards)` form.

    `raw` is `[type_str, rank_str, card_strings_or_None]` as returned by
    `bridge.legal_moves` or logged in a trace's `action_list`. `level_char`
    is the round level as a rank character (`"7"`, `"T"`, ...).
    """
    ogd_type, rank_str, raw_cards = raw[0], raw[1], raw[2]
    our_type = OGD_TYPE_TO_OURS[ogd_type]
    level = RANK_INDEX[level_char]

    if our_type == "Pass":
        return ("Pass", None, tuple())

    cards = tuple(sorted(card_id(c) for c in raw_cards)) if raw_cards else tuple()

    if our_type in ("Single", "Pair", "Triple", "FullHouse"):
        key: Key = rank_char_power(rank_str, level)
    elif our_type in ("Straight", "StraightFlush"):
        key = straight_window(rank_str)
    elif our_type == "Tube":
        key = tube_window(rank_str)
    elif our_type == "Plate":
        key = plate_window(rank_str)
    elif our_type == "Bomb":
        key = (len(cards), rank_char_power(rank_str, level))
    elif our_type == "JokerBomb":
        key = None
    elif our_type in ("Tribute", "BackTribute"):
        # raw[1] is the literal string "tribute"/"back", not a rank; the
        # power comes from the one card actually being moved.
        card_rank_char = raw_cards[0][1]
        key = rank_char_power(card_rank_char, level)
    else:  # pragma: no cover - exhaustive above
        raise ValueError(f"unhandled OGD type {ogd_type!r}")

    return (our_type, key, cards)


def normalize_action_list(raw_list: list[list], level_char: str) -> set[NormalizedAction]:
    """Normalize a full `actionList` into our full-mode legal action set."""
    return {normalize_action(a, level_char) for a in raw_list}


def wild_card_id(level: int) -> int:
    """The heart card of the round level: the wild card (docs/RULES.md 4)."""
    return level * 4 + SUIT_INDEX["H"]
