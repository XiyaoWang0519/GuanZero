"""Pure numpy reimplementation of DanZero-V1T's input encoding (964 floats per row).

Black-box reconstruction from DanLM's compiled ``danzero.encoding.encoder_v1``
(``encode_batch_v1t``, ``encode_tribute_state``, ``cards_to_actions_80``),
probed from Python and verified bit for bit by ``eval/v1t/verify.py``. No
DanLM code is copied; everything is in DanLM's card numbering (``suit * 13 +
rank``, suits H S C D, jokers 52 and 53) and play encoding (54 card counts,
11-way type one-hot, 15-way rank one-hot), see ``eval/danlm/bridge.py``.

One row is ``state (884) + action (80)``. The state:

====== ===== ==========================================================
start  size  field
====== ===== ==========================================================
0      54    own hand, card counts
54     54    "other unseen": 2 - hand - cards played by the OTHER three
             seats (the player's own played cards are not subtracted).
             In the tribute phase jokers count 1 - hand instead of 2.
108    320   current trick: four 80-dim plays, slot i = seat player + i;
             zero for a seat that has not acted since the trick began
428    216   cards played this round, slot i = seat player + i
644    4     finished seats, slot (seat - player) mod 4
648    13    round level one-hot (level rank index)
661    3     phase one-hot: give, back, play
664    4     tribute receiver, relative: 0 down, 1 teammate, 2 up,
             3 "opponent" (double tribute, receiver not yet known);
             zero in the play phase
668    216   cards received through tribute (give and back), slot i =
             seat player + i; an anti-tribute instead marks the big-joker
             holders' own slot at card 53
====== ===== ==========================================================
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

DIM_CARDS = 54
DIM_PLAY = 80
DIM_STATE = 884
DIM_ROW = DIM_STATE + DIM_PLAY
SMALL_JOKER, BIG_JOKER = 52, 53

OFF_HAND, OFF_OTHER, OFF_TRICK, OFF_PLAYED = 0, 54, 108, 428
OFF_FINISHED, OFF_LEVEL, OFF_PHASE, OFF_RECEIVER, OFF_TRIBUTE = 644, 648, 661, 664, 668
PHASES = {"give": 0, "back": 1, "play": 2}

TYPE_INDEX = {"pass": 0, "single": 1, "double": 2, "triple": 3, "fullhouse": 4, "tube": 5,
              "plate": 6, "straight": 7, "normalbomb": 8, "flushbomb": 9, "jokerbomb": 10}


def card_rank(card: int) -> int:
    """Natural rank index 0..12 of a DanLM card; 13 and 14 for the jokers."""
    return 13 + card - SMALL_JOKER if card >= SMALL_JOKER else card % 13


def single_play(card: int) -> np.ndarray:
    """The 80-dim "single" play of one card (DanLM's ``cards_to_actions_80`` row)."""
    row = np.zeros(DIM_PLAY, np.float32)
    row[card] = 1.0
    row[DIM_CARDS + TYPE_INDEX["single"]] = 1.0
    row[DIM_CARDS + 11 + card_rank(card)] = 1.0
    return row


def tribute_received_rel(records: Iterable[tuple[int, int, int]], player: int) -> np.ndarray:
    """(4, 54) cards received per seat, relative to ``player``; card -1 is skipped."""
    out = np.zeros((4, DIM_CARDS), np.float32)
    for giver, receiver, card in records:
        if card >= 0:
            out[(receiver - player) % 4, card] += 1.0
    return out


def relative(absolute: np.ndarray | None, player: int) -> np.ndarray | None:
    if absolute is None:
        return None
    return np.stack([absolute[(player + i) % 4] for i in range(4)]).astype(np.float32)


def _tail(state: np.ndarray, phase: str, receiver_rel: int | None,
          tribute_rel: np.ndarray | None) -> None:
    state[OFF_PHASE + PHASES[phase]] = 1.0
    if receiver_rel is not None:
        state[OFF_RECEIVER + receiver_rel] = 1.0
    if tribute_rel is not None:
        state[OFF_TRIBUTE:OFF_TRIBUTE + 216] = np.asarray(tribute_rel, np.float32).reshape(-1)


def play_state(player: int, hand: np.ndarray, played: Sequence[np.ndarray],
               trick: Sequence[np.ndarray | None], finished: Iterable[int], level: int,
               tribute_rel: np.ndarray | None) -> np.ndarray:
    """884-dim play-phase state. ``level`` is DanLM's 2..14; arrays are DanLM vectors."""
    state = np.zeros(DIM_STATE, np.float32)
    hand = np.asarray(hand, np.int64)
    state[OFF_HAND:OFF_HAND + 54] = hand
    others = np.zeros(DIM_CARDS, np.int64)
    for seat in range(4):
        if seat != player:
            others += np.asarray(played[seat], np.int64)
    # int8 wrap as in DanLM's Observation.other_unseen (values stay in -2..2 anyway)
    state[OFF_OTHER:OFF_OTHER + 54] = (2 - hand - others).astype(np.int8)
    for i in range(4):
        seat = (player + i) % 4
        if trick[seat] is not None:
            state[OFF_TRICK + 80 * i:OFF_TRICK + 80 * (i + 1)] = trick[seat]
        state[OFF_PLAYED + 54 * i:OFF_PLAYED + 54 * (i + 1)] = played[seat]
    for seat in finished:
        state[OFF_FINISHED + (seat - player) % 4] = 1.0
    state[OFF_LEVEL + level - 2] = 1.0
    _tail(state, "play", None, tribute_rel)
    return state


def tribute_state(hand: np.ndarray, level: int, phase: str, receiver_rel: int,
                  tribute_rel: np.ndarray | None) -> np.ndarray:
    """884-dim give/back state as the V1T agent builds it (``encode_tribute_state``)."""
    state = np.zeros(DIM_STATE, np.float32)
    hand = np.asarray(hand, np.int64)
    state[OFF_HAND:OFF_HAND + 54] = hand
    other = 2 - hand
    other[SMALL_JOKER:] -= 1          # the agent counts one copy per joker here (measured)
    state[OFF_OTHER:OFF_OTHER + 54] = other
    state[OFF_LEVEL + level - 2] = 1.0
    _tail(state, phase, receiver_rel, tribute_rel)
    return state


def batch(state: np.ndarray, plays: np.ndarray) -> np.ndarray:
    """(N, 964) float32 rows: the state repeated beside each 80-dim play."""
    plays = np.asarray(plays, np.float32)
    out = np.empty((len(plays), DIM_ROW), np.float32)
    out[:, :DIM_STATE] = state
    out[:, DIM_STATE:] = plays
    return out
