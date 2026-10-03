"""Rebuild a Botzone round from what one seat has been told.

The bot never sees the other hands, but the engine (``pyengine``, the
pure-Python port of ``gd``) keeps four hands. ``rebuild`` therefore deals
the unseen cards to the other seats in a way consistent with everything
public: each seat starts with every card it later plays or hands over (net of
the cards it received), a tribute payer holds nothing that outranks its
tribute card, and the big jokers sit where the announced anti-tribute
(``resist``) says. It then replays the round (tribute, back-tribute, every
play and pass, ``auto_pass`` off) and appends each applied action to a public
token stream exactly as ``eval.history_policy.apply_and_observe`` does.

Nothing the actor reads depends on that filling: the observation and the
candidate encodings of the acting seat are functions of its own hand and of
public events (``tests/test_botzone.py`` checks this across fillings).

Plays are taken as Botzone declares them: the reading comes from the claim
(``pyengine.classify``), the cards from the action. The replay stops at our
pending decision. If an event it needs has not been announced yet (the judge
asks the two tribute payers, and the two receivers, in its own order), a
stand-in legal choice of that seat is applied; this only happens before our
own tribute or back-tribute, which use the tribute heuristic on our own hand.
Passes the history no longer shows (see ``_ImpliedPasses``) are restored
as passes.

Python 3.6 compatible, NumPy only.
"""
from collections import Counter
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import pyengine as pe
from .protocol import BIG_JOKER, Move, RoundLog, card_to_gd, power, rank_of

TOKEN_DIM = 4 + 154 + 28


class TokenStream(object):
    """The public tokens of one round, laid out as ``train.logs.public_token``.

    A torch-free stand-in for ``train.history_model.PublicStream``;
    ``tests/test_botzone.py`` checks the two agree token for token.
    """

    def __init__(self) -> None:
        self.tokens = []     # type: List[np.ndarray]
        self.rounds = []     # type: List[int]
        self.phases = []     # type: List[int]

    def append(self, event) -> None:
        token = np.zeros(TOKEN_DIM, dtype=np.uint8)
        token[int(event.seat)] = 1
        token[4:158] = event.encoded_action
        token[4 + 146:4 + 154] = 0        # tribute structure is private
        token[158 + int(event.cards_left)] = 1
        self.tokens.append(token)
        self.rounds.append(int(event.round_index))
        self.phases.append(int(event.phase))

    def arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        tokens = (np.stack(self.tokens) if self.tokens
                  else np.zeros((0, TOKEN_DIM), dtype=np.uint8))
        return (tokens, np.asarray(self.rounds, dtype=np.int64),
                np.asarray(self.phases, dtype=np.int64))


class _Event(object):
    __slots__ = ("seat", "phase", "round_index", "encoded_action", "cards_left")

    def __init__(self, seat, phase, round_index, encoded_action, cards_left) -> None:
        self.seat, self.phase, self.round_index = seat, phase, round_index
        self.encoded_action, self.cards_left = encoded_action, cards_left


def apply_and_record(state: pe.State, action: pe.Action, stream) -> None:
    """``state.apply`` plus the public token, like ``apply_and_observe``:
    seat, phase and round before the action, the actor's count after it."""
    seat, phase = state.to_move, state.phase
    encoded = pe.encode_action(action, state, seat)
    encoded[pe.ACT_TRIBUTE_FLAGS:] = 0.0
    state.apply(action)
    stream.append(_Event(seat, phase, state.round_index, encoded, sum(state.hands[seat])))


class MirrorError(RuntimeError):
    """The public record cannot be replayed."""


def partner(seat: int) -> int:
    return (seat + 2) % 4


def tribute_power(face: int, level: int) -> int:
    return power(rank_of(face), level)


def prev_order(log: RoundLog) -> Optional[List[int]]:
    """The previous finishing order the engine needs, or None without tribute.

    Botzone tells only the first and last finisher. With a double tribute
    the first's partner was second and the last's partner third; with a
    single tribute the first's partner was third.
    """
    if log.tribute <= 0:
        return None
    first, last = log.first, log.last
    if first < 0 or last < 0 or (first - last) % 2 == 0:
        raise MirrorError("tribute %d needs first and last on opposite teams, got first=%d "
                          "last=%d" % (log.tribute, first, last))
    if log.tribute >= 2:
        return [first, partner(first), partner(last), last]
    return [first, partner(last), partner(first), last]


def routing(log: RoundLog, faces: Dict[int, int]) -> Dict[int, int]:
    """Payer -> receiver of the tribute cards under the ``house`` rules."""
    order = prev_order(log)
    if order is None:
        return {}
    banker, follower, third, dweller = order
    if log.tribute < 2:
        return {dweller: banker}
    if third not in faces or dweller not in faces:
        return {}
    down, up = (banker + 1) % 4, (banker + 3) % 4
    p_down = tribute_power(faces[down], log.level)
    p_up = tribute_power(faces[up], log.level)
    to_banker = down if p_down >= p_up else up      # tie: TributeTie::Downstream
    return {to_banker: banker, (down if to_banker == up else up): follower}


class Rebuilt(object):
    """A replayed round at our pending decision."""

    def __init__(self, state: pe.State, stream) -> None:
        self.state = state
        self.stream = stream
        self.events = 0
        self.stand_ins = 0
        self.implied_passes = 0
        self.notes = []      # type: List[str]


def _faces(cards) -> Counter:
    return Counter(card_to_gd(c) for c in cards)


def fill_hands(log: RoundLog, rng: random.Random) -> List[List[int]]:
    """Four hands of gd faces consistent with the log (ours is the real one)."""
    me, level = log.me, log.level
    mine = _faces(log.deliver)
    if sum(mine.values()) != 27:
        raise MirrorError("deal of %d cards" % sum(mine.values()))
    tribute_faces = {s: card_to_gd(c) for s, c in log.tribute_cards.items()}
    return_faces = {s: card_to_gd(c) for s, c in log.return_cards.items()}
    route = routing(log, tribute_faces) if not log.resist else {}
    out_of = {s: Counter() for s in range(4)}
    into = {s: Counter() for s in range(4)}
    for move in log.moves:
        out_of[move.seat].update(_faces(move.action))
    for payer, face in tribute_faces.items():
        out_of[payer][face] += 1
        if payer in route:
            into[route[payer]][face] += 1
    for payer, receiver in route.items():
        if receiver in return_faces:
            out_of[receiver][return_faces[receiver]] += 1
            into[payer][return_faces[receiver]] += 1
    hold = {s: Counter({f: n - into[s][f] for f, n in out_of[s].items() if n > into[s][f]})
            for s in range(4) if s != me}
    pool = Counter({f: 2 for f in range(54)})
    pool.subtract(mine)
    for s, counter in hold.items():
        pool.subtract(counter)
    if any(n < 0 for n in pool.values()):
        raise MirrorError("public cards exceed the deck")
    order = prev_order(log)
    payers = []          # type: List[int]
    if order is not None:
        payers = [order[2], order[3]] if log.tribute >= 2 else [order[3]]
    others = [s for s in range(4) if s != me]
    need = {s: 27 - sum(hold[s].values()) for s in others}
    if any(n < 0 for n in need.values()):
        raise MirrorError("a seat played more than 27 cards")

    def take(seat: int, face: int) -> None:
        pool[face] -= 1
        hold[seat][face] += 1
        need[seat] -= 1

    # Big jokers first: the anti-tribute is a fact about the payers' hands.
    payer_others = [s for s in payers if s != me]
    jokers = [sum((mine if s == me else hold[s])[BIG_JOKER] for s in payers)]
    if order is not None and log.resist:
        for seat in payer_others:
            while jokers[0] < 2 and pool[BIG_JOKER] > 0 and need[seat] > 0:
                take(seat, BIG_JOKER)
                jokers[0] += 1
        if jokers[0] < 2:
            raise MirrorError("anti-tribute announced but the payers cannot hold both big jokers")
    capped = {}
    for seat in payer_others:
        if seat in tribute_faces and not log.resist:
            capped[seat] = tribute_power(tribute_faces[seat], level)
    blocked_joker = order is not None and not log.resist

    def allowed(seat: int, face: int) -> bool:
        if (seat in capped and face != level * 4 + 1
                and tribute_power(face, level) > capped[seat]):
            return False
        if blocked_joker and seat in payers and face == BIG_JOKER and jokers[0] >= 1:
            return False
        return True

    # Most constrained cards first (fewest seats may hold them): the caps
    # are nested, so placing the high cards before the low ones cannot strand
    # a card that only an already full seat could take.
    cards = list(pool.elements())
    rng.shuffle(cards)
    cards.sort(key=lambda f: sum(allowed(s, f) for s in others))
    for face in cards:
        seats = [s for s in others if need[s] > 0 and allowed(s, face)]
        if not seats:
            raise MirrorError("no seat can hold card %d consistently" % face)
        seat = rng.choice(seats)
        if face == BIG_JOKER and seat in payers:
            jokers[0] += 1
        take(seat, face)
    hands = []
    for s in range(4):
        counter = mine if s == me else hold[s]
        hands.append(sorted(counter.elements()))
    return hands


def make_state(log: RoundLog, hands: Sequence[Sequence[int]]) -> pe.State:
    leader = log.leader_hint if log.leader_hint >= 0 else 0
    return pe.State(hands, log.level, [log.level, log.level], leader, prev_order(log))


def _card_action(state: pe.State, face: int) -> Optional[pe.Action]:
    for action in state.legal_actions():
        if action.cards == [face]:
            return action
    return None


def _move_action(move: Move, level: int) -> pe.Action:
    try:
        return pe.classify([card_to_gd(c) for c in move.claim],
                           [card_to_gd(c) for c in move.action], level)
    except ValueError as error:
        raise MirrorError(str(error))


def _playable(state: pe.State, action: pe.Action) -> bool:
    """Whether the seat to move may make this play now (Botzone's reading)."""
    top = state.top
    if action.is_pass:
        return not top.is_pass
    hand = state.hands[state.to_move]
    if any(hand[c] < n for c, n in enumerate(action.counts)):
        return False
    return pe.beats_reading(action.type, action.key, action.bomb_size,
                            top.type, top.key, top.bomb_size)


class _ImpliedPasses(object):
    """Passes the play history no longer shows.

    A seat moves twice between two of our turns only when the lead jumps to
    a finished seat's partner (jiefeng). The platform's last-four list then
    keeps every move unless more than four happened; positional slots keep
    each seat's latest move only and lose the passes that closed the
    finished seat's trick. A pass may be restored for a seat other than ours
    that still moves later in the same window, at most once per window.
    """

    def __init__(self, moves: Sequence[Move], me: int) -> None:
        self.me = me
        self.used = set()    # type: set
        # For each position: the end of its window (our next move) and the
        # seats that still move in the window from there on.
        self.end = [len(moves)] * (len(moves) + 1)
        self.later = [frozenset()] * (len(moves) + 1)
        for i in range(len(moves) - 1, -1, -1):
            if moves[i].seat == me:
                self.end[i], self.later[i] = i, frozenset()
            else:
                self.end[i] = self.end[i + 1]
                self.later[i] = self.later[i + 1] | {moves[i].seat}

    def allowed(self, position: int, seat: int) -> bool:
        return (seat != self.me and seat in self.later[position]
                and (self.end[position], seat) not in self.used)

    def use(self, position: int, seat: int) -> None:
        self.used.add((self.end[position], seat))

    def jiefeng_pending(self, state: pe.State, seat: int, holder: int, position: int) -> bool:
        """The trick holder has finished, ``seat`` is its partner and every
        other seat that must still pass before the lead comes back may have
        lost that pass: read the partner's play as the jiefeng lead.

        The alternative, the partner beating its own finished partner's last
        play, is legal but almost never sensible, and the history cannot tell
        the two apart.
        """
        if holder < 0 or any(state.hands[holder]) or seat != partner(holder):
            return False
        between = [s for s in ((seat + 1) % 4,) if any(state.hands[s])]
        return all(self.allowed(position, s) for s in between)


def rebuild(log: RoundLog, seed: int = 0, stream_factory=None) -> Rebuilt:
    """Replay the round up to our pending decision (``state.to_move == log.me``)."""
    state = make_state(log, fill_hands(log, random.Random(seed)))
    stream = (stream_factory or TokenStream)()
    built = Rebuilt(state, stream)
    me, level = log.me, log.level
    tribute_faces = {s: card_to_gd(c) for s, c in log.tribute_cards.items()}
    return_faces = {s: card_to_gd(c) for s, c in log.return_cards.items()}
    if prev_order(log) is not None and log.resist != state.anti_tribute:
        raise MirrorError("anti-tribute mismatch: botzone %s, replay %s"
                          % (log.resist, state.anti_tribute))
    moves = list(log.moves)
    implied = _ImpliedPasses(moves, me)
    holder = -1
    position = 0
    guard = 0
    while True:
        guard += 1
        if guard > 1000:
            raise MirrorError("replay does not terminate")
        phase, seat = state.phase, state.to_move
        if phase in (pe.PHASE_TRIBUTE, pe.PHASE_BACK_TRIBUTE):
            known = (tribute_faces if phase == pe.PHASE_TRIBUTE else return_faces).get(seat)
            if known is None:
                if seat == me:
                    return built
                # Not announced yet: a stand-in for a choice ours does not depend on.
                action = state.legal_actions()[0]
                built.stand_ins += 1
            else:
                action = _card_action(state, known)
                if action is None:
                    raise MirrorError("%s %d of seat %d is not legal" % (
                        "tribute" if phase == pe.PHASE_TRIBUTE else "return", known, seat))
            apply_and_record(state, action, stream)
            built.events += 1
            continue
        if phase != pe.PHASE_PLAY:
            raise MirrorError("replay reached phase %d with %d moves left"
                              % (phase, len(moves) - position))
        if position == len(moves):
            if seat != me:
                raise MirrorError("replay ends with seat %d to move, not ours (%d)" % (seat, me))
            if built.stand_ins:
                raise MirrorError("a stand-in tribute choice reached the play phase")
            return built
        move = moves[position]
        action = _move_action(move, level)
        following = not state.top.is_pass
        if following and implied.allowed(position, seat):
            playable = move.seat == seat and _playable(state, action)
            # Only positional slots can hide the passes before a jiefeng lead;
            # the platform's last-four list keeps them.
            if not playable or (log.positional and not move.is_pass
                                and implied.jiefeng_pending(state, seat, holder, position)):
                apply_and_record(state, pe.PASS_ACTION, stream)
                implied.use(position, seat)
                built.events += 1
                built.implied_passes += 1
                continue
        if move.seat != seat:
            raise MirrorError("move %d by seat %d, the replay has seat %d to move"
                              % (position, move.seat, seat))
        if not _playable(state, action):
            raise MirrorError("move %d %r is not playable" % (position, move))
        if not following:
            holder = -1                       # a new trick starts with this lead
        if not action.is_pass:
            holder = seat
        apply_and_record(state, action, stream)
        built.events += 1
        position += 1


def own_hand(log: RoundLog, built: Rebuilt) -> List[int]:
    """Our physical ids, matched to the replay's view of our hand."""
    me = log.me
    physical = list(log.deliver)
    tribute_route = routing(log, {s: card_to_gd(c) for s, c in log.tribute_cards.items()})
    if log.my_tribute is not None:
        physical.remove(log.my_tribute)
    for payer, receiver in tribute_route.items():
        if receiver == me and payer in log.tribute_cards:
            physical.append(log.tribute_cards[payer])
    if log.my_return is not None:
        physical.remove(log.my_return)
    for payer, receiver in tribute_route.items():
        if payer == me and receiver in log.return_cards:
            physical.append(log.return_cards[receiver])
    for move in log.moves:
        if move.seat == me:
            for card in move.action:
                physical.remove(card)
    if Counter(card_to_gd(c) for c in physical) != Counter(built.state.hand(me)):
        built.notes.append("own_hand_mismatch")
    return physical


def decision_arrays(built: Rebuilt, actions: Sequence[pe.Action]
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Observation and candidate encodings of the seat to move."""
    state = built.state
    seat = state.to_move
    obs = pe.encode_observation(state, seat)
    cand = np.stack([pe.encode_action(a, state, seat) for a in actions])
    return obs, cand
