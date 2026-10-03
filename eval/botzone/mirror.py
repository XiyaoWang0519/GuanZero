"""Rebuild a Botzone round in ``gd`` from what one seat has been told.

The bot never sees the other hands, but ``gd`` needs four hands to apply
plays. ``rebuild`` therefore deals the unseen cards to the other seats in a
way consistent with everything public: each seat starts with every card it
later plays or hands over (net of the cards it received), a tribute payer
holds nothing that outranks its tribute card, and the big jokers sit where
the announced anti-tribute (``resist``) says. It then replays the round
(tribute, back-tribute, every play and pass) with ``auto_pass`` off and
feeds each applied action to a public token stream exactly as the evaluators
do (``eval.history_policy.apply_and_observe``).

Nothing the actor reads depends on that filling: the observation and the
candidate encodings of the acting seat are functions of its own hand and of
public events (``tests/test_botzone.py`` checks this across fillings).

The replay stops at the seat's pending decision. If an event the replay
needs has not been announced yet (the judge asks the two tribute payers, and
the two receivers, in its own order), a stand-in legal choice of that seat
is applied; this only happens before our own tribute or back-tribute, which
use the engine's tribute heuristic on our own hand. Passes the positional
history overwrote (see ``rebuild``) are restored as passes.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import random

import gd
import numpy as np

from eval.botzone.protocol import (BIG_JOKER, Move, RoundLog, card_to_gd, claim_matches,
                                   power, rank_of)
from eval.history_policy import apply_and_observe

TOKEN_DIM = 4 + 154 + 28

TRIBUTE = int(gd.Phase.Tribute)
BACK_TRIBUTE = int(gd.Phase.BackTribute)
PLAY = int(gd.Phase.Play)


class TokenStream:
    """The public tokens of one round, laid out as ``train.logs.public_token``.

    A torch-free stand-in for ``train.history_model.PublicStream`` (which
    imports torch); ``tests/test_botzone.py`` checks the two agree token for
    token.
    """

    def __init__(self) -> None:
        self.tokens: list[np.ndarray] = []
        self.rounds: list[int] = []
        self.phases: list[int] = []

    def append(self, event) -> None:
        token = np.zeros(TOKEN_DIM, dtype=np.uint8)
        token[int(event.seat)] = 1
        token[4:158] = event.encoded_action
        token[4 + 146:4 + 154] = 0        # tribute structure is private
        token[158 + int(event.cards_left)] = 1
        self.tokens.append(token)
        self.rounds.append(int(event.round_index))
        self.phases.append(int(event.phase))

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tokens = (np.stack(self.tokens) if self.tokens
                  else np.zeros((0, TOKEN_DIM), dtype=np.uint8))
        return (tokens, np.asarray(self.rounds, dtype=np.int64),
                np.asarray(self.phases, dtype=np.int64))


class MirrorError(RuntimeError):
    """The public record cannot be replayed in ``gd``."""


def partner(seat: int) -> int:
    return (seat + 2) % 4


def tribute_power(face: int, level: int) -> int:
    return power(rank_of(face), level)


def prev_order(log: RoundLog) -> list[int] | None:
    """The previous finishing order gd needs, or None without tribute.

    Botzone tells only the first and last finisher. With a double tribute
    the first's partner was second and the last's partner third; with a
    single tribute the first's partner was third.
    """
    if log.tribute <= 0:
        return None
    first, last = log.first, log.last
    if first < 0 or last < 0 or (first - last) % 2 == 0:
        raise MirrorError(f"tribute {log.tribute} needs first and last on opposite teams, "
                          f"got first={first} last={last}")
    if log.tribute >= 2:
        return [first, partner(first), partner(last), last]
    return [first, partner(last), partner(first), last]


def routing(log: RoundLog, faces: dict[int, int]) -> dict[int, int]:
    """Payer -> receiver of the tribute cards, as gd's ``house`` rules route them."""
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


@dataclass
class Rebuilt:
    """A replayed round at our pending decision."""
    engine: gd.Engine
    canon: gd.Engine
    state: gd.MatchState
    stream: object
    events: int
    stand_ins: int = 0
    implied_passes: int = 0
    notes: list[str] = field(default_factory=list)


def _faces(cards) -> Counter:
    return Counter(card_to_gd(c) for c in cards)


def fill_hands(log: RoundLog, rng: random.Random) -> list[list[int]]:
    """Four gd hands consistent with the log (ours is the real one)."""
    me, level = log.me, log.level
    mine = _faces(log.deliver)
    if sum(mine.values()) != 27:
        raise MirrorError(f"deal of {sum(mine.values())} cards")
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
    payers = []
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
    jokers_with_payers = sum((mine if s == me else hold[s])[BIG_JOKER] for s in payers)
    if order is not None and log.resist:
        for seat in payer_others:
            while jokers_with_payers < 2 and pool[BIG_JOKER] > 0 and need[seat] > 0:
                take(seat, BIG_JOKER)
                jokers_with_payers += 1
        if jokers_with_payers < 2:
            raise MirrorError("anti-tribute announced but the payers cannot hold both big jokers")
    capped = {}
    for seat in payer_others:
        if seat in tribute_faces and not log.resist:
            capped[seat] = tribute_power(tribute_faces[seat], level)
    blocked_joker = order is not None and not log.resist

    def allowed(seat: int, face: int) -> bool:
        if seat in capped and not (face == level * 4 + 1) and tribute_power(face, level) > capped[seat]:
            return False
        if blocked_joker and seat in payers and face == BIG_JOKER and jokers_with_payers >= 1:
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
            raise MirrorError(f"no seat can hold {gd.card_str(face)} consistently")
        seat = rng.choice(seats)
        if face == BIG_JOKER and seat in payers:
            jokers_with_payers += 1
        take(seat, face)
    hands = []
    for s in range(4):
        counter = mine if s == me else hold[s]
        hands.append(sorted(counter.elements()))
    return hands


def make_deal(log: RoundLog, hands: list[list[int]]) -> gd.DealSpec:
    deal = gd.DealSpec()
    deal.hands = hands
    deal.level = log.level
    deal.team_levels = [log.level, log.level]
    deal.owner = -1
    order = prev_order(log)
    if order is not None:
        deal.prev_order = order
    else:
        deal.leader = log.leader_hint if log.leader_hint >= 0 else 0
    return deal


def _match_play(actions, move: Move, level: int):
    if move.is_pass:
        for action in actions:
            if action.is_pass:
                return action, "exact"
        return None, "missing"
    faces = sorted(card_to_gd(c) for c in move.action)
    same = [a for a in actions if not a.is_pass and sorted(a.cards) == faces]
    for action in same:
        if claim_matches(action.type, int(action.key), list(action.cards), level, move.claim):
            return action, "exact"
    if same:
        return same[0], "reading"
    return None, "missing"


def _match_card(engine: gd.Engine, state: gd.MatchState, face: int):
    for action in engine.legal_actions(state):
        if list(action.cards) == [face]:
            return action
    return None


class _ImpliedPasses:
    """Passes the positional play history overwrote.

    Between two of our turns the history keeps each seat's latest move only.
    A seat moves twice in that window only when the lead jumps to a finished
    seat's partner (jiefeng): the passes that closed the finished seat's
    trick are lost. A pass may therefore be restored for a seat other than
    ours that still moves later in the same window, at most once per window.
    """

    def __init__(self, moves: list[Move], me: int) -> None:
        self.me = me
        self.used: set[tuple[int, int]] = set()
        # For each position: the end of its window (our next move) and the
        # seats that still move in the window from there on.
        self.end = [len(moves)] * (len(moves) + 1)
        self.later: list[frozenset] = [frozenset()] * (len(moves) + 1)
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

    def jiefeng_pending(self, state: gd.MatchState, seat: int, holder: int,
                        position: int) -> bool:
        """The trick holder has finished, ``seat`` is its partner and every
        other seat that must still pass before the lead comes back may have
        lost that pass: read the partner's play as the jiefeng lead.

        The alternative, the partner beating its own finished partner's last
        play, is legal but almost never sensible, and the history cannot tell
        the two apart.
        """
        if holder < 0 or len(state.hand(holder)) or seat != partner(holder):
            return False
        between = [s for s in ((seat + 1) % 4,) if len(state.hand(s))]
        return all(self.allowed(position, s) for s in between)


def rebuild(log: RoundLog, *, seed: int = 0, stream_factory=None) -> Rebuilt:
    """Replay the round up to our pending decision (``state.to_move == log.me``)."""
    rules = gd.RuleConfig.house()
    engine = gd.Engine(rules, gd.ActionConfig.full())
    canon = gd.Engine(rules, gd.ActionConfig())
    engine.auto_pass = False
    canon.auto_pass = False
    state = gd.MatchState()
    hands = fill_hands(log, random.Random(seed))
    engine.set_deal(state, make_deal(log, hands))
    stream = (stream_factory or TokenStream)()
    built = Rebuilt(engine, canon, state, stream, 0)

    class _Listener:
        def observe(self, event) -> None:
            stream.append(event)

    listeners = (_Listener(),)
    me, level = log.me, log.level
    tribute_faces = {s: card_to_gd(c) for s, c in log.tribute_cards.items()}
    return_faces = {s: card_to_gd(c) for s, c in log.return_cards.items()}
    if prev_order(log) is not None and log.resist != (int(state.phase) == PLAY):
        raise MirrorError(f"anti-tribute mismatch: botzone {log.resist}, "
                          f"gd {int(state.phase) == PLAY}")
    moves = list(log.moves)
    implied = _ImpliedPasses(moves, me)
    holder = -1
    position = 0
    guard = 0
    while True:
        guard += 1
        if guard > 1000:
            raise MirrorError("replay does not terminate")
        phase = int(state.phase)
        seat = int(state.to_move)
        if phase in (TRIBUTE, BACK_TRIBUTE):
            known = (tribute_faces if phase == TRIBUTE else return_faces).get(seat)
            if known is None:
                if seat == me:
                    return built
                # Not announced yet: a stand-in for a choice ours does not depend on.
                action = engine.legal_actions(state)[0]
                built.stand_ins += 1
            else:
                action = _match_card(engine, state, known)
                if action is None:
                    raise MirrorError(f"{'tribute' if phase == TRIBUTE else 'return'} "
                                      f"{gd.card_str(known)} of seat {seat} is not legal in gd")
            apply_and_observe(engine, state, action, listeners)
            built.events += 1
            continue
        if phase != PLAY:
            raise MirrorError(f"replay reached phase {phase} with {len(moves) - position} "
                              "moves left")
        if position == len(moves):
            if seat != me:
                raise MirrorError(f"replay ends with seat {seat} to move, not ours ({me})")
            if built.stand_ins:
                raise MirrorError("a stand-in tribute choice reached the play phase")
            return built
        move = moves[position]
        actions = engine.legal_actions(state)
        pass_action = next((a for a in actions if a.is_pass), None)
        if pass_action is not None and implied.allowed(position, seat):
            matched = None if move.seat != seat else _match_play(actions, move, level)[0]
            if matched is None or (not move.is_pass
                                   and implied.jiefeng_pending(state, seat, holder, position)):
                apply_and_observe(engine, state, pass_action, listeners)
                implied.use(position, seat)
                built.events += 1
                built.implied_passes += 1
                continue
        if move.seat != seat:
            raise MirrorError(f"move {position} by seat {move.seat}, gd has seat {seat} to move")
        action, how = _match_play(actions, move, level)
        if action is None:
            raise MirrorError(f"move {position} {move} is not legal in gd")
        if how != "exact":
            built.notes.append(f"reading:{position}")
        if pass_action is None:
            holder = -1                       # a new trick starts with this lead
        if not action.is_pass:
            holder = seat
        apply_and_observe(engine, state, action, listeners)
        built.events += 1
        position += 1


def own_hand(log: RoundLog, built: Rebuilt) -> list[int]:
    """Our physical ids, matched to gd's view of our hand."""
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


def decision_arrays(built: Rebuilt, actions) -> tuple[np.ndarray, np.ndarray]:
    """Observation and candidate encodings of the seat to move."""
    state, canon = built.state, built.canon
    seat = int(state.to_move)
    obs = np.asarray(state.observation(seat), dtype=np.float32)
    cand = np.stack([np.asarray(canon.encode_action(a, state, seat), dtype=np.float32)
                     for a in actions])
    return obs, cand
