"""DanZero-V1T as a ``gd`` player: pure Python, numpy and onnxruntime.

The compiled DanLM agent (``danzero.eval.agents.MLPTributeAgent``, spec
``v1t:<onnx>``) was probed as a black box (``eval/v1t/verify.py``); this is
what it does, reproduced on our engine's state:

* **Play.** One 964-float row per legal play (``encoder.play_state`` +
  the play's 80-dim vector), one ONNX call, the first row with the largest
  Q value is chosen (``np.argmax``). Nothing is sampled; no rule overrides
  the network. The agent keeps no history beyond the tribute exchange: the
  state reads only the current hand, cards played per seat, the current
  trick, finished seats, the round level and the tribute record.
* **Legal plays** are DanLM's set, rebuilt from our full-mode list: plays
  identical in DanLM's 80-dim encoding are merged; a full house made of
  level-rank cards only (wilds standing in for another rank) is dropped,
  since DanLM lists those cards as a bomb only; and DanLM's extra wild
  declarations are added: a lone wild card as a single of any natural rank,
  two wild cards as a pair of any natural rank (``RULES.md`` 13.3). When the
  network picks such a declaration, our engine plays the same cards at the
  level rank, as the lockstep arena does (``their_choice_reading``).
* **Tribute and back-tribute** are model-driven: one row per legal card (the
  card as a single), state from ``encoder.tribute_state`` with the receiver's
  relative seat (3 when a double tribute's receiver is not yet known) and the
  cards received so far, argmax over the rows.
* **Anti-tribute**: the record marks big-joker holders (2 for a lone holder,
  1 each for two), exactly as ``eval/danlm/arena.py`` tells the compiled agent.

The player follows the ``eval.policies`` protocol (``select``) plus the
history protocol (``needs_history``, ``start_match``, ``observe``): every
applied action of every seat must reach ``observe``, with forced passes
explicit. ``begin_round(deal, state)`` must be called once per round after
``set_deal`` with the round's ``gd.DealSpec`` (the previous finishing order is
public and decides who receives tribute).
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence

import gd
import numpy as np

from eval.danlm.bridge import (BIG_JOKER, card_ours_to_theirs, card_theirs_to_ours,
                               hand_ours_to_theirs, power, their_rank_from_key)
from eval.v1t import encoder as enc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / ".work/external/DanLM/ckpts/DanZero_v3_rep_v1t/v3_rep_v1t_best_eval_001_int8.onnx"
MODEL_ENV = "V1T_ONNX"

_TYPE_TO_DANLM = {"Pass": "pass", "Single": "single", "Pair": "double", "Triple": "triple",
                  "FullHouse": "fullhouse", "Tube": "tube", "Plate": "plate",
                  "Straight": "straight", "Bomb": "normalbomb", "StraightFlush": "flushbomb",
                  "JokerBomb": "jokerbomb"}
_RANKLESS = ("Pass", "JokerBomb")
PLAY_PHASE = int(gd.Phase.Play)
TRIBUTE_PHASE = int(gd.Phase.Tribute)
BACK_PHASE = int(gd.Phase.BackTribute)


def default_model_path() -> Path:
    return Path(os.environ.get(MODEL_ENV, DEFAULT_MODEL))


class V1TModel:
    """The int8 ONNX Q-network, one thread, the session options DanLM uses."""

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        import onnxruntime as ort

        self.path = Path(path) if path else default_model_path()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.path), options,
                                            providers=["CPUExecutionProvider"])

    def __call__(self, rows: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"input": np.ascontiguousarray(rows, np.float32)})[0]


# -- plays -------------------------------------------------------------------

def play_vector(type_name: str, cards_ours: Sequence[int], rank: int | None) -> np.ndarray:
    """DanLM 80-dim play (float32) from a gd type name, gd cards and DanLM rank."""
    row = np.zeros(enc.DIM_PLAY, np.float32)
    for card in cards_ours:
        row[card_ours_to_theirs(int(card))] += 1.0
    row[enc.DIM_CARDS + enc.TYPE_INDEX[_TYPE_TO_DANLM[type_name]]] = 1.0
    if rank is not None:
        row[enc.DIM_CARDS + 11 + rank] = 1.0
    return row


def action_vector(action, level: int) -> np.ndarray:
    """A ``gd.Action`` in DanLM's encoding; pass and joker bomb carry no rank bit."""
    rank = None if action.type in _RANKLESS else their_rank_from_key(action.type, int(action.key), level)
    return play_vector(action.type, action.cards, rank)


def action_identity(action) -> tuple:
    key = None if action.type in _RANKLESS else int(action.key)
    return action.type, tuple(sorted(int(c) for c in action.cards)), key


def _vector_power(vector: np.ndarray, level: int) -> int:
    """gd power of a single/pair DanLM vector (natural rank -> power, jokers 13/14)."""
    return power(int(vector[enc.DIM_CARDS + 11:].argmax()), level)


def danlm_order_key(play: np.ndarray, level: int) -> tuple:
    """Where DanLM's move generator lists a play, as far as probing could pin it down.

    ``hand_calculator_v2`` lists plays by type index, then rank; inside one type
    and rank, singles, pairs, triples and bombs come with the fewest wild cards
    first, then in ascending lexicographic order of their 54 card counts (for
    singles: the highest card id first). Measured on 400 random hands: exact
    for singles, pairs and triples; sequences and full houses follow another
    (unidentified) order, and wild-card plays of other ranks can be listed out
    of rank order.
    """
    type_index = int(play[enc.DIM_CARDS:enc.DIM_CARDS + 11].argmax())
    rank_bits = play[enc.DIM_CARDS + 11:]
    rank = int(rank_bits.argmax()) if rank_bits.any() else -1
    return type_index, rank, int(play[level]), tuple(play[:enc.DIM_CARDS].tolist())


def break_ties(q: np.ndarray, plays: np.ndarray, level: int) -> int:
    """Index of the row DanLM's ``argmax`` would pick among rows tied at the maximum.

    DanLM takes the first maximum in its own legal-list order, which we do not
    rebuild in full. Ties are rare (about 1 in 30,000 decisions) and arise
    between near-identical plays (the int8 network can map different suits of
    one rank to the same value), so ``danlm_order_key`` decides them.
    """
    best = np.flatnonzero(q == q.max())
    if len(best) == 1:
        return int(best[0])
    return int(min(best, key=lambda i: danlm_order_key(plays[i], level)))


@dataclass
class LegalPlays:
    rows: np.ndarray                 # (N, 80) float32, DanLM encoding
    actions: list                    # gd action played for each row
    declared: list[bool]             # row is a DanLM-only wild declaration


def danlm_legal_plays(actions: Sequence, hand: Sequence[int], level: int,
                      lead: np.ndarray | None) -> LegalPlays:
    """DanLM's legal set for a decision, from our FULL-mode legal list.

    ``lead`` is the trick's current top play (DanLM vector) or ``None`` when
    leading. Order: our engine's order, then the declarations by rank.
    """
    rows: list[np.ndarray] = []
    chosen: list = []
    declared: list[bool] = []
    seen: set[bytes] = set()
    for action in actions:
        if action.type == "FullHouse" and all(int(c) // 4 == level for c in action.cards):
            continue
        row = action_vector(action, level)
        key = row.tobytes()
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
        chosen.append(action)
        declared.append(False)
    wild = level * 4 + 1                       # heart of the level rank
    wilds = sum(1 for c in hand if int(c) == wild)
    lead_type = None if lead is None else int(lead[enc.DIM_CARDS:enc.DIM_CARDS + 11].argmax())
    for count, type_name, type_index in ((1, "Single", 1), (2, "Pair", 2)):
        if wilds < count or (lead is not None and lead_type != type_index):
            continue
        reading = next((a for a in actions if a.type == type_name
                        and [int(c) for c in a.cards] == [wild] * count), None)
        for rank in range(13):
            if rank == level:
                continue
            if lead is not None and power(rank, level) <= _vector_power(lead, level):
                continue
            row = play_vector(type_name, [wild] * count, rank)
            if row.tobytes() in seen:
                continue
            seen.add(row.tobytes())
            rows.append(row)
            chosen.append(reading)
            declared.append(True)
    return LegalPlays(np.stack(rows) if rows else np.zeros((0, enc.DIM_PLAY), np.float32),
                      chosen, declared)


# -- public round record -------------------------------------------------------

def danlm_card_power(card: int, level: int) -> int:
    """DanLM ``single_card_power`` (DanLM card, gd level 0..12): rank, level 98, jokers 99/100."""
    if card >= enc.SMALL_JOKER:
        return 99 + card - enc.SMALL_JOKER
    return 98 if card % 13 == level else card % 13


class RoundRecord:
    """What the V1T state needs from the public history of one round."""

    def __init__(self, level: int, prev_order: Sequence[int] | None,
                 anti: np.ndarray | None) -> None:
        self.level = level                         # gd 0..12
        self.prev_order = list(prev_order) if prev_order else None
        self.anti = anti
        self.records: list[tuple[int, int, int]] = []     # (giver, receiver, DanLM card)
        self.gives: dict[int, int] = {}
        self.receivers: dict[int, int] = {}
        self.played = [np.zeros(enc.DIM_CARDS, np.float32) for _ in range(4)]
        self.trick: list[np.ndarray | None] = [None] * 4
        self.trick_open = True
        self.top: np.ndarray | None = None
        self.holder = -1
        self.passes = 0
        self.finished: list[int] = []

    @property
    def is_double(self) -> bool:
        p3, p4 = self.prev_order[2], self.prev_order[3]
        return (p3 - p4) % 4 == 2

    def tribute_absolute(self) -> np.ndarray | None:
        """``MLPTributeAgent._tribute_received_abs`` after ``notify_tribute``."""
        if self.anti is not None:
            return self.anti
        if self.prev_order is None:
            return None
        absolute = np.zeros((4, enc.DIM_CARDS), np.float32)
        for _, receiver, card in self.records:
            if card >= 0:
                absolute[receiver, card] += 1.0
        return absolute

    def back_to(self, seat: int) -> int:
        return next((payer for payer, r in self.receivers.items() if r == seat), self.prev_order[3])

    def on_tribute(self, seat: int, card: int) -> None:
        p1, p2, p3, p4 = self.prev_order
        self.gives[seat] = card
        if not self.is_double:
            self.receivers[seat] = p1
            self.records.append((seat, p1, card))
        elif len(self.gives) == 2:
            power4 = danlm_card_power(self.gives[p4], self.level)
            power3 = danlm_card_power(self.gives[p3], self.level)
            if power4 == power3:
                self.receivers.update({p4: (p4 + 3) % 4, p3: (p3 + 3) % 4})
            else:
                self.receivers.update({p4: p1, p3: p2} if power4 > power3 else {p4: p2, p3: p1})
            for payer in (p4, p3):
                self.records.append((payer, self.receivers[payer], self.gives[payer]))

    def on_back(self, seat: int, card: int) -> None:
        self.records.append((seat, self.back_to(seat), card))

    def on_play(self, seat: int, vector: np.ndarray, is_pass: bool, cards_left: int) -> None:
        if is_pass:
            self.trick[seat] = vector
            self.passes += 1
            active = sum(1 for s in range(4) if s != self.holder and s not in self.finished)
            if self.passes >= active:
                self.trick_open = True
            return
        if self.trick_open:
            self.trick = [None] * 4
            self.trick_open = False
        self.trick[seat] = vector
        self.top = vector
        self.holder = seat
        self.passes = 0
        self.played[seat] = self.played[seat] + vector[:enc.DIM_CARDS]
        if cards_left == 0 and seat not in self.finished:
            self.finished.append(seat)

    def visible_trick(self) -> list[np.ndarray | None]:
        return [None] * 4 if self.trick_open else list(self.trick)


@dataclass
class Decision:
    """One V1T decision: model rows, Q values, the chosen row and gd action."""
    phase: str                       # "play", "give" or "back"
    rows: np.ndarray                 # (N, 964)
    q: np.ndarray                    # (N,)
    choice: int
    vector: np.ndarray               # chosen 80-dim play
    action: Any                      # gd action to apply
    declared: bool = False           # DanLM-only wild declaration, played at the level rank
    legal_cards: list[int] | None = None   # tribute: DanLM card ints, ascending


def anti_tribute_matrix(deal: gd.DealSpec, state: gd.MatchState) -> np.ndarray | None:
    """The arena's anti-tribute record when the engine skipped the tribute phase."""
    order = [int(s) for s in deal.prev_order]
    if sorted(order) != [0, 1, 2, 3] or state.phase != gd.Phase.Play:
        return None
    p1, p2, p3, p4 = order
    is_double = (p3 - p4) % 4 == 2
    holders = [s for s in ((p3, p4) if is_double else (p4,))
               if list(state.hand(s)).count(BIG_JOKER) >= 2]
    if not holders and is_double:
        holders = [p4, p3]
    anti = np.zeros((4, enc.DIM_CARDS), np.float32)
    for seat in holders:
        anti[seat, BIG_JOKER] += 2.0 if len(holders) == 1 else 1.0
    return anti


class V1TPlayer:
    """DanZero-V1T behind the scalar ``Policy`` and history protocols."""

    needs_history = True

    def __init__(self, model: Any = None, name: str = "danzero-v1t") -> None:
        # a path (or None for the default file), or any callable rows -> Q values
        self.model = V1TModel(model) if model is None or isinstance(model, (str, os.PathLike)) else model
        self.name = name
        self.rules = gd.RuleConfig.house()
        self.engine_full = gd.Engine(self.rules, gd.ActionConfig.full())
        self.engine_full.auto_pass = False
        self.events = 0
        self.round: RoundRecord | None = None
        self.pending: tuple[int, list[int] | None, np.ndarray | None] | None = None
        self.decisions = 0
        self.declared_choices = 0

    # -- history protocol -----------------------------------------------------

    def begin_round(self, deal: gd.DealSpec, state: gd.MatchState) -> None:
        """Public round context: level, previous finishing order, anti-tribute."""
        order = [int(s) for s in deal.prev_order]
        has_tribute = sorted(order) == [0, 1, 2, 3]
        self.pending = (int(deal.level), order if has_tribute else None,
                        anti_tribute_matrix(deal, state) if has_tribute else None)
        self.round = RoundRecord(*self.pending)

    def start_match(self, match_id: int = -1) -> None:
        self.events = 0
        self.round = RoundRecord(*self.pending) if self.pending else None

    @property
    def events_seen(self) -> int:
        return self.events

    def observe(self, event: Any) -> None:
        self.events += 1
        record = self._record()
        action = event.action
        if action is None:
            raise ValueError("V1TPlayer needs the gd action in every event")
        phase = int(event.phase)
        if phase == TRIBUTE_PHASE:
            record.on_tribute(int(event.seat), card_ours_to_theirs(int(action.cards[0])))
        elif phase == BACK_PHASE:
            record.on_back(int(event.seat), card_ours_to_theirs(int(action.cards[0])))
        elif phase == PLAY_PHASE:
            record.on_play(int(event.seat), action_vector(action, record.level),
                           bool(action.is_pass), int(event.cards_left))

    def _record(self) -> RoundRecord:
        if self.round is None:
            raise RuntimeError(f"{self.name}: begin_round(deal, state) was not called")
        return self.round

    # -- decisions --------------------------------------------------------------

    def decide(self, state: gd.MatchState) -> Decision:
        record = self._record()
        seat = int(state.to_move)
        level = record.level
        level_t = level + 2
        hand_ours = [int(c) for c in state.hand(seat)]
        hand = hand_ours_to_theirs(hand_ours)
        phase = state.phase
        if phase in (gd.Phase.Tribute, gd.Phase.BackTribute):
            actions = self.engine_full.legal_actions(state)
            cards = sorted({card_ours_to_theirs(int(a.cards[0])) for a in actions})
            if phase == gd.Phase.Tribute:
                receiver = None if record.is_double else record.prev_order[0]
                receiver_rel = 3 if receiver is None else (receiver - seat) % 4 - 1
                tribute_rel = enc.tribute_received_rel([], seat)
                name = "give"
            else:
                receiver_rel = (record.back_to(seat) - seat) % 4 - 1
                tribute_rel = enc.tribute_received_rel(record.records, seat)
                name = "back"
            state_row = enc.tribute_state(hand, level_t, name, receiver_rel, tribute_rel)
            rows = enc.batch(state_row, np.stack([enc.single_play(c) for c in cards]))
            q = self.model(rows)
            choice = int(np.argmax(q))
            card = card_theirs_to_ours(cards[choice])
            action = next(a for a in actions if int(a.cards[0]) == card)
            self.decisions += 1
            return Decision(name, rows, q, choice, rows[choice, enc.DIM_STATE:], action,
                            legal_cards=cards)
        if phase != gd.Phase.Play:
            raise ValueError(f"no decision in phase {phase}")
        actions = self.engine_full.legal_actions(state)
        leading = not any(a.is_pass for a in actions)
        legal = danlm_legal_plays(actions, hand_ours, level, None if leading else record.top)
        state_row = enc.play_state(seat, hand, record.played, record.visible_trick(),
                                   record.finished, level_t, enc.relative(record.tribute_absolute(), seat))
        rows = enc.batch(state_row, legal.rows)
        q = self.model(rows)
        choice = break_ties(q, legal.rows, level)
        self.decisions += 1
        self.declared_choices += int(legal.declared[choice])
        return Decision("play", rows, q, choice, legal.rows[choice], legal.actions[choice],
                        declared=legal.declared[choice])

    def select(self, engine: gd.Engine, state: gd.MatchState, actions: Sequence,
               rng: Any = None) -> int:
        """Index into ``actions`` (pass a FULL-mode list: V1T tells suits apart)."""
        action = self.decide(state).action
        target = action_identity(action)
        for index, candidate in enumerate(actions):
            if action_identity(candidate) == target:
                return index
        for index, candidate in enumerate(actions):        # same cards, other reading
            if candidate.type == action.type and \
                    sorted(int(c) for c in candidate.cards) == sorted(int(c) for c in action.cards):
                return index
        raise RuntimeError(f"{self.name}: chosen {action} is not among the offered actions; "
                           "offer the full-mode legal list")
