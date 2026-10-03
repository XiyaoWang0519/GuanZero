"""Our policy's seats at an OpenGuanDan table, through a ``gd`` mirror.

The table referees. Every round is replayed in a ``gd`` engine under the
``ogd`` rule profile so that our policy sees its own observation encoding:
the mirror is dealt the four hands the table dealt, applies every choice any
seat makes (tribute, back-tribute and play), and our seats choose among the
mirror's canonical candidates that the table also offers. A history policy
hears every action applied to the mirror and gets a fresh stream per round,
as in ``eval.danlm.arena``.

When the mirror cannot follow the table (a seat or candidate it cannot
match), the round is marked ``mirror_failed`` and our seats finish it with
the table's first non-pass option (or pass); such rounds are counted and
reported separately, never dropped silently.
"""
from __future__ import annotations

from collections import Counter
import random
from typing import Any

import gd

from eval.history_policy import apply_and_observe, needs_history
from eval.ogd_adapter import normalize as N

_RANKS = "23456789TJQKA"


def level_char(value: Any) -> str:
    """The table's level as a rank character: ``"7"`` or the int ``7``."""
    if isinstance(value, str):
        return value
    return _RANKS[int(value) - 2]


def triple(action: gd.Action) -> tuple:
    """Our action in ``normalize``'s vocabulary (see ``replay_diff._as_triple``)."""
    kind, key, cards = action.as_tuple()
    if kind == "Pass":
        return ("Pass", None, ())
    if kind in ("JokerBomb", "Tribute", "BackTribute"):
        return (kind, None, tuple(sorted(cards)))
    return (kind, key, tuple(sorted(cards)))


def their_triple(raw: list, level: str) -> tuple:
    kind, key, cards = N.normalize_action(raw, level)
    if kind in ("Tribute", "BackTribute", "JokerBomb"):
        key = None
    return (kind, key, cards)


class OurSeats:
    """Plays every seat of ``seats`` for one table, one round at a time."""

    def __init__(self, policy: Any, seats: tuple[int, ...], rng: random.Random) -> None:
        self.policy, self.seats, self.rng = policy, set(seats), rng
        self.rules = gd.RuleConfig.ogd()
        self.engine_full = gd.Engine(self.rules, gd.ActionConfig.full())
        self.engine_canon = gd.Engine(self.rules, gd.ActionConfig())
        self.engine_full.auto_pass = False
        self.engine_canon.auto_pass = False
        self.history = policy if needs_history(policy) else None
        self.stats: Counter = Counter()
        self.examples: dict[str, list[str]] = {}
        self.prev_order: list[int] | None = None
        self._new_round()

    # -- bookkeeping -------------------------------------------------------

    def note(self, kind: str, detail: str = "") -> None:
        self.stats[kind] += 1
        bucket = self.examples.setdefault(kind, [])
        if detail and len(bucket) < 3:
            bucket.append(detail)

    def _new_round(self) -> None:
        self.hands: dict[int, list[str]] = {}
        self.ranks: dict[int, tuple[int, int]] = {}
        self.level: str | None = None
        self.anti = False
        self.state: gd.MatchState | None = None
        self.failed = False
        self.applied = 0
        self.pending: tuple[int, tuple] | None = None

    def fail(self, kind: str, detail: str = "") -> None:
        if not self.failed:
            self.failed = True
            self.note("mirror_failed")
            self.note(f"mirror_failed/{kind}", detail)

    # -- table messages ------------------------------------------------------

    def notify(self, seat: int, msg: dict) -> None:
        """Every message for every seat passes here first (the referee's view)."""
        stage = msg.get("stage")
        if msg.get("type") != "notify":
            return
        if stage == "beginning":
            self.hands[seat] = list(msg["handCards"])
            self.level = level_char(msg["curRank"])
            self.ranks[seat] = (int(msg["selfRank"]), int(msg["oppoRank"]))
        elif stage == "anti-tribute" and seat == 0:
            self.anti = True
        elif stage == "episodeOver" and seat == 0:
            self._end_round([int(s) for s in msg["order"]])

    def _start(self, seat: int, stage: str) -> None:
        """Deal the mirror lazily, at the round's first decision."""
        if len(self.hands) != 4 or self.level is None:
            self.fail("no_deal", f"hands for {sorted(self.hands)}")
            return
        deal = gd.DealSpec()
        deal.hands = [sorted(N.card_id(c) for c in self.hands[s]) for s in range(4)]
        deal.level = N.RANK_INDEX[self.level]
        team = [self.ranks.get(0, (2, 2))[0], self.ranks.get(1, (2, 2))[0]]
        deal.team_levels = [N.RANK_INDEX[level_char(v)] for v in team]
        deal.owner = -1
        tribute_round = stage in ("tribute", "back") or self.anti
        if tribute_round:
            if self.prev_order is None or sorted(self.prev_order) != [0, 1, 2, 3]:
                self.fail("no_previous_order")
                return
            deal.prev_order = self.prev_order
        else:
            deal.leader = seat
        self.state = gd.MatchState()
        self.engine_full.set_deal(self.state, deal)
        if self.history is not None:
            self.history.start_match()

    def _apply(self, action: gd.Action) -> None:
        self.applied += 1
        if self.history is None:
            self.engine_full.apply(self.state, action)
        else:
            apply_and_observe(self.engine_full, self.state, action, (self.history,))
        self._flush_pending()

    def _sync_to(self, seat: int, theirs: list[tuple]) -> bool:
        """Bring the mirror to ``seat``'s turn, applying lone passes the table skipped."""
        guard = 0
        while int(self.state.to_move) != seat:
            guard += 1
            actions = self.engine_full.legal_actions(self.state)
            if len(actions) == 1 and actions[0].is_pass and guard < 4:
                self._apply(actions[0])
                continue
            return False
        return True

    def act(self, seat: int, msg: dict, choice: int | None) -> int | None:
        """Called for every ``act`` message. For our seats ``choice`` is None and
        the return value is the table index; for others ``choice`` is what the
        seat chose and the mirror just applies it."""
        level = level_char(msg["curRank"])
        stage = msg["stage"]
        raw = msg["actionList"]
        if self.state is None and not self.failed:
            self._start(seat, stage)
        theirs = [their_triple(a, level) for a in raw]
        only_pass = theirs == [("Pass", None, ())]
        if not self.failed and int(self.state.phase) == int(gd.Phase.Play) and stage == "play":
            if int(self.state.to_move) != seat and only_pass:
                # The table asks a seat with nothing but pass; ours may have
                # moved on already. Answer without touching the mirror.
                pass
            elif not self._sync_to(seat, theirs):
                self.fail("seat_mismatch", f"table {seat} mirror {self.state.to_move}")
        elif not self.failed and int(self.state.to_move) != seat:
            return self._out_of_order(seat, stage, theirs, choice)
        if seat in self.seats:
            return self._choose(seat, msg, theirs, only_pass)
        if not self.failed:
            if int(self.state.to_move) != seat and only_pass:
                return None
            self._follow(theirs[choice])
        return None

    def _out_of_order(self, seat: int, stage: str, theirs: list[tuple], choice: int | None) -> int | None:
        """A double tribute or double back-tribute asked in the other order.

        The table asks the Dweller before the Follower; the mirror asks them the
        other way round. The two payments come from separate hands, so the
        table's early payer is decided on a copy of the mirror where the other
        payer has paid (any legal card), and applied to the mirror right after
        the other payer's real choice.
        """
        phase = int(self.state.phase)
        if phase not in (int(gd.Phase.Tribute), int(gd.Phase.BackTribute)) or self.pending:
            self.fail("seat_mismatch", f"{stage}: table {seat} mirror {self.state.to_move} "
                                       f"phase {self.state.phase}")
            return self._choose(seat, {}, theirs, False) if seat in self.seats else None
        probe = gd.MatchState.deserialize(self.state.serialize())
        self.engine_full.apply(probe, self.engine_full.legal_actions(probe)[0])
        if int(probe.to_move) != seat or int(probe.phase) != phase:
            self.fail("seat_mismatch", f"{stage}: table {seat} mirror {self.state.to_move}")
            return self._choose(seat, {}, theirs, False) if seat in self.seats else None
        self.note("tribute_reordered")
        if seat in self.seats:
            self.stats["our_decisions"] += 1
            actions = self.engine_canon.legal_actions(probe)
            pick = actions[self.engine_canon.greedy(probe)]
            row = self._match(pick, {t: i for i, t in enumerate(theirs)},
                              {(t[0], t[2]): i for i, t in enumerate(theirs)})
            if row is None:
                self.fail("tribute_unmatched", str(triple(pick)))
                return 0
            self.pending = (seat, triple(pick))
            return row
        self.pending = (seat, theirs[choice])
        return None

    def _flush_pending(self) -> None:
        if self.pending and not self.failed and int(self.state.to_move) == self.pending[0]:
            _, chosen = self.pending
            self.pending = None
            self._follow(chosen)

    def _match(self, action: gd.Action, index: dict[tuple, int], loose: dict[tuple, int]) -> int | None:
        t = triple(action)
        if t in index:
            return index[t]
        return loose.get((t[0], t[2]))

    def _choose(self, seat: int, msg: dict, theirs: list[tuple], only_pass: bool) -> int:
        self.stats["our_decisions"] += 1
        if only_pass:
            if not self.failed and int(self.state.to_move) == seat:
                self._apply(self.engine_full.legal_actions(self.state)[0])
            return 0
        if self.failed:
            self.stats["our_fallback_decisions"] += 1
            return next((i for i, t in enumerate(theirs) if t[0] != "Pass"), 0)
        index = {t: i for i, t in enumerate(theirs)}
        loose = {(t[0], t[2]): i for i, t in enumerate(theirs)}
        candidates = self.engine_canon.legal_actions(self.state)
        usable, rows = [], []
        for action in candidates:
            row = self._match(action, index, loose)
            if row is None:
                self.note("our_candidate_unmatched", str(triple(action)))
                continue
            usable.append(action)
            rows.append(row)
        if not usable:
            self.fail("all_candidates_unmatched", str(theirs[:4]))
            self.stats["our_fallback_decisions"] += 1
            return next((i for i, t in enumerate(theirs) if t[0] != "Pass"), 0)
        pick = int(self.policy.select(self.engine_canon, self.state, usable, self.rng))
        self._apply(usable[pick])
        return rows[pick]

    def _follow(self, chosen: tuple) -> None:
        actions = self.engine_full.legal_actions(self.state)
        exact = [a for a in actions if triple(a) == chosen]
        if not exact:
            exact = [a for a in actions if triple(a)[0] == chosen[0] and triple(a)[2] == chosen[2]]
            if exact:
                self.note("their_choice_reading")
        if not exact:
            self.fail("their_choice_missing", str(chosen))
            return
        self._apply(exact[0])

    def _end_round(self, order: list[int]) -> None:
        self.stats["rounds"] += 1
        if self.state is not None and not self.failed:
            # Seats left with cards may still owe lone passes in the mirror.
            guard = 0
            while int(self.state.phase) == int(gd.Phase.Play) and guard < 8:
                guard += 1
                actions = self.engine_full.legal_actions(self.state)
                if len(actions) == 1 and actions[0].is_pass:
                    self._apply(actions[0])
                else:
                    break
            if int(self.state.phase) != int(gd.Phase.RoundEnd):
                self.fail("round_end_mismatch", f"phase {self.state.phase}")
            else:
                result = self.engine_full.end_round(self.state)
                ours = list(result.order)[:4]
                if ours[:len(order)] != order[:len(ours)]:
                    self.note("finish_order_mismatch", f"mirror {ours} table {order}")
            if self.history is not None and self.history.events_seen != self.applied:
                self.note("history_event_gap")
        if self.failed:
            self.stats["rounds_mirror_failed"] += 1
        self.prev_order = order
        self._new_round()
