"""Local Botzone GuanDan judge: our engine referees, bots speak the Botzone protocol.

The referee deals 108 physical cards, runs the tribute phases and the play in
``gd`` (``house`` rules, ``auto_pass`` off) and talks to each seat exactly the
way the Botzone judge does, as far as the wiki and the platform's match logs
show it: requests per stage, tribute payers asked last-finisher first,
back-tribute receivers first-finisher first, the play history as the last
four moves in order, padded with ``[]`` or ``{}`` as the platform sends it
(``--history positional`` for four positional slots instead), ``done`` and
``pass_on``. Every response is checked: the cards must be in the seat's
physical hand and the claim must name a legal ``gd`` reading of them. An
illegal or failed response ends the game as a loss for
that seat's team, as on Botzone.

Seats: ``bot`` (``eval.botzone.bot`` in-process, the traditional full-replay
path, fast), ``proc:<command>`` (a subprocess, keep-running unless the
command has ``--traditional``), ``greedy`` (the engine's greedy bot).

``--reference CHECKPOINT`` also runs the torch ``HistoryPolicy`` on the
referee's own state and stream at every in-process bot play decision and
counts how often the bot's play equals it: the end-to-end check of the bot's
round reconstruction.

    python -m eval.botzone.judge --weights .work/botzone/u15094.npz \\
        --seats bot,greedy,bot,greedy --games 200
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shlex
import subprocess
import sys
import time

import gd
import numpy as np

from eval.botzone.bot import KEEP_RUNNING, Bot, answer, find_weights
from eval.botzone.mirror import partner, prev_order, routing
from eval.botzone.protocol import (LEVEL_NAMES, RoundLog, card_to_gd, claim_faces,
                                   claim_matches, physical_claim)
from eval.history_policy import apply_and_observe

PLAY = int(gd.Phase.Play)


def make_deal(log: RoundLog, hands: list[list[int]]) -> gd.DealSpec:
    """The referee's gd deal for a Botzone setup (same previous order as the bot)."""
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


class IllegalResponse(RuntimeError):
    def __init__(self, seat: int, message: str) -> None:
        super().__init__(f"seat {seat}: {message}")
        self.seat = seat


# ---- seats -----------------------------------------------------------------------

class InProcessSeat:
    """``eval.botzone.bot`` called in-process with the full request history."""
    kind = "bot"

    def __init__(self, bot: Bot) -> None:
        self.bot = bot
        self.requests: list = []
        self.responses: list = []
        self.max_ms = 0.0
        self.notes: Counter = Counter()
        self.samples: list[str] = []

    def start(self) -> None:
        self.requests, self.responses = [], []

    def turn(self, request: dict) -> object:
        self.requests.append(request)
        started = time.perf_counter()
        out = answer(self.bot, self.requests, self.responses)
        self.max_ms = max(self.max_ms, 1000 * (time.perf_counter() - started))
        for note in json.loads(out["debug"]).get("notes", []):
            self.notes[note.split(":")[0]] += 1
            if len(self.samples) < 5:
                self.samples.append(note)
        self.responses.append(out["response"])
        return out["response"]


class ProcessSeat:
    """A bot subprocess; keep-running unless the command asks for ``--traditional``."""
    kind = "proc"

    def __init__(self, command: str) -> None:
        self.traditional = "--traditional" in command
        self.command = command.replace("--traditional", "").strip()
        self.proc = None
        self.requests: list = []
        self.responses: list = []
        self.max_ms = 0.0
        self.notes: Counter = Counter()
        self.samples: list[str] = []

    def start(self) -> None:
        self.close()
        self.requests, self.responses = [], []

    def _read(self, stream) -> dict:
        line = stream.readline()
        if not line:
            raise RuntimeError("bot process closed its output")
        return json.loads(line)

    def turn(self, request: dict) -> object:
        self.requests.append(request)
        started = time.perf_counter()
        payload = json.dumps({"requests": self.requests, "responses": self.responses})
        if self.traditional:
            proc = subprocess.run(shlex.split(self.command), input=payload + "\n",
                                  capture_output=True, text=True, timeout=60)
            lines = [l for l in proc.stdout.splitlines() if l.strip()]
            if len(lines) != 1:
                raise RuntimeError(f"traditional bot printed {len(lines)} lines: {proc.stderr[-500:]}")
            out = json.loads(lines[0])
        else:
            if self.proc is None:
                self.proc = subprocess.Popen(shlex.split(self.command) + ["--keep-running"],
                                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                             text=True, bufsize=1)
                self.proc.stdin.write(payload + "\n")
            else:
                self.proc.stdin.write(json.dumps(request) + "\n")
            self.proc.stdin.flush()
            out = self._read(self.proc.stdout)
            marker = self.proc.stdout.readline()
            if KEEP_RUNNING not in marker:
                raise RuntimeError(f"missing keep-running marker: {marker!r}")
        self.max_ms = max(self.max_ms, 1000 * (time.perf_counter() - started))
        for note in json.loads(out.get("debug", "{}")).get("notes", []):
            self.notes[note.split(":")[0]] += 1
            if len(self.samples) < 5:
                self.samples.append(note)
        self.responses.append(out["response"])
        return out["response"]

    def close(self) -> None:
        if self.proc is not None:
            self.proc.kill()
            self.proc.wait()
            self.proc = None


class GreedySeat:
    """The engine's greedy bot on the referee's state (it reads only its own hand)."""
    kind = "greedy"

    def __init__(self) -> None:
        self.referee = None
        self.seat = -1
        self.max_ms = 0.0
        self.notes: Counter = Counter()
        self.samples: list[str] = []

    def start(self) -> None:
        pass

    def turn(self, request: dict) -> object:
        return self.referee.greedy_response(self.seat, request)


# ---- referee ---------------------------------------------------------------------

class Referee:
    def __init__(self, seats: list, rng: random.Random, reference=None,
                 history_format: str = "botzone") -> None:
        if history_format not in ("botzone", "positional"):
            raise ValueError("history_format is botzone or positional")
        self.history_format = history_format
        self.seats = seats
        self.rng = rng
        self.reference = reference
        rules = gd.RuleConfig.house()
        self.engine = gd.Engine(rules, gd.ActionConfig.full())
        self.canon = gd.Engine(rules, gd.ActionConfig())
        self.engine.auto_pass = False
        self.canon.auto_pass = False
        for index, seat in enumerate(seats):
            if isinstance(seat, GreedySeat):
                seat.referee, seat.seat = self, index
        self.agree = Counter()

    # -- helpers -------------------------------------------------------------------

    def padding(self) -> object:
        """Empty history entry: the platform has sent both [] and {}."""
        return {} if self.tribute else []

    def _global(self, full: bool) -> dict:
        g = {"level": LEVEL_NAMES[self.level], "tribute": self.tribute,
             "first": self.first, "last": self.last}
        if full:
            g.update(tribute_cards={str(k): v for k, v in self.tribute_cards.items()},
                     return_cards={str(k): v for k, v in self.return_cards.items()},
                     resist=self.resist)
        return g

    def _apply(self, action) -> None:
        listeners = (self.reference,) if self.reference is not None else ()
        apply_and_observe(self.engine, self.state, action, listeners)

    def _take(self, seat: int, cards: list[int]) -> None:
        hand = self.hands[seat]
        for card in cards:
            if card not in hand:
                raise IllegalResponse(seat, f"card {card} is not in hand")
            hand.remove(card)

    def _card_action(self, seat: int, card: int):
        for action in self.engine.legal_actions(self.state):
            if list(action.cards) == [card_to_gd(card)]:
                return action
        raise IllegalResponse(seat, f"{gd.card_str(card_to_gd(card))} is not a legal "
                                    f"{'tribute' if int(self.state.phase) == 1 else 'return'}")

    def greedy_response(self, seat: int, request: dict) -> object:
        """The greedy bot's answer for ``seat``, in Botzone form."""
        stage = request["stage"]
        if stage == "deal":
            return []
        if stage in ("tribute", "return"):
            choices = (gd.tribute_choices if stage == "tribute" else gd.back_tribute_choices)(
                [card_to_gd(c) for c in self.hands[seat]], self.level)
            face = min(choices) if stage == "return" else max(choices)
            return [next(c for c in self.hands[seat] if card_to_gd(c) == face)]
        if int(self.state.to_move) != seat:
            raise RuntimeError(f"greedy seat {seat} asked while seat {self.state.to_move} moves")
        actions = self.canon.legal_actions(self.state)
        action = actions[self.canon.greedy(self.state)]
        if action.is_pass:
            return [[], []]
        faces = list(action.cards)
        pool, used = list(self.hands[seat]), []
        for face in faces:
            card = next(c for c in pool if card_to_gd(c) == face)
            pool.remove(card)
            used.append(card)
        declared = claim_faces(action.type, int(action.key), faces, self.level)
        if declared is None:
            raise RuntimeError(f"greedy chose an inexpressible reading {action}")
        return [used, physical_claim(faces, declared, used)]

    # -- one game ------------------------------------------------------------------

    def play_game(self, kind: str) -> dict:
        rng = self.rng
        deck = list(range(108))
        rng.shuffle(deck)
        self.hands = [sorted(deck[27 * s:27 * (s + 1)]) for s in range(4)]
        self.level = rng.randrange(13)
        self.tribute, self.first, self.last = 0, -1, -1
        if kind != "none":
            self.first = rng.randrange(4)
            self.last = rng.choice([(self.first + 1) % 4, (self.first + 3) % 4])
            self.tribute = 2 if kind == "double" else 1
        self.tribute_cards: dict[int, int] = {}
        self.return_cards: dict[int, int] = {}
        self.resist = False
        log = RoundLog()
        log.level, log.tribute, log.first, log.last = self.level, self.tribute, self.first, self.last
        log.leader_hint = 0
        self.state = gd.MatchState()
        self.engine.set_deal(self.state, make_deal(
            log, [sorted(card_to_gd(c) for c in h) for h in self.hands]))
        for seat in self.seats:
            seat.start()
        if self.reference is not None:
            self.reference.start_match()
        for s in range(4):
            response = self.seats[s].turn({"stage": "deal", "deliver": list(self.hands[s]),
                                           "your_id": s, "global": self._global(False)})
            if response != []:
                raise IllegalResponse(s, f"deal answer {response!r}")
        if self.tribute and int(self.state.phase) == PLAY:
            self.resist = True
            payers = [self.last] + ([partner(self.last)] if self.tribute == 2 else [])
            for p in payers:
                self.tribute_cards[p] = -1
        elif self.tribute:
            self._tribute_phase()
        return self._play_phase()

    def _tribute_phase(self) -> None:
        payers = [self.last] + ([partner(self.last)] if self.tribute == 2 else [])
        given: dict[int, int] = {}
        for p in payers:                                   # Botzone order: last first
            response = self.seats[p].turn({"stage": "tribute", "global": self._global(True)})
            card = int(response[0])
            if card not in self.hands[p] or card_to_gd(card) not in gd.tribute_choices(
                    [card_to_gd(c) for c in self.hands[p]], self.level):
                raise IllegalResponse(p, f"tribute {card} is not the largest card")
            given[p] = card
            self.tribute_cards[p] = card
        while int(self.state.phase) == 1:                  # gd order
            p = int(self.state.to_move)
            self._apply(self._card_action(p, given[p]))
        for p, card in given.items():
            self.hands[p].remove(card)
        # gd routes the cards (house rules, tie: TributeTie::Downstream);
        # the physical cards follow the same routing.
        log = RoundLog()
        log.level, log.tribute, log.first, log.last = self.level, self.tribute, self.first, self.last
        route = routing(log, {p: card_to_gd(c) for p, c in given.items()})
        received = {receiver: payer for payer, receiver in route.items()}
        for payer, receiver in route.items():
            self.hands[receiver].append(given[payer])
        receivers = [self.first] + ([partner(self.first)] if self.tribute == 2 else [])
        for r in receivers:
            expected = Counter(card_to_gd(c) for c in self.hands[r])
            if expected != Counter(self.state.hand(r)):
                raise RuntimeError("referee routing disagrees with gd")
        returned: dict[int, int] = {}
        for r in receivers:                                # Botzone order: first first
            response = self.seats[r].turn({"stage": "return", "global": self._global(True)})
            card = int(response[0])
            if card not in self.hands[r] or card_to_gd(card) not in gd.back_tribute_choices(
                    [card_to_gd(c) for c in self.hands[r]], self.level):
                raise IllegalResponse(r, f"return {card} is not allowed")
            returned[r] = card
            self.return_cards[r] = card
        while int(self.state.phase) == 2:
            r = int(self.state.to_move)
            self._apply(self._card_action(r, returned[r]))
        for r, card in returned.items():
            self.hands[r].remove(card)
            self.hands[received[r]].append(card)

    def _play_phase(self) -> dict:
        history: list[tuple[int, list]] = []
        done: list[int] = []
        pass_on = -1
        decisions = 0
        while int(self.state.phase) == PLAY:
            seat = int(self.state.to_move)
            if self.history_format == "positional":
                last_self = max([i for i, (p, _) in enumerate(history) if p == seat], default=-1)
                slots: list = [[], [], [], []]
                if last_self >= 0:
                    slots[0] = history[last_self][1]
                for p, response in history[last_self + 1:]:
                    slots[(p - seat) % 4] = response
            else:
                # The platform: the last four moves in order, front-padded.
                slots = [{"player": p, "response": r} for p, r in history[-4:]]
                slots = [self.padding() for _ in range(4 - len(slots))] + slots
            request = {"stage": "play", "history": slots, "done": list(done),
                       "pass_on": pass_on, "global": self._global(True)}
            reference_action = None
            if self.reference is not None and isinstance(self.seats[seat], (InProcessSeat, ProcessSeat)):
                actions = self.canon.legal_actions(self.state)
                if len(actions) > 1:
                    reference_action = actions[self.reference.select(
                        self.canon, self.state, actions, random.Random(0))]
            response = self.seats[seat].turn(request)
            action = self._match(seat, response)
            if reference_action is not None:
                same = (action.type == reference_action.type
                        and int(action.key) == int(reference_action.key)
                        and sorted(action.cards) == sorted(reference_action.cards))
                self.agree["same" if same else "different"] += 1
            if not action.is_pass:
                self._take(seat, response[0])
            self._apply(action)
            decisions += 1
            history.append((seat, response))
            if action.is_pass:
                pass
            else:
                pass_on = -1
                if not self.hands[seat]:
                    done.append(seat)
                    pass_on = seat
        result = self.engine.end_round(self.state)
        returns = list(result.seat_return)
        return {"order": list(result.order), "returns": returns, "decisions": decisions,
                "resist": self.resist}

    def _match(self, seat: int, response: object):
        if not (isinstance(response, list) and len(response) == 2):
            raise IllegalResponse(seat, f"play answer {response!r}")
        action_cards, claim = response
        actions = self.engine.legal_actions(self.state)
        if not claim:
            if action_cards:
                raise IllegalResponse(seat, "cards with an empty claim")
            for action in actions:
                if action.is_pass:
                    return action
            raise IllegalResponse(seat, "pass while leading")
        if len(action_cards) != len(claim):
            raise IllegalResponse(seat, "claim and action differ in length")
        for card in action_cards:
            if card not in self.hands[seat]:
                raise IllegalResponse(seat, f"card {card} is not in hand")
        faces = sorted(card_to_gd(c) for c in action_cards)
        for action in actions:
            if (not action.is_pass and sorted(action.cards) == faces
                    and claim_matches(action.type, int(action.key), list(action.cards),
                                      self.level, claim)):
                return action
        raise IllegalResponse(seat, f"no legal reading for {response!r}")


# ---- driver ----------------------------------------------------------------------

def make_seat(spec: str, bot: Bot | None):
    if spec == "bot":
        return InProcessSeat(bot)
    if spec == "greedy":
        return GreedySeat()
    if spec.startswith("proc:"):
        return ProcessSeat(spec[5:])
    raise ValueError(f"unknown seat {spec!r}")


def run(seats_spec: list[str], games: int, seed: int, weights: str | None,
        reference: str | None = None, swap: bool = True, history_format: str = "botzone") -> dict:
    bot = Bot(find_weights(weights)) if "bot" in seats_spec else None
    ref = None
    if reference:
        from eval.history_policy import load_history_policy
        ref = load_history_policy(reference)
    rng = random.Random(seed)
    kinds = ("none", "single", "double")
    totals = Counter()
    team_score = []
    illegal = []
    samples: list[str] = []
    started = time.perf_counter()
    for game in range(games):
        lineup = list(seats_spec)
        flipped = swap and game % 2 == 1
        if flipped:
            lineup = lineup[1:] + lineup[:1]      # the A team moves to seats 1 and 3
        seats = [make_seat(s, bot) for s in lineup]
        referee = Referee(seats, random.Random(rng.getrandbits(63)), ref, history_format)
        kind = kinds[game % 3] if game % 6 < 3 else kinds[(game // 2) % 3]
        try:
            result = referee.play_game(kind)
            a_seat = 1 if flipped else 0
            team_score.append(result["returns"][a_seat])
            totals["decisions"] += result["decisions"]
            totals["resist"] += int(result["resist"])
        except IllegalResponse as error:
            illegal.append({"game": game, "seat": error.seat, "error": str(error)})
            a_team = (1 if flipped else 0)
            team_score.append(-3 if error.seat % 2 == a_team % 2 else 3)
        totals.update({f"agree_{k}": v for k, v in referee.agree.items()})
        for seat in seats:
            totals.update({f"note_{k}": v for k, v in seat.notes.items()})
            totals["max_ms"] = max(totals["max_ms"], int(seat.max_ms))
            samples.extend(f"game {game}: {note}" for note in seat.samples[:5 - len(samples)])
            if isinstance(seat, ProcessSeat):
                seat.close()
    score = np.asarray(team_score, dtype=np.float64)
    report = {
        "seats": seats_spec, "games": games, "seed": seed, "weights": str(find_weights(weights)),
        "team_a_levels_per_round": float(score.mean()) if len(score) else 0.0,
        "team_a_round_wins": int((score > 0).sum()),
        "illegal": illegal, "counts": dict(totals), "note_samples": samples,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seats", default="bot,greedy,bot,greedy",
                        help="four seat specs: bot, greedy, proc:<command>")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--reference", default=None,
                        help="history_ppo checkpoint to compare in-process bot plays against")
    parser.add_argument("--no-swap", action="store_true", help="keep team A in seats 0 and 2")
    parser.add_argument("--history", default="botzone", choices=("botzone", "positional"),
                        help="play history format sent to the bots")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    seats = args.seats.split(",")
    if len(seats) != 4:
        parser.error("--seats needs four entries")
    report = run(seats, args.games, args.seed, args.weights, args.reference, not args.no_swap,
                 args.history)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    sys.exit(main())
