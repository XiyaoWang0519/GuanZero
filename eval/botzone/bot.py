"""Botzone GuanDan bot: the history Transformer behind the Botzone JSON protocol.

One turn: rebuild the round from every request so far (``eval.botzone.mirror``),
then answer. Tribute and back-tribute use the engine's tribute heuristic on
our own hand, exactly as every evaluation of the history player does; plays
are the greedy argmax of the NumPy actor (``eval.botzone.numpy_actor``) over
the canonical candidates Botzone can express. A failure anywhere falls back
to a safe legal answer and says so in ``debug``.

Modes, as on Botzone: by default one JSON line in (``{"requests", "responses"}``),
one line out, exit. ``--keep-running`` prints Botzone's keep-running marker
and then reads one bare request per line (local judge speed).

Weights: ``--weights PATH``, else ``$GZ_BOTZONE_WEIGHTS``, else ``data/gz_actor.npz``
(Botzone user storage), else ``gz_actor.npz`` next to this file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from eval.botzone.protocol import (RoundLog, card_to_gd, claim_faces, physical_claim)

KEEP_RUNNING = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"


def find_weights(explicit: str | None = None) -> Path | None:
    here = Path(__file__).resolve().parent
    for candidate in (explicit, os.environ.get("GZ_BOTZONE_WEIGHTS"),
                      os.path.join("data", "gz_actor.npz"), str(here / "gz_actor.npz")):
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


class Bot:
    def __init__(self, weights: Path | None) -> None:
        from eval.botzone.numpy_actor import NumpyHistoryActor

        self.actor = NumpyHistoryActor.load(weights) if weights is not None else None
        self.notes: list[str] = []
        self.last_choice: dict | None = None

    # -- answers ---------------------------------------------------------------

    def respond(self, log: RoundLog) -> object:
        self.notes = []
        self.last_choice = None
        if log.stage == "deal":
            return []
        try:
            return self._respond(log)
        except Exception as error:   # the platform scores a crash as a loss
            self.notes.append("fallback: " + "".join(
                traceback.format_exception_only(type(error), error)).strip()[:300])
            return self._fallback(log)

    def _respond(self, log: RoundLog) -> object:
        from eval.botzone.mirror import decision_arrays, own_hand, rebuild

        built = rebuild(log)
        self.notes.extend(built.notes)
        state, canon = built.state, built.canon
        if int(state.to_move) != log.me:
            raise RuntimeError(f"replay has seat {state.to_move} to move, not {log.me}")
        hand = own_hand(log, built)
        self.notes.extend(n for n in built.notes if n not in self.notes)
        actions = canon.legal_actions(state)
        if log.stage in ("tribute", "return"):
            action = actions[canon.greedy(state)]
            face = list(action.cards)[0]
            return [self._physical([face], hand)[0]]
        if log.stage != "play":
            return []
        expressible = []
        declared = []
        for action in actions:
            if action.is_pass:
                expressible.append(True)
                declared.append([])
                continue
            claim = claim_faces(action.type, int(action.key), list(action.cards), log.level)
            expressible.append(claim is not None)
            declared.append(claim)
        if not any(expressible):
            raise RuntimeError("no candidate Botzone can express")
        if self.actor is None or len(actions) == 1:
            choice = next(i for i, ok in enumerate(expressible) if ok)
        else:
            obs, cand = decision_arrays(built, actions)
            tokens, rounds, phases = built.stream.arrays()
            logits = self.actor.logits(tokens, rounds, phases, obs, log.me, cand)
            masked = np.where(np.asarray(expressible), logits, -np.inf)
            choice = int(np.argmax(masked))
            if choice != int(np.argmax(logits)):
                self.notes.append("inexpressible_top")
        action = actions[choice]
        self.last_choice = {"type": action.type, "key": int(action.key),
                            "cards": sorted(int(c) for c in action.cards)}
        if action.is_pass:
            return [[], []]
        faces = list(action.cards)
        used = self._physical(faces, hand)
        return [used, physical_claim(faces, declared[choice], used)]

    @staticmethod
    def _physical(faces: list[int], hand: list[int]) -> list[int]:
        pool = list(hand)
        out = []
        for face in faces:
            for card in pool:
                if card_to_gd(card) == face:
                    pool.remove(card)
                    out.append(card)
                    break
            else:
                raise RuntimeError(f"face {face} is not in our physical hand")
        return out

    def _fallback(self, log: RoundLog) -> object:
        """A legal answer from the log alone: pass when following, else the lowest card."""
        hand = list(log.deliver)
        for card in (log.my_tribute, log.my_return):
            if card is not None and card in hand:
                hand.remove(card)
        for move in log.moves:
            if move.seat == log.me:
                for card in move.action:
                    if card in hand:
                        hand.remove(card)
        if log.stage in ("tribute", "return"):
            return [hand[0]] if hand else []
        last_own = max([i for i, m in enumerate(log.moves) if m.seat == log.me], default=-1)
        following = any(not m.is_pass and m.seat != log.me for m in log.moves[last_own + 1:])
        if following or not hand:
            return [[], []]
        card = min(hand, key=lambda c: (card_to_gd(c) // 4 if card_to_gd(c) < 52 else 13, c))
        return [[card], [card]]


def answer(bot: Bot, requests: list, responses: list) -> dict:
    started = time.perf_counter()
    log = RoundLog.from_turns(requests, responses)
    response = bot.respond(log)
    debug = {"ms": round(1000 * (time.perf_counter() - started), 1)}
    if bot.notes:
        debug["notes"] = bot.notes
    return {"response": response, "debug": json.dumps(debug)[:1000]}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    keep_running = "--keep-running" in argv
    explicit = None
    if "--weights" in argv:
        explicit = argv[argv.index("--weights") + 1]
    bot = Bot(find_weights(explicit))
    line = sys.stdin.readline()
    if not line:
        return
    data = json.loads(line)
    requests = list(data.get("requests", [data])) if isinstance(data, dict) else []
    responses = list(data.get("responses", [])) if isinstance(data, dict) else []
    while True:
        out = answer(bot, requests, responses)
        sys.stdout.write(json.dumps(out) + "\n")
        if not keep_running:
            sys.stdout.flush()
            return
        sys.stdout.write(KEEP_RUNNING + "\n")
        sys.stdout.flush()
        responses.append(out["response"])
        line = sys.stdin.readline()
        if not line:
            return
        data = json.loads(line)
        if isinstance(data, dict) and "requests" in data:
            requests = list(data["requests"])
            responses = list(data.get("responses", responses))
        else:
            requests.append(data)


if __name__ == "__main__":
    main()
