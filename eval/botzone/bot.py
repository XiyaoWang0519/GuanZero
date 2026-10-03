"""Botzone GuanDan bot: the history Transformer behind the Botzone JSON protocol.

One turn: rebuild the round from every request so far (``mirror``), then
answer. Tribute and back-tribute use the engine's tribute heuristic on our
own hand, exactly as every evaluation of the history player does; plays are
the greedy argmax of the NumPy actor (``numpy_actor``) over the canonical
candidates Botzone can express. A failure anywhere falls back to a safe legal
answer and says so in ``debug``.

Modes, as on Botzone: by default one JSON line in (``{"requests", "responses"}``),
one line out, exit. ``--keep-running`` prints Botzone's keep-running marker
and then reads one bare request per line.

Weights: ``--weights PATH``, else ``$GZ_BOTZONE_WEIGHTS``, else
``data/gz_actor.npz`` (Botzone user storage), else ``gz_actor.npz`` next to
this file or inside the uploaded zip.

Needs only NumPy and the standard library; Python 3.6 compatible.
"""
import os

# One BLAS thread: the matrices are tiny, and OpenBLAS otherwise reserves a
# buffer per core at import, which pushed the process to 254 of Botzone's
# 256 MB. Must run before numpy is imported.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import io
import json
import sys
import time
import traceback
import zipfile
from typing import List, Optional

import numpy as np

from . import pyengine as pe
from .protocol import RoundLog, card_to_gd, claim_faces, physical_claim

KEEP_RUNNING = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"
WEIGHTS_NAME = "gz_actor.npz"


def find_weights(explicit: Optional[str] = None):
    """A weights path, an open file object from inside our zip, or None."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (explicit, os.environ.get("GZ_BOTZONE_WEIGHTS"),
                      os.path.join("data", WEIGHTS_NAME), os.path.join(here, WEIGHTS_NAME)):
        if candidate and os.path.isfile(candidate):
            return candidate
    # Running from an uploaded zip: the weights sit at the archive root.
    archive = here
    while archive and not os.path.isfile(archive):
        parent = os.path.dirname(archive)
        archive = "" if parent == archive else parent
    if archive and zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            if WEIGHTS_NAME in z.namelist():
                return io.BytesIO(z.read(WEIGHTS_NAME))
    return None


class Bot(object):
    def __init__(self, weights) -> None:
        from .numpy_actor import NumpyHistoryActor

        self.actor = NumpyHistoryActor.load(weights) if weights is not None else None
        self.notes = []          # type: List[str]
        self.last_choice = None

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
        from .mirror import decision_arrays, own_hand, rebuild

        built = rebuild(log)
        state = built.state
        if state.to_move != log.me:
            raise RuntimeError("replay has seat %d to move, not %d" % (state.to_move, log.me))
        hand = own_hand(log, built)
        self.notes.extend(built.notes)
        actions = state.legal_actions()
        if log.stage in ("tribute", "return"):
            action = actions[pe.tribute_choice(state, actions)]
            return [self._physical(action.cards, hand)[0]]
        if log.stage != "play":
            return []
        declared = []
        for action in actions:
            declared.append([] if action.is_pass else
                            claim_faces(action.type_name, action.key, action.cards, log.level))
        expressible = [d is not None for d in declared]
        if not any(expressible):
            raise RuntimeError("no candidate Botzone can express")
        if self.actor is None or len(actions) == 1:
            choice = expressible.index(True)
        else:
            obs, cand = decision_arrays(built, actions)
            tokens, rounds, phases = built.stream.arrays()
            logits = self.actor.logits(tokens, rounds, phases, obs, log.me, cand)
            masked = np.where(np.asarray(expressible), logits, -np.inf)
            choice = int(np.argmax(masked))
            if choice != int(np.argmax(logits)):
                self.notes.append("inexpressible_top")
        action = actions[choice]
        self.last_choice = {"type": action.type_name, "key": action.key, "cards": action.cards}
        if action.is_pass:
            return [[], []]
        faces = action.cards
        used = self._physical(faces, hand)
        return [used, physical_claim(faces, declared[choice], used)]

    @staticmethod
    def _physical(faces: List[int], hand: List[int]) -> List[int]:
        pool = list(hand)
        out = []
        for face in faces:
            for card in pool:
                if card_to_gd(card) == face:
                    pool.remove(card)
                    out.append(card)
                    break
            else:
                raise RuntimeError("face %d is not in our physical hand" % face)
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
        own = [i for i, m in enumerate(log.moves) if m.seat == log.me]
        last_own = own[-1] if own else -1
        following = any(not m.is_pass and m.seat != log.me for m in log.moves[last_own + 1:])
        if following or not hand:
            return [[], []]
        card = min(hand, key=lambda c: (card_to_gd(c) // 4 if card_to_gd(c) < 52 else 13, c))
        return [[card], [card]]


def memory_mb() -> dict:
    """Resident and virtual peak of this process from /proc, when present."""
    out = {}
    try:
        with open("/proc/self/status") as status:
            for line in status:
                key = line.split(":")[0]
                if key in ("VmHWM", "VmPeak"):
                    out[key] = int(line.split()[1]) // 1024
    except (IOError, OSError, ValueError, IndexError):
        pass
    return out


def answer(bot: Bot, requests: list, responses: list) -> dict:
    started = time.time()
    log = RoundLog.from_turns(requests, responses)
    response = bot.respond(log)
    debug = {"ms": round(1000 * (time.time() - started), 1)}
    debug.update(memory_mb())
    if bot.notes:
        debug["notes"] = bot.notes
    return {"response": response, "debug": json.dumps(debug)[:1000]}


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    keep_running = "--keep-running" in argv
    explicit = argv[argv.index("--weights") + 1] if "--weights" in argv else None
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
