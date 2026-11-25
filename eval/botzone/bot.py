"""Botzone GuanDan bot: the history Transformer behind the Botzone JSON protocol.

One turn: rebuild the round from every request so far (``mirror``), then
answer. Tribute and back-tribute use the engine's tribute heuristic on our
own hand, exactly as every evaluation of the history player does; plays are
the greedy argmax of the NumPy actor (``numpy_actor``) over the canonical
candidates Botzone can express. ``--search`` adds deadline-bounded root
search using the same actor for rollouts. A failure anywhere falls back to a safe legal
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
from .protocol import (RoundLog, card_to_gd, claim_faces, is_wild, physical_claim,
                       power, rank_of)

KEEP_RUNNING = ">>>BOTZONE_REQUEST_KEEP_RUNNING<<<"
WEIGHTS_NAME = "gz_actor.npz"


def find_weights(explicit: Optional[str] = None):
    """A weights path, an open file object from inside our zip, or None."""
    if explicit:
        if not os.path.isfile(explicit):
            raise FileNotFoundError("explicit weights not found: " + explicit)
        return explicit
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
    def __init__(self, weights, search_seconds=0.0) -> None:
        from .numpy_actor import NumpyHistoryActor

        self.actor = NumpyHistoryActor.load(weights) if weights is not None else None
        self.search_seconds = float(search_seconds)
        self.search_stats = {}
        self.notes = []          # type: List[str]
        self.last_choice = None

    def respond(self, log: RoundLog) -> object:
        self.notes = []
        self.search_stats = {}
        self.deadline = time.perf_counter() + self.search_seconds
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
            if self.search_seconds > 0:
                from .search import select
                try:
                    choice, self.search_stats = select(
                        self.actor, log, built, actions, masked, choice, self.deadline)
                except Exception as error:
                    self.notes.append("search_fallback: " + str(error)[:150])
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
        """Answer from our cards and the pending request, without mirror or actor.

        Exchanges are simultaneous in the platform judge: tribute/return
        choices come from the original deal; received cards enter play only
        after every return. During play, pass when following and lead a single.
        """
        if log.stage in ("tribute", "return") and log.resist:
            return []
        hand = list(log.deliver)
        if log.stage == "tribute":
            eligible = [c for c in hand if not is_wild(card_to_gd(c), log.level)]
            card = max(eligible, key=lambda c: (power(rank_of(card_to_gd(c)), log.level), -c))
            return [card]
        if log.stage == "return":
            # The supplied judge allows natural ranks <=9, except level 9
            # where the cap is 8. Jokers and the elevated level are excluded.
            cap = 6 if log.level == 7 else 7  # gd ranks: 2=0, 8=6, 9=7
            eligible = [c for c in hand
                        if rank_of(card_to_gd(c)) <= cap
                        and rank_of(card_to_gd(c)) != log.level]
            return [min(eligible, key=lambda c: (rank_of(card_to_gd(c)), c))]
        if log.stage != "play":
            return []
        if log.tribute and not log.resist:
            if log.tribute == 1:
                route = {log.last: log.first}
            else:
                down, up = (log.first + 1) % 4, (log.first + 3) % 4
                key = lambda seat: power(rank_of(card_to_gd(log.tribute_cards[seat])), log.level)
                banker_payer = down if key(down) >= key(up) else up
                other_payer = up if banker_payer == down else down
                route = {banker_payer: log.first, other_payer: (log.first + 2) % 4}
            # Remove outgoing physical ids before adding incoming cards.
            # Use public exchange dictionaries even if our prior response
            # was not separately cached in my_tribute/my_return.
            for payer, receiver in route.items():
                if payer == log.me:
                    hand.remove(log.tribute_cards[payer])
                if receiver == log.me:
                    hand.remove(log.return_cards[receiver])
            for payer, receiver in route.items():
                if receiver == log.me:
                    hand.append(log.tribute_cards[payer])
                if payer == log.me:
                    hand.append(log.return_cards[receiver])
        for move in log.moves:
            if move.seat == log.me:
                for card in move.action:
                    hand.remove(card)
        if not log.leading:
            return [[], []]
        card = min(hand, key=lambda c: (power(rank_of(card_to_gd(c)), log.level), c))
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
    if bot.actor is not None:
        debug["checkpoint"] = bot.actor.config.get("checkpoint_id", "unknown")[:12]
    if bot.search_stats:
        debug["search"] = bot.search_stats
    if bot.notes:
        debug["notes"] = bot.notes
    return {"response": response, "debug": json.dumps(debug)[:1000]}


def main(argv: Optional[List[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    keep_running = "--keep-running" in argv
    explicit = argv[argv.index("--weights") + 1] if "--weights" in argv else None
    # Traditional Python limit is 6 s. Include interpreter/import/load CPU time
    # and keep 0.4 s for serialization and platform variance.
    search = "--search" in argv
    if search and keep_running:
        raise ValueError("search package requires traditional mode")
    bot = Bot(find_weights(explicit))
    line = sys.stdin.readline()
    if not line:
        return
    data = json.loads(line)
    requests = list(data.get("requests", [data])) if isinstance(data, dict) else []
    responses = list(data.get("responses", [])) if isinstance(data, dict) else []
    while True:
        if search:
            bot.search_seconds = max(0.0, 5.6 - time.process_time())
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
