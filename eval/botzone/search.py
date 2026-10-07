"""Deadline-bounded public-information root search (Python 3.6, NumPy only).

Same hand-or-unsure trigger, top-eight candidates, terminal returns and
0.5 margin as the evaluated search. Only complete paired worlds count.
The platform CPU budget replaces the offline fixed world count.
"""
import copy
import random
import time

import numpy as np

from . import pyengine as pe
from .mirror import Rebuilt, TokenStream, apply_and_record, decision_arrays, rebuild
from .protocol import claim_faces
from .numpy_actor import IncrementalStream


def terminal_return(state, seat):
    winner = state.order[0] % 2
    partner_pos = state.order.index((state.order[0] + 2) % 4)
    levels = 4 - partner_pos
    return float(levels if seat % 2 == winner else -levels)


def rollout(actor, sample, prefix, action, seat, deadline):
    state = copy.deepcopy(sample)
    stream = prefix.fork()
    apply_and_record(state, action, stream)
    built = Rebuilt(state, stream)
    while state.phase == pe.PHASE_PLAY:
        if time.perf_counter() >= deadline:
            return None
        actions = state.legal_actions()
        if len(actions) == 1:
            choice = 0
        else:
            obs, cand = decision_arrays(built, actions)
            logits = actor.candidate_logits(actor.decision_state(stream.encoded, obs, state.to_move), cand)
            allowed = [a.is_pass or claim_faces(a.type_name, a.key, a.cards, state.level)
                       is not None for a in actions]
            choice = int(np.argmax(np.where(allowed, logits, -np.inf)))
        apply_and_record(state, actions[choice], stream)
    if time.perf_counter() >= deadline:
        return None
    return terminal_return(state, seat)


def select(actor, log, built, actions, logits, baseline, deadline,
           top_actions=8, min_worlds=16, max_worlds=1000000):
    """Sample only from the seat's log; never access a referee's hidden hands."""
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    stats = {"worlds": 0, "choice": baseline, "baseline": baseline}
    if min(sum(h) for h in built.state.hands) > 10 and probs[baseline] >= 0.6:
        stats["skip"] = "confident_early"
        return baseline, stats
    candidates = sorted((i for i in range(len(actions)) if np.isfinite(logits[i])),
                        key=lambda i: (-float(logits[i]), i))[:top_actions]
    if len(candidates) < 2:
        return baseline, stats
    totals = np.zeros(len(candidates), dtype=np.float64)
    rng = random.Random(20261004 + len(log.moves) * 4 + log.me)
    started = time.perf_counter()
    prefix = IncrementalStream(actor)
    for token, rnd, phase in zip(*built.stream.arrays()):
        if time.perf_counter() >= deadline:
            return baseline, stats
        prefix.append_token(token, int(rnd), int(phase))
    for _ in range(max_worlds):
        if time.perf_counter() >= deadline:
            break
        sample = rebuild(log, seed=rng.getrandbits(64)).state
        values = []
        for index in candidates:
            value = rollout(actor, sample, prefix, actions[index], log.me, deadline)
            if value is None:
                break
            values.append(value)
        if len(values) != len(candidates):
            break
        totals += values
        stats["worlds"] += 1
    worlds = stats["worlds"]
    stats.update(candidates=len(candidates), ms=round((time.perf_counter()-started)*1000, 1))
    if worlds >= min_worlds:
        means = totals / worlds
        winner = int(np.argmax(means))
        base = candidates.index(baseline)
        if means[winner] >= means[base] + 0.5:
            stats["choice"] = candidates[winner]
        stats["means"] = means.tolist()
    return stats["choice"], stats
