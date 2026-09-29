"""Public opponent-response targets for the candidate-conditioned T7 ablation.

For an executed learner action, predict the first opposing-seat Play event
after that action, before the observer acts again or the round ends. Partner
events may intervene. Class 0 is no response within that horizon; the other
22 classes are (relative seat 1 or 3, one of the 11 public Play action types).
An engine-resolved pass is just Pass. No forced flag, other-seat hand or
legal list is read. Future public events supply labels, never actor inputs.
Only completed rounds can supply the no-response label.
"""
from __future__ import annotations

import numpy as np

PLAY_ACTION_TYPES = 11   # encoder.h: Pass .. JokerBomb; excludes exchanges
RESPONSE_CLASSES = 1 + 2 * PLAY_ACTION_TYPES
RESPONSE_SCHEMA = "first_opponent_before_observer_or_round_end_v1"


def opponent_response_labels(buffer, store, rows: np.ndarray) -> np.ndarray:
    """One target for each row's executed action; never label unchosen actions."""
    data = buffer.compact()
    labels = np.zeros(len(rows), np.int64)
    for index, row in enumerate(np.asarray(rows, np.int64).tolist()):
        if not buffer.trajectories[int(data["traj"][row])].complete:
            raise ValueError("response labels require a completed round")
        stream = store.stream(int(data["env"][row]), int(data["match"][row]))
        prefix, seat, rnd = (int(data[k][row]) for k in ("prefix", "seat", "round"))
        if prefix >= stream.prefix:
            raise ValueError("executed action missing from public history")
        tokens, rounds, phases = stream.arrays()
        own = tokens[prefix]
        chosen = data["cand"][int(data["cand_start"][row] + data["chosen"][row])]
        if (rounds[prefix] != rnd or phases[prefix] != int(data["phase"][row])
                or own[:4].sum() != 1 or int(own[:4].argmax()) != seat
                or not np.array_equal(own[4:150], chosen[:146])):
            raise ValueError("stored action does not match the public event at its prefix")
        for j in range(prefix + 1, stream.prefix):
            if rounds[j] != rnd:
                break
            if phases[j] != int(data["phase"][row]):
                raise ValueError("exchange inside a completed Play round")
            token = tokens[j]
            relative_seat = (int(token[:4].argmax()) - seat) % 4
            if relative_seat == 0:
                break
            if relative_seat == 2:
                continue
            action_type = token[4 + 108:4 + 121]
            if action_type.sum() != 1 or int(action_type.argmax()) >= PLAY_ACTION_TYPES:
                raise ValueError("invalid public Play action type")
            labels[index] = (1 + (relative_seat == 3) * PLAY_ACTION_TYPES
                             + int(action_type.argmax()))
            break
    return labels
