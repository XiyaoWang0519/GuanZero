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


def opponent_response_labels(buffer, store, rows: np.ndarray,
                             round_streams: bool = False) -> np.ndarray:
    """One target for each row's executed action; never label unchosen actions.

    ``round_streams``: rows of the planted-habit ROUND view cite their round's
    own stream (``RoundEventStore.round_stream``); the labels are the same
    events, since a response never crosses the round boundary.
    """
    rows = np.asarray(rows, np.int64)
    if len(rows) < 8:
        return _scalar_opponent_response_labels(buffer, store, rows, round_streams)
    data = buffer.compact()
    labels = np.zeros(len(rows), np.int64)
    if any(not buffer.trajectories[trajectory].complete
           for trajectory in np.unique(data["traj"][rows]).tolist()):
        raise ValueError("response labels require a completed round")
    # Group rows only for this call: streams and buffers may grow, reset or be
    # compacted between minibatches, so no target or prefix cache is retained.
    fields = ("env", "match", "round") if round_streams else ("env", "match")
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, key in enumerate(zip(*(data[field][rows].tolist() for field in fields))):
        groups.setdefault(key, []).append(index)
    for key, positions in groups.items():
        if len(positions) < 8:
            labels[positions] = _scalar_opponent_response_labels(
                buffer, store, rows[positions], round_streams)
            continue
        stream = store.round_stream(*key) if round_streams else store.stream(*key)
        labels[positions] = _stream_labels(data, rows[positions], stream)
    return labels


def _scalar_opponent_response_labels(buffer, store, rows: np.ndarray,
                                     round_streams: bool = False) -> np.ndarray:
    """Original scan for small groups where vector setup would cost more."""
    data = buffer.compact()
    labels = np.zeros(len(rows), np.int64)
    for index, row in enumerate(np.asarray(rows, np.int64).tolist()):
        if not buffer.trajectories[int(data["traj"][row])].complete:
            raise ValueError("response labels require a completed round")
        stream = (store.round_stream(int(data["env"][row]), int(data["match"][row]),
                                     int(data["round"][row])) if round_streams else
                  store.stream(int(data["env"][row]), int(data["match"][row])))
        tokens, rounds, phases = stream.arrays()
        prefix, seat, rnd = (int(data[k][row]) for k in ("prefix", "seat", "round"))
        if prefix >= stream.prefix:
            raise ValueError("executed action missing from public history")
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


def _stream_labels(data, rows: np.ndarray, stream) -> np.ndarray:
    """Advance all requested horizons together, without scanning old history."""
    tokens, rounds, phases = stream.arrays()
    prefix, seat, rnd, phase = (data[field][rows] for field in ("prefix", "seat", "round", "phase"))
    if (prefix >= stream.prefix).any():
        raise ValueError("executed action missing from public history")
    own = tokens[prefix]
    chosen = data["cand"][data["cand_start"][rows] + data["chosen"][rows], :146]
    if ((rounds[prefix] != rnd).any() or (phases[prefix] != phase).any()
            or (own[:, :4].sum(axis=1) != 1).any()
            or (own[:, :4].argmax(axis=1) != seat).any()
            or not np.array_equal(own[:, 4:150], chosen)):
        raise ValueError("stored action does not match the public event at its prefix")
    labels = np.zeros(len(rows), np.int64)
    cursor = prefix + 1
    active = np.arange(len(rows))
    while len(active):
        active = active[cursor[active] < stream.prefix]
        active = active[rounds[cursor[active]] == rnd[active]]
        if (phases[cursor[active]] != phase[active]).any():
            raise ValueError("exchange inside a completed Play round")
        relative = (tokens[cursor[active], :4].argmax(axis=1) - seat[active]) % 4
        opposing = (relative == 1) | (relative == 3)
        responding = active[opposing]
        action_type = tokens[cursor[responding], 4 + 108:4 + 121]
        kinds = action_type.argmax(axis=1)
        if (action_type.sum(axis=1) != 1).any() or (kinds >= PLAY_ACTION_TYPES).any():
            raise ValueError("invalid public Play action type")
        labels[responding] = 1 + (relative[opposing] == 3) * PLAY_ACTION_TYPES + kinds
        active = active[relative == 2]
        cursor[active] += 1
    return labels
