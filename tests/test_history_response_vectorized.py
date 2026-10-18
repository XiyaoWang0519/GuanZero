"""Batched response targets retain the original scalar horizon and validation."""
from types import SimpleNamespace

import numpy as np
import pytest

from train.history_habit import RoundEventStore
from train.history_response import (_scalar_opponent_response_labels,
                                    opponent_response_labels)
from test_history_response import action, add, synthetic


@pytest.mark.parametrize("round_streams", [False, True])
@pytest.mark.parametrize("seed", [7, 29, 101])
def test_vectorized_targets_match_scalar_for_reordered_repeated_rows(round_streams, seed):
    rng = np.random.default_rng(seed)
    store = RoundEventStore()
    records, candidates = [], []
    for env, match in [(0, 0), (1, 0), (1, 1)]:
        full = store.stream(env, match)
        for index in range(148):
            rnd = index // 37
            short = store.round_stream(env, match, rnd)
            seat, kind = int(rng.integers(4)), int(rng.integers(11))
            records.append(dict(env=env, match=match, round=rnd, seat=seat,
                                prefix=short.prefix if round_streams else full.prefix,
                                phase=0, traj=0, cand_start=len(candidates), chosen=1))
            candidates.extend([action((kind + 1) % 11), action(kind), action((kind + 2) % 11)])
            add(full, seat, kind, rnd)
            add(short, seat, kind, rnd)
    data = {field: np.array([record[field] for record in records], np.int64)
            for field in records[0]}
    data["cand"] = np.stack(candidates)
    buffer = SimpleNamespace(compact=lambda: data,
                             trajectories=[SimpleNamespace(complete=True)])
    for rows in (np.arange(len(records)), rng.permutation(len(records)),
                 rng.integers(len(records), size=600), np.array([0, 147, 148]),
                 np.array([], np.int64)):
        actual = opponent_response_labels(buffer, store, rows, round_streams)
        expected = _scalar_opponent_response_labels(buffer, store, rows, round_streams)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("change,message", [
    ("incomplete", "completed round"),
    ("missing", "executed action missing"),
    ("own_seat", "stored action"),
    ("own_candidate", "stored action"),
    ("exchange", "exchange inside"),
    ("empty_type", "invalid public Play"),
    ("exchange_type", "invalid public Play"),
])
def test_vectorized_targets_preserve_validation(change, message):
    buffer, store = synthetic([dict(seat=2), dict(seat=1)])
    stream, data = store.stream(0, 0), buffer.compact()
    if change == "incomplete":
        buffer.trajectories[0].complete = False
    elif change == "missing":
        data["prefix"][0] = stream.prefix
    elif change == "own_seat":
        stream._tokens[1, 1] = 1
    elif change == "own_candidate":
        data["cand"][0] = action(3)
    elif change == "exchange":
        stream._phases[2] = 1
    else:
        stream._tokens[3, 112:125] = 0
        if change == "exchange_type":
            stream._tokens[3, 123] = 1
    rows = np.zeros(12, np.int64)  # Enough rows to exercise the batched path.
    for function in (opponent_response_labels, _scalar_opponent_response_labels):
        with pytest.raises(ValueError, match=message):
            function(buffer, store, rows)


def test_vectorized_targets_recompute_after_stream_growth_and_round_boundary():
    buffer, store = synthetic([dict(seat=2)])
    stream = store.stream(0, 0)
    rows = np.zeros(12, np.int64)
    assert not opponent_response_labels(buffer, store, rows).any()
    add(stream, seat=3, kind=10)
    assert (opponent_response_labels(buffer, store, rows) == 22).all()
    # A new round ends the horizon before an exchange can trigger validation.
    stream._rounds[3] = 1
    stream._phases[3] = 1
    assert not opponent_response_labels(buffer, store, rows).any()
