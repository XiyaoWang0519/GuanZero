"""Decision grouping preserves first-use order through buffer lifecycle changes."""
import numpy as np
import pytest
import torch

from bench.history_batch_host import fixture
from train.history_model import PublicStream, StreamBatch


def expected_groups(buffer, rows, round_streams=False):
    data = buffer.compact()
    names = ("env", "match", "round") if round_streams else ("env", "match")
    keys = [tuple(int(data[name][row]) for name in names) for row in rows]
    unique = list(dict.fromkeys(keys))
    return unique, np.asarray([unique.index(key) for key in keys], np.int64)


@pytest.mark.parametrize("round_streams", [False, True])
def test_grouping_preserves_duplicate_and_arbitrary_row_order(round_streams):
    _, buffer = fixture(matches=4, decisions=4, tokens=8, candidates=3)
    for rows in (np.array([15, 0, 7, 15, 8, 10, 1]), np.zeros(0, np.int64),
                 buffer.samples, buffer.samples[::-1]):
        keys, index = buffer._group_rows(rows, round_streams)
        expected_keys, expected_index = expected_groups(buffer, rows, round_streams)
        assert keys == expected_keys
        np.testing.assert_array_equal(index, expected_index)


def test_training_batch_preserves_candidate_and_stream_order():
    store, buffer = fixture(matches=4, decisions=4, tokens=8, candidates=3)
    rows = np.array([15, 0, 7, 15, 8, 10, 1])
    keys, index = expected_groups(buffer, rows)
    batch = buffer.training_batch(rows, store, "cpu", extra={"label": rows + 7})
    data, inputs = buffer.compact(), batch.inputs
    np.testing.assert_array_equal(inputs.match_index, index)
    np.testing.assert_array_equal(inputs.obs, data["obs"][rows])
    np.testing.assert_array_equal(inputs.prefix, data["prefix"][rows])
    expected_streams = StreamBatch.from_streams([store.stream(*key) for key in keys], "cpu")
    for name in ("tokens", "rounds", "phases", "lengths"):
        assert torch.equal(getattr(inputs.streams, name), getattr(expected_streams, name))
    for position, row in enumerate(rows):
        start, count = data["cand_start"][row], data["cand_count"][row]
        np.testing.assert_array_equal(inputs.cand[inputs.offsets[position]:inputs.offsets[position + 1]],
                                      data["cand"][start:start + count])
        assert batch.chosen[position] == inputs.offsets[position] + data["chosen"][row]
    np.testing.assert_array_equal(batch.fields["label"], rows + 7)


def test_grouping_rebuilt_after_append_and_carryover():
    _, buffer = fixture(matches=3, decisions=4, tokens=8, candidates=3)
    # Keep one trajectory open, including non-contiguous rows, across the update.
    retained = buffer.trajectories[-1]
    retained.complete = False
    buffer.open[(retained.env, retained.team)] = len(buffer.trajectories) - 1
    buffer._group_rows(buffer.samples)
    buffer._group_rows(buffer.samples, round_streams=True)
    buffer.next_iteration()
    before = buffer.compact()
    # Append an independent new match through the public storage API.
    buffer.add_step(
        keep=np.ones(1, bool), env_id=np.array([20]), match_id=np.array([4]),
        round_index=np.array([2]), seat=np.array([1]), phase=np.array([2]),
        obs=before["obs"][:1], hidden=before["hidden"][:1], cand=before["cand"][:1],
        offsets=np.array([0, 1]), chosen=np.array([0]), logp=np.array([-1.], np.float32),
        prefix=np.array([0]), version=2)
    rows = np.arange(len(buffer) - 1, -1, -1)
    for round_streams in (False, True):
        keys, index = buffer._group_rows(rows, round_streams)
        expected_keys, expected_index = expected_groups(buffer, rows, round_streams)
        assert keys == expected_keys
        np.testing.assert_array_equal(index, expected_index)


def test_round_view_separates_rounds_of_the_same_match():
    store, buffer = fixture(matches=2, decisions=4, tokens=8, candidates=3)
    data = buffer.compact()
    buffer._group_rows(buffer.samples, round_streams=True)
    buffer.add_step(
        keep=np.ones(2, bool), env_id=np.array([0, 1]), match_id=np.array([0, 1]),
        round_index=np.ones(2, np.int64), seat=np.array([0, 1]), phase=np.array([2, 2]),
        obs=data["obs"][:2], hidden=data["hidden"][:2], cand=data["cand"][:2],
        offsets=np.array([0, 1, 2]), chosen=np.array([0, 0]),
        logp=np.array([-1., -1.], np.float32), prefix=np.array([0, 0]), version=2)
    rows = np.array([8, 6, 9, 7, 8])
    round_streams = {(0, 0, 0): store.stream(0, 0), (1, 1, 0): store.stream(1, 1),
                     (0, 0, 1): PublicStream(0), (1, 1, 1): PublicStream(1)}
    store.round_stream = lambda *key: round_streams[key]
    batch = buffer.training_batch(rows, store, "cpu", fields=(), round_streams=True)
    keys, index = expected_groups(buffer, rows, round_streams=True)
    assert keys == [(0, 0, 1), (0, 0, 0), (1, 1, 1), (1, 1, 0)]
    np.testing.assert_array_equal(batch.inputs.match_index, index)
    np.testing.assert_array_equal(batch.inputs.streams.lengths, [0, 8, 0, 8])
    assert len(buffer._group_rows(rows)[0]) == 2


def test_prefix_validation_honors_shortened_stream_override():
    store, buffer = fixture(matches=3, decisions=4, tokens=8, candidates=3)
    rows = np.array([11, 0, 7, 11])
    buffer.training_batch(rows, store, "cpu")
    key = expected_groups(buffer, rows)[0][0]
    with pytest.raises(ValueError, match="more history"):
        buffer.training_batch(rows, store, "cpu", streams={key: PublicStream(key[1])})


def test_empty_training_batch_and_match_groups():
    store, buffer = fixture(matches=2, decisions=2, tokens=2, candidates=1)
    buffer.samples = np.zeros(0, np.int64)
    assert buffer.match_groups() == {}
    batch = buffer.training_batch(buffer.samples, store, "cpu")
    assert batch.inputs.match_index.numel() == batch.chosen.numel() == 0
    assert batch.inputs.streams.lengths.numel() == 0
