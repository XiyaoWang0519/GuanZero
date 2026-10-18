"""A replay fingerprint must distinguish nested structure and array values."""
import numpy as np
import torch

from bench.training_revision import fingerprint


def test_fingerprint_frames_nested_dictionaries():
    assert fingerprint({"a": {"b": 1}, "c": 2}) != fingerprint({"a": {"b": 1, "c": 2}})
    assert fingerprint({"a": 1, "b": 2}) == fingerprint({"b": 2, "a": 1})


def test_fingerprint_keeps_types_shapes_and_values():
    empty = np.empty(0, dtype=np.float32)
    assert fingerprint(empty) != fingerprint((str(empty.dtype), tuple(empty.shape)))
    assert fingerprint(empty) != fingerprint(torch.from_numpy(empty))
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert fingerprint(a) == fingerprint(a.copy())
    assert fingerprint(a) != fingerprint(a.astype(np.float64))
    assert fingerprint(a) != fingerprint(a.reshape(4, 3))
    b = a.copy()
    b[0, 0] = 1
    assert fingerprint(a) != fingerprint(b)
