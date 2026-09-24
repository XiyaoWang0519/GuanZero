"""DanLM decomposition cache behavior without its optional compiled package."""
from types import SimpleNamespace

import numpy as np

from eval.danlm.fast_obs import _hand_plays


def test_hand_cache_is_per_seat_bounded_and_invalidates_on_hand_or_level():
    calls = []
    scratch = np.zeros((2, 80), np.int8)

    def calculate(hand, level):
        calls.append((hand.copy(), level))
        scratch[:] = len(calls)
        return scratch

    actions = SimpleNamespace(hand_calculator_v2=calculate)
    rnd = SimpleNamespace()
    hand = np.ones(54, np.int8)
    first = _hand_plays(rnd, actions, hand, 0, 3)
    assert _hand_plays(rnd, actions, hand.copy(), 0, 3) is first
    assert len(calls) == 1 and not first.flags.writeable
    _hand_plays(rnd, actions, hand, 1, 3)
    # A compiled scratch buffer must not corrupt earlier cached results.
    np.testing.assert_array_equal(first, np.ones((2, 80), np.int8))
    hand[0] -= 1
    assert _hand_plays(rnd, actions, hand, 0, 3) is not first
    _hand_plays(rnd, actions, hand, 0, 4)
    assert len(calls) == 4 and len(rnd._gd_hand_plays) == 2
    # A new round cannot see another round's cache.
    other = SimpleNamespace()
    _hand_plays(other, actions, hand, 0, 4)
    assert len(calls) == 5 and len(other._gd_hand_plays) == 1
