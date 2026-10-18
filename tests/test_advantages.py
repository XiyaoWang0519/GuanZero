"""Shared advantage semantics: terminal boundaries, padding and explicit bootstrap."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from train.advantages import compute_gae

ROOT = Path(__file__).resolve().parents[1]


def test_advantage_module_does_not_import_a_training_runtime():
    script = """
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'gd'}:
            raise RuntimeError(f'advantage estimation imported {fullname}')
sys.meta_path.insert(0, NoRuntime())
from train.advantages import compute_gae
assert compute_gae([1.], [3.], [True])[1].tolist() == [3.]
"""
    done = subprocess.run([sys.executable, '-I', '-c', script, str(ROOT)],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr


def test_bootstrap_broadcasts_over_batch_axes_and_stops_at_terminal():
    values = np.broadcast_to([1., 2.], (2, 2, 2))
    rewards = np.broadcast_to([3., 4.], values.shape)
    dones = np.zeros(values.shape, bool)
    dones[0, 1, -1] = True
    final = np.array([[10., 20.], [30., 40.]])
    advantages, returns = compute_gae(values, rewards, dones, gamma=0.5, lam=1.,
                                      last_value=final)
    # With lambda=1, returns are discounted rewards plus the nonterminal bootstrap.
    last_return = np.array([[9., 4.], [19., 24.]])
    expected = np.stack((3. + 0.5 * last_return, last_return), axis=-1)
    np.testing.assert_array_equal(returns, expected)
    np.testing.assert_array_equal(advantages, expected - values)
    assert returns.dtype == advantages.dtype == np.float32


def test_padding_discards_rewards_and_bootstrap_beyond_the_real_prefix():
    advantages, returns = compute_gae([1., 2., 99.], [3., 4., 99.],
                                      [False, False, False], gamma=0.5, lam=1.,
                                      mask=[True, True, False], last_value=1000.)
    np.testing.assert_array_equal(returns, [5., 4., 0.])
    np.testing.assert_array_equal(advantages, [4., 2., 0.])


def test_legacy_api_and_history_terminal_only_contract_are_preserved():
    from train.history_rollout import compute_gae as history_gae
    from train.rollout_buffer import compute_gae as legacy_gae

    assert legacy_gae is compute_gae
    advantages, returns = history_gae([1., 2.], [3., 4.], [False, False], 0.5, 1.)
    np.testing.assert_array_equal(returns, [5., 4.])
    np.testing.assert_array_equal(advantages, [4., 2.])
    with pytest.raises(TypeError):
        history_gae([1.], [3.], [False], last_value=10.)
