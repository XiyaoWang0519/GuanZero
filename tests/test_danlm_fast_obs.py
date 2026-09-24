"""Stage C C0: the pure-Python replacement for DanLM's observation.

Runs only where DanLM's compiled package imports (the Python 3.12
environment with ``DANLM_ROOT`` on ``sys.path``); skipped elsewhere.
"""
import os
import sys

import pytest

root = os.environ.get("DANLM_ROOT", ".work/external/DanLM")
if os.path.isdir(root) and root not in sys.path:
    sys.path.insert(0, root)
pytest.importorskip("danzero.engine.game")

from eval.danlm import fast_obs  # noqa: E402


def test_replacement_matches_original_on_random_play():
    report = fast_obs.verify(rounds=3, seed=5)
    assert report["steps"] > 100
    assert report["mismatching_steps"] == 0, report["mismatches_by_field"]


def test_install_and_uninstall_swap_the_method():
    from danzero.engine import game

    fast_obs.uninstall()
    original = game.GuanDanRound.get_observation
    fast_obs.install()
    assert game.GuanDanRound.get_observation is fast_obs.fast_get_observation
    fast_obs.uninstall()
    assert game.GuanDanRound.get_observation is original
