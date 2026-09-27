"""Capacity x pool 2x2: each cell differs from the reused cell only as declared."""
import pytest

from infra import history_factorial_experiment as factorial
from train.history_ppo import HistoryPPOConfig

T7_PRESENT = (factorial.base.T7 / "seed-2026092801/run-manifest.json").exists()


@pytest.mark.skipif(not T7_PRESENT, reason="T7 run kits are local artifacts")
def test_cells_differ_from_the_reused_entropy_arm_only_by_their_settings():
    seed = factorial.SEEDS[0]
    reused = dict(factorial.base.screen_config(seed), **factorial.base.ARMS["entropy"])
    assert factorial.cell_config(seed, "small_recent") == reused
    for cell, settings in factorial.CELLS.items():
        config = factorial.cell_config(seed, cell)
        HistoryPPOConfig(**config)
        assert {k for k in config if config[k] != reused.get(k)} == set(settings)
        assert config["entropy"] == 0.03 and config["updates"] == factorial.UPDATES


def test_declared_settings_are_valid_and_factorial():
    assert set(factorial.CELLS) == {"small_recent", "small_wide", "large_recent", "large_wide"}
    assert factorial.CELLS["large_wide"] == dict(factorial.LARGE, **factorial.WIDE)
    HistoryPPOConfig(**factorial.WIDE, **factorial.LARGE)
    assert factorial.LARGE["width"] % factorial.LARGE["heads"] == 0
