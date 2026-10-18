"""Run preparation uses history configuration without loading the training runtime."""
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from train.history_config import (HistoryPPOConfig, build_parser, config_from_args,
                                  parse_resume_overrides)

ROOT = Path(__file__).resolve().parents[1]


def test_configuration_import_and_parsing_need_only_standard_library():
    script = textwrap.dedent("""
        import importlib.abc
        import sys
        from dataclasses import asdict

        sys.path.insert(0, sys.argv[1])

        class NoTrainingRuntime(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'torch', 'gd', 'numpy'}:
                    raise RuntimeError(f'configuration imported {fullname}')
                if fullname in {'train.history_model', 'train.history_ppo'}:
                    raise RuntimeError(f'configuration imported {fullname}')

        sys.meta_path.insert(0, NoTrainingRuntime())
        from train.history_config import (
            HistoryPPOConfig, REWARD_SEMANTICS, build_parser, config_from_args,
            parse_arm_schedule, parse_resume_overrides,
        )
        config = config_from_args(build_parser().parse_args([
            '--output', 'kit', '--width', '32', '--rollout-kv-cache',
            '--rollout-paged-cache', '--rollout-trim-cuda-cache', 'auto',
        ]))
        assert asdict(config)['width'] == 32
        assert config.rollout_paged_cache and config.rollout_trim_cuda_cache is None
        assert parse_arm_schedule('2:off,1:on') == [(2, False), (1, True)]
        assert parse_resume_overrides(['num-envs=4', 'rollout_trim_cuda_cache=false']) == {
            'num_envs': 4, 'rollout_trim_cuda_cache': False,
        }
        assert REWARD_SEMANTICS['reward']
        try:
            HistoryPPOConfig(rollout_paged_cache=True)
        except ValueError:
            pass
        else:
            raise AssertionError('configuration did not validate cache dependencies')
    """)
    done = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT)],
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr


def test_trainer_keeps_configuration_imports_compatible():
    from train import history_config, history_ppo

    for name in ("HistoryPPOConfig", "REWARD_SEMANTICS", "RESUME_OVERRIDES",
                 "parse_arm_schedule", "parse_resume_overrides", "build_parser",
                 "optional_bool", "config_from_args"):
        assert getattr(history_ppo, name) is getattr(history_config, name)


def test_policy_configuration_is_materialized_on_request():
    from train.history_model import HistoryPolicyConfig

    config = HistoryPPOConfig(width=32, layers=1, heads=2, window=8,
                              max_rounds=12, response_mode="explicit")
    assert config.policy_config() == HistoryPolicyConfig(
        width=32, layers=1, heads=2, window=8, max_rounds=12, response_mode="explicit")


def test_parser_defaults_and_payload_filter_preserve_run_configuration():
    config = config_from_args(build_parser().parse_args(["--output", "kit"]))
    assert config == HistoryPPOConfig()
    assert HistoryPPOConfig.from_payload({**asdict(config), "stage": "history_ppo"},
                                        updates=5) == HistoryPPOConfig(updates=5)


@pytest.mark.parametrize("item", ["width=32", "lr=0.1", "rollout_trim_cuda_cache=maybe"])
def test_resume_parser_preserves_change_boundary(item):
    with pytest.raises(ValueError):
        parse_resume_overrides([item])
