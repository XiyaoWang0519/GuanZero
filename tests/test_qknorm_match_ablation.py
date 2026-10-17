from types import SimpleNamespace

import numpy as np
import pytest
import torch

import bench.qknorm_match_ablation as diagnostic
from bench.qknorm_match_summary import bootstrap, interval


def test_seat_random_streams_ignore_batch_order_and_other_matches():
    first = diagnostic.MatchSampler([91, 91, 92])
    second = diagnostic.MatchSampler([91, 91, 92])
    expected = first.draw([0, 2], [1, 3])
    second.draw([1, 1, 1], [0, 0, 2])
    actual = second.draw([2, 0], [3, 1])
    assert np.array_equal(expected, actual[::-1])
    assert first.draw([0], [2])[0] == first.draw([1], [2])[0]


def test_inverse_cdf_respects_ragged_padding_and_probability_mass():
    table = torch.tensor([[.25, .75, 0.], [.1, .2, .7]]).log()
    choice, _ = diagnostic.inverse_cdf(table, torch.tensor([.999999, .15]))
    assert choice.tolist() == [1, 1]
    with pytest.raises(FloatingPointError):
        diagnostic.inverse_cdf(torch.tensor([[float("nan")]]), torch.tensor([.5]))


def test_bootstrap_retains_paired_conditions_and_frozen_opponent_weights():
    # Unequal opponent strata: the population mean must retain their weights.
    opponents = np.asarray([1, 1, 1, 2])
    values = np.asarray([[0., 0.], [0., 0.], [0., 0.], [1., 1.]])
    draws = bootstrap(values, opponents, draws=100)
    assert np.all(draws == .25)
    contrast = interval(0., draws[:, 0]-draws[:, 1])
    assert contrast["ci95"] == [0., 0.] and contrast["ci975"] == [0., 0.]


def test_complete_match_schedule_preserves_explicit_seeds_and_waits_for_every_slot(monkeypatch):
    class Env:
        def __init__(self, **kwargs):
            self.reset_calls = []
        def reset(self, **kwargs):
            self.reset_calls.append(kwargs)
    instance = Env()
    monkeypatch.setattr(diagnostic.gd, "VecEnv", lambda **kwargs: instance)
    class Collector:
        def __init__(self, *args, seeds, **kwargs):
            self.started = False
            self.done = np.zeros(len(seeds), bool)
            self.first_round = [__import__("hashlib").sha256() for _ in seeds]
            self.play_decisions = np.zeros(len(seeds), int)
            self.calls = 0
        def collect(self, steps):
            assert self.started, "collector must not reset explicit match seeds"
            self.calls += 1
            self.results = []
            for e in range(2):
                if self.calls == 1 + 2*e:
                    self.done[e] = True
                    self.results.append(SimpleNamespace(env_id=e, match_id=0,
                        seat_return=[1., -1., 1., -1.], gain=1, winning_team=0,
                        match_winner=0))
    monkeypatch.setattr(diagnostic, "EvalCollector", Collector)
    result = diagnostic.run_wave(None, {1: None}, [91, 91], "cpu", "full", 5)
    assert instance.reset_calls == [{"match_seeds": [91, 91]}]
    assert result["steps"] == 3
    assert [r["win"] for r in result["records"]] == [1, 0]
