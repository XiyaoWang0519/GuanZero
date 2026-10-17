"""Numerical/state-slice checks for the offline diagnostic, independent of real fixtures."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bench.qknorm_plasticity import migrate, scale_slices


class SmallReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(width=4, heads=2)
        self.q_proj = nn.Linear(4, 4)
        self.kv_proj = nn.Linear(4, 8)
        self.other = nn.Linear(4, 1)


def saved_state(actor):
    opt = torch.optim.Adam(actor.parameters(), lr=.0003)
    for p in actor.parameters():
        p.grad = torch.ones_like(p)
    opt.step()
    return copy.deepcopy(opt.state_dict())


def test_scaling_preserves_v_and_unrelated_parameters():
    torch.manual_seed(5)
    actor = SmallReader()
    before = copy.deepcopy(actor.state_dict())
    scale = {"q": torch.tensor([.1, .2]), "k": torch.tensor([.3, .4])}
    scale_slices(actor, dict(actor.named_parameters()), scale)
    assert torch.equal(actor.kv_proj.weight[4:], before["kv_proj.weight"][4:])
    assert torch.equal(actor.kv_proj.bias[4:], before["kv_proj.bias"][4:])
    assert torch.equal(actor.other.weight, before["other.weight"])
    x = torch.randn(7, 4)
    for prefix, side in (("q_proj", "q"), ("kv_proj", "k")):
        previous = torch.nn.functional.linear(x, before[prefix+".weight"], before[prefix+".bias"])[:, :4]
        current = getattr(actor, prefix)(x)[:, :4]
        expected = previous * scale[side].repeat_interleave(2)
        torch.testing.assert_close(current, expected)


@pytest.mark.parametrize("mode", ["transport", "reset_slice_old_step", "keep"])
def test_state_surgery_preserves_v_clock_other_state_and_source(mode):
    actor = SmallReader()
    saved = saved_state(actor)
    original = copy.deepcopy(saved)
    scale = {"q": torch.tensor([.1, .2]), "k": torch.tensor([.3, .4])}
    result = migrate(actor, saved, scale, mode)
    for (name, p), index in zip(actor.named_parameters(), saved["param_groups"][0]["params"]):
        old, new = saved["state"][index], result["state"][index]
        assert torch.equal(old["step"], new["step"])
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(saved["state"][index][key], original["state"][index][key])
            if name.startswith("kv_proj"):
                assert torch.equal(old[key][4:], new[key][4:])
            elif name.startswith("other"):
                assert torch.equal(old[key], new[key])
            if name.startswith(("q_proj", "kv_proj")):
                side = "q" if name.startswith("q_proj") else "k"
                factor = scale[side].repeat_interleave(2)
                factor = factor.reshape((-1,) + (1,) * (p.ndim-1))
                expected = old[key][:4]
                if mode == "transport":
                    expected = expected / factor.pow(1 if key == "exp_avg" else 2)
                elif mode == "reset_slice_old_step":
                    expected = torch.zeros_like(expected)
                torch.testing.assert_close(new[key][:4], expected)


def scalar_step(theta, grad, m, v, step, lr, eps):
    param = nn.Parameter(torch.tensor(theta, dtype=torch.float64))
    opt = torch.optim.Adam([param], lr=lr, eps=eps)
    opt.state[param] = dict(step=torch.tensor(float(step)),
                            exp_avg=torch.tensor(m, dtype=torch.float64),
                            exp_avg_sq=torch.tensor(v, dtype=torch.float64))
    param.grad = torch.tensor(grad, dtype=torch.float64)
    opt.step()
    return param.item()


def test_coordinate_preserving_adam_requires_lr_and_epsilon_transform():
    # Epsilon dominates here: transforming moments alone cannot preserve the trajectory.
    a, theta, g, m, v = .1, 2., 1e-11, 1e-10, 1e-20
    baseline = scalar_step(theta, g, m, v, 1000, .0003, 1e-8)
    correct = scalar_step(a*theta, g/a, m/a, v/a**2, 1000, a*.0003, 1e-8/a)
    assert correct == pytest.approx(a*baseline, abs=1e-14)
    wrong = scalar_step(a*theta, g/a, m/a, v/a**2, 1000, .0003, 1e-8)
    assert abs(wrong-a*baseline) > 1e-7


def test_zero_moments_with_old_clock_are_not_a_fresh_adam_step():
    mature = scalar_step(1., 1., 0., 0., 1_000_000, .0003, 1e-8)
    fresh = scalar_step(1., 1., 0., 0., 0, .0003, 1e-8)
    assert (1-mature)/(1-fresh) == pytest.approx(10**.5, rel=1e-6)
