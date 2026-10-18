"""Gradient reduction reuse preserves clipping, metrics and optimizer state."""
import copy

import pytest
import torch

from train.history_gradients import ActorGradientClipper


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"))]


class OriginalActorGradientClipper:
    """The trainer's original norm/clip operations, kept as the test oracle."""

    def __init__(self, parameters, encoder_parameters):
        self.parameters = tuple(parameters)
        self.encoder_parameters = tuple(encoder_parameters)

    @torch.no_grad()
    def __call__(self, max_norm):
        grads = [p.grad.detach().float() for p in self.encoder_parameters if p.grad is not None]
        encoder = (torch.stack(torch._foreach_norm(grads)) ** 2).unbind() if grads else ()
        return torch.nn.utils.clip_grad_norm_(self.parameters, max_norm), encoder


def assert_exact(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_exact(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_exact(a, b)
    else:
        assert actual == expected


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["dense", "views", "missing", "none", "zero", "overflow", "nan"])
def test_norms_and_clipping_match_original(device, kind):
    torch.manual_seed(641)
    shapes = [(513,), (7, 11), (), (0,), (13, 3)]
    baseline = [torch.nn.Parameter(torch.randn(shape, device=device)) for shape in shapes]
    candidate = [torch.nn.Parameter(p.detach().clone()) for p in baseline]
    flat = torch.randn(sum(p.numel() for p in baseline), device=device)
    views = flat.clone()
    cursor = 0
    for index, (a, b) in enumerate(zip(baseline, candidate)):
        n = a.numel()
        if kind == "none" or (kind == "missing" and index in (0, 4)):
            continue
        gradient = flat[cursor:cursor + n].view_as(a)
        a.grad = gradient.clone()
        b.grad = views[cursor:cursor + n].view_as(b) if kind == "views" else gradient.clone()
        if kind == "zero":
            a.grad.zero_()
            b.grad.zero_()
        elif kind in ("overflow", "nan") and index == 0:
            value = 1e30 if kind == "overflow" else float("nan")
            a.grad.fill_(value)
            b.grad.fill_(value)
        cursor += n
    pointers = [p.grad.data_ptr() if p.grad is not None else None for p in candidate]
    # A different encoder parameter order must preserve the metric's sum order.
    encoder_indices = (4, 0, 2)
    original = OriginalActorGradientClipper(baseline, [baseline[i] for i in encoder_indices])
    optimized = ActorGradientClipper(candidate, [candidate[i] for i in encoder_indices])
    expected = original(0.7)
    actual = optimized(0.7)
    assert_exact(actual, expected)
    for p, q, pointer in zip(candidate, baseline, pointers):
        assert_exact(p.grad, q.grad)
        assert (p.grad.data_ptr() if p.grad is not None else None) == pointer
    if kind == "overflow":
        # Keep the existing FP32 overflow detection; do not widen its reduction.
        assert torch.isinf(actual[0])


@pytest.mark.parametrize("device", DEVICES)
def test_missing_gradients_and_adam_state_across_steps(device):
    torch.manual_seed(831)
    baseline = [torch.nn.Parameter(torch.randn(9, 13, device=device)) for _ in range(4)]
    candidate = [torch.nn.Parameter(p.detach().clone()) for p in baseline]
    original = OriginalActorGradientClipper(baseline, baseline[:2])
    optimized = ActorGradientClipper(candidate, candidate[:2])
    optimizers = [torch.optim.Adam(params, lr=1e-3) for params in (baseline, candidate)]
    for step in range(4):
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        for index, (a, b) in enumerate(zip(baseline, candidate)):
            if index == step:
                continue
            gradient = torch.randn_like(a)
            a.grad = gradient.clone()
            b.grad = gradient.clone()
        assert_exact(optimized(1.0), original(1.0))
        for optimizer in optimizers:
            optimizer.step()
        assert_exact(candidate, baseline)
        assert_exact(optimizers[1].state_dict(), optimizers[0].state_dict())


def test_mixed_dtype_retains_original_path():
    baseline = [torch.nn.Parameter(torch.ones(9, dtype=dtype))
                for dtype in (torch.float32, torch.float64)]
    candidate = copy.deepcopy(baseline)
    for a, b in zip(baseline, candidate):
        a.grad = torch.linspace(-0.3, 0.7, 9, dtype=a.dtype)
        b.grad = a.grad.clone()
    original = OriginalActorGradientClipper(baseline, baseline)
    optimized = ActorGradientClipper(candidate, candidate)
    assert not optimized.reuse_norms
    assert_exact(optimized(0.5), original(0.5))
    assert_exact([p.grad for p in candidate], [p.grad for p in baseline])


def test_empty_actor_and_invalid_encoder():
    assert_exact(ActorGradientClipper([], [])(1.0), (torch.tensor(0.0), ()))
    with pytest.raises(ValueError, match="belong to the actor"):
        ActorGradientClipper([torch.nn.Parameter(torch.ones(1))],
                             [torch.nn.Parameter(torch.ones(1))])


def test_complete_ppo_updates_match_original(tmp_path, monkeypatch):
    from train import history_gradients
    from train.history_ppo import HistoryPPOConfig, HistoryTrainer

    config = HistoryPPOConfig(width=32, layers=1, heads=4, num_envs=4,
                              steps_per_update=90, seed=3, updates=2, epochs=1,
                              minibatch_matches=2, aux_heads="next,belief,outcome",
                              aux_coef=0.1, snapshot_updates=1)
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    torch.use_deterministic_algorithms(True)
    try:
        with monkeypatch.context() as context:
            context.setattr(history_gradients, "ActorGradientClipper", OriginalActorGradientClipper)
            baseline = HistoryTrainer(config, tmp_path / "baseline")
            expected = [baseline.update() for _ in range(config.updates)]
        candidate = HistoryTrainer(config, tmp_path / "candidate")
        actual = [candidate.update() for _ in range(config.updates)]
        for a, b in zip(actual, expected):
            assert_exact({key: a[key] for key in candidate.STAT_KEYS},
                         {key: b[key] for key in baseline.STAT_KEYS})
            assert a["update_samples"] == b["update_samples"]
        assert_exact(candidate.actor.state_dict(), baseline.actor.state_dict())
        assert_exact(candidate.critic.state_dict(), baseline.critic.state_dict())
        assert_exact(candidate.actor_optimizer.state_dict(), baseline.actor_optimizer.state_dict())
        assert_exact(candidate.critic_optimizer.state_dict(), baseline.critic_optimizer.state_dict())
        assert_exact(candidate.generator.get_state(), baseline.generator.get_state())
        assert candidate.rng.bit_generator.state == baseline.rng.bit_generator.state
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
