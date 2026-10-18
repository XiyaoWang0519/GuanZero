"""The known CUDA private-attention workaround covers every actor route."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from train import history_attention, history_model
from train.history_model import DecisionInputs, HistoryActor, HistoryPolicyConfig, StreamBatch
from train.logs import TOKEN_DIM


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("grad_enabled", [False, True])
def test_private_sdpa_only_forces_math_for_cuda_training(monkeypatch, cuda, grad_enabled):
    """Check dispatch without requiring CUDA hardware, retaining identical operands."""
    query = SimpleNamespace(is_cuda=cuda)
    keys, values, mask, expected = object(), object(), object(), object()
    calls, backends = [], []

    @contextmanager
    def kernel(backend):
        backends.append(backend)
        yield

    def attend(q, k, v, *, attn_mask):
        calls.append((q, k, v, attn_mask))
        return expected

    monkeypatch.setattr(history_attention, "sdpa_kernel", kernel)
    monkeypatch.setattr(history_attention.F, "scaled_dot_product_attention", attend)
    with torch.set_grad_enabled(grad_enabled):
        actual = history_attention.private_sdpa(query, keys, values, attn_mask=mask)
    assert actual is expected
    assert calls == [(query, keys, values, mask)]
    assert backends == ([SDPBackend.MATH] if cuda and grad_enabled else [])


@pytest.mark.parametrize("route,decisions,matches,expected_calls", [
    ("default", 3, 2, 2),
    ("singleton", 1, 1, 1),
    ("window", 3, 2, 1),
    ("independent", 3, 3, 1),
    ("grouped", 3, 2, 1),
])
def test_every_private_actor_route_uses_shared_dispatch(monkeypatch, route, decisions,
                                                       matches, expected_calls):
    torch.manual_seed(35)
    config = HistoryPolicyConfig(width=16, layers=1, heads=2, window=3 if route == "window" else 0)
    actor = HistoryActor(config).train()
    actor.causal_sdpa = True
    actor.batched_match_attention = route in ("singleton", "grouped")
    actor.batched_private_attention = route == "independent"
    inputs = DecisionInputs(
        streams=StreamBatch(torch.zeros(matches, 4, TOKEN_DIM, dtype=torch.uint8),
                            torch.zeros(matches, 4, dtype=torch.long),
                            torch.full((matches, 4), 2, dtype=torch.long),
                            torch.full((matches,), 4, dtype=torch.long)),
        match_index=torch.arange(decisions) % matches,
        prefix=torch.arange(decisions) * 2,
        obs=torch.randn(decisions, config.obs_dim),
        seat=torch.arange(decisions) % 4,
        cand=None, offsets=None,
        one_decision_per_stream=route == "independent")
    calls = []
    private_sdpa = history_attention.private_sdpa

    def tracked(query, keys, values, *, attn_mask):
        calls.append((query.shape, attn_mask.shape))
        return private_sdpa(query, keys, values, attn_mask=attn_mask)

    monkeypatch.setattr(history_model, "private_sdpa", tracked)
    states = actor.decision_states(None, inputs)
    assert len(calls) == expected_calls
    assert torch.isfinite(states).all()
    (states * torch.randn_like(states)).sum().backward()
    assert actor.q_proj.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA private SDPA backward acceptance")
def test_large_private_scores_match_math_outputs_and_gradients_on_cuda():
    """Large Q/K values remain on the math path, including singleton queries."""
    for decisions in (1, 7):
        generator = torch.Generator().manual_seed(761 + decisions)
        query = (torch.randn(2, 4, decisions, 8, generator=generator) * 1e4).cuda().requires_grad_()
        keys = (torch.randn(2, 4, 19, 8, generator=generator) * 1e4).cuda().requires_grad_()
        values = torch.randn(2, 4, 19, 8, generator=generator).cuda().requires_grad_()
        prefix = torch.arange(decisions, device="cuda") * 2
        mask = (torch.arange(19, device="cuda")[None] <= prefix[:, None])[None, None]
        upstream = torch.randn(query.shape, generator=generator).cuda()
        with sdpa_kernel(SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask)
        actual = history_attention.private_sdpa(query, keys, values, attn_mask=mask)
        expected_grad = torch.autograd.grad(expected, (query, keys, values), upstream)
        actual_grad = torch.autograd.grad(actual, (query, keys, values), upstream)
        for reference, candidate in zip((expected, *expected_grad), (actual, *actual_grad)):
            assert torch.isfinite(candidate).all()
            torch.testing.assert_close(candidate, reference, rtol=0, atol=0)
