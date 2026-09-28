"""Opt-in batched learner attention (``learner_batched_attention``).

The learner's minibatches hold many decisions per match. By default each
match's decisions attend in their own call, found with a device ``nonzero``
per match. The opt-in path places every decision at (match, rank) in one
padded attention call. Same FP32 math, possibly different reduction order:
values and gradients must agree to tight tolerance, and the default path
must stay untouched.
"""
import json

import numpy as np
import pytest
import torch

from train.history_model import HistoryActor, HistoryPolicyConfig
from train.history_ppo import HistoryPPOConfig, HistoryTrainer, build_parser, config_from_args

RTOL = 1e-5
ATOL = 1e-6


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA learner-attention comparison")
    return request.param


@pytest.fixture(scope="module", autouse=True)
def strict_fp32():
    previous = (torch.get_num_threads(), torch.backends.cuda.matmul.allow_tf32,
                torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_num_threads(previous[0])
        torch.backends.cuda.matmul.allow_tf32 = previous[1]
        torch.backends.cudnn.allow_tf32 = previous[2]


def per_match_reference(actor, query, keys, values, match_index, prefix):
    positions = torch.arange(keys.shape[1], device=keys.device)
    attended = torch.zeros_like(query)
    for b in range(keys.shape[0]):
        rows = (match_index == b).nonzero(as_tuple=True)[0]
        if len(rows):
            allowed = positions[None] <= prefix[rows][:, None]
            attended = attended.index_put((rows,), actor._attend(
                query[rows], keys[b:b + 1], values[b:b + 1], allowed))
    return attended


def problem(device, width, matches, length, decisions, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    actor = HistoryActor(HistoryPolicyConfig(width=width, layers=1, heads=4)).to(device)
    query = torch.randn(decisions, width, generator=generator).to(device).requires_grad_()
    encoded = torch.randn(matches, length, width, generator=generator).to(device)
    # Uneven, unsorted assignment; the last match gets no decisions at all.
    match_index = torch.randint(matches - 1, (decisions,), generator=generator).to(device)
    prefix = torch.randint(length, (decisions,), generator=generator).to(device)
    return actor, query, encoded, match_index, prefix


def test_defaults_are_off():
    assert HistoryActor(HistoryPolicyConfig(width=32, layers=1, heads=4)).batched_match_attention is False
    assert HistoryPPOConfig().learner_batched_attention is False
    args = build_parser().parse_args(["--output", "x"])
    assert config_from_args(args).learner_batched_attention is False
    args = build_parser().parse_args(["--output", "x", "--learner-batched-attention"])
    assert config_from_args(args).learner_batched_attention is True


@pytest.mark.parametrize("width,matches,length,decisions,seed",
                         [(32, 3, 17, 40, 1), (64, 5, 64, 200, 2), (32, 2, 1, 7, 3),
                          (64, 9, 130, 513, 4)])
def test_batched_matches_per_match_values_and_gradients(device, width, matches, length,
                                                        decisions, seed):
    actor, query, encoded, match_index, prefix = problem(device, width, matches, length,
                                                         decisions, seed)
    keys, values = actor.kv_proj(encoded).chunk(2, dim=-1)
    expected = per_match_reference(actor, query, keys, values, match_index, prefix)
    actual = actor._attend_by_match(query, keys, values, match_index, prefix)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)

    weights = torch.randn(expected.shape, generator=torch.Generator().manual_seed(seed)).to(device)
    grads = []
    for output in (expected, actual):
        actor.zero_grad()
        query.grad = None
        (output * weights).sum().backward(retain_graph=True)
        grads.append([query.grad.clone()] + [p.grad.clone() for p in actor.parameters()
                                             if p.grad is not None])
    assert len(grads[0]) == len(grads[1])
    for reference, batched in zip(*grads):
        torch.testing.assert_close(batched, reference, rtol=1e-4, atol=1e-5)


def test_trainer_recomputation_matches_and_learns(tmp_path):
    config = HistoryPPOConfig(width=32, layers=1, heads=4, num_envs=4, steps_per_update=120,
                              seed=7, updates=1, epochs=1, minibatch_matches=2)
    trainer = HistoryTrainer(config, tmp_path / "default")
    trainer.collect()
    rows = np.arange(len(trainer.buffer))
    assert rows.size
    with torch.no_grad():
        exact, _ = trainer.recompute_log_probs(rows)
        trainer.actor.batched_match_attention = True
        batched, _ = trainer.recompute_log_probs(rows)
    assert np.abs(batched.numpy() - exact.numpy()).max() < 1e-5

    flagged = HistoryTrainer(HistoryPPOConfig(**{**config.__dict__,
                                                 "learner_batched_attention": True}),
                             tmp_path / "flagged")
    assert flagged.actor.batched_match_attention is True
    manifest = json.loads((tmp_path / "flagged" / "manifest.json").read_text())
    assert manifest["inference"]["learner_batched_attention"] is True
    flagged.collect()
    stats = flagged.learn()
    assert stats["minibatches"] > 0
    assert all(torch.isfinite(p).all() for p in flagged.actor.parameters())
