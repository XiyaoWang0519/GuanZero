"""Opt-in wide private projection (``rollout_wide_projection``).

With batched private attention the default path keeps each q/out projection's
original one-row GEMM shape so rollouts stay bitwise. The wide option runs one
linear call over all rows instead: the same FP32 math with a possibly different
reduction order. These tests pin both halves of that contract: the default
stays bitwise equal to the per-row reference, and the wide path agrees with it
to tight FP32 tolerance (never through TF32 or a lower precision).
"""
import json

import numpy as np
import pytest
import torch

from train.history_model import DecisionInputs, HistoryActor, HistoryPolicyConfig, fresh_player
from train.history_ppo import HistoryPPOConfig, HistoryTrainer, build_parser, config_from_args

RTOL = 1e-5
ATOL = 1e-6


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA wide-projection comparison")
    return request.param


@pytest.fixture(scope="module", autouse=True)
def strict_fp32():
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1], warn_only=previous[2])
        torch.backends.cuda.matmul.allow_tf32 = previous[3]
        torch.backends.cudnn.allow_tf32 = previous[4]


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype == torch.float32
    assert actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8))


def one_row_problem(actor, device, batch, length, seed):
    width = actor.config.width
    generator = torch.Generator(device=device).manual_seed(seed + 100)
    query = torch.randn(batch, width, generator=generator, device=device)
    encoded = torch.randn(batch, length, width, generator=generator, device=device)
    keys, values = actor.kv_proj(encoded).chunk(2, dim=-1)
    prefix = torch.randint(length, (batch,), generator=generator, device=device)
    allowed = torch.arange(length, device=device)[None] <= prefix[:, None]
    return query, encoded, keys, values, prefix, allowed


def test_defaults_are_off():
    assert HistoryActor(HistoryPolicyConfig(width=32, layers=1, heads=4)).wide_private_projection is False
    assert HistoryPPOConfig().rollout_wide_projection is False
    args = build_parser().parse_args(["--output", "x"])
    assert config_from_args(args).rollout_wide_projection is False
    args = build_parser().parse_args(["--output", "x", "--rollout-batched-attention",
                                      "--rollout-wide-projection"])
    assert config_from_args(args).rollout_wide_projection is True


def test_wide_projection_requires_batched_attention():
    with pytest.raises(ValueError, match="rollout_batched_attention"):
        HistoryPPOConfig(rollout_wide_projection=True)
    HistoryPPOConfig(rollout_batched_attention=True, rollout_wide_projection=True)


@pytest.mark.parametrize("width", [32, 64, 128])
@pytest.mark.parametrize("batch,length", [(1, 1), (4, 127), (32, 513), (16, 2403), (257, 64)])
@pytest.mark.parametrize("seed", [0, 7])
def test_wide_projection_matches_exact_path(device, width, batch, length, seed):
    actor, _ = fresh_player(HistoryPolicyConfig(width=width, layers=1, heads=4), seed)
    actor.to(device)
    with torch.no_grad():
        query, _, keys, values, _, allowed = one_row_problem(actor, device, batch, length, seed)
        reference = torch.cat([actor._attend(query[b:b + 1], keys[b:b + 1],
                                             values[b:b + 1], allowed[b:b + 1])
                               for b in range(batch)])
        exact = actor._attend_independent(query, keys, values, allowed)
        actor.wide_private_projection = True
        wide = actor._attend_independent(query, keys, values, allowed)
    # Default (flag off) stays bit-for-bit the per-row reference.
    assert_bits(exact, reference)
    assert wide.dtype == torch.float32 and wide.shape == reference.shape
    assert torch.allclose(wide, reference, rtol=RTOL, atol=ATOL), \
        float((wide - reference).abs().max())


@pytest.mark.parametrize("batch", [2, 9, 64])
def test_decision_states_wide_vs_default(device, batch):
    """Whole private decision state, full-history one-decision-per-stream layout."""
    actor, _ = fresh_player(HistoryPolicyConfig(width=64, layers=2, heads=4), 11)
    actor.to(device)
    generator = torch.Generator(device=device).manual_seed(batch)
    length = 97
    encoded = torch.randn(batch, length, 64, generator=generator, device=device)
    inputs = DecisionInputs(
        streams=None, match_index=torch.arange(batch, device=device),
        prefix=torch.randint(length, (batch,), generator=generator, device=device),
        obs=torch.randint(0, 2, (batch, actor.config.obs_dim), generator=generator,
                          device=device, dtype=torch.uint8),
        seat=torch.randint(0, 4, (batch,), generator=generator, device=device),
        cand=None, offsets=None, one_decision_per_stream=True)
    with torch.no_grad():
        loop = actor.decision_states(encoded, inputs)
        actor.batched_private_attention = True
        exact = actor.decision_states(encoded, inputs)
        actor.wide_private_projection = True
        wide = actor.decision_states(encoded, inputs)
    assert_bits(exact, loop)
    assert torch.allclose(wide, loop, rtol=RTOL, atol=ATOL), float((wide - loop).abs().max())


def test_trainer_update_with_wide_projection(tmp_path):
    config = HistoryPPOConfig(width=32, layers=1, heads=4, num_envs=4, steps_per_update=120,
                              seed=5, updates=1, epochs=1, minibatch_matches=2,
                              rollout_kv_cache=True, rollout_batched_attention=True,
                              rollout_wide_projection=True)
    trainer = HistoryTrainer(config, tmp_path)
    assert trainer.actor.wide_private_projection is True
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["inference"]["rollout_wide_projection"] is True
    trainer.collect()
    rows = np.arange(len(trainer.buffer))
    assert rows.size
    with torch.no_grad():
        log_prob, _ = trainer.recompute_log_probs(rows)
    # Learner recomputation uses the unchanged per-match path; behaviour
    # log-probabilities agree to FP32 reduction-order noise.
    assert np.abs(log_prob.numpy() - trainer.buffer.compact()["logp"][rows]).max() < 1e-5
    stats = trainer.learn()
    assert stats["update_samples"] > 0 and np.isfinite(stats["policy_loss"])
