"""First fusion layer factoring keeps the existing model and checkpoint contract."""

import pytest
import torch

from train.model import GuandanModel, ModelConfig, select_actions


CONFIG = ModelConfig(obs_dim=19, act_dim=7, state_width=24, state_layers=2,
                     action_width=12, action_layers=2, fusion_width=16,
                     fusion_layers=2)


def unfactored_scores(model, obs, cand, offsets, phase, phase_code, chunk_size):
    state = model.state_tower(obs)
    rows = torch.repeat_interleave(torch.arange(len(obs)), offsets.diff(),
                                   output_size=len(cand))
    return torch.cat([
        model._fuse(state[rows[start:start + chunk_size]],
                    model.action_tower(cand[start:start + chunk_size]),
                    phase[rows[start:start + chunk_size]], phase_code)
        for start in range(0, len(cand), chunk_size)
    ])


@pytest.mark.parametrize("phase_code", [1, 2, 3])
def test_static_scoring_matches_original_values_and_gradients(phase_code):
    torch.set_num_threads(1)
    torch.manual_seed(71)
    model = GuandanModel(CONFIG)
    obs = torch.randn(4, CONFIG.obs_dim)
    cand = torch.randn(70, CONFIG.act_dim)
    offsets = torch.tensor([0, 12, 12, 48, 70])
    phase = torch.tensor([3, 1, 2, 3])

    actual = model.score_candidates(obs, cand, offsets, phase,
                                    chunk_size=4, phase_code=phase_code)
    expected = unfactored_scores(model, obs, cand, offsets, phase, phase_code, 4)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

    (actual.square().sum()).backward()
    actual_grad = {name: param.grad.clone() for name, param in model.named_parameters()
                   if param.grad is not None}
    model.zero_grad(set_to_none=True)
    (expected.square().sum()).backward()
    assert set(actual_grad) == {name for name, param in model.named_parameters()
                                if param.grad is not None}
    for name, param in model.named_parameters():
        if param.grad is not None:
            torch.testing.assert_close(actual_grad[name], param.grad,
                                       rtol=1e-4, atol=2e-5)

    with torch.no_grad():
        state = model.state_tower(obs)
        cached = model.score_candidates(obs, cand, offsets, phase, chunk_size=2,
                                        phase_code=phase_code, state=state)
        torch.testing.assert_close(cached, expected, rtol=2e-5, atol=2e-6)


def test_exact_score_ties_keep_first_index_and_checkpoint_keys():
    torch.set_num_threads(1)
    torch.manual_seed(79)
    model = GuandanModel(CONFIG).eval()
    original_keys = set(model.state_dict())
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.phase_heads["3"][1].bias.fill_(1)
    obs = torch.randn(2, CONFIG.obs_dim)
    action_a, action_b = torch.randn(2, CONFIG.act_dim)
    cand = torch.cat((action_a.expand(33, -1), action_b.expand(32, -1)))
    offsets = torch.tensor([0, 33, 65])
    phase = torch.full((2,), 3)
    with torch.inference_mode():
        scores = model.score_candidates(obs, cand, offsets, phase,
                                        chunk_size=33, phase_code=3)
        expected = unfactored_scores(model, obs, cand, offsets, phase, 3, 33)
    assert torch.equal(scores, torch.ones_like(scores))
    assert torch.equal(expected, scores)
    assert select_actions(scores, offsets).tolist() == [0, 0]
    assert select_actions(expected, offsets).tolist() == [0, 0]
    assert set(model.state_dict()) == original_keys


def test_bfloat16_autocast_keeps_original_fused_calculation():
    torch.set_num_threads(1)
    torch.manual_seed(83)
    model = GuandanModel(CONFIG).eval()
    obs = torch.randn(3, CONFIG.obs_dim)
    cand = torch.randn(72, CONFIG.act_dim)
    offsets = torch.tensor([0, 16, 40, 72])
    phase = torch.full((3,), 3)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        actual = model.score_candidates(obs, cand, offsets, phase,
                                        chunk_size=24, phase_code=3)
        expected = unfactored_scores(model, obs, cand, offsets, phase, 3, 24)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tiny_static_batch_keeps_original_fused_calculation():
    torch.set_num_threads(1)
    torch.manual_seed(89)
    model = GuandanModel(CONFIG).eval()
    obs = torch.randn(2, CONFIG.obs_dim)
    cand = torch.randn(40, CONFIG.act_dim)
    offsets = torch.tensor([0, 19, 40])
    phase = torch.full((2,), 3)
    with torch.inference_mode():
        actual = model.score_candidates(obs, cand, offsets, phase,
                                        chunk_size=24, phase_code=3)
        expected = unfactored_scores(model, obs, cand, offsets, phase, 3, 24)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sparse_static_batch_keeps_original_fused_calculation():
    torch.set_num_threads(1)
    torch.manual_seed(97)
    model = GuandanModel(CONFIG).eval()
    obs = torch.randn(10, CONFIG.obs_dim)
    cand = torch.randn(70, CONFIG.act_dim)
    offsets = torch.arange(0, 71, 7)
    phase = torch.full((10,), 3)
    with torch.inference_mode():
        actual = model.score_candidates(obs, cand, offsets, phase,
                                        chunk_size=24, phase_code=3)
        expected = unfactored_scores(model, obs, cand, offsets, phase, 3, 24)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
