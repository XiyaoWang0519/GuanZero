"""Information boundaries and capacity controls for the second belief gate."""
import pytest
import torch
from torch import nn

from train.belief_model import HistoryBelief, count_parameters, matched_models
from train.logs import TOKEN_DIM


def inputs():
    torch.manual_seed(31)
    return (torch.randn(3, 1849), torch.randn(3, 5, TOKEN_DIM),
            torch.tensor([0, 2, 5]), torch.tensor([0, 1, 3]))


def test_capacity_matching_and_paired_independent_initialization():
    torch.set_num_threads(1)
    models = matched_models(1849, 16, 1)
    assert set(models) == {"v1", "v2", "no_history"}
    sizes = {name: count_parameters(model) for name, model in models.items()}
    assert sizes["no_history"] == sizes["v2"]
    assert abs(sizes["v1"] - sizes["v2"]) / sizes["v2"] < .02
    for name, value in models["v2"].state_dict().items():
        control = models["no_history"].state_dict()[name]
        torch.testing.assert_close(value, control, rtol=0, atol=0)
        assert value.data_ptr() != control.data_ptr()
    assert sum(isinstance(module, nn.ReLU) for module in models["v2"].private) == 2
    assert any(isinstance(module, nn.ReLU) for module in models["v2"].feed_forward)


def test_future_padding_empty_prefix_and_public_causality():
    torch.set_num_threads(1)
    model = HistoryBelief(1849, 16, 1).eval()
    obs, tokens, lengths, seats = inputs()
    with torch.inference_mode():
        expected = model(obs, tokens, lengths, seats)
        before, _ = model.encode_public(tokens, torch.full_like(lengths, 5))
        changed = tokens.clone()
        changed[0] = torch.randn_like(changed[0]) * 100
        changed[1, 2:] = torch.randn_like(changed[1, 2:]) * 100
        actual = model(obs, changed, lengths, seats)
        after, _ = model.encode_public(changed, torch.full_like(lengths, 5))
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    # Even when future tokens are unmasked, earlier PUBLIC states stay causal.
    torch.testing.assert_close(before[1, :3], after[1, :3])


def test_private_query_never_changes_public_stream():
    torch.set_num_threads(1)
    model = HistoryBelief(1849, 16, 1).eval()
    obs, tokens, lengths, seats = inputs()
    encoded = []
    handle = model.stream.register_forward_hook(
        lambda _module, _args, result: encoded.append(result.detach().clone()))
    with torch.inference_mode():
        first = model(obs, tokens, lengths, seats)
        second = model(obs * -7, tokens, lengths, (seats + 1) % 4)
    handle.remove()
    assert len(encoded) == 2
    torch.testing.assert_close(encoded[0], encoded[1], rtol=0, atol=0)
    assert not torch.allclose(first, second)


def test_paired_models_agree_without_history_and_history_path_is_trainable():
    torch.set_num_threads(1)
    models = matched_models(1849, 16, 1)
    history, control = models["v2"], models["no_history"]
    obs, tokens, lengths, seats = inputs()
    empty = torch.zeros_like(lengths)
    with torch.inference_mode():
        # Slicing padding away may select different matmul shapes, so allow
        # normal float32 roundoff while requiring the same initial function.
        torch.testing.assert_close(history(obs, tokens, empty, seats),
                                   control(obs, tokens, lengths, seats))
        present = history(obs, tokens, lengths, seats)
        absent = control(obs, tokens, lengths, seats)
    assert not torch.allclose(present[1:], absent[1:])
    history(obs, tokens, lengths, seats).square().mean().backward()
    assert history.public.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("training", [False, True])
def test_no_history_cannot_read_tokens_in_training_or_evaluation(training):
    torch.set_num_threads(1)
    model = matched_models(1849, 16, 1)["no_history"].train(training)
    obs, tokens, lengths, seats = inputs()
    expected = model(obs, tokens, lengths, seats)
    other_tokens = torch.randn(3, 11, TOKEN_DIM) * 100
    actual = model(obs, other_tokens, torch.tensor([11, 8, 3]), seats)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # The control is actually fitted; private and post-attention paths receive gradients.
    actual.square().mean().backward()
    assert model.private[0].weight.grad.abs().sum() > 0
    assert model.feed_forward[0].weight.grad.abs().sum() > 0
    assert model.public.weight.grad is None or model.public.weight.grad.count_nonzero() == 0


@pytest.mark.parametrize("kwargs", [{"width": 14}, {"layers": 0}, {"obs_dim": 0}])
def test_invalid_architectures_rejected(kwargs):
    with pytest.raises(ValueError):
        HistoryBelief(**{"obs_dim": 1849, **kwargs})
