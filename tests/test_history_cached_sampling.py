"""Supplied inference probabilities preserve the original sampler and RNG."""
from dataclasses import fields

import pytest
import torch

from test_history_model import Rollout
from train.history_model import HistoryPolicyConfig, fresh_player


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8),
                       expected.contiguous().view(torch.uint8))


@pytest.fixture(scope="module", autouse=True)
def exact_backend():
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.set_num_threads(previous[0])
    torch.use_deterministic_algorithms(previous[1], warn_only=previous[2])
    torch.backends.cuda.matmul.allow_tf32 = previous[3]
    torch.backends.cudnn.allow_tf32 = previous[4]


@pytest.fixture(scope="module")
def snapshot():
    return Rollout(num_envs=4, steps=12, seed=19, capture=(0, 11)).last


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA cached sampler acceptance")
    return request.param


@pytest.fixture(scope="module", params=["none", "auxiliary", "explicit"])
def model_inputs(request, snapshot, device):
    actor, _ = fresh_player(HistoryPolicyConfig(width=32, layers=1, action_width=16,
                                               fusion_width=16, response_mode=request.param), 7)
    actor.to(device)
    inputs = snapshot.inputs(device)
    with torch.no_grad():
        encoded = actor.encode_batch(inputs.streams)
        log_probs = actor.candidate_log_probs(inputs, encoded)
    assert len(set(inputs.counts.cpu().tolist())) > 1
    # A non-contiguous supplied vector must also retain every value.
    storage = torch.empty(2 * len(log_probs), device=device)
    supplied = storage[::2]
    supplied.copy_(log_probs)
    return actor, inputs, encoded, supplied


@pytest.mark.parametrize("max_hint", [False, True])
@pytest.mark.parametrize("mode,temperature,epsilon", [
    ("act", 1.0, 0.0), ("greedy", 1.0, 0.0), ("explore", 1.0, 0.0),
    ("explore", 0.7, 0.0), ("explore", 1.0, 0.2),
    ("explore", 1.7, 0.2), ("explore", 1.0, 1.0),
])
def test_supplied_probabilities_preserve_every_sampling_field_and_generator(
        model_inputs, device, max_hint, mode, temperature, epsilon, monkeypatch):
    actor, inputs, encoded, supplied = model_inputs
    longest = int(inputs.counts.max()) if max_hint else None
    generator = torch.Generator(device=device).manual_seed(29)
    initial_rng = generator.get_state()
    before = supplied.clone()

    def sample(**kwargs):
        if mode == "explore":
            value = actor.explore(inputs, generator, temperature=temperature, epsilon=epsilon,
                                  max_candidates=longest, **kwargs)
            return tuple(getattr(value, field.name) for field in fields(value))
        return actor.act(inputs, generator, greedy=mode == "greedy", max_candidates=longest, **kwargs)

    expected = sample(encoded=encoded)
    expected_rng = generator.get_state()
    generator.set_state(initial_rng)

    def forbidden_recomputation(*args, **kwargs):
        raise AssertionError("supplied probabilities unexpectedly recomputed the actor")

    with monkeypatch.context() as patch:
        patch.setattr(actor, "candidate_log_probs", forbidden_recomputation)
        actual = sample(inference_log_probs=supplied)
    for current, original in zip(actual, expected):
        assert_bits(current, original)
    assert_bits(generator.get_state(), expected_rng)
    assert_bits(supplied, before)
    if mode == "greedy":
        assert_bits(expected_rng, initial_rng)


@pytest.mark.parametrize("invalid", ["shape", "dtype", "device"])
def test_supplied_probabilities_reject_incomplete_or_incompatible_vectors(model_inputs, invalid):
    actor, inputs, _, supplied = model_inputs
    if invalid == "shape":
        supplied = supplied[:-1]
    elif invalid == "dtype":
        supplied = supplied.double()
    else:
        supplied = torch.empty(supplied.shape, device="meta")
    with pytest.raises(ValueError, match="every candidate"):
        actor.act(inputs, inference_log_probs=supplied)
