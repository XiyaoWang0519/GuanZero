"""Incremental/full-prefix parity and shared-public/private-query boundaries."""
import pytest
import torch

from train.belief_model import HistoryBelief
from train.history_cache import PublicHistoryCache
from train.logs import TOKEN_DIM


def fixture(dtype=torch.float32):
    torch.set_num_threads(1)
    torch.manual_seed(47)
    model = HistoryBelief(1849, 16, 2).to(dtype=dtype).eval()
    tokens = torch.randint(0, 2, (9, TOKEN_DIM), dtype=torch.uint8)
    tokens[:, :4] = 0
    tokens[torch.arange(len(tokens)), torch.arange(len(tokens)) % 4] = 1
    tokens[:, 150:158] = 0
    obs = torch.randn(4, 1849, dtype=dtype)
    return model, tokens, obs


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_every_prefix_matches_full_causal_model_for_all_seats(dtype):
    model, tokens, obs = fixture(dtype)
    cache = PublicHistoryCache(model, max_tokens=len(tokens))
    seats = torch.arange(4)
    for length in range(len(tokens) + 1):
        if length:
            cache.append_public(tokens[length - 1])
        with torch.inference_mode():
            prefix = tokens[:length].to(dtype=dtype)
            expected = model(obs, prefix[None].expand(4, -1, -1),
                             torch.full((4,), length), seats)
            public, _ = model.encode_public(prefix[None], torch.tensor([length]))
        tolerance = 2e-6 if dtype == torch.float32 else 1e-12
        torch.testing.assert_close(cache.query(obs, seats), expected, atol=tolerance, rtol=tolerance)
        torch.testing.assert_close(cache.public_memory, public, atol=tolerance, rtol=tolerance)
        assert cache.public_length == length
        assert not cache.query(obs, seats).requires_grad


def test_multiappend_and_private_queries_cannot_change_shared_cache():
    model, tokens, obs = fixture()
    batch = PublicHistoryCache(model)
    single = PublicHistoryCache(model)
    batch.append_public(tokens[:4])
    for token in tokens[:4]:
        single.append_public(token)
    torch.testing.assert_close(batch.public_memory, single.public_memory, rtol=0, atol=0)
    before = batch.public_memory
    before_keys = [value.clone() for value in batch._keys]
    before_values = [value.clone() for value in batch._values]
    output = batch.query(obs, torch.arange(4))
    private = batch.encode_private(obs, torch.arange(4))
    assert private.shape == (4, model.width)
    with torch.inference_mode():
        torch.testing.assert_close(model.head(private).reshape(4, 3, 54, 3), output)
    batch.query(-obs * 9, torch.arange(4).flip(0))
    torch.testing.assert_close(batch.public_memory, before, rtol=0, atol=0)
    for actual, expected in zip(batch._keys + batch._values, before_keys + before_values):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert output.shape == (4, 3, 54, 3)
    assert batch.query(obs[0], 3).shape == (1, 3, 54, 3)
    snapshot = batch.public_memory
    snapshot.zero_()
    torch.testing.assert_close(batch.public_memory, before, rtol=0, atol=0)
    batch.append_public(tokens[4:])
    single.append_public(tokens[4:])
    torch.testing.assert_close(batch.public_memory, single.public_memory, rtol=0, atol=0)


def test_clear_starts_new_round_and_accepts_updated_weights_only_after_replay():
    model, tokens, obs = fixture()
    cache = PublicHistoryCache(model)
    cache.append_public(tokens)
    cache.clear()
    fresh = PublicHistoryCache(model)
    assert cache.public_length == 0
    torch.testing.assert_close(cache.public_memory, fresh.public_memory, rtol=0, atol=0)
    cache.append_public(tokens[2:5])
    fresh.append_public(tokens[2:5])
    torch.testing.assert_close(cache.query(obs, torch.arange(4)), fresh.query(obs, torch.arange(4)))
    with torch.no_grad():
        model.private[0].weight.add_(.01)
    with pytest.raises(ValueError, match="model changed"):
        cache.query(obs, 0)
    with pytest.raises(ValueError, match="model changed"):
        cache.append_public(tokens[0])
    cache.clear()
    cache.append_public(tokens[2:5])
    updated = PublicHistoryCache(model)
    updated.append_public(tokens[2:5])
    torch.testing.assert_close(cache.query(obs, 0), updated.query(obs, 0))


def test_overflow_and_invalid_events_leave_prefix_intact():
    model, tokens, _ = fixture()
    cache = PublicHistoryCache(model, max_tokens=3)
    cache.append_public(tokens[:2])
    before = cache.public_memory
    with pytest.raises(ValueError, match="max_tokens"):
        cache.append_public(tokens[2:4])
    invalid = tokens[2].clone()
    invalid[150] = 1
    with pytest.raises(ValueError, match="private tribute"):
        cache.append_public(invalid)
    with pytest.raises(ValueError, match="shape"):
        cache.append_public(torch.zeros(5))
    with pytest.raises(ValueError, match="finite binary"):
        cache.append_public(torch.full((TOKEN_DIM,), float("nan")))
    with pytest.raises(ValueError, match="finite binary"):
        cache.append_public(torch.zeros(TOKEN_DIM))
    cache.append_public(tokens[:0])
    torch.testing.assert_close(cache.public_memory, before, rtol=0, atol=0)
    assert cache.public_length == 2


def test_training_no_history_and_changed_device_dtype_rejected():
    model, tokens, obs = fixture()
    with pytest.raises(ValueError, match="evaluation-mode"):
        PublicHistoryCache(model.train())
    with pytest.raises(ValueError, match="history enabled"):
        PublicHistoryCache(HistoryBelief(1849, 16, 1, no_history=True).eval())
    cache = PublicHistoryCache(model.eval())
    model.train()
    with pytest.raises(ValueError, match="evaluation-mode"):
        cache.query(obs, 0)
    model.eval().double()
    with pytest.raises(ValueError, match="model changed"):
        cache.append_public(tokens[0])
    cache.clear()
    cache.append_public(tokens[:2])
    assert cache.query(obs, 0).dtype == torch.float64


@pytest.mark.parametrize("seat", [-1, 4, 1.5, True, [0, 1]])
def test_invalid_private_queries_rejected(seat):
    model, _, obs = fixture()
    cache = PublicHistoryCache(model)
    with pytest.raises(ValueError, match="absolute seats"):
        cache.query(obs[0], seat)
    with pytest.raises(ValueError, match="private observations"):
        cache.query(obs[0, :3], 0)


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_cache_limit_must_be_positive_integer(limit):
    model, _, _ = fixture()
    with pytest.raises(ValueError, match="max_tokens"):
        PublicHistoryCache(model, max_tokens=limit)
