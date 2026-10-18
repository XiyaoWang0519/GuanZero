"""Batched paired search vs the scalar reference ``team_rollout``."""
import numpy as np
import pytest

import gd

pytestmark = pytest.mark.skipif(not hasattr(gd.VecEnv, 'load'), reason='needs the VecEnv.load engine build')

from bench.batched_rollout import SearchRequest, batched_search, public_tokens, splitmix_uniform  # noqa: E402
from bench.history_habit_round import team_rollout  # noqa: E402
from tests.test_history_habit_value import styled_pair  # noqa: E402
from tests.test_history_value_tasks import fixture, small_actor  # noqa: E402
from train.logs import public_token  # noqa: E402


def test_vectorised_public_tokens_match_the_scalar_encoder():
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    enc = (rng.random((5, 154)) < .1).astype(np.float32)
    seat, left = np.array([0, 1, 2, 3, 1]), np.array([27, 0, 5, 12, 1])
    got = public_tokens(seat, enc, left)
    for i in range(5):
        want = public_token(SimpleNamespace(seat=seat[i], encoded_action=enc[i], cards_left=left[i]))
        np.testing.assert_array_equal(got[i], want)


def test_counter_uniforms_are_in_range_and_depend_on_every_key():
    u = splitmix_uniform(np.array([1, 1, 1, 2]), np.array([0, 1, 0, 0]), np.array([0, 0, 1, 0]))
    assert ((0 <= u) & (u < 1)).all() and len(set(u.tolist())) == 4


@pytest.mark.parametrize('axis,b', [('pass', 2.0), ('bomb', 4.0), ('lead_single', 3.0)])
def test_reference_sampler_reproduces_team_rollout(axis, b):
    actor = small_actor()
    state, arrays, cand = fixture()
    seeds = np.array([11, 12, 13])
    want = team_rollout(actor, styled_pair(actor, axis, b), state, arrays, cand, seeds)
    got = batched_search(actor, [SearchRequest(state, arrays, cand, seeds, (-1, 1), axis, b)],
                         sampler='reference')[0]
    np.testing.assert_array_equal(got, want)


def test_several_requests_in_one_batch_match_separate_runs():
    actor = small_actor()
    state, arrays, cand = fixture()
    a = SearchRequest(state, arrays, cand, np.array([3, 4]), (-1, 1), 'pass', 2.0)
    b = SearchRequest(state, arrays, cand[:1], np.array([5]), (1,), 'bomb', 1.0)
    together = batched_search(actor, [a, b], sampler='reference')
    for req, q in zip((a, b), together):
        np.testing.assert_array_equal(q, batched_search(actor, [req], sampler='reference')[0])


def test_fast_sampler_completes_with_valid_outcomes():
    actor = small_actor()
    state, arrays, cand = fixture()
    q = batched_search(actor, [SearchRequest(state, arrays, cand, np.arange(1, 9), (-1, 1), 'pass', 2.0)])[0]
    assert q.shape == (2, len(cand), 8, 2) and np.isin(q[..., 0], [0, 1]).all()


def test_device_inverse_cdf_matches_the_reference_rule():
    import torch
    from bench.batched_rollout import device_inverse_cdf
    from bench.qknorm_match_ablation import inverse_cdf
    g = torch.Generator().manual_seed(0)
    table = torch.log_softmax(torch.randn(50, 7, generator=g), 1)
    table[::3, 5:] = float('-inf')
    u = torch.rand(50, generator=g, dtype=torch.float64)
    torch.testing.assert_close(device_inverse_cdf(table, u), inverse_cdf(table, u)[0])
