"""Generator/scheduler stage 0b vs the scalar play_match."""
import numpy as np
import pytest

import gd

pytestmark = pytest.mark.skipif(not hasattr(gd.VecEnv, 'load'), reason='needs the VecEnv.load engine build')

from bench.habit_crossround_batched import batched_probs, run_matches  # noqa: E402
from bench.habit_crossround_value import play_match  # noqa: E402
from bench.history_value_tasks import decision_probabilities  # noqa: E402
from tests.test_history_habit_value import styled_pair  # noqa: E402
from tests.test_history_value_tasks import fixture, small_actor  # noqa: E402

TIMING = {'prefix_seconds', 'seconds', 'config', 'index'}


def strip(rec):
    return {k: v for k, v in rec.items() if k not in TIMING}


def test_batched_probs_match_single_decisions():
    actor = small_actor()
    state, arrays, _ = fixture()
    other = state.determinize_uniform(0, 3)
    got = batched_probs(actor, [(state, arrays), (other, arrays)])
    for (s, a), p in zip([(state, arrays), (other, arrays)], got):
        np.testing.assert_allclose(p, decision_probabilities(actor, s, a), rtol=0, atol=1e-6)


@pytest.mark.parametrize('axis,b', [('pass', 4.0), ('bomb', 2.0)])
def test_exact_scheduler_reproduces_play_match(axis, b):
    actor = small_actor()
    kw = dict(worlds=3, candidates=2, margin=0., skip_gap=0.)
    specs = [dict(config=0, index=i, axis=axis, b=b, deal_seed=31 + i, z=(-1, 1)[i % 2], seed=7, **kw)
             for i in range(3)]
    got = sorted(run_matches(actor, specs, exact=True, concurrency=2), key=lambda r: r['index'])
    pair = styled_pair(actor, axis, b)
    for spec, rec in zip(specs, got):
        want = play_match(actor, pair, axis, b, spec['deal_seed'], spec['z'], spec['seed'], **kw)
        assert strip(rec) == strip(want)
    assert any(r['searches'] for r in got)
