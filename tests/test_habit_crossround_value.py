"""Cross-round value: prefix rounds, posterior tracking and fork logic."""
import numpy as np

from bench.habit_crossround_value import play_match
from tests.test_history_habit_value import styled_pair
from tests.test_history_value_tasks import small_actor


def test_identical_styles_give_no_evidence_and_never_fork():
    actor = small_actor()
    pair = styled_pair(actor, 'pass', 0.0)
    r = play_match(actor, pair, 'pass', 0.0, deal_seed=21, z=1, seed=5, worlds=2, candidates=2, margin=0.)
    assert r['prefix_llr'] == 0.0 and not r['forked'] and r['full'] == r['round']
    assert all(x == .5 for x in r['post_round'] + r['post_full'])


def test_prefix_rounds_build_full_but_not_round_evidence():
    actor = small_actor()
    pair = styled_pair(actor, 'pass', 4.0)
    r = play_match(actor, pair, 'pass', 4.0, deal_seed=21, z=1, seed=5, worlds=2, candidates=2, margin=0.,
                   skip_gap=0.)
    assert r['prefix_llr'] > 0                     # pass-heavy opponents seen in rounds 1-2
    assert r['searches'] > 0
    assert set(r['full']) == {'win', 'ret', 'searches'} and set(r['round']) == {'win', 'ret', 'searches'}
    if not r['forked']:
        assert r['full'] == r['round']
    # At the first target-round search the full view has strictly more evidence.
    assert r['post_full'][0] >= r['post_round'][0] or np.isclose(r['post_full'][0], r['post_round'][0])


def test_skip_gap_replaces_search_where_posteriors_agree():
    actor = small_actor()
    pair = styled_pair(actor, 'pass', 4.0)
    never = play_match(actor, pair, 'pass', 4.0, deal_seed=21, z=1, seed=5, worlds=2, candidates=2,
                       margin=0., skip_gap=2.0)            # gap can never reach 2: no search
    assert never['searches'] == 0 and not never['forked'] and never['full'] == never['round']
    assert never['skipped'] > 0
    always = play_match(actor, pair, 'pass', 4.0, deal_seed=21, z=1, seed=5, worlds=2, candidates=2,
                        margin=0., skip_gap=0.0)
    assert always['skipped'] == 0 and always['searches'] > 0
