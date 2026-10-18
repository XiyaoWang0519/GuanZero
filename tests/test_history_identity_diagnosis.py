"""Statistical controls distinguish strength, matching, and selection noise."""
import json
import numpy as np
import pytest
from bench import history_identity_diagnosis as run


def test_matching_generic_decomposition_preserves_each_paired_observation():
    q = np.random.default_rng(11).integers(0, 2, size=(4, 3, 12, 2)).astype(float)
    result = run.contrasts(q, [0, 1, 1, 2], 0)
    np.testing.assert_allclose(result['gain'], result['matching']+result['generic'])
    # A type-independent action can improve generally without matching benefit.
    q[:, 1] = q[:, 0]+.2
    same = run.contrasts(q, [1]*4, 0)
    np.testing.assert_allclose(same['matching'], 0)
    np.testing.assert_allclose(same['generic'], .2)


def test_different_opponent_strength_is_not_identity_decision_value():
    q = np.array([.0, .2, .4, .6])[:, None]+np.array([.1, .2, .3])[None]
    assert run.oracle_value(q) == pytest.approx(0)
    np.testing.assert_allclose(run.additive(q), q)
    np.testing.assert_allclose(q.argmax(1), [2]*4)


def test_crossed_preferences_have_real_matching_value():
    q = np.array([[.9, .1], [.1, .9], [.8, .2], [.2, .8]])
    assert run.oracle_value(q) == pytest.approx(.35)
    assert np.square(q-run.additive(q)).mean() > 0
    expanded = np.repeat(q[:, :, None, None], 24, axis=2)
    differences = run.contrasts(expanded, [0, 1, 0, 1], 0)
    assert differences['gain'].mean() == pytest.approx(.35)
    assert differences['matching'].mean() == pytest.approx(.35)


def test_shrinkage_zero_keeps_blind_choice_despite_noisy_type_optima():
    q = np.zeros((4, 3, 16, 2))
    for z in range(4): q[z, z%2, :, 0] = 1
    q[:, 2, :, 0] = .6
    p = run.policies(q)
    assert p['shrink-0'] == [2]*4
    assert p['shrink-1'] == [0, 1, 0, 1]


def test_simultaneous_value_bound_includes_deterministic_oracle_gain():
    q = np.zeros((4, 2, 2048, 2))
    q[:2, 0, :, 0] = 1; q[2:, 1, :, 0] = 1
    bound = run.value_bound(q, 0, comparisons=4)
    assert .5 <= bound['upper'] < .53
    np.testing.assert_array_equal(np.asarray(bound['simultaneous_upper_gaps'])[:, 0], 0)


def test_all_new_seeds_are_disjoint_from_previous_and_reserved_worlds():
    result = run.seed_validation()
    assert result['disjoint'] and result['new_unique'] == 6*(1024+2048)


def test_evaluation_refuses_modified_choices_before_rollout(tmp_path, monkeypatch):
    monkeypatch.setattr(run.previous, 'check_choices', lambda p: 6)
    monkeypatch.setattr(run, 'sources', lambda: {'frozen': 'identity'})
    (tmp_path/'protocol.json').write_text(json.dumps({'sources': run.sources(), 'old_artifacts': {}}))
    file = tmp_path/'choices.json'; file.write_text('{"actions":[0,0,0,0]}')
    (tmp_path/'estimation-complete.json').write_text(json.dumps({'choices': {'choices.json': run.BASE.sha(file)}}))
    run.check(tmp_path, True)
    file.write_text('{"actions":[1,1,1,1]}')
    monkeypatch.setattr(run, 'dispatch', lambda *args: pytest.fail('changed choices must stop evaluation'))
    with pytest.raises(ValueError, match='choice seal changed'): run.evaluate(tmp_path)
