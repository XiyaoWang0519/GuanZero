"""Information-set integration, holdout selection and real rollout boundary tests."""
import numpy as np
import pytest
import torch

from bench.history_decision_value import (choose_actions, evaluate, root_state,
    rollout_matrix, select_candidates)


def test_hidden_worlds_are_averaged_before_action_selection():
    # A per-world oracle would get 10; the legal fixed action gets at most 6.
    q = np.array([[[10.,0.],[0.,10.],[6.,6.]]])
    assert int(choose_actions(q,np.array([1.])))==2
    assert q.mean(-1).max()==6
    assert q.max(1).mean()==10


def test_evaluation_does_not_reselect_using_test_returns():
    selection = np.array([[[2.],[0.]],[[0.],[3.]]])
    test = np.array([[[0.],[2.]],[[4.],[0.]]])
    probs = np.array([[.1,.9],[.9,.1],[.1,.9]])
    result = evaluate(selection,test,np.eye(2),probs,np.array([0,1]))
    assert result['baseline_action']==1
    assert result['identity_actions']==[0,1]
    # A test-return oracle would reverse both; honest out-of-sample gain is negative.
    assert result['gains']['identity']['mean']==-1
    assert result['selection_plugin_identity_value']>0


def test_candidate_set_always_contains_all_native_greedy_actions():
    round_prob = np.arange(10,dtype=float)/45
    history_prob = np.eye(10)[[1,2,3,4]]
    choices = select_candidates(round_prob,history_prob)
    assert set(choices)=={9,1,2,3,4}
    assert len(select_candidates(np.array([.8,.2]),np.array([[.9,.1]]*4)))==2


def test_sampling_same_information_is_independent_of_original_hidden_truth():
    state = root_state(991)
    alternate = state.determinize_uniform(0,37)
    for seed in (11,12,13):
        left,right = (s.determinize_uniform(0,seed) for s in (state,alternate))
        assert [left.hand(i) for i in range(4)]==[right.hand(i) for i in range(4)]
        assert left.hand(0)==state.hand(0)
        np.testing.assert_array_equal(left.observation(0),state.observation(0))


def test_round_rollout_forces_each_root_and_finishes_without_training():
    from train.history_model import HistoryPolicyConfig, fresh_player
    torch.set_num_threads(1)
    actor,_ = fresh_player(HistoryPolicyConfig(width=16,layers=1,heads=2,
        action_width=16,fusion_width=16,critic_width=16,critic_layers=1),seed=71)
    actor.eval().requires_grad_(False)
    before = {k:v.clone() for k,v in actor.state_dict().items()}
    state = root_state(123)
    q,meta = rollout_matrix(actor,{1:actor},state,np.array([0,1]),np.array([12,13]))
    assert q.shape==(1,2,2) and np.isfinite(q).all()
    assert meta['rounds']==4 and meta['root_information_equal']
    assert np.isin(q,[-3,-2,-1,1,2,3]).all()
    for k,v in actor.state_dict().items():
        assert torch.equal(v,before[k])


def test_invalid_posterior_is_rejected():
    with pytest.raises(ValueError,match='probability'):
        choose_actions(np.zeros((2,2,3)),np.array([-.1,1.1]))
