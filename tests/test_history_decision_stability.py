"""Replication distinguishes ranking noise, practical gaps and held-out value."""
import numpy as np
import torch

from bench.history_decision_stability import (stability_metrics,score_choices,
    world_seeds,q_uncertainty)
from bench.history_decision_value import root_state,rollout_matrix


def test_near_tied_first_choices_can_disagree_without_large_value_gap():
    a=np.array([[1.,.99,0.]])
    b=np.array([[.99,1.,0.]])
    result=stability_metrics(a,b)
    assert result['top1_agreement']==0
    assert result['within_tolerance']==1
    assert result['symmetric_cross_gap']<.011
    assert result['pairwise_order_agreement']==2/3


def test_same_top_action_does_not_prove_complete_ranking_stability():
    result=stability_metrics(np.array([[3.,2.,1.]]),np.array([[3.,1.,2.]]))
    assert result['top1_agreement']==1
    assert result['pairwise_order_agreement']==2/3


def test_paired_gap_uncertainty_cancels_shared_world_noise():
    q=np.array([[[10.,-10.,10.,-10.],[9.,-11.,9.,-11.]]])
    result=q_uncertainty(q,np.array([[1.]]))
    assert result['q_se'][0][0]>0
    assert result['paired_top_gap_se']==[0]
    assert result['top_gap']==[1]


def test_scoring_respects_sealed_choices_even_when_test_optimum_reverses():
    q=np.array([[[0.,0.],[2.,2.]],[[4.,4.],[0.,0.]]])
    choices=dict(prior=1,identity=[0,1],posterior=[1,1],actor_round=[0,0],actor_full=[0,0])
    result=score_choices(q,choices)
    assert result['gains']['identity']['mean']==-1
    assert result['gains']['posterior']['mean']==0


def test_all_replica_and_evaluation_seeds_are_disjoint():
    seeds=[world_seeds(s,r,128 if s==2 else 64) for r in range(8) for s in range(3)]
    joined=np.concatenate(seeds)
    assert len(np.unique(joined))==len(joined)


def test_world_chunking_does_not_change_sampled_rollout_returns():
    from train.history_model import HistoryPolicyConfig,fresh_player
    torch.set_num_threads(1)
    actor,_=fresh_player(HistoryPolicyConfig(width=16,layers=1,heads=2,action_width=16,
        fusion_width=16,critic_width=16,critic_layers=1),seed=83)
    actor.eval().requires_grad_(False)
    state=root_state(1001);candidates=np.array([0,1]);seeds=np.array([873,874])
    together,_=rollout_matrix(actor,{1:actor},state,candidates,seeds)
    pieces=[rollout_matrix(actor,{1:actor},state,candidates,np.array([s]))[0] for s in seeds]
    np.testing.assert_array_equal(together,np.concatenate(pieces,axis=-1))
