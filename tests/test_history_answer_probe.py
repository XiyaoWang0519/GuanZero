"""Acceptance checks for public fact labels and isolated reader learning."""
import copy
import numpy as np
import pytest
import torch
from torch import nn
from bench.history_answer_probe import past_facts, Reader, shifted_history
from bench.history_answer_score import metrics,aggregate_types,bootstrap_scores
from train.history_model import HistoryPolicyConfig,fresh_player


def token(seat,kind):
    t=np.zeros(186,np.uint8);t[seat]=1;t[112+kind]=1;t[158+10]=1;return t


def test_fact_labels_use_previous_round_public_play_and_relative_opponents():
    # Own bomb, previous-previous bomb, current bomb and tribute bomb must not count.
    tokens=np.array([token(0,8),token(1,8),token(1,8),token(3,8),token(1,2),token(3,5)])
    rounds=np.array([2,1,3,2,2,2]);phases=np.array([3,3,3,1,3,3])
    y,counts=past_facts((tokens,rounds,phases),3,0)
    np.testing.assert_array_equal(y,[0,1,0,0,0,0,0,0,0,1,0,0])
    assert counts[2]==1 and counts[13+5]==1 and counts[8]==0 and counts[13+8]==0
    tokens[:,0:4]=np.roll(tokens[:,0:4],1,axis=1)
    rotated,_=past_facts((tokens,rounds,phases),3,1)
    np.testing.assert_array_equal(y,rotated)
    tokens[0,150]=1
    with pytest.raises(ValueError,match='private'):past_facts((tokens,rounds,phases),3,1)


def small_actor():
    torch.set_num_threads(1)
    a,_=fresh_player(HistoryPolicyConfig(width=16,layers=1,heads=2),77)
    return a.eval().requires_grad_(False)


@pytest.mark.parametrize('normalized',[False,True])
def test_reader_matches_owning_attention_and_excludes_padding(normalized):
    actor=small_actor();reader=Reader(actor,normalized)
    if normalized:actor.q_norm=nn.LayerNorm(8,elementwise_affine=False);actor.k_norm=nn.LayerNorm(8,elementwise_affine=False)
    torch.manual_seed(33);q=torch.randn(2,16);m=torch.randn(2,7,16);mask=torch.arange(7)[None]<torch.tensor([3,7])[:,None]
    with torch.no_grad():
        k,v=actor.project_keys_values(m);native=actor._attend(q,k,v,mask)
        actual=reader(q,m,mask);altered=m.clone();altered[0,3:]=123.
        torch.testing.assert_close(actual,native,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(reader(q,altered,mask),actual,atol=0,rtol=0)


def test_training_copy_restores_qk_gradient_without_touching_source():
    actor=small_actor()
    with torch.no_grad():actor.q_proj.weight.mul_(1e4);actor.kv_proj.weight[:16].mul_(1e4)
    before={k:v.clone() for k,v in actor.state_dict().items()}
    torch.manual_seed(17);q=torch.randn(3,16);m=torch.randn(3,5,16);mask=torch.ones(3,5,dtype=torch.bool)
    vanilla=Reader(actor,False);normal=Reader(actor,True)
    vanilla(q,m,mask).square().mean().backward();normal(q,m,mask).square().mean().backward()
    assert vanilla.q_proj.weight.grad.norm()==0
    assert normal.q_proj.weight.grad.norm()>0 and normal.kv_proj.weight.grad[:16].norm()>0
    opt=torch.optim.Adam(normal.parameters(),lr=.001);opt.step()
    for k,v in actor.state_dict().items():assert torch.equal(v,before[k])


def test_history_swap_preserves_current_events_and_rotates_donor_seats():
    receiver={'original':(np.array([token(1,2),token(3,5)]),np.array([2,3]),np.array([3,3])),'round':3,'seat':0}
    donor={'original':(np.array([token(2,8),token(1,2),token(1,5)]),np.array([1,2,3]),np.array([3,3,3])),'round':3,'seat':1}
    swapped=shifted_history(receiver,donor)
    for a,b in zip(swapped,receiver['original']):np.testing.assert_array_equal(a[-1:],b[-1:])
    assert swapped[0][0,:4].argmax()==1 and swapped[0][1,:4].argmax()==0
    assert swapped[0][0,112:125].argmax()==8 and len(swapped[0])==3


def test_balanced_accuracy_does_not_reward_majority_guessing():
    y=np.array([[1.]]*90+[[0.]]*10)
    result=metrics(np.full_like(y,.9),y,np.ones(100))
    assert result['accuracy']==.9 and result['balanced_accuracy']==.5


def test_token_decoder_aggregates_types_into_same_history_questions():
    predicted=np.array([8,0,2,5,4,1,9]);owners=np.array([7,7,7,7,9,9,9]);relative=np.array([0,0,1,1,0,1,1])
    answer=aggregate_types(predicted,owners,relative,np.array([7,9]),
        ['opp_1_bomb','opp_1_full_house','opp_3_pair','opp_3_straight','opp_3_bomb'])
    np.testing.assert_array_equal(answer,[[1,0,1,1,0],[0,1,0,0,1]])


def test_bootstrap_keeps_paired_teams_as_one_seed_cluster():
    meta=[dict(opponent=z,replicate=r,team=t) for z in range(20) for r in range(24,32) for t in (0,1)]
    y=np.array([[m['team'],1-m['team']] for m in meta],float)
    draws=np.random.default_rng(7).integers(0,8,(15,20,8));probs=np.tile(y[None],(3,1,1))
    scores=bootstrap_scores(probs,y,meta,np.full(20,.05),draws)
    np.testing.assert_array_equal(scores,np.ones(15))
    with pytest.raises(AssertionError,match='paired team'):
        bootstrap_scores(probs[:,:-1],y[:-1],meta[:-1],np.full(20,.05),draws)
