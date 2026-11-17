"""Controls that matter to the isolated reader causal comparison."""
import numpy as np
import pytest
import torch

from bench.history_answer_probe import Reader
from bench.history_reader_mechanism import (
    DiagnosticHead, SlotHead, SlotReader, target_mask, padded_batch,
)
from train.history_model import HistoryPolicyConfig, fresh_player
from bench.history_reader_mechanism_score import bootstrap


def data_and_actor():
    torch.set_num_threads(1)
    torch.manual_seed(71)
    actor,_=fresh_player(HistoryPolicyConfig(width=128,layers=1,heads=8),72)
    data=dict(memory=[torch.randn(6,128) for _ in range(5)],
              masks=[torch.tensor([[0,1,1,0,0,0],[0,0,0,1,1,0]],dtype=torch.bool) for _ in range(5)],
              query=torch.randn(5,128),seats=torch.tensor([0,1,2,3,0]),
              split=torch.tensor([0,0,0,0,2]),sample_weight=torch.ones(5))
    return data,actor.eval().requires_grad_(False)


def test_leaky_head_keeps_gradient_when_every_relu_unit_would_be_off():
    relu=DiagnosticHead(3,2,17,False);leaky=DiagnosticHead(3,2,17,True)
    for a,b in zip(relu.parameters(),leaky.parameters()):torch.testing.assert_close(a,b)
    with torch.no_grad():
        for head in (relu,leaky):
            head.first.weight.fill_(1);head.first.bias.fill_(-10);head.last.weight.fill_(1)
    for head,alive in ((relu,False),(leaky,True)):
        x=torch.ones(2,3,requires_grad=True);head(x).sum().backward()
        assert bool(x.grad.norm()>0)==alive


def test_address_mask_uses_no_type_answer_or_current_or_bos():
    tokens=np.zeros((5,186),np.uint8);tokens[np.arange(5),[1,3,1,1,0]]=1
    arrays=(tokens,np.array([2,2,3,2,2]),np.array([3,3,3,1,3]))
    actual=target_mask(arrays,0)
    torch.testing.assert_close(actual,torch.tensor([[0,1,0,0,0,0],[0,0,1,0,0,0]],dtype=torch.bool))
    tokens[:,112:125]=np.random.default_rng(1).integers(0,2,(5,13))
    torch.testing.assert_close(target_mask(arrays,0),actual)


def test_repeated_actual_slots_match_single_query_attention():
    data,actor=data_and_actor();slots=SlotReader(actor,data,'actual_all',71)
    single=Reader(actor,True);args=padded_batch(data,torch.arange(4))
    with torch.no_grad():
        actual=slots(*args);expected=single(*args[:3])
        torch.testing.assert_close(actual,expected[:,None].expand_as(actual),atol=2e-6,rtol=2e-6)
    data['query'][4].fill_(1e6)
    changed=SlotReader(actor,data,'mean_all',71)
    torch.testing.assert_close(slots.query_mean,changed.query_mean)
    torch.testing.assert_close(slots.query_scale.mean,changed.query_scale.mean)


@pytest.mark.parametrize('arm',['actual_target','task_target','late_mean','early_mean','early_max'])
def test_target_readout_excludes_other_seats_rounds_and_padding(arm):
    data,actor=data_and_actor();reader=SlotReader(actor,data,arm,71)
    args=list(padded_batch(data,torch.arange(4)))
    with torch.no_grad():
        expected=reader(*args);args[1][:,0]=1e5;args[1][:,5]=-1e5
        torch.testing.assert_close(reader(*args),expected,atol=0,rtol=0)


def test_pooling_order_has_matched_parameters_but_preserves_different_information():
    data,actor=data_and_actor();late=SlotReader(actor,data,'late_mean',71)
    early=SlotReader(actor,data,'early_mean',71)
    for a,b in zip(late.parameters(),early.parameters()):torch.testing.assert_close(a,b)
    args=list(padded_batch(data,torch.arange(4)))
    with torch.no_grad():
        # Opposite token perturbations leave the pre-MLP mean unchanged.
        before_late=late(*args);before_early=early(*args)
        args[1][:,1,0]+=100;args[1][:,2,0]-=100
        torch.testing.assert_close(late(*args),before_late,atol=2e-6,rtol=2e-6)
        assert (early(*args)-before_early).abs().max()>.01


def test_recalibrated_head_retains_gradient_after_large_reader_translation():
    torch.manual_seed(2);head=SlotHead(71);x=torch.randn(20,12,128)+1000
    head.eval();head.calibrate(x)
    torch.testing.assert_close(head.norm.running_mean,x.flatten(0,1).mean(0))
    x.requires_grad_();head(x).square().mean().backward()
    assert torch.isfinite(x.grad).all() and (x.grad.abs().sum(-1)>0).all()


def test_fresh_bootstrap_uses_actual_replicates_and_pairs():
    meta=[dict(opponent=z,replicate=r,team=t,seed=z*100+r)
          for z in range(20) for r in range(32,40) for t in (0,1)]
    y=np.array([[r['team'],1-r['team']] for r in meta],float)
    probs=np.stack([y]*3);draw=np.random.default_rng(7).integers(0,8,(20,20,8))
    np.testing.assert_array_equal(bootstrap(probs,y,meta,np.full(20,.05),draw),np.ones(20))
    meta[1]['seed']=-1
    with pytest.raises(AssertionError):bootstrap(probs,y,meta,np.full(20,.05),draw)
