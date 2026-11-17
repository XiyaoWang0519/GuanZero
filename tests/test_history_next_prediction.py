import copy
import numpy as np
import pytest
import torch

from bench.history_next_prediction import (
    future_target,history_groups,group_pool,PredictorReader,CurrentModel,ResidualHead,
)
from bench.history_next_prediction_score import clusters,row_metrics


def fake_data():
    torch.set_num_threads(1);torch.manual_seed(9)
    return dict(x=torch.randn(16,21),query=torch.randn(16,128),
                split=torch.tensor([0]*8+[1]*4+[2]*4),sample_weight=torch.ones(16),
                memory_scale={'mean':torch.zeros(128),'std':torch.ones(128)},
                prevalence=torch.full((11,),1/11))


def test_future_target_stays_in_round_and_skips_own_team_and_nonplay():
    # meta: env, round, step, seat, cards left, phase
    meta=np.array([[0,3,i,s,20,p] for i,(s,p) in enumerate([(0,3),(2,3),(1,1),(3,3),(1,3)])])
    a=np.zeros((5,146));a[np.arange(5),108+np.array([1,8,11,2,5])]=1
    assert future_target(meta,a,0,3,0)==(2,3)
    assert future_target(meta,a,3,3,0)==(5,4)
    meta[4,1]=4
    assert future_target(meta,a,3,3,0) is None


def test_grouping_uses_only_public_seat_and_completed_round():
    t=np.zeros((4,186));t[np.arange(4),[0,1,2,3]]=1
    arrays=(t,np.array([0,1,2,2]),np.array([3,1,2,3]))
    torch.testing.assert_close(history_groups(arrays,1),torch.tensor([12,11,4,1,2]))
    t[:,112:125]=1
    torch.testing.assert_close(history_groups(arrays,1),torch.tensor([12,11,4,1,2]))
    t[0,150]=1
    with pytest.raises(ValueError,match='private'):history_groups(arrays,1)
    t[0,150]=0;arrays[1][0]=3
    with pytest.raises(ValueError,match='completed'):history_groups(arrays,1)


def test_group_mean_and_max_exclude_padding_and_handle_empty_groups():
    x=torch.tensor([[[2.,-1.],[4.,3.],[999.,999.]]],requires_grad=True)
    groups=torch.tensor([[0,0,13]]);valid=groups!=13
    mean,maximum,count=group_pool(x,groups,valid)
    torch.testing.assert_close(mean[0,0],torch.tensor([3.,1.]))
    torch.testing.assert_close(maximum[0,0],torch.tensor([4.,3.]))
    assert count[0,0]==2 and torch.isfinite(maximum).all()
    assert (mean[0,1:]==0).all() and (maximum[0,1:]==0).all()
    (mean.sum()+maximum.sum()).backward()
    assert (x.grad[0,-1]==0).all() and torch.isfinite(x.grad).all()


@pytest.mark.parametrize('arm',['original_small','original_large','improved_small','improved_large'])
def test_readers_ignore_padding_and_heldout_values_cannot_change_scaling(arm):
    data=fake_data();reader=PredictorReader(data,arm,71)
    memory=torch.randn(2,7,128);groups=torch.tensor([[12,0,1,4,5,13,13],[12,0,1,4,5,8,9]])
    valid=groups!=13;query=data['query'][:2]
    expected=reader(query,memory,valid,groups)
    changed=memory.clone();changed[0,5:]=1e5
    torch.testing.assert_close(reader(query,changed,valid,groups),expected,atol=0,rtol=0)
    data['query'][12:]=1e7
    other=PredictorReader(data,arm,71)
    torch.testing.assert_close(reader.query_scale.mean,other.query_scale.mean)
    torch.testing.assert_close(reader.query_scale.std,other.query_scale.std)
    expected.square().mean().backward()
    assert reader.q_proj.weight.grad.norm()>0


def test_reader_parameter_budgets_match_across_structures():
    data=fake_data()
    count={a:sum(p.numel() for p in PredictorReader(data,a,71).parameters()) for a in
           ['original_small','original_large','improved_small','improved_large']}
    assert count==dict(original_small=67840,original_large=133760,improved_small=67872,improved_large=133760)


def test_zero_residual_preserves_current_baseline_for_any_history():
    data=fake_data();base=CurrentModel(data,71);context=base.encode(data['x']).detach()
    head=ResidualHead(context[:8],torch.ones(8),71).eval()
    for value in (torch.randn(16,128),torch.ones(16,128)*1000):
        torch.testing.assert_close(head(context,value),torch.zeros(16,11),atol=0,rtol=0)
    original=copy.deepcopy(base.state_dict());base.requires_grad_(False)
    head.train();opt=torch.optim.Adam(head.parameters(),lr=.001)
    logits=base(data['x'])+head(context,torch.randn(16,128))
    torch.nn.functional.cross_entropy(logits,torch.arange(16)%11).backward();opt.step()
    assert all(torch.equal(v,original[k]) for k,v in base.state_dict().items())


def test_cluster_scores_do_not_overweight_matches_with_more_captures():
    meta=[];values=[]
    for z in range(20):
        for r in range(40,48):
            for t in (0,1):
                for threshold in (0,20,50):
                    if z==0 and r==40 and t==0 and threshold==50:continue
                    meta.append(dict(opponent=z,replicate=r,team=t,seed=z*100+r,threshold=threshold))
                    values.append(t+threshold/100)
    actual=clusters(np.array(values),meta)
    assert actual[0,0]==pytest.approx((.1+(1+1.2+1.5)/3)/2)
    assert actual[0,1]==pytest.approx(.5+(.0+.2+.5)/3)


def test_scoring_uses_all_eleven_types_and_coarse_groups_consistently():
    y=np.arange(11);p=np.stack([np.eye(11)]*3)
    scores=row_metrics(p,y)
    for key in ('brier','coarse_brier','log_loss'):np.testing.assert_array_equal(scores[key],np.zeros((3,11)))
    for key in ('accuracy','coarse_accuracy'):np.testing.assert_array_equal(scores[key],np.ones((3,11)))
