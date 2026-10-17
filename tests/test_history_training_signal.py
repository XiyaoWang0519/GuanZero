from types import SimpleNamespace
import numpy as np
import torch
from bench.history_training_signal import policy_surrogate, visible_rounds
from train.history_rollout import MatchEventStore,SequenceRolloutBuffer,HIDDEN_DIM,compute_gae


def test_training_batch_keeps_previous_rounds_but_excludes_future_suffix():
    store=MatchEventStore();stream=store.stream(0,0);token=np.zeros(186,np.uint8);token[0]=token[158]=1
    for k in [0,0,0,0,1,1]:stream.append_token(token,k,3)
    b=SequenceRolloutBuffer()
    def add(k,prefix):
        b.add_step(keep=np.array([True]),env_id=np.array([0]),match_id=np.array([0]),
            round_index=np.array([k]),seat=np.array([0]),phase=np.array([3]),obs=np.zeros((1,1849),np.uint8),
            hidden=np.zeros((1,HIDDEN_DIM),np.uint8),cand=np.zeros((1,154),np.uint8),offsets=np.array([0,1]),
            chosen=np.array([0]),logp=np.array([0.]),prefix=np.array([prefix]),version=1)
    add(0,2);b.finish_round(0,0,0,[1,-1,1,-1]);add(1,5)
    batch=b.training_batch(np.array([1]),store,'cpu',fields=(),length_groups=2,width=32)
    assert batch.inputs.prefix.tolist()==[5]
    assert batch.inputs.streams.rounds.tolist()==[[0,0,0,0,1]]
    assert batch.inputs.streams.lengths.tolist()==[5]


def test_round_boundary_blocks_later_reward_credit():
    values=np.zeros((1,4));rewards=np.array([[0.,1.,0.,100.]])
    done=np.array([[False,True,False,True]]);mask=np.ones((1,4),bool)
    advantages,_=compute_gae(values,rewards,done,1.,.95,mask=mask)
    assert np.isclose(advantages[0,0],.95) and advantages[0,1]==1
    assert advantages[0,3]==100


def test_ppo_gradient_reaches_selected_actions_except_when_clipped():
    cfg=SimpleNamespace(normalize_advantages=False,clip=.2,behaviour_weight_cap=1.)
    x=torch.zeros(2,requires_grad=True);old=torch.zeros(2);adv=torch.tensor([2.,-2.])
    loss=policy_surrogate(x,old,old,adv,cfg,None).mean()
    torch.testing.assert_close(torch.autograd.grad(loss,x)[0],torch.tensor([-1.,1.]))
    x=torch.tensor([np.log(1.5),np.log(.5)],requires_grad=True)
    loss=policy_surrogate(x,old,old,adv,cfg,None).mean()
    torch.testing.assert_close(torch.autograd.grad(loss,x)[0],torch.zeros_like(x))


def test_current_round_bound_never_reads_the_next_event():
    inputs=SimpleNamespace(prefix=torch.tensor([0,2]),match_index=torch.tensor([0,0]),
        streams=SimpleNamespace(rounds=torch.tensor([[0,0,9]])))
    assert visible_rounds(inputs).tolist()==[0,0]
