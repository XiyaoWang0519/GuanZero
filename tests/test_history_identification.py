import math
from pathlib import Path
import numpy as np
import torch
import pytest

from bench.history_identification import decode_cards, logmean, replay, sample_initial, ResidualSampler


def test_public_card_decoder_preserves_double_copies():
    x=np.zeros(154,np.uint8);x[[2,54+2,17]]=1
    result=decode_cards(x)
    assert result[2]==2 and result[17]==1 and result.sum()==3


def test_round_likelihood_integrates_whole_trajectories_not_independent_steps():
    # Same persistent hidden world explains both steps. Re-sampling at each
    # step would assign 1/4, whereas a coherent world gives 1/2.
    logp=np.log(np.array([[.99*.99,.01*.01],[.5*.5,.5*.5]]))
    got,ess=logmean(logp,np.zeros(2))
    np.testing.assert_allclose(np.exp(got[0]-got[1]),(.99**2+.01**2)/(.5**2+.5**2))
    assert 1<=ess.min()<=ess.max()<=2


def test_duplicate_card_importance_factor_corrects_uniform_residual_proposal():
    # Small analog: card X has two copies, Y/Z one each. Each seat gets two.
    # Observed public play: A played one X, B played Y. Residual proposal
    # assigns remaining X/Z either way equally, but deal probabilities are 1:2.
    residuals=[np.array([[1,0,0],[0,0,1]]),np.array([[0,0,1],[1,0,0]])]
    played=np.array([[1,0,0],[0,1,0]])
    weights=[math.exp(math.log(2)*((r==2).sum()-((r+played)==2).sum())) for r in residuals]
    np.testing.assert_allclose(weights,[.5,1.])


def test_conditional_residual_sampler_keeps_hypergeometric_multiplicity():
    deck=np.repeat(np.arange(54),2).reshape(4,27)
    played=np.stack([np.bincount(h,minlength=54) for h in deck])
    played[0,0]-=2;played[1,14]-=2
    meta=dict(played=played,left=[2,2,0,0],transfers=[],level=0,start=0,end=0)
    sampler=ResidualSampler(meta,(np.zeros((0,186)),np.zeros(0),np.zeros(0)))
    rng=np.random.default_rng(71);counts=np.zeros(3)
    for _ in range(2000):
        hand,_=sampler.draw(rng);counts[int(hand[0,0])]+=1
        assert np.all(hand.sum(1)==27) and np.all(hand.sum(0)==2)
    np.testing.assert_allclose(counts/2000,[1/6,4/6,1/6],atol=.035)


def test_reconstructed_public_round_matches_private_audit_without_receiving_it():
    folder=Path('.work/history-identification-smoke2-2026-11-15/histories/z00-r000')
    if not (folder/'public.pt').exists():pytest.skip('local frozen smoke artifact absent')
    public=torch.load(folder/'public.pt',weights_only=False)
    audit=torch.load(folder/'audit-private.pt',weights_only=False)
    rng=np.random.default_rng(991);checked=0
    for meta,truth in zip(public['rounds'],audit):
        if not (sum(c>0 for c in meta['left'])==1 or meta['left'][1]==meta['left'][3]==0):continue
        accepted=[]
        for _ in range(10000):
            initial,_=sample_initial(meta,rng)
            if initial is None:continue
            frames=replay(meta,public['segments'][meta['segment']],initial)
            if frames is None:continue
            assert [f['prefix'] for f in frames]==[p for p,_ in truth['decision_observations']]
            for f,(_,obs) in zip(frames,truth['decision_observations']):
                np.testing.assert_array_equal(f['obs'],obs)
            accepted.append(frames)
            if len(accepted)==2:break
        assert len(accepted)==2
        checked+=1
    assert checked>0


def test_question_transfer_score_equals_explicit_probability_brier():
    from bench.history_identification_score import gain_matrix
    a=np.array([[.8,.2],[.3,.7]]);b=np.array([[.7,.3],[.4,.6]])
    c=((a@b.T+b@a.T)/2)[None];s=((a@a.T+b@b.T)/2)[None]
    prior=np.array([.4,.6]);q=np.array([[.9,.1],[.2,.8]]);z=np.array([0,1])
    result=gain_matrix(q,z,prior,c,s)[:,0]
    loss=lambda p,t:1+p@p-2*p@t
    expected=[]
    for p,target in zip(q,z):
        expected.append(.5*(loss(prior@a,b[target])+loss(prior@b,a[target])-
                            loss(p@a,b[target])-loss(p@b,a[target])))
    np.testing.assert_allclose(result,expected,atol=1e-15)


def test_sequence_probe_does_not_mix_histories_and_has_live_gradients():
    from bench.history_identification_probes import Probe
    torch.set_num_threads(1);prior=torch.full((20,),.05)
    model=Probe(7,prior,False,71)
    x=torch.randn(6,7);groups=torch.tensor([0,1,4,40,41,44])
    out=model(x,groups,2)
    torch.testing.assert_close(out.softmax(-1),prior.expand(2,-1))
    # Open the zero-initialized output gate, then check the token path.
    with torch.no_grad():model.output.weight.normal_(0,.1)
    first=model(x[:3],groups[:3],1)
    together=model(x,groups,2)
    torch.testing.assert_close(first,together[:1],atol=1e-6,rtol=1e-5)
    torch.nn.functional.cross_entropy(together,torch.tensor([1,2])).backward()
    assert model.token[0].weight.grad.norm()>0


def test_public_likelihood_positive_control_and_identical_policy_null():
    from bench.history_identification_score import posterior
    prior=np.array([.5,.5]);prob=np.array([.2,.8])
    # Ten public actions from each planted style, without hidden-state inputs.
    successes=np.array([2,8])
    loglik=successes[:,None]*np.log(prob)+(10-successes[:,None])*np.log(1-prob)
    result=posterior(loglik,prior)
    assert (result.argmax(1)==np.array([0,1])).all()
    assert np.diag(result).min()>.99
    null=posterior(np.repeat(loglik[:,:1],2,axis=1),prior)
    np.testing.assert_allclose(null,np.tile(prior,(2,1)))


def test_public_action_score_feature_equals_last_layer_loglik_gradient():
    torch.manual_seed(71)
    h=torch.randn(7,5,dtype=torch.float64);weight=torch.randn(5,dtype=torch.float64,requires_grad=True)
    prob=(h@weight).softmax(0);matches=torch.tensor([1,4])
    prob[matches].sum().log().backward()
    observed=prob[matches].detach();observed/=observed.sum()
    expected=observed@h[matches]-prob.detach()@h
    torch.testing.assert_close(weight.grad,expected,atol=1e-14,rtol=1e-12)


def test_opportunity_ridge_never_uses_test_rows_to_select_or_scale():
    from bench.history_opportunity_probe import ridge
    rng=np.random.default_rng(71);x=rng.normal(size=(100,7));y=np.arange(100)%20
    split=np.array([0]*60+[1]*20+[2]*20);prior=np.full(20,.05)
    p,fit=ridge(x,y,split,prior)
    changed=x.copy();changed[split==2]*=1000
    changed_labels=y.copy();changed_labels[split==2]=np.roll(y[split==2],5)
    _,other=ridge(changed,changed_labels,split,prior)
    assert fit==other
    assert np.isfinite(p).all() and (p>=0).all()
    np.testing.assert_allclose(p.sum(1),1)
