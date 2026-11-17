import numpy as np
import torch

from bench.filtered_identity_oracle import (
    aggregate, cross_metrics, encode_world, identity_gain, predict_worlds,
    probability_summary, synthetic_pair,
)


def test_weighted_brier_identity_gain_matches_expected_forecast_loss():
    p = np.array([[.7,.2,.1], [.1,.3,.6]])
    weights = np.array([.25,.75]); blind = weights@p
    loss_blind = sum(w*(1+blind@blind-2*blind@q) for w,q in zip(weights,p))
    loss_known = sum(w*(1-q@q) for w,q in zip(weights,p))
    np.testing.assert_allclose(identity_gain(p, weights), loss_blind-loss_known)
    s = probability_summary(p, weights)
    np.testing.assert_allclose(s['pair_top1_disagreement'], .375)


def test_hidden_hand_average_precedes_identity_comparison():
    # Opponents disagree completely at each known hand, but agree after the
    # observer marginalizes a fair hidden hand. Per-world gain would leak it.
    worlds = np.array([[[1.,0.],[0.,1.]], [[0.,1.],[1.,0.]]])
    w = np.array([.5,.5])
    assert np.mean([identity_gain(p,w) for p in worlds]) == .5
    assert identity_gain(worlds.mean(0), w) == 0


def test_independent_cross_moment_removes_monte_carlo_positive_bias():
    a = np.array([[.9,.1],[.1,.9]])
    b = np.array([[.1,.9],[.9,.1]])
    w = np.array([.5,.5])
    got = cross_metrics(a,b,w)
    assert got['cross_gain'] < 0 and got['split_noise'] > 0
    loss = lambda p,q: 1+np.sum(p*p,axis=-1)-2*np.sum(p*q,axis=-1)
    honest = .5*(w@(loss(w@a,b)-loss(a,b)) + w@(loss(w@b,a)-loss(b,a)))
    np.testing.assert_allclose(got['heldout_gain'], honest)
    exact = cross_metrics(a,a,w)
    np.testing.assert_allclose(exact['cross_gain'], identity_gain(a,w))
    np.testing.assert_allclose(exact['heldout_gain'], identity_gain(a,w))


def test_public_action_alignment_aggregates_duplicate_encodings():
    p = np.array([[.2,.3,.5],[.1,.1,.8]])
    ids = np.array([2,0,2])
    np.testing.assert_allclose(aggregate(p,ids,4), [[.3,0,.7,0],[.1,0,.9,0]])


def test_positive_control_is_normalized_and_identical_label_null_is_zero():
    p = np.array([.2,.3,.5])
    positive = synthetic_pair(p,np.array([0.,1.,5.]))
    np.testing.assert_allclose(positive.sum(1),1)
    assert identity_gain(positive,np.array([.5,.5])) > .01
    assert identity_gain(np.stack([p,p]),np.array([.5,.5])) == 0


def test_shared_world_forward_matches_serial_and_preserves_hidden_boundary():
    import gd
    from train.history_model import HistoryPolicyConfig, fresh_player, StreamBatch, PublicStream
    torch.set_num_threads(1)
    actor,_ = fresh_player(HistoryPolicyConfig(width=16,layers=1,heads=2,
        action_width=16,fusion_width=16,critic_width=16,critic_layers=1),seed=71)
    actor.eval().requires_grad_(False)
    before = {k:v.clone() for k,v in actor.state_dict().items()}
    engine=gd.Engine(); state=gd.MatchState(); engine.new_match(state,991)
    observer=(int(state.to_move)+3)%4
    states=[state.determinize_uniform(observer,s) for s in (71,72,73)]
    alt=states[1].determinize_uniform(observer,71)
    assert [alt.hand(i) for i in range(4)] == [states[0].hand(i) for i in range(4)]
    assert all(np.array_equal(s.observation(observer),state.observation(observer)) for s in states)
    sb=StreamBatch.from_arrays([PublicStream().arrays()],'cpu')
    worlds=[encode_world(engine,s) for s in states]
    with torch.inference_mode():
        encoded=actor.encode_batch(sb)
        batch=predict_worlds(actor,sb,encoded,worlds)
        single=predict_worlds(actor,sb,encoded,worlds,batch_size=1)
    for a,b in zip(batch,single):
        np.testing.assert_allclose(a,b,atol=1e-6,rtol=1e-5)
        np.testing.assert_allclose(a.sum(),1)
    assert all(torch.equal(v,before[k]) for k,v in actor.state_dict().items())
