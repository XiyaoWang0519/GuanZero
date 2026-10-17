import numpy as np
import pytest
from bench.qknorm_history_stages import public_profile, ridge_predict, seed_permutation, brier_probabilities


def test_closed_form_readout_generalizes_without_test_labels():
    rng=np.random.default_rng(7)
    classes=np.repeat(np.arange(4),24)
    centers=np.eye(4)*4
    x=centers[classes]+rng.normal(0,.1,(len(classes),4))
    target=np.eye(4)[classes]
    test=centers+np.array([.02,-.02,.01,.03])
    p=ridge_predict(x,target,test)
    assert np.array_equal(p.argmax(1),np.arange(4))
    assert (p>=0).all() and np.allclose(p.sum(1),1)
    # An uninformative control cannot exploit the unrelated test examples.
    null=ridge_predict(np.ones_like(x),target,test)
    np.testing.assert_allclose(null,np.full((4,4),.25))


def test_seed_shuffle_preserves_both_team_positions_and_has_no_original_seed():
    seeds=np.repeat(np.arange(20),2)
    permutation=seed_permutation(seeds,71)
    assert (seeds[permutation]!=seeds).all()
    assert np.array_equal(permutation%2,np.arange(40)%2)
    assert len(set(permutation))==40
    assert np.array_equal(permutation,seed_permutation(seeds,71))


def test_same_card_scoring_uses_only_public_answers_and_predictive_mixture():
    pi=np.array([[[1.,0,0],[0,1.,0],[0,0,1.],[.5,.5,0]]])
    frequencies=pi.copy()
    p=np.eye(4)
    scores=brier_probabilities(p,pi,frequencies,np.arange(4))
    np.testing.assert_allclose(scores[:,0],[0,0,0,.5])
    wrong=brier_probabilities(p[[1,0,2,3]],pi,frequencies,np.arange(4))
    assert wrong[0,0]==2 and wrong[1,0]==2


def test_public_profile_omits_own_actions_and_rejects_private_flags():
    tokens=np.zeros((3,186),np.uint8)
    tokens[0,0]=1;tokens[1,1]=1;tokens[2,3]=1
    tokens[0,113]=1;tokens[1,112]=1;tokens[2,120]=1
    np.testing.assert_allclose(public_profile(tokens,0),[1.5/4,.5/4,1.5/4,.5/4])
    tokens[0,150]=1
    with pytest.raises(AssertionError): public_profile(tokens,0)
