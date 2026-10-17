import numpy as np
from bench.qknorm_fusion_isolation import normalization_parts, features_for_view, ridge
from bench.qknorm_fusion_geometry import geometry
from bench.qknorm_history_stages import ridge_predict


def test_centering_removes_only_row_mean_and_full_inverse_recovers_input():
    rng=np.random.default_rng(81);x=rng.normal(size=(8,128))+rng.normal(size=(8,1))*5
    parts=normalization_parts(x,1e-5)
    np.testing.assert_allclose(parts['unit']*parts['scale']+parts['mean'],x,atol=1e-14)
    np.testing.assert_allclose(normalization_parts(x+7,1e-5)['unit'],parts['unit'],atol=1e-14)
    np.testing.assert_allclose(parts['centered'].mean(1),0,atol=1e-14)


def test_query_subtraction_recovers_reader_without_using_identity():
    rng=np.random.default_rng(82);q=rng.normal(size=(20,128))*30;r=rng.normal(size=q.shape)
    out=features_for_view(q,r,np.ones(128),np.zeros(128),1e-5)
    np.testing.assert_allclose(out['inverse_then_subtract'],r,atol=1e-13)


def test_layernorm_geometry_removes_mean_and_radial_but_preserves_tangent():
    rng=np.random.default_rng(83);x=rng.normal(size=128);center=x-x.mean()
    mean=geometry(x,np.ones(128)*.0001,1e-5)
    assert mean['mean_fraction']>.999999 and mean['finite_rescaled_energy_fraction']<1e-15
    radial=geometry(x,center*.0001,1e-5)
    assert radial['radial_fraction']>.999999 and radial['finite_rescaled_energy_fraction']<1e-8
    delta=rng.normal(size=128);delta-=delta.mean();delta-=center*np.dot(delta,center)/np.dot(center,center)
    tangent=geometry(x,delta*1e-5,1e-5)
    assert tangent['tangent_fraction']>.999999
    assert abs(tangent['finite_rescaled_energy_fraction']-1)<1e-4


def test_new_floor_control_matches_existing_closed_form_readout():
    rng=np.random.default_rng(84);x=rng.normal(size=(32,12));y=np.eye(4)[np.arange(32)%4];test=rng.normal(size=(10,12))
    np.testing.assert_array_equal(ridge(x,y,test),ridge_predict(x,y,test))
