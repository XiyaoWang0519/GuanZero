import numpy as np
import torch
import gd
from bench.filtered_history_collect import DiagnosticStore, raw_profile
from bench.public_history_probe import PublicCollector
from train.history_rollout import MatchEventStore,SequenceRolloutBuffer
from train.history_model import HistoryPolicyConfig,fresh_player
from bench.filtered_history_analyze import fit_probe
from bench.filtered_history_value import choose_actions,world_seeds
import pytest


def test_public_instrumentation_keeps_rollout_and_records_batch_events():
    torch.set_num_threads(1)
    actor,_=fresh_player(HistoryPolicyConfig(width=32,layers=1,heads=4),71)
    out=[]
    for store in (MatchEventStore(),DiagnosticStore()):
        env=gd.VecEnv(num_envs=2,num_threads=1,seed=71,log_public_actions=True,log_env_limit=2)
        env.reset(match_seeds=[7001,7002])
        c=PublicCollector(env,actor,store,SequenceRolloutBuffer(),device='cpu',kv_cache=True,seeds=[7001,7002])
        c.started=True
        with torch.inference_mode():c.collect(16)
        out.append((c, [tuple(a.copy() for a in store.stream(e,0).arrays()) for e in range(2)]))
    for a,b in zip(out[0][1],out[1][1]):
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    assert len(out[1][0].tap.meta)>0
    assert len(out[1][0].tap.meta)==sum(len(a[0]) for a in out[1][1])
    for arrays in out[1][1]:
        f=raw_profile(arrays,0)
        assert f.shape==(128,) and np.isfinite(f).all()


def test_public_profile_rejects_private_exchange_flags():
    tokens=np.zeros((1,186),np.uint8);tokens[0,150]=1
    with pytest.raises(AssertionError):raw_profile((tokens,np.array([0]),np.array([3])),0)


def test_probe_selection_cannot_read_heldout_labels():
    rng=np.random.default_rng(55);z=np.tile(np.arange(20),6)
    split=np.repeat([0,1,2],40);x=rng.normal(size=(120,6));y=np.eye(3)[rng.integers(0,3,120)]
    p,meta=fit_probe(x,y,split,z,np.full(20,.05))
    alternate=y.copy();alternate[split==2]=alternate[split==2,::-1]
    q,other=fit_probe(x,alternate,split,z,np.full(20,.05))
    np.testing.assert_array_equal(p,q);assert meta==other


def test_value_choices_average_worlds_before_maximizing():
    q=np.zeros((2,2,2,2))
    q[0,0,:,1]=[3,-3];q[0,1,:,1]=[1,1]
    q[1,0,:,1]=[2,2];q[1,1,:,1]=[-1,-1]
    choice=choose_actions(q,np.array([.5,.5]),np.array([[1.,0.],[0.,1.]]))
    assert choice==dict(prior=0,oracle=[1,0],history=[1,0])


def test_value_selection_and_evaluation_worlds_are_disjoint():
    seeds=np.concatenate([world_seeds(s,i) for s in ('select','evaluate') for i in range(12)])
    assert len(seeds)==len(np.unique(seeds))==1152
