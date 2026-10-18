"""Larger budget, pre-evaluation seals and local process reproducibility."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import numpy as np
import pytest
import torch

from bench import history_decision_precision as run
from bench.history_decision_value import root_state,rollout_matrix


def _small_cpu_rollout(seed):
    from train.history_model import HistoryPolicyConfig,fresh_player
    torch.set_num_threads(1)
    actor,_=fresh_player(HistoryPolicyConfig(width=16,layers=1,heads=2,action_width=16,
        fusion_width=16,critic_width=16,critic_layers=1),seed=113)
    actor.eval().requires_grad_(False)
    q,_=rollout_matrix(actor,{1:actor},root_state(1001),np.array([0,1]),np.array([seed]))
    return q.tolist()


def test_spawned_cpu_workers_reproduce_serial_rollouts():
    seeds=[191,192]
    serial=[_small_cpu_rollout(seed) for seed in seeds]
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
        parallel=list(pool.map(_small_cpu_rollout,seeds))
    np.testing.assert_array_equal(serial,parallel)


def test_larger_world_seeds_are_disjoint_from_both_previous_runs():
    record=run.check_seed_disjointness()
    assert record['disjoint']
    assert record['new_unique_seeds']==8*(512+512+1024)


def test_larger_nested_stability_checks_the_full_512_world_replicas():
    # A 64-world implementation accidentally reused from the older runner
    # would miss the winner changing after the prefix.
    q=np.zeros((4,5,512));q[:,1,:64]=2;q[:,2,64:]=2
    rows=run.stability(q,q,np.eye(4))
    assert set(rows)=={'64','128','256','512'}
    assert rows['64']['identity']['choices_a']==[1]*4
    assert rows['512']['identity']['choices_a']==[2]*4
    summary=run.stability_summary([rows]*8)
    assert summary['descriptive_stability_gate']
    assert summary['by_replica_world_count']['512']['identity']['q_rmse']==0


def test_evaluation_refuses_a_modified_choice_seal(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'assert_identity',lambda p:None)
    (tmp_path/'protocol.json').write_text('{}')
    stability=tmp_path/'stability-summary.json';stability.write_text('{}')
    hashes={}
    for i in range(8):
        folder=tmp_path/f'root-{i:02d}';folder.mkdir()
        p=folder/'sealed-choices.json';p.write_text('{}');hashes[str(i)]=run.base.sha(p)
    (tmp_path/'estimation-complete.json').write_text(json.dumps(dict(choice_hashes=hashes,stability_sha=run.base.sha(stability))))
    run.check_seal(tmp_path)
    (tmp_path/'root-03'/'sealed-choices.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='choice seal'):run.check_seal(tmp_path)
