"""Midgame public information boundaries and discovery/confirmation controls."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import gd

from bench.history_value_tasks import (Listener,candidates_for,rollout)
from bench import history_value_curriculum as run
from eval.history_policy import apply_and_observe
from train.history_model import PublicStream,HistoryPolicyConfig,fresh_player


def fixture():
    engine=gd.Engine();engine.auto_pass=False
    for seed in range(33,45):
        state=gd.MatchState();engine.new_match(state,seed);stream=PublicStream()
        for step in range(256):
            if state.phase==gd.Phase.RoundEnd:break
            legal=engine.legal_actions(state);counts=[len(state.hand(s)) for s in range(4)]
            if state.to_move==0 and len(legal)>=3 and min(counts)>0 and counts[0]<=14 and sum(counts)<=56:
                return state,tuple(x.copy() for x in stream.arrays()),np.array([0,1])
            apply_and_observe(engine,state,legal[engine.greedy(state)],[Listener(stream)])
    raise AssertionError('no fixture')


def small_actor():
    torch.set_num_threads(1)
    actor,_=fresh_player(HistoryPolicyConfig(width=16,layers=1,heads=2,action_width=16,
        fusion_width=16,critic_width=16,critic_layers=1),seed=113)
    return actor.eval().requires_grad_(False)


def test_cached_midgame_rollouts_match_full_recomputation_and_keep_weights():
    actor=small_actor();state,arrays,candidates=fixture();before={k:v.clone() for k,v in actor.state_dict().items()}
    cached=rollout(actor,{1:actor},state,arrays,candidates,np.array([73,74]))
    fresh=rollout(actor,{1:actor},state,arrays,candidates,np.array([73,74]),use_cache=False)
    np.testing.assert_array_equal(cached,fresh)
    assert cached.shape==(1,2,2,2)
    assert np.isin(cached[:,:,:,0],[0,1]).all()
    for k,v in actor.state_dict().items():assert torch.equal(v,before[k])


def test_midgame_choices_and_worlds_do_not_depend_on_original_private_truth():
    actor=small_actor();state,arrays,candidates=fixture();alternate=state.determinize_uniform(0,991)
    q1=rollout(actor,{1:actor},state,arrays,candidates,np.array([17]))
    q2=rollout(actor,{1:actor},alternate,arrays,candidates,np.array([17]))
    np.testing.assert_array_equal(q1,q2)
    for seed in (1,2):
        a=state.determinize_uniform(0,seed);b=alternate.determinize_uniform(0,seed)
        assert [a.hand(s) for s in range(4)]==[b.hand(s) for s in range(4)]
        np.testing.assert_array_equal(a.observation(0),state.observation(0))


def test_candidate_set_preserves_pass_bomb_and_actor_choice():
    legal=[SimpleNamespace(is_pass=False,type='Single',key=7,cards=[1]),
        SimpleNamespace(is_pass=True,type='Pass',key=0,cards=[]),
        SimpleNamespace(is_pass=False,type='Bomb',key=9,cards=[2]*4),
        SimpleNamespace(is_pass=False,type='Pair',key=4,cards=[3]*2)]
    p=np.array([.7,.1,.05,.15]);roster=np.tile(p,(4,1))
    assert set(candidates_for(legal,p,roster))=={0,1,2,3}


def test_crossfit_screen_rejects_selection_only_maximum():
    a=np.zeros((4,2,16,2));b=np.zeros_like(a)
    for z in range(4):a[z,z%2,:,0]=1;b[z,1-z%2,:,0]=1
    assert run.screening_score(a,b)['rank_score']<0


def test_history_cannot_start_when_identity_confirmation_failed(tmp_path,monkeypatch):
    import json
    monkeypatch.setattr(run,'check_choices',lambda p:6)
    monkeypatch.setattr(run,'dispatch',lambda *args:pytest.fail('history rollout must not start'))
    (tmp_path/'confirm-summary.json').write_text(json.dumps(dict(gate_passed=False)))
    with pytest.raises(ValueError,match='identity gate failed'):run.evaluate(tmp_path,'history')


def test_all_reserved_new_world_seeds_are_unique():
    parts=[run.seeds(stage,i) for stage in run.STAGES for i in range(32 if stage.startswith('screen') else 6)]
    joined=np.concatenate(parts)
    assert len(np.unique(joined))==len(joined)==10240
