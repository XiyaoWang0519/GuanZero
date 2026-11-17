"""Bounded, frozen decision-value diagnostic for the pinned filtered pool."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import gd
import numpy as np
import torch
from bench.filtered_history_collect import ROOT, SEED, load_players, save_json
from bench.history_value_tasks import (Listener, candidates_for, load_task, rollout)
from bench.qknorm_match_ablation import sha
from eval.history_policy import apply_and_observe
from train.history_inference import BatchedHistoryCache
from train.history_model import PublicStream, DecisionInputs


def generate(root: Path) -> None:
    torch.set_num_threads(1)
    dest=root/'value';dest.mkdir(exist_ok=True)
    if (dest/'roots.json').exists():return
    actor,_,_=load_players(root,indices=[])
    engine=gd.Engine();engine.auto_pass=False;records=[]
    seeds=np.random.default_rng(SEED+300).integers(1,2**63-1,size=64)
    for source,seed in enumerate(seeds):
        state=gd.MatchState();engine.new_match(state,int(seed))
        stream=PublicStream();remaining=np.full(4,27);cache=BatchedHistoryCache(actor)
        for step in range(512):
            if state.phase==gd.Phase.RoundEnd:break
            legal=engine.legal_actions(state);seat=int(state.to_move)
            with torch.inference_mode():
                metadata,memory=cache.encode([(0,0)],[stream])
                inputs=DecisionInputs(metadata,torch.zeros(1,dtype=torch.long),metadata.lengths,
                    torch.tensor(np.asarray(state.observation(seat))[None]),torch.tensor([seat]),
                    torch.tensor(np.asarray([engine.encode_action(a,state,seat) for a in legal])),
                    torch.tensor([0,len(legal)]),one_decision_per_stream=True)
                lp=actor._log_prob_table(inputs,memory,len(legal))[0].double().numpy()
                prob=np.exp(lp);prob/=prob.sum()
            eligible=(seat==0 and 4<=remaining[0]<=14 and min(remaining[1],remaining[3])<=8
                and (remaining>0).all() and 16<=remaining.sum()<=56 and len(legal)>=3)
            if eligible:
                # Public/own-hand semantic alternatives, not hidden-payoff screening.
                cand=candidates_for(legal,prob,prob[None],cap=4)
                arrays=tuple(a.copy() for a in stream.arrays())
                folder=dest/f'root-{len(records):02d}';folder.mkdir(exist_ok=False)
                (folder/'simulator-state.bin').write_bytes(state.serialize())
                np.savez_compressed(folder/'public-inputs.npz',tokens=arrays[0],rounds=arrays[1],phases=arrays[2],
                    candidates=cand,actor_prob=prob,observation=state.observation(0))
                records.append(dict(index=len(records),source=source,seed=int(seed),step=step,
                    remaining=remaining.tolist(),candidates=cand.tolist(),native_choice=int(prob.argmax()),
                    kinds=[str(legal[a].type) for a in cand],public_events=stream.prefix))
                break
            event=apply_and_observe(engine,state,legal[int(prob.argmax())],[Listener(stream)])
            remaining[event.seat]=event.cards_left
        else:raise RuntimeError('neutral source round incomplete')
        if len(records)==12:break
    assert len(records)==12,'insufficient independent roots'
    save_json(dest/'roots.json',dict(roots=records,source_sha=sha(__file__),protocol_sha=sha(root/'protocol.json'),
        selection='first eligible root per independent neutral source; no payoff or opponent disagreement screen'))
    print(json.dumps({'roots':len(records),'candidate_counts':[len(r['candidates']) for r in records]}),flush=True)


_ACTOR=None
_OPPONENTS=None
_ROOT=None


def init_worker(root_string: str) -> None:
    global _ACTOR,_OPPONENTS,_ROOT
    torch.set_num_threads(1);_ROOT=Path(root_string)
    _ACTOR,_OPPONENTS,_=load_players(_ROOT)


def world_seeds(stage: str, root_index: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([SEED+400,{'select':0,'evaluate':1}[stage],root_index])).integers(
        1,2**63-1,size=32 if stage=='select' else 64)


def branch_chunk(task: tuple) -> dict:
    stage,index,start=task;folder=_ROOT/'value'/stage/f'root-{index:02d}'
    folder.mkdir(parents=True,exist_ok=True);path=folder/f'chunk-{start:02d}.npz'
    if path.exists():return dict(stage=stage,root=index,chunk=start,resumed=True)
    state,arrays,candidates=load_task(_ROOT/'value'/f'root-{index:02d}')
    seeds=world_seeds(stage,index)[start:start+4];now=time.monotonic()
    result=rollout(_ACTOR,_OPPONENTS,state,arrays,candidates,seeds)
    assert np.isfinite(result).all()
    np.savez_compressed(path,values=result,seeds=seeds,candidates=candidates)
    return dict(stage=stage,root=index,chunk=start,seconds=time.monotonic()-now,branches=int(np.prod(result.shape[:-1])))


def dispatch(root: Path,stage: str,workers: int=4) -> None:
    generate(root)
    if stage=='evaluate':
        seal=json.loads((root/'value'/'choices.json').read_text())
        assert seal['selection_sha']==selection_digest(root)
        assert seal['posterior_sha']==sha(root/'posterior.npz')
    count=32 if stage=='select' else 64
    tasks=[(stage,i,s) for i in range(12) for s in range(0,count,4)]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=init_worker,
                             initargs=(str(root),)) as pool:
        futures=[pool.submit(branch_chunk,t) for t in tasks]
        for n,f in enumerate(as_completed(futures)):
            result=f.result()
            if n%8==0 or n==len(futures)-1:print(json.dumps(dict(result,completed=n+1,total=len(futures))),flush=True)
    save_json(root/'value'/f'{stage}-complete.json',dict(chunks=len(tasks),source_sha=sha(__file__)))


def selection_digest(root: Path) -> str:
    import hashlib
    return hashlib.sha256(''.join(sha(p) for p in sorted((root/'value'/'select').glob('*/*.npz'))).encode()).hexdigest()


def read_values(root: Path,stage: str,index: int) -> np.ndarray:
    paths=sorted((root/'value'/stage/f'root-{index:02d}').glob('*.npz'))
    assert len(paths)==(8 if stage=='select' else 16)
    values=np.concatenate([np.load(p)['values'] for p in paths],axis=2)
    assert values.shape[0]==20 and np.isfinite(values).all()
    return values


def choose_actions(q: np.ndarray,prior: np.ndarray,posterior: np.ndarray) -> dict:
    """Average hidden worlds before selecting any action; never use evaluation."""
    mean=q[:,:,:,1].mean(2)
    return dict(prior=int((prior@mean).argmax()),oracle=mean.argmax(1).tolist(),
                history=(posterior@mean).argmax(1).tolist())


def seal(root: Path) -> None:
    assert not (root/'value'/'evaluate').exists(),'seal before any evaluation outcomes'
    prior=np.array(json.loads((root/'protocol.json').read_text())['identity']['weights'])
    posterior=np.load(root/'posterior.npz');p=posterior['old_raw']
    records=[]
    for i in range(12):
        q=read_values(root,'select',i);choice=choose_actions(q,prior,p)
        left=choose_actions(q[:,:,:16],prior,p);right=choose_actions(q[:,:,16:],prior,p)
        choice.update(root=i,oracle_half_agreement=float(prior@(np.array(left['oracle'])==right['oracle'])),
            prior_half_agreement=int(left['prior']==right['prior']))
        records.append(choice)
    save_json(root/'value'/'choices.json',dict(choices=records,selection_sha=selection_digest(root),
        posterior_sha=sha(root/'posterior.npz'),source_sha=sha(__file__),
        rule='maximize mean PPO return after averaging hidden worlds; evaluation worlds not read'))
    print(json.dumps({'sealed_roots':12,'oracle_half_agreement':np.mean([r['oracle_half_agreement'] for r in records])}),flush=True)


def summarize(root: Path) -> dict:
    seal=json.loads((root/'value'/'choices.json').read_text())
    assert seal['selection_sha']==selection_digest(root) and seal['posterior_sha']==sha(root/'posterior.npz')
    prior=np.array(json.loads((root/'protocol.json').read_text())['identity']['weights'])
    post=np.load(root/'posterior.npz');z=post['opponent'];rep=post['replicate']
    # Histories are independent of score roots; each type has eight heldout paired seeds.
    history_weight=prior[z]/np.bincount(z,minlength=20)[z]
    oracle=[];history=[];blind=[];native=[];selected=[];history_choices=[]
    for i,r in enumerate(seal['choices']):
        q=read_values(root,'evaluate',i);base=q[:,r['prior']]
        oracle.append(q[np.arange(20),r['oracle']]-base)
        history.append(q[z,np.array(r['history'])]-base[z])
        blind.append(np.einsum('z,zwm->wm',prior,base))
        native.append(np.einsum('z,zwm->wm',prior,q[:,0]))
        qs=read_values(root,'select',i)[:,:,:,1].mean(2)
        selected.append(float(prior@(qs[np.arange(20),r['oracle']]-qs[:,r['prior']])))
        history_choices.append(float(history_weight@(np.array(r['history'])!=r['prior'])))
    oracle=np.asarray(oracle);history=np.asarray(history)
    root_oracle=np.einsum('z,rzwm->rwm',prior,oracle)
    root_history=np.einsum('h,rhwm->rwm',history_weight,history)
    rng=np.random.default_rng(SEED+501);boots_o=[];boots_h=[]
    # Crossed bootstrap: independent roots, hidden worlds, and heldout history seed clusters.
    for _ in range(10000):
        roots=rng.integers(0,12,12);worlds=rng.integers(0,64,(12,64))
        boots_o.append(root_oracle[roots[:,None],worlds].mean((0,1)))
        hw=np.zeros(len(z))
        for zz in range(20):
            for rr in rng.integers(24,32,8):
                mask=(z==zz)&(rep==rr);hw[mask]+=prior[zz]/8/mask.sum()
        rows=np.einsum('h,rhwm->rwm',hw,history[roots])
        boots_h.append(rows[np.arange(12)[:,None],worlds].mean((0,1)))
    def stats(point,boots):
        return {name:dict(gain=float(point[j]),ci95=np.quantile(np.asarray(boots)[:,j],[.025,.975]).tolist())
                for j,name in enumerate(('win_probability','PPO_round_return'))}
    result=dict(roots=12,opponents=20,selection_worlds=32,evaluation_worlds=64,
        branches=sum(int(np.prod(read_values(root,s,i).shape[:-1])) for s in ('select','evaluate') for i in range(12)),
        oracle=stats(root_oracle.mean((0,1)),boots_o),history=stats(root_history.mean((0,1)),boots_h),
        oracle_half_agreement=float(np.mean([r['oracle_half_agreement'] for r in seal['choices']])),
        prior_half_agreement=float(np.mean([r['prior_half_agreement'] for r in seal['choices']])),
        oracle_action_change=float(np.mean([prior@(np.array(r['oracle'])!=r['prior']) for r in seal['choices']])),
        history_action_change=float(np.mean(history_choices)),selection_oracle_return_gain=float(np.mean(selected)),
        prior_win=float(np.mean(blind,axis=(0,1))[0]),prior_return=float(np.mean(blind,axis=(0,1))[1]),
        native_win=float(np.mean(native,axis=(0,1))[0]),native_return=float(np.mean(native,axis=(0,1))[1]),
        source_sha=sha(__file__),choices_sha=sha(root/'value'/'choices.json'),
        limitation='12 neutral independent roots, finite candidates, one changed action; uniform hidden redraws; fixed continuation; not optimal/full-match information value')
    save_json(root/'value-summary.json',result);print(json.dumps(result,indent=2),flush=True);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['generate','select','seal','evaluate','summarize'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.command in ('select','evaluate'):dispatch(a.root,a.command,a.workers)
    else:globals()[a.command](a.root)

if __name__=='__main__':main()
