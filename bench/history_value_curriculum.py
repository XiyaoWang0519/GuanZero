"""Bounded midgame discovery -> identity confirmation -> public-history test.

New primary metric is round team win, fixed before data generation. PPO round
return is retained as a secondary metric. No training or production mutation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import gd
import numpy as np
import torch

from bench import history_decision_precision as old
from bench import history_value_tasks as tasks

SEED=17101770
DISCOVER=32
SELECT=6
CAP=8
WORKERS=4
CHUNK=8
BASE=old.base
STAGES={'screen-a':(0,16),'screen-b':(1,16),'estimate-a':(2,256),'estimate-b':(3,256),'confirm':(4,512),'history':(5,512)}


def seeds(stage: str,index: int) -> np.ndarray:
    tag,n=STAGES[stage]
    return np.random.default_rng(np.random.SeedSequence([SEED,tag,index])).integers(1,2**63-1,size=n)


def sources() -> dict:
    import train.history_inference as inference
    import eval.history_policy as policy
    return {**old.sources(),__file__:BASE.sha(__file__),tasks.__file__:BASE.sha(tasks.__file__),
            inference.__file__:BASE.sha(inference.__file__),policy.__file__:BASE.sha(policy.__file__)}


def assert_sources(output: Path) -> None:
    p=json.loads((output/'protocol.json').read_text())
    if p['sources']!=sources():raise ValueError('source changed')
    for r in [p['endpoint'],*p['opponents'].values()]:
        if BASE.sha(r['path'])!=r['checkpoint_sha']:raise ValueError('checkpoint changed')


def check_seeds(output: Path) -> dict:
    used=set()
    for i in range(8):
        for split in ('selection','evaluation'):
            with np.load(BASE.PILOT/f'root-{i:02d}-{split}.npz') as p:used.update(map(int,p['world_seeds']))
        for root in (old.PREVIOUS,Path('.work/history-decision-precision-2026-10-17')):
            for split in ('estimate-a','estimate-b','evaluation'):
                with np.load(root/f'root-{i:02d}'/split/'all.npz') as p:used.update(map(int,p['world_seeds']))
    before=len(used)
    generation=np.random.default_rng(SEED+1).integers(1,2**63-1,size=64)
    if used.intersection(map(int,generation)):raise ValueError('generation seed collision')
    used.update(map(int,generation))
    for stage in STAGES:
        for i in range(DISCOVER if stage.startswith('screen') else SELECT):
            s=seeds(stage,i);new=set(map(int,s))
            if len(new)!=len(s) or used.intersection(new):raise ValueError('world seed collision')
            used.update(new)
    return dict(prior_unique_seeds=before,new_reserved_including_generation=len(used)-before,disjoint=True)


def select_choices(q: np.ndarray,posterior: np.ndarray) -> dict:
    win=q[:,:,:,0];nz=len(win)
    return dict(prior=int(BASE.choose_actions(win,np.full(nz,1/nz))),
        identity=BASE.choose_actions(win,np.eye(nz)).tolist(),
        posterior=BASE.choose_actions(win,posterior).tolist(),
        shuffled_posterior=BASE.choose_actions(win,np.roll(posterior,1,axis=0)).tolist())


def score(q: np.ndarray,choices: dict) -> dict:
    z=np.arange(len(q));baseline=q[:,choices['prior'],:,:]
    values={name:q[z,np.asarray(choices[name]),:,:] for name in ('identity','posterior','shuffled_posterior')}
    identity_mix=q[:,np.asarray(choices['identity']),:,:].mean(1)
    gains={name:v-baseline for name,v in values.items()}
    gains['identity_alignment']=values['identity']-identity_mix
    gains['history_vs_shuffle']=values['posterior']-values['shuffled_posterior']
    return gains


def screening_score(a: np.ndarray,b: np.ndarray) -> dict:
    post=np.full((4,4),.25)
    ca,cb=select_choices(a,post),select_choices(b,post)
    ga,gb=score(b,ca),score(a,cb)
    identity=float((ga['identity'][:,: ,0].mean()+gb['identity'][:,:,0].mean())/2)
    alignment=float((ga['identity_alignment'][:,:,0].mean()+gb['identity_alignment'][:,:,0].mean())/2)
    return dict(crossfit_identity_win=identity,crossfit_alignment_win=alignment,
                rank_score=min(identity,alignment))


def root_folder(output: Path,index: int,stage: str) -> Path:
    if stage.startswith('screen'):return output/'discovery'/f'case-{index:02d}'
    return output/'selected'/f'case-{index:02d}'


def worker(output_path: str,index: int,stages: list[str]) -> dict:
    actor,opponents=old._MODELS
    output=Path(output_path);folder=root_folder(output,index,stages[0])
    metadata=json.loads((folder/'task.json').read_text())
    state,arrays,candidates=tasks.load_task(output/'generated'/f"pool-{metadata['index']:03d}")
    with (folder/f'{stages[0]}-worker.log').open('x') as log,redirect_stdout(log):
        for stage in stages:
            target=folder/stage;target.mkdir()
            ss=seeds(stage,index);pieces=[];started=time.monotonic()
            for start in range(0,len(ss),CHUNK):
                part=ss[start:start+CHUNK]
                q=tasks.rollout(actor,opponents,state,arrays,candidates,part)
                np.savez_compressed(target/f'chunk-{start//CHUNK:03d}.npz',returns=q,world_seeds=part,candidates=candidates)
                pieces.append(q)
                print(json.dumps(dict(stage=stage,worlds_done=start+len(part),total=len(ss))),flush=True)
            q=np.concatenate(pieces,axis=2)
            np.savez_compressed(target/'all.npz',returns=q,world_seeds=ss,candidates=candidates)
            BASE.save_json(target/'complete.json',dict(rounds=int(np.prod(q.shape[:3])),seconds=time.monotonic()-started))
    return dict(case=index,stages=stages,complete=True)


def dispatch(output: Path,count: int,stages: list[str]) -> None:
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=old.init_worker) as pool:
        jobs=[pool.submit(worker,str(output),i,stages) for i in range(count)]
        for job in as_completed(jobs):print(json.dumps(job.result()),flush=True)
    BASE.save_json(output/f"{stages[0]}-timing.json",dict(wall_seconds=time.monotonic()-started))


def prepare(output: Path) -> None:
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    actor,identity=BASE.load_actor(BASE.ENDPOINT);opponents,opponent_meta=BASE.load_selected(BASE.POOL)
    protocol=dict(seed=SEED,source_round_cap=64,public_pool_cap=96,discovery_cap=DISCOVER,selected_cap=SELECT,
        candidate_cap=CAP,screen_replica_worlds=16,estimate_replica_worlds=256,confirmation_worlds=512,history_worlds=512,
        max_score_branches=327680,max_generation_rounds=64,workers=WORKERS,chunk_worlds=CHUNK,
        primary='Round winning_team == 0',secondary='PPO RoundResult.seat_return[0]',
        gate='Identity-vs-prior mean >= .02 AND paired-world 97.5% CI lower >0; identity-vs-type-independent identical action mixture 97.5% CI lower >0',
        history_gate='Only after identity gate: posterior-vs-prior AND posterior-vs-shuffled-history 97.5% CI lower >0',
        task='Public-consistent midgame hidden-hand redraw; frozen opponent type assigned at scoring node; prefix generated by type-neutral frozen control. Not natural-match conditional inference.',
        selection='Public policy-disagreement prefilter, then independent crossfit win/alignment discovery; no confirmation-data root selection',
        sources=sources(),endpoint=identity,opponents=opponent_meta,torch=torch.__version__,numpy=np.__version__)
    BASE.save_json(output/'protocol.json',protocol);BASE.save_json(output/'seed-validation.json',check_seeds(output))
    snap=output/'source-snapshot';snap.mkdir()
    for n,p in enumerate(protocol['sources']):
        if p.endswith('.py'):shutil.copyfile(p,snap/f'{n:02d}-{Path(p).name}')
    generation=tasks.generate_tasks(actor,opponents,output/'generated',SEED+1)
    BASE.save_json(output/'generation.json',generation)
    (output/'discovery').mkdir()
    for i,record in enumerate(generation['discovery']):
        folder=output/'discovery'/f'case-{i:02d}';folder.mkdir();BASE.save_json(folder/'task.json',record)
    assert_sources(output)


def discover(output: Path) -> None:
    prepare(output)
    generation=json.loads((output/'generation.json').read_text());count=len(generation['discovery'])
    dispatch(output,count,['screen-a','screen-b'])
    rows=[]
    for i,record in enumerate(generation['discovery']):
        folder=output/'discovery'/f'case-{i:02d}'
        with np.load(folder/'screen-a/all.npz') as raw:a=raw['returns']
        with np.load(folder/'screen-b/all.npz') as raw:b=raw['returns']
        rows.append(dict(discovery_case=i,task=record,**screening_score(a,b)))
    chosen=sorted(rows,key=lambda r:(-r['rank_score'],r['discovery_case']))[:SELECT]
    (output/'selected').mkdir()
    for i,row in enumerate(chosen):
        folder=output/'selected'/f'case-{i:02d}';folder.mkdir();BASE.save_json(folder/'task.json',row['task'])
    BASE.save_json(output/'selection.json',dict(discovery=rows,selected=chosen))
    assert_sources(output);BASE.save_json(output/'discovery-complete.json',dict(complete=True,selection_sha=BASE.sha(output/'selection.json')))


def estimate(output: Path) -> None:
    seal=json.loads((output/'discovery-complete.json').read_text())
    if BASE.sha(output/'selection.json')!=seal['selection_sha']:raise ValueError('selection changed')
    assert_sources(output);selection=json.loads((output/'selection.json').read_text());count=len(selection['selected'])
    dispatch(output,count,['estimate-a','estimate-b'])
    tables=BASE.Tables()
    for match in BASE.iter_matches(BASE.POOL/'raw',range(8)):tables.add(match)
    tables.finish();hashes={};rows=[]
    for i in range(count):
        folder=output/'selected'/f'case-{i:02d}'
        with np.load(folder/'estimate-a/all.npz') as raw:a=raw['returns']
        with np.load(folder/'estimate-b/all.npz') as raw:b=raw['returns']
        _,posterior,history_records=BASE.public_histories(i,tables)
        pooled=np.concatenate([a,b],axis=2)
        choices=select_choices(pooled,posterior)
        BASE.save_json(folder/'sealed-choices.json',dict(choices=choices,posterior=posterior.tolist(),histories=history_records))
        hashes[str(i)]=BASE.sha(folder/'sealed-choices.json')
        stability=BASE.stability_metrics(a[:,:,:,0].mean(2),b[:,:,:,0].mean(2))
        BASE.save_json(folder/'stability.json',stability);rows.append(stability)
    BASE.save_json(output/'estimation-complete.json',dict(complete=True,choice_hashes=hashes,
        selection_sha=BASE.sha(output/'selection.json'),stability=rows))
    assert_sources(output)


def check_choices(output: Path) -> int:
    assert_sources(output);seal=json.loads((output/'estimation-complete.json').read_text())
    if BASE.sha(output/'selection.json')!=seal['selection_sha']:raise ValueError('selection changed after estimate')
    for i,h in seal['choice_hashes'].items():
        if BASE.sha(output/'selected'/f'case-{int(i):02d}'/'sealed-choices.json')!=h:raise ValueError('choice changed')
    return len(seal['choice_hashes'])


def analysis(output: Path,stage: str,count: int) -> dict:
    gains=[];absolute=[]
    for i in range(count):
        folder=output/'selected'/f'case-{i:02d}'
        choices=json.loads((folder/'sealed-choices.json').read_text())['choices']
        with np.load(folder/stage/'all.npz') as raw:q=raw['returns']
        gains.append(score(q,choices));absolute.append(dict(prior=q[:,choices['prior']].mean((0,1)).tolist(),
            identity=q[np.arange(4),choices['identity']].mean((0,1)).tolist(),posterior=q[np.arange(4),choices['posterior']].mean((0,1)).tolist()))
    rng=np.random.default_rng(SEED+20+(stage=='history'));draws={k:np.zeros((10000,2)) for k in gains[0]}
    for g in gains:
        weights=rng.multinomial(512,np.full(512,1/512),size=10000)/512
        for name in draws:draws[name]+=(weights@g[name].mean(0))/count
    stats={}
    for name in draws:
        means=np.asarray([g[name].mean((0,1)) for g in gains])
        rr=np.random.default_rng(SEED+22).integers(count,size=(10000,count))
        stats[name]=dict(mean=means.mean(0).tolist(),by_root=means.tolist(),
            world_ci975=np.quantile(draws[name],[.0125,.9875],axis=0).T.tolist(),
            world_ci95=np.quantile(draws[name],[.025,.975],axis=0).T.tolist(),
            root_ci95=np.quantile(means[rr].mean(1),[.025,.975],axis=0).T.tolist())
    if stage=='confirm':
        gate=bool(stats['identity']['mean'][0]>=.02 and stats['identity']['world_ci975'][0][0]>0 and stats['identity_alignment']['world_ci975'][0][0]>0)
    else:gate=bool(stats['posterior']['world_ci975'][0][0]>0 and stats['history_vs_shuffle']['world_ci975'][0][0]>0)
    return dict(stage=stage,gate_passed=gate,metrics=['round win probability','PPO round return'],gains=stats,
        absolute={name:np.asarray([r[name] for r in absolute]).mean(0).tolist() for name in absolute[0]},
        scope='Fixed selected roots and sealed policies; selection set reused for discovery only, independent scoring worlds; not full-match or unknown-opponent generalization')


def evaluate(output: Path,stage: str) -> None:
    count=check_choices(output)
    if stage=='history':
        result=json.loads((output/'confirm-summary.json').read_text())
        if not result['gate_passed']:raise ValueError('identity gate failed; history budget must remain unused')
    dispatch(output,count,[stage]);result=analysis(output,stage,count)
    BASE.save_json(output/f'{stage}-summary.json',result);check_choices(output)
    if stage=='history' or not result['gate_passed']:
        total=sum(json.loads(p.read_text())['rounds'] for p in output.glob('**/complete.json'))
        BASE.save_json(output/'complete.json',dict(complete=True,history_tested=stage=='history',identity_gate_passed=result['gate_passed'] if stage=='confirm' else True,
            score_branches=total,budget_cap=327680,no_training=True))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=('discover','estimate','confirm','history'),required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();torch.set_num_threads(1)
    if args.stage=='discover':discover(args.output)
    elif args.stage=='estimate':estimate(args.output)
    else:evaluate(args.output,args.stage)


if __name__=='__main__':main()
