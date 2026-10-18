"""Larger fixed-budget local CPU replication on the same eight information sets.

Four single-threaded local processes; completed estimation seals ALL choices
before evaluation can start. Prior evidence and measured runners are unchanged.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time
from typing import Any

import gd
import numpy as np
import torch

from bench import history_decision_stability as base

ROOTS = 8
ESTIMATE_WORLDS = 512
EVALUATION_WORLDS = 1024
WORKERS = 4
SEED = 17101750
NESTED = (64,128,256,512)
PREVIOUS = Path('.work/history-decision-stability-2026-10-17')
_MODELS: tuple[Any,dict] | None = None


def world_seeds(split: int, root: int, count: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([SEED,split,root])).integers(1,2**63-1,size=count)


def sources() -> dict[str,str]:
    return {**base.source_files(),__file__:base.sha(__file__)}


def assert_identity(protocol: dict) -> None:
    if sources()!=protocol['sources']:
        raise ValueError('source identity changed')
    for record in [protocol['endpoint'],*protocol['opponents'].values()]:
        if base.sha(record['path'])!=record['checkpoint_sha']:
            raise ValueError('checkpoint identity changed')


def check_seed_disjointness() -> dict:
    used: set[int] = set()
    for i in range(ROOTS):
        for split in ('selection','evaluation'):
            with np.load(base.PILOT/f'root-{i:02d}-{split}.npz') as raw:
                used.update(map(int,raw['world_seeds']))
        for split in ('estimate-a','estimate-b','evaluation'):
            with np.load(PREVIOUS/f'root-{i:02d}'/split/'all.npz') as raw:
                used.update(map(int,raw['world_seeds']))
    previous_count = len(used)
    for i in range(ROOTS):
        for split,count in ((0,ESTIMATE_WORLDS),(1,ESTIMATE_WORLDS),(2,EVALUATION_WORLDS)):
            seeds = world_seeds(split,i,count)
            fresh = set(map(int,seeds))
            if len(fresh)!=count or fresh.intersection(used):
                raise ValueError('seed collision with prior evidence or another split')
            used.update(fresh)
    return dict(previous_unique_seeds=previous_count,new_unique_seeds=len(used)-previous_count,disjoint=True)


def stability(a: np.ndarray,b: np.ndarray,posterior: np.ndarray) -> dict:
    weights = dict(prior=np.full((1,4),.25),identity=np.eye(4),posterior=posterior)
    return {str(n):{name:base.stability_metrics(w@a[:,:,:n].mean(-1),w@b[:,:,:n].mean(-1))
                    for name,w in weights.items()} for n in NESTED}


def stability_summary(rows: list[dict]) -> dict:
    metrics = ('top1_agreement','pairwise_order_agreement','q_rmse','symmetric_cross_gap','within_tolerance')
    values = {str(n):{kind:{metric:float(np.mean([r[str(n)][kind][metric] for r in rows]))
                             for metric in metrics} for kind in ('prior','identity','posterior')} for n in NESTED}
    final=values[str(ESTIMATE_WORLDS)]['identity']
    return dict(by_replica_world_count=values,
        descriptive_stability_gate=bool(final['top1_agreement']>=.8 and final['pairwise_order_agreement']>=.8),
        gate='At 512 worlds per independent replica: identity top1 AND pairwise order agreement >=80%; descriptive only',
        practical_tolerance=.1,unit='PPO team round seat_return')


def prepare(output: Path) -> None:
    output.mkdir(parents=True,exist_ok=False)
    actor,identity=base.load_actor(base.ENDPOINT)
    opponents,opponent_meta=base.load_selected(base.POOL)
    protocol=dict(roots=ROOTS,replica_worlds=ESTIMATE_WORLDS,
        pooled_selection_worlds=2*ESTIMATE_WORLDS,evaluation_worlds=EVALUATION_WORLDS,
        candidate_cap=5,max_rounds=ROOTS*4*5*(2*ESTIMATE_WORLDS+EVALUATION_WORLDS),
        workers=WORKERS,torch_threads_per_worker=1,engine_threads_per_worker=1,
        chunk_worlds=base.CHUNK,seed=SEED,sources=sources(),endpoint=identity,opponents=opponent_meta,
        previous_protocol_sha=base.sha(PREVIOUS/'protocol.json'),torch=torch.__version__,numpy=np.__version__,
        stage_order='ALL A/B estimates and stability, then SHA-sealed choices, then independent evaluation',
        conditions='Same 8 opening roots, own hands, public histories, posteriors, 5 candidates and frozen round-only continuation',
        budget='Fixed sample counts, no outcome-dependent expansion, root filtering, PPO, GPU or remote work',
        stability_gate='512-world independent identity top1 and pairwise ranking >=80%; practical tolerance .1 reported separately',
        uncertainty='Exploratory; reused roots and histories; conditional paired-world intervals hold sealed choices fixed')
    base.save_json(output/'protocol.json',protocol)
    snapshot=output/'source-snapshot';snapshot.mkdir()
    for k,path in enumerate(protocol['sources']):
        if path.endswith('.py'):shutil.copyfile(path,snapshot/f'{k:02d}-{Path(path).name}')
    base.save_json(output/'seed-validation.json',check_seed_disjointness())
    tables=base.Tables()
    for match in base.iter_matches(base.POOL/'raw',range(8)):tables.add(match)
    tables.finish()
    for i in range(ROOTS):
        folder=output/f'root-{i:02d}';folder.mkdir()
        pilot=json.loads((base.PILOT/f'root-{i:02d}-inputs.json').read_text())
        previous=json.loads((PREVIOUS/f'root-{i:02d}'/'inputs.json').read_text())
        state=base.root_state(pilot['seed'])
        histories,post,history_meta=base.public_histories(i,tables)
        probs=base.root_probabilities(actor,state,histories)
        candidates=base.select_candidates(probs[0],probs[1:])
        if (candidates.tolist()!=pilot['candidates'] or history_meta!=pilot['histories'] or
            state.hand(0)!=pilot['own_hand'] or not np.array_equal(post,previous['posterior']) or
            probs.argmax(1).tolist()!=previous['actor_greedy']):
            raise ValueError('root, candidates, posterior or actor choice changed')
        base.save_json(folder/'inputs.json',dict(root=i,seed=pilot['seed'],candidates=candidates.tolist(),
            histories=history_meta,posterior=post.tolist(),actor_greedy=probs.argmax(1).tolist()))
        np.savez_compressed(folder/'inputs.npz',posterior=post,probs=probs,candidates=candidates)
    assert_identity(protocol)
    # Release parent models before spawning independent CPU workers.
    del actor,opponents


def init_worker() -> None:
    global _MODELS
    torch.set_num_threads(1)
    actor,_=base.load_actor(base.ENDPOINT)
    opponents,_=base.load_selected(base.POOL)
    _MODELS=(actor,opponents)


def work_root(output_path: str,root: int,stage: str) -> dict:
    if _MODELS is None:
        raise RuntimeError('worker must load frozen CPU models first')
    actor,opponents=_MODELS
    output=Path(output_path);folder=output/f'root-{root:02d}'
    inp=json.loads((folder/'inputs.json').read_text())
    with np.load(folder/'inputs.npz') as raw:
        post,probs,candidates=(raw[k] for k in ('posterior','probs','candidates'))
    state=base.root_state(inp['seed'])
    with (folder/f'{stage}-worker.log').open('x') as log,redirect_stdout(log):
        if stage=='estimate':
            replicas=[]
            for split,name in ((0,'estimate-a'),(1,'estimate-b')):
                q,_=base.chunked_rollouts(actor,opponents,state,candidates,
                    world_seeds(split,root,ESTIMATE_WORLDS),folder/name)
                replicas.append(q)
            a,b=replicas;pooled=np.concatenate([a,b],axis=-1)
            base.save_json(folder/'stability.json',stability(a,b,post))
            base.save_json(folder/'q-uncertainty.json',{name:base.q_uncertainty(pooled,w) for name,w in
                dict(prior=np.full((1,4),.25),identity=np.eye(4),posterior=post).items()})
            choices={name:base.choices_from(q,post,probs,candidates) for name,q in
                     [('a512',a),('b512',b),('pooled1024',pooled)]}
            choices['nested']={str(n):base.choices_from(np.concatenate([a[:,:,:n],b[:,:,:n]],axis=-1),post,probs,candidates)
                               for n in NESTED}
            base.save_json(folder/'sealed-choices.json',choices)
        else:
            choices=json.loads((folder/'sealed-choices.json').read_text())
            q,_=base.chunked_rollouts(actor,opponents,state,candidates,
                world_seeds(2,root,EVALUATION_WORLDS),folder/'evaluation')
            rows={recipe:base.score_choices(q,choices[recipe]) for recipe in ('a512','b512','pooled1024')}
            rows['nested']={n:base.score_choices(q,c) for n,c in choices['nested'].items()}
            base.save_json(folder/'evaluation-result.json',rows)
    return dict(root=root,stage=stage,complete=True)


def run_workers(output: Path,stage: str) -> None:
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
        jobs=[pool.submit(work_root,str(output),i,stage) for i in range(ROOTS)]
        for job in as_completed(jobs):
            print(json.dumps(job.result()),flush=True)
    base.save_json(output/f'{stage}-timing.json',dict(wall_seconds=time.monotonic()-started,workers=WORKERS))


def estimate(output: Path) -> None:
    prepare(output)
    run_workers(output,'estimate')
    rows=[json.loads((output/f'root-{i:02d}'/'stability.json').read_text()) for i in range(ROOTS)]
    base.save_json(output/'stability-summary.json',stability_summary(rows))
    protocol=json.loads((output/'protocol.json').read_text());assert_identity(protocol)
    base.save_json(output/'estimation-complete.json',dict(complete=True,
        choice_hashes={str(i):base.sha(output/f'root-{i:02d}'/'sealed-choices.json') for i in range(ROOTS)},
        stability_sha=base.sha(output/'stability-summary.json'),rounds=ROOTS*4*5*2*ESTIMATE_WORLDS))


def check_seal(output: Path) -> None:
    protocol=json.loads((output/'protocol.json').read_text())
    seal=json.loads((output/'estimation-complete.json').read_text())
    assert_identity(protocol)
    if base.sha(output/'stability-summary.json')!=seal['stability_sha']:
        raise ValueError('stability seal changed')
    for i in range(ROOTS):
        if base.sha(output/f'root-{i:02d}'/'sealed-choices.json')!=seal['choice_hashes'][str(i)]:
            raise ValueError('choice seal changed')


def evaluation_summary(output: Path) -> dict:
    rows=[];paired=[]
    for i in range(ROOTS):
        folder=output/f'root-{i:02d}'
        rows.append(json.loads((folder/'evaluation-result.json').read_text()))
        selected=json.loads((folder/'sealed-choices.json').read_text())['pooled1024']
        with np.load(folder/'evaluation/all.npz') as raw:q=raw['returns']
        z=np.arange(4);baseline=q[:,selected['prior'],:]
        gains={name:q[z,np.asarray(selected[name]),:]-baseline for name in ('identity','posterior','actor_round','actor_full')}
        gains['actor_history_effect']=gains['actor_full']-gains['actor_round']
        paired.append(gains)
    recipes={recipe:dict(gains={metric:base.root_intervals(np.asarray([r[recipe]['gains'][metric]['mean'] for r in rows]))
        for metric in rows[0][recipe]['gains']},values={metric:float(np.mean([r[recipe]['values'][metric]['mean'] for r in rows]))
        for metric in rows[0][recipe]['values']}) for recipe in ('a512','b512','pooled1024')}
    # Stream bootstrap matrices per root to bound memory at the larger sample size.
    rng=np.random.default_rng(SEED+11)
    metrics=list(paired[0]);draws={metric:np.zeros(10000) for metric in metrics}
    for gain in paired:
        weights=rng.multinomial(EVALUATION_WORLDS,np.full(EVALUATION_WORLDS,1/EVALUATION_WORLDS),size=10000)/EVALUATION_WORLDS
        for metric in metrics:draws[metric]+=(weights@gain[metric].mean(0))/ROOTS
    conditional={metric:dict(mean=float(np.mean([g[metric].mean() for g in paired])),
        ci95=np.quantile(draws[metric],[.025,.975]).tolist()) for metric in metrics}
    return dict(recipes=recipes,pooled1024_conditional_world_bootstrap=conditional,roots=ROOTS,
        rounds=ROOTS*4*5*(2*ESTIMATE_WORLDS+EVALUATION_WORLDS),unit='PPO team round seat_return',
        uncertainty='Exploratory reused eight roots. Conditional intervals hold roots, histories, models and sealed choices fixed; root bootstrap has only eight clusters. No exact oracle or unseen-root confirmation.')


def evaluate(output: Path) -> None:
    check_seal(output)
    run_workers(output,'evaluate')
    base.save_json(output/'evaluation-summary.json',evaluation_summary(output))
    check_seal(output)
    base.save_json(output/'complete.json',dict(complete=True,rounds=ROOTS*4*5*(2*ESTIMATE_WORLDS+EVALUATION_WORLDS),
        sources_unchanged=True,choices_unchanged=True,no_training=True))


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('estimate','evaluate'),required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    if args.stage=='estimate':estimate(args.output)
    else:evaluate(args.output)


if __name__=='__main__':main()
