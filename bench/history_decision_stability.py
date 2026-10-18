"""Fixed-budget CPU Q replication, sealed selection, independent value evaluation.

Run this file with checkpoint-matching train source first on PYTHONPATH.
Two stages: estimate completes and seals all decisions before evaluate starts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import gd
import numpy as np
import torch

from bench.history_decision_value import (
    ENDPOINT, POOL, Tables, choose_actions, iter_matches, load_actor, load_selected,
    public_histories, root_probabilities, root_state, rollout_matrix, select_candidates,
)
from bench.qknorm_match_ablation import sha

ROOTS = 8
ESTIMATE_WORLDS = 64  # each of two independent replicas
EVALUATION_WORLDS = 128
CHUNK = 16
SEED = 17101740
PILOT = Path('.work/history-decision-value-2026-10-17/run')


def world_seeds(split: int, root: int, count: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([SEED,split,root])).integers(
        1,2**63-1,size=count)


def stability_metrics(a: np.ndarray, b: np.ndarray, tolerance: float = .1) -> dict:
    """Rows are fixed prior/identity/posterior contexts; columns are candidate Qs."""
    if a.ndim!=2 or a.shape!=b.shape or a.shape[1]<2:
        raise ValueError('need matching context-by-action matrices')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite Q')
    aa,bb = a.argmax(1),b.argmax(1)
    row = np.arange(len(a))
    cross_a = a.max(1)-a[row,bb]
    cross_b = b.max(1)-b[row,aa]
    left,right = np.triu_indices(a.shape[1],1)
    da,db = a[:,left]-a[:,right],b[:,left]-b[:,right]
    # Ties count as half agreement unless both sides tie.
    pair = np.where(da*db>0,1.,np.where(da*db<0,0.,
                    np.where((da==0)&(db==0),1.,.5)))
    return dict(contexts=len(a),top1_agreement=float(np.mean(aa==bb)),
        pairwise_order_agreement=float(pair.mean()),q_rmse=float(np.sqrt(np.mean((a-b)**2))),
        symmetric_cross_gap=float(np.mean((cross_a+cross_b)/2)),
        within_tolerance=float(np.mean((cross_a<=tolerance+1e-12)&(cross_b<=tolerance+1e-12))),
        max_cross_gap=float(max(cross_a.max(),cross_b.max())),
        choices_a=aa.tolist(),choices_b=bb.tolist())


def choices_from(q: np.ndarray, posterior: np.ndarray, probs: np.ndarray,
                 candidates: np.ndarray) -> dict:
    nz = len(q)
    def local(a: int) -> int:
        return int(np.flatnonzero(candidates==a)[0])
    return dict(prior=int(choose_actions(q,np.full(nz,1/nz))),
        identity=choose_actions(q,np.eye(nz)).tolist(),
        posterior=choose_actions(q,posterior).tolist(),
        actor_round=[local(int(probs[0].argmax()))]*nz,
        actor_full=[local(int(a)) for a in probs[1:].argmax(1)])


def score_choices(q_test: np.ndarray, choices: dict) -> dict:
    """No selection estimates are accessible to this scoring function."""
    q = q_test.mean(-1)
    z = np.arange(len(q))
    base = q[:,choices['prior']]
    values = {'prior':base}
    for name in ('identity','posterior','actor_round','actor_full'):
        values[name] = q[z,np.asarray(choices[name])]
    gains = {name:values[name]-base for name in values if name!='prior'}
    gains['actor_history_effect'] = values['actor_full']-values['actor_round']
    return dict(values={k:dict(mean=float(v.mean()),by_type=v.tolist()) for k,v in values.items()},
                gains={k:dict(mean=float(v.mean()),by_type=v.tolist()) for k,v in gains.items()})


def estimate_stability(a: np.ndarray, b: np.ndarray, posterior: np.ndarray) -> dict:
    weights = dict(prior=np.full((1,len(a)),1/len(a)),identity=np.eye(len(a)),posterior=posterior)
    result = {}
    for n in (8,16,32,64):
        result[str(n)] = {name:stability_metrics(w@a[:,:,:n].mean(-1),w@b[:,:,:n].mean(-1))
                          for name,w in weights.items()}
    return result


def q_uncertainty(q: np.ndarray, weights: np.ndarray) -> dict:
    """Paired-world SEs; selected top gaps are descriptive, not corrected tests."""
    values = np.einsum('cz,zaw->caw',weights,q)
    means = values.mean(-1)
    order = np.argsort(-means,axis=1,kind='stable')
    row = np.arange(len(means))
    gaps = values[row,order[:,0],:]-values[row,order[:,1],:]
    se = gaps.std(-1,ddof=1)/np.sqrt(q.shape[-1])
    return dict(q_mean=means.tolist(),q_se=(values.std(-1,ddof=1)/np.sqrt(q.shape[-1])).tolist(),
        top1=order[:,0].tolist(),runner_up=order[:,1].tolist(),top_gap=gaps.mean(-1).tolist(),
        paired_top_gap_se=se.tolist(),
        top_gap_normal95=np.stack([gaps.mean(-1)-1.96*se,gaps.mean(-1)+1.96*se],axis=-1).tolist(),
        caveat='Top two selected on these same estimates; intervals are descriptive and uncorrected for selection or multiplicity')


def source_files() -> dict[str,str]:
    import train.history_model as model
    import train.history_rollout as rollout
    return {p:sha(p) for p in [__file__,'bench/history_decision_value.py',
        'bench/public_history_analysis.py','bench/public_history_probe.py',
        'bench/public_history_diverse.py','bench/qknorm_match_ablation.py',
        model.__file__,rollout.__file__,gd._gd_core.__file__]}


def assert_sources(protocol: dict) -> None:
    if source_files()!=protocol['sources']:
        raise ValueError('runtime source identity changed')
    for record in [protocol['endpoint'],*protocol['opponents'].values()]:
        if sha(record['path'])!=record['checkpoint_sha']:
            raise ValueError('checkpoint identity changed')


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value,indent=2)+'\n')


def chunked_rollouts(actor: Any, opponents: dict, state: Any, candidates: np.ndarray,
                     seeds: np.ndarray, folder: Path) -> tuple[np.ndarray,dict]:
    """Fixed chunks bound memory and persist every completed chunk without censoring."""
    folder.mkdir(exist_ok=False)
    arrays,details = [],[]
    for start in range(0,len(seeds),CHUNK):
        take = seeds[start:start+CHUNK]
        q,meta = rollout_matrix(actor,opponents,state,candidates,take)
        np.savez_compressed(folder/f'chunk-{start//CHUNK:02d}.npz',returns=q,world_seeds=take,candidates=candidates)
        arrays.append(q);details.append(meta)
        print(json.dumps(dict(split=folder.name,chunk=start//CHUNK,worlds_done=start+len(take),
                              total_worlds=len(seeds),rounds=meta['rounds'])),flush=True)
    q = np.concatenate(arrays,axis=-1)
    np.savez_compressed(folder/'all.npz',returns=q,world_seeds=seeds,candidates=candidates)
    meta = dict(rounds=sum(d['rounds'] for d in details),
                seconds=sum(d['seconds'] for d in details),chunks=details)
    save_json(folder/'complete.json',meta)
    return q,meta


def aggregate_stability(rows: list[dict]) -> dict:
    out = {}
    metrics = ('top1_agreement','pairwise_order_agreement','q_rmse',
               'symmetric_cross_gap','within_tolerance')
    for n in ('8','16','32','64'):
        out[n] = {kind:{metric:float(np.mean([r[n][kind][metric] for r in rows]))
                       for metric in metrics} for kind in ('prior','identity','posterior')}
    final = out['64']['identity']
    return dict(by_replica_world_count=out,
        descriptive_stability_gate=bool(final['top1_agreement']>=.8 and final['pairwise_order_agreement']>=.8),
        gate='At 64 worlds per independent replica: identity top1 agreement >=80% AND pairwise ordering agreement >=80%; descriptive screen, no inferential guarantee',
        practical_tolerance=.1,practical_unit='PPO team round seat_return')


def estimate(output: Path) -> None:
    output.mkdir(parents=True,exist_ok=False)
    actor,identity = load_actor(ENDPOINT)
    opponents,opponent_meta = load_selected(POOL)
    protocol = dict(roots=ROOTS,replica_worlds=ESTIMATE_WORLDS,
        pooled_selection_worlds=2*ESTIMATE_WORLDS,evaluation_worlds=EVALUATION_WORLDS,
        candidate_cap=5,max_rounds=ROOTS*4*5*(2*ESTIMATE_WORLDS+EVALUATION_WORLDS),
        chunk_worlds=CHUNK,seed=SEED,sources=source_files(),endpoint=identity,opponents=opponent_meta,
        pilot_protocol_sha=sha(PILOT/'protocol.json'),torch=torch.__version__,numpy=np.__version__,
        task='Same eight pilot roots, histories, candidate sets, fixed round-only frozen continuation; all new independent rollout worlds',
        stage_order='Complete both estimation replicas for ALL roots, seal choices and stability, THEN run independent evaluation',
        budget_policy='Exactly fixed sample counts; no dropping roots, optional stopping, new candidates, PPO, GPU or remote work',
        stability_gate='64-world independent identity top1 >=80% and pairwise ranking >=80%; tolerance .1 reported separately',
        reporting='Exploratory root cluster intervals plus paired-world conditional intervals; fixed reused roots/histories, no new-population confirmation')
    save_json(output/'protocol.json',protocol)
    snapshot = output/'source-snapshot';snapshot.mkdir()
    for k,path in enumerate(protocol['sources']):
        if path.endswith('.py'):shutil.copyfile(path,snapshot/f'{k:02d}-{Path(path).name}')
    # Verify all generated seed sets are disjoint from each other AND the pilot.
    used = set()
    for i in range(ROOTS):
        for split in ('selection','evaluation'):
            with np.load(PILOT/f'root-{i:02d}-{split}.npz') as raw:
                used.update(map(int,raw['world_seeds']))
    for i in range(ROOTS):
        for split,count in ((0,ESTIMATE_WORLDS),(1,ESTIMATE_WORLDS),(2,EVALUATION_WORLDS)):
            seeds = world_seeds(split,i,count)
            if len(set(map(int,seeds)))!=count or used.intersection(map(int,seeds)):
                raise ValueError('seed collision')
            used.update(map(int,seeds))
    tables = Tables()
    for match in iter_matches(POOL/'raw',range(8)):tables.add(match)
    tables.finish()
    stability_rows,choice_hashes = [],{}
    for i in range(ROOTS):
        folder = output/f'root-{i:02d}';folder.mkdir()
        pilot = json.loads((PILOT/f'root-{i:02d}-inputs.json').read_text())
        state = root_state(pilot['seed'])
        histories,post,history_meta = public_histories(i,tables)
        probs = root_probabilities(actor,state,histories)
        candidates = select_candidates(probs[0],probs[1:])
        if candidates.tolist()!=pilot['candidates'] or history_meta!=pilot['histories'] or state.hand(0)!=pilot['own_hand']:
            raise ValueError('pilot roots/candidates/history identity changed')
        save_json(folder/'inputs.json',dict(root=i,seed=pilot['seed'],candidates=candidates.tolist(),histories=history_meta,
            posterior=post.tolist(),actor_greedy=probs.argmax(1).tolist()))
        replicas = []
        for split,name in ((0,'estimate-a'),(1,'estimate-b')):
            q,_ = chunked_rollouts(actor,opponents,state,candidates,world_seeds(split,i,ESTIMATE_WORLDS),folder/name)
            replicas.append(q)
        a,b = replicas
        stability = estimate_stability(a,b,post)
        stability_rows.append(stability)
        save_json(folder/'stability.json',stability)
        pooled = np.concatenate([a,b],axis=-1)
        save_json(folder/'q-uncertainty.json',{name:q_uncertainty(pooled,w) for name,w in
            dict(prior=np.full((1,4),.25),identity=np.eye(4),posterior=post).items()})
        choices = {name:choices_from(q,post,probs,candidates) for name,q in
                   [('a64',a),('b64',b),('pooled128',pooled)]}
        # Also seal convergence choices, without using any future evaluation returns.
        choices['nested'] = {str(n):choices_from(np.concatenate([a[:,:,:n],b[:,:,:n]],axis=-1),post,probs,candidates)
                             for n in (8,16,32,64)}
        save_json(folder/'sealed-choices.json',choices)
        choice_hashes[str(i)] = sha(folder/'sealed-choices.json')
        print(json.dumps(dict(root=i,stage='estimate-complete',identity_stability=stability['64']['identity'])),flush=True)
    save_json(output/'stability-summary.json',aggregate_stability(stability_rows))
    assert_sources(protocol)
    save_json(output/'estimation-complete.json',dict(complete=True,choice_hashes=choice_hashes,
        stability_sha=sha(output/'stability-summary.json'),rounds=ROOTS*4*5*2*ESTIMATE_WORLDS))


def root_intervals(values: np.ndarray) -> dict:
    draws = np.random.default_rng(SEED+10).integers(len(values),size=(10000,len(values)))
    return dict(mean=float(values.mean()),ci95=np.quantile(values[draws].mean(1),[.025,.975]).tolist(),by_root=values.tolist())


def evaluation_summary(rows: list[dict], paired_gains: list[dict]) -> dict:
    out = {}
    for recipe in ('a64','b64','pooled128'):
        out[recipe] = dict(gains={metric:root_intervals(np.asarray([r[recipe]['gains'][metric]['mean'] for r in rows]))
                                  for metric in rows[0][recipe]['gains']},
                           values={metric:float(np.mean([r[recipe]['values'][metric]['mean'] for r in rows]))
                                   for metric in rows[0][recipe]['values']})
    # Conditional uncertainty for the SAME fixed roots and sealed decisions.
    # Pair types and actions by world, independently resample worlds within each root.
    conditional = {}
    rng = np.random.default_rng(SEED+11)
    weights = [rng.multinomial(EVALUATION_WORLDS,np.full(EVALUATION_WORLDS,1/EVALUATION_WORLDS),size=10000)/EVALUATION_WORLDS
               for _ in paired_gains]
    for metric in paired_gains[0]:
        means = [g[metric].mean(0) for g in paired_gains]
        draws = np.mean([w@m for w,m in zip(weights,means)],axis=0)
        conditional[metric] = dict(mean=float(np.mean(means)),ci95=np.quantile(draws,[.025,.975]).tolist())
    return dict(recipes=out,pooled128_conditional_world_bootstrap=conditional,roots=ROOTS,
        unit='PPO team round seat_return',
        uncertainty='Root intervals resample only eight reused roots. Conditional world intervals hold those roots, histories, weights and sealed choices fixed. Neither establishes unseen-root generalization or exact-ID optimal value.',
        rounds=ROOTS*4*5*(2*ESTIMATE_WORLDS+EVALUATION_WORLDS))


def evaluate(output: Path) -> None:
    protocol = json.loads((output/'protocol.json').read_text())
    seal = json.loads((output/'estimation-complete.json').read_text())
    assert_sources(protocol)
    if sha(output/'stability-summary.json')!=seal['stability_sha']:
        raise ValueError('stability result changed after sealing')
    for i in range(ROOTS):
        if sha(output/f'root-{i:02d}'/'sealed-choices.json')!=seal['choice_hashes'][str(i)]:
            raise ValueError('sealed choice changed')
    actor,_ = load_actor(ENDPOINT)
    opponents,_ = load_selected(POOL)
    rows,paired = [],[]
    for i in range(ROOTS):
        folder = output/f'root-{i:02d}'
        inputs = json.loads((folder/'inputs.json').read_text())
        choices = json.loads((folder/'sealed-choices.json').read_text())
        candidates = np.asarray(inputs['candidates'])
        q,_ = chunked_rollouts(actor,opponents,root_state(inputs['seed']),candidates,
                              world_seeds(2,i,EVALUATION_WORLDS),folder/'evaluation')
        row = {recipe:score_choices(q,choices[recipe]) for recipe in ('a64','b64','pooled128')}
        row['nested'] = {n:score_choices(q,c) for n,c in choices['nested'].items()}
        save_json(folder/'evaluation-result.json',row)
        rows.append(row)
        selected = choices['pooled128'];z=np.arange(len(q))
        base = q[:,selected['prior'],:]
        gains = {name:q[z,np.asarray(selected[name]),:]-base
                 for name in ('identity','posterior','actor_round','actor_full')}
        gains['actor_history_effect'] = gains['actor_full']-gains['actor_round']
        paired.append(gains)
        print(json.dumps(dict(root=i,stage='evaluation-complete',gains={k:v['mean'] for k,v in row['pooled128']['gains'].items()})),flush=True)
    save_json(output/'evaluation-summary.json',evaluation_summary(rows,paired))
    assert_sources(protocol)
    for i in range(ROOTS):
        if sha(output/f'root-{i:02d}'/'sealed-choices.json')!=seal['choice_hashes'][str(i)]:
            raise ValueError('choice seal changed during evaluation')
    save_json(output/'complete.json',dict(complete=True,rounds=protocol['max_rounds'],sources_unchanged=True,
        choices_unchanged=True,no_training=True))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('estimate','evaluate'),required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.stage=='estimate':estimate(args.output)
    else:evaluate(args.output)


if __name__=='__main__':
    main()
