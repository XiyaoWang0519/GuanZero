"""Frozen width-preserving fusion counterfactuals; no optimizer or model updates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from bench.public_history_diverse import load_actor
from bench.public_history_analysis import cluster_weights
from bench.qknorm_history_stages import ROOT as STAGE_ROOT, POOL, ridge_predict, brier_probabilities
from bench.qknorm_history_swap import ROOT as SWAP_ROOT, donor_bank, splice
from bench.qknorm_match_summary import ENDPOINTS
from train.history_model import StreamBatch, DecisionInputs, segment_log_softmax

ROOT=Path('.work/qknorm-fusion-variants-2026-10-17')
VARIANTS=('native','history2','history4','restore_mean','postnorm1','postnorm4','query_half')


def load_data(arm: str) -> dict[str,np.ndarray]:
    parts=[]
    for w in range(16):
        with np.load(STAGE_ROOT/f'wave-{w:02d}'/f'{arm}.npz') as f:
            parts.append({k:f[k] for k in f.files})
    return {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}


def calibration(d: dict[str,np.ndarray]) -> dict[str,torch.Tensor]:
    take=d['wave']<8
    return dict(query=torch.tensor(d['context'][take,:128].mean(0),dtype=torch.float32),
                reader=torch.tensor(d['reader'][take].mean(0),dtype=torch.float32))


def fusion_states(actor: Any, query: torch.Tensor, reader: torch.Tensor,
                  reference: dict[str,torch.Tensor]) -> dict[str,torch.Tensor]:
    """All alternatives preserve width and the original candidate/feed-forward heads."""
    qbar,rbar=reference['query'],reference['reader']
    delta=reader-rbar
    anchor=query+rbar
    scale=torch.sqrt(anchor.var(1,unbiased=False,keepdim=True)+actor.attention_norm.eps)
    gamma=actor.attention_norm.weight
    base=actor.attention_norm(query+reader)
    h=dict(native=base,
        history2=actor.attention_norm(query+rbar+2*delta),
        history4=actor.attention_norm(query+rbar+4*delta),
        restore_mean=base+gamma*delta.mean(1,keepdim=True)/scale,
        postnorm1=actor.attention_norm(anchor)+gamma*delta/scale,
        postnorm4=actor.attention_norm(anchor)+4*gamma*delta/scale,
        query_half=actor.attention_norm(qbar+.5*(query-qbar)+reader))
    return {k:actor.output_norm(v+actor.feed_forward(v)) for k,v in h.items()}


def readouts() -> None:
    torch.set_num_threads(1);result={}
    with np.load(POOL/'raw/questions/public_answers.npz') as f:
        pi=(f['counts'][:,:,0]+.5)/33.5;freq=f['counts'][:,:,1]/32
    for arm in ENDPOINTS:
        d=load_data(arm);ref=calibration(d)
        actor,identity=load_actor(Path(f'.work/qknorm-short-ppo-2026-10-16/remote/runs/{arm}/endpoint.pt'))
        assert identity['checkpoint_sha']==ENDPOINTS[arm]
        (ROOT/f'{arm}-calibration.json').write_text(json.dumps({k:v.tolist() for k,v in ref.items()})+'\n')
        query=torch.tensor(d['context'][:,:128],dtype=torch.float32)
        arrays={}
        with torch.inference_mode():
            for view,reader in [('full',d['reader']),('round',d['context'][:,128:256])]:
                for name,state in fusion_states(actor,query,torch.tensor(reader,dtype=torch.float32),ref).items():
                    arrays[name+'/'+view]=state.numpy().astype(float)
        error=float(np.abs(arrays['native/full']-d['state']).max());assert error==0
        train=d['wave']<8;test=~train;ids=d['opponent'][test];seeds=d['seed'][test]
        y=np.eye(4)[d['opponent'][train]]
        groups=np.array([np.flatnonzero(seeds==s) for s in np.unique(seeds)])
        weights=cluster_weights(ids[groups[:,0]],10000,np.random.default_rng(17102741))
        qw=np.random.default_rng(17102742).multinomial(256,np.full(256,1/256),size=10000)/256
        pred={k:ridge_predict(v[train],y,v[test]) for k,v in arrays.items()}
        scores={k:brier_probabilities(p,pi,freq,ids) for k,p in pred.items()}
        base_gain=scores['native/round']-scores['native/full'];values={}
        for name in VARIANTS:
            gain=scores[name+'/round']-scores[name+'/full']
            row={}
            for title,contrast in [('history_gain',gain),('gain_over_native',gain-base_gain)]:
                matrix=contrast[groups].mean(1);draws=np.sum((weights@matrix)*qw,1)
                row[title]=dict(estimate=float(matrix.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())
            state=arrays[name+'/full']
            row.update(full_accuracy=float(np.mean(pred[name+'/full'].argmax(1)==ids)),
                full_question_brier=float(scores[name+'/full'].mean()),
                state_rms_drift=float(np.sqrt(np.mean((state[test]-arrays['native/full'][test])**2))),
                calibration_columns_below_floor=int((state[train].std(0)<.01).sum()))
            values[name]=row
        np.savez_compressed(ROOT/f'{arm}-readout-predictions.npz',**{k.replace('/','__'):v for k,v in pred.items()},ids=ids,seeds=seeds)
        result[arm]=dict(native_state_exact=True,conditions=values,calibration=256,heldout=256,test_seed_clusters=128)
    (ROOT/'readout-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def policy_measure(actor: Any, sample: dict, donor: tuple,
                   reference: dict[str,torch.Tensor]) -> dict:
    original=sample['original'];other=splice(original,donor,sample['round'],True)
    sb=StreamBatch.from_arrays([original,other],'cpu');c=len(sample['cand'])
    inputs=DecisionInputs(sb,torch.arange(2),sb.lengths,
        torch.tensor(np.repeat(sample['obs'][None],2,0)),torch.full((2,),sample['seat']),
        torch.tensor(np.tile(sample['cand'],(2,1))),torch.arange(3)*c,one_decision_per_stream=True)
    encoded=actor.encode_batch(sb)
    query=actor.private(inputs.obs.float())+actor.seat(inputs.seat)
    assert torch.equal(query[0],query[1])
    reader=actor._attend_encoded(query,encoded,inputs.match_index,inputs.prefix,True)
    states=fusion_states(actor,query,reader,reference)
    tables={}
    for name,state in states.items():
        logits=actor.candidate_logits(state,inputs.cand,inputs.offsets,rows=inputs.rows)
        table=segment_log_softmax(logits,inputs.rows,2).reshape(2,c).double()
        assert torch.isfinite(table).all()
        tables[name]=table
    cached=float((tables['native'][0]-torch.tensor(sample['cached_logp'])).abs().max());assert cached<1e-4
    native=actor._log_prob_table(inputs,encoded,c).double()
    assert torch.equal(native,tables['native'])
    probs={k:v.exp()/v.exp().sum(1,keepdim=True) for k,v in tables.items()}
    p0=probs['native'][0];out={}
    for name,p in probs.items():
        out[name]=dict(history_tv=float((p[0]-p[1]).abs().sum()/2),
            history_flip=int(p[0].argmax()!=p[1].argmax()),
            original_policy_drift=float((p[0]-p0).abs().sum()/2),
            original_flip=int(p[0].argmax()!=p0.argmax()),
            original_entropy=float(-(p[0]*p[0].log()).sum()),
            max_probability=float(p[0].max()))
    return dict(conditions=out,cache_error=cached)


def policy_wave(wave: int) -> None:
    torch.set_num_threads(1);folder=ROOT/f'wave-{wave:02d}';folder.mkdir(exist_ok=False)
    reference_rows=json.loads((SWAP_ROOT/f'wave-{wave:02d}'/'rows.json').read_text());rows=[]
    for arm in ENDPOINTS:
        actor,identity=load_actor(Path(f'.work/qknorm-short-ppo-2026-10-16/remote/runs/{arm}/endpoint.pt'))
        assert identity['checkpoint_sha']==ENDPOINTS[arm]
        ref=calibration(load_data(arm))
        samples=torch.load(SWAP_ROOT/f'wave-{wave:02d}'/f'{arm}-samples.pt',weights_only=False)
        bank=donor_bank(arm);lookup={(d['source'],d['wave'],d['env'],d['round'],d['window_start']):d for d in bank}
        with torch.inference_mode():
            for row in (r for r in reference_rows if r['arm']==arm):
                d=lookup[(row['donor_source'],row['donor_wave'],row['donor_env'],row['round'],row['donor_window_start'])]
                donor=tuple(x.copy() for x in d['arrays'])
                if row['donor_seat_rotation']:donor[0][:,:4]=np.roll(donor[0][:,:4],1,1)
                v=policy_measure(actor,samples[row['index']],donor,ref)
                assert abs(v['conditions']['native']['history_tv']-row['fixed_metadata']['tv'])<2e-5
                assert v['conditions']['native']['history_flip']==row['fixed_metadata']['flip']
                v.update(arm=arm,wave=wave,env=row['env'],seed=row['seed'],round=row['round']);rows.append(v)
    (folder/'policy.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(wave=wave,rows=len(rows),native_parity=True)),flush=True)


def summarize_policy() -> None:
    rows=[]
    for w in range(8,12):rows+=json.loads((ROOT/f'wave-{w:02d}'/'policy.json').read_text())
    result={}
    for arm in ENDPOINTS:
        rs=[r for r in rows if r['arm']==arm];seeds=np.unique([r['seed'] for r in rs]);out={}
        rng=np.random.default_rng(17102743);draws=rng.integers(len(seeds),size=(10000,len(seeds)))
        for name in VARIANTS:
            values={}
            for metric in ('history_tv','history_flip','original_policy_drift','original_flip','original_entropy','max_probability'):
                a=np.array([r['conditions'][name][metric] for r in rs])
                seed_means=np.array([np.mean([r['conditions'][name][metric] for r in rs if r['seed']==s]) for s in seeds])
                values[metric]=dict(row_mean=float(a.mean()),seed_mean=float(seed_means.mean()),
                    ci95=np.quantile(seed_means[draws].mean(1),[.025,.975]).tolist(),max=float(a.max()))
            values.update(history_flips=sum(r['conditions'][name]['history_flip'] for r in rs),
                original_flips=sum(r['conditions'][name]['original_flip'] for r in rs))
            out[name]=values
        result[arm]=dict(rows=len(rs),seed_clusters=len(seeds),conditions=out,max_cache_error=max(r['cache_error'] for r in rs))
    (ROOT/'policy-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--readouts',action='store_true');p.add_argument('--wave',type=int);p.add_argument('--summary',action='store_true');args=p.parse_args()
    if args.readouts:readouts()
    elif args.summary:summarize_policy()
    else:policy_wave(args.wave)
