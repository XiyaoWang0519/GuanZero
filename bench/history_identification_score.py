"""History posterior calibration and independent-question predictive recovery."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from bench.history_identification import ROOT,LENGTHS,logmean
from bench.history_identification_probes import ARMS,SEEDS,EVAL_LENGTHS
from bench.filtered_history_collect import save_json
from bench.qknorm_match_ablation import sha


def posterior(loglik:np.ndarray,prior:np.ndarray,temperature:float=1) -> np.ndarray:
    x=temperature*loglik+np.log(prior);x-=x.max(-1,keepdims=True)
    p=np.exp(x);return p/p.sum(-1,keepdims=True)


def question_kernels(root:Path) -> dict:
    path=root/'question-kernels.npz'
    if path.exists():return dict(np.load(path))
    sources=Path('.work/filtered-identity-oracle-2026-11-15')
    c={k:[] for k in ('full_action','action_type')};s={k:[] for k in c}
    for folder in sorted(sources.glob('root-*')):
        data=np.load(folder/'probabilities.npz')
        pairs={'full_action':(data['full_blocks'][:4].mean(0),data['full_blocks'][4:].mean(0)),
            'action_type':(data['type_prob'][:,1:129].mean(1),data['type_prob'][:,129:].mean(1))}
        for k,(a,b) in pairs.items():
            c[k].append((a@b.T+b@a.T)/2)
            s[k].append((a@a.T+b@b.T)/2)
    result={f'{k}_cross':np.stack(c[k]) for k in c}
    result.update({f'{k}_square':np.stack(s[k]) for k in c})
    np.savez_compressed(path,**result);return result


def gain_matrix(q:np.ndarray,labels:np.ndarray,prior:np.ndarray,cross:np.ndarray,square=None) -> np.ndarray:
    """Histories x independent questions, no sampled target actions."""
    square=cross if square is None else square
    baseline=np.einsum('i,qij,j->q',prior,square,prior)[None,:]-2*np.einsum('i,qiz->zq',prior,cross)[labels]
    forecast=np.einsum('hi,qij,hj->hq',q,square,q)-2*np.einsum('hi,qih->hq',q,cross[:,:,labels])
    result=baseline-forecast
    return np.where(np.abs(result)<1e-14,0.,result)


def history_rows(root:Path,particles=32):
    records=[];all_logs=[];exact_logs=[];halves=[];metadata=[]
    for z in range(20):
        for rep in range(20,32):
            folder=root/'histories'/f'z{z:02d}-r{rep:03d}'
            data=np.load(folder/f'likelihood-{particles}.npz')
            meta=json.loads((folder/f'likelihood-{particles}.json').read_text())
            pooled=[]
            for r,n in enumerate(data['valid']):
                ll,_=logmean(data['raw_logp'][r,:,:n],data['logweight'][r,:n]);pooled.append(ll)
            pooled=np.array(pooled);exact=np.array([r['exact'] for r in meta['rounds']])
            all_logs.append(np.stack([np.zeros(20) if L==0 else pooled[:L].sum(0) for L in LENGTHS]))
            exact_logs.append(np.stack([np.zeros(20) if L==0 else (pooled*exact[:,None])[:L].sum(0) for L in LENGTHS]))
            halves.append(np.stack([np.zeros((2,20)) if L==0 else data['round_loglik'][:L].sum(0) for L in LENGTHS]))
            records.append((z,rep));metadata.append(meta)
    return np.array(records),np.array(all_logs),np.array(exact_logs),np.array(halves),metadata


def calibration(logs,labels,prior):
    weights=prior[labels];weights/=weights.sum();curve=[]
    for temp in (0.,.01,.03,.1,.3,.5,1.,2.):
        p=posterior(logs,prior,temp)
        score=float(-weights@np.log(p[np.arange(len(labels)),labels].clip(1e-30)))
        curve.append(dict(temperature=temp,ce=score))
    return min(curve,key=lambda r:r['ce'])['temperature'],curve


def bootstrap_weights(labels,prior,repetitions=3000):
    rng=np.random.default_rng(20261115991)
    h=np.zeros((repetitions,len(labels)))
    for z in range(20):
        ids=np.flatnonzero(labels==z);draw=rng.choice(ids,size=(repetitions,len(ids)))
        rows=np.repeat(np.arange(repetitions),len(ids))
        np.add.at(h,(rows,draw.ravel()),prior[z]/len(ids))
    # Stage stratification inherited from the independent question experiment.
    q=np.zeros((repetitions,192))
    for stage in range(3):
        ids=np.arange(stage,192,3);draw=rng.choice(ids,size=(repetitions,len(ids)))
        rows=np.repeat(np.arange(repetitions),len(ids))
        np.add.at(q,(rows,draw.ravel()),1/192)
    return h,q


def summarize_posterior(q,labels,prior,kernels,hboot,qboot,oracle_gains):
    weights=prior[labels];weights/=weights.sum()
    ce=-np.log(q[np.arange(len(labels)),labels].clip(1e-30))
    brier=((q-np.eye(20)[labels])**2).sum(1)
    result=dict(identity_ce=float(weights@ce),identity_brier=float(weights@brier),
        identity_accuracy=float(weights@(q.argmax(1)==labels)),
        identity_entropy=float(weights@(-q*np.log(q.clip(1e-30))).sum(1)))
    for label in ('action_type','full_action'):
        gains=gain_matrix(q,labels,prior,kernels[f'{label}_cross'])
        draws=((hboot@gains)*qboot).sum(1)
        mean=float(weights@gains.mean(1))
        known=oracle_gains[label]
        denom=((hboot@known)*qboot).sum(1)
        honest=gain_matrix(q,labels,prior,kernels[f'{label}_cross'],kernels[f'{label}_square'])
        result[label]=dict(gain=mean,ci95=np.quantile(draws,[.025,.975]).tolist(),
            recovered_fraction=mean/float(weights@known.mean(1)),
            fraction_ci95=np.quantile(draws/denom,[.025,.975]).tolist(),
            heldout_forecast_gain=float(weights@honest.mean(1)),
            history_only_ci95=np.quantile((hboot@gains).mean(1),[.025,.975]).tolist())
    return result


def precision_audit(root,prior,kernels):
    paths=sorted((root/'histories').glob('*/likelihood-128.npz'))
    if len(paths)!=40:return dict(complete=False,completed=len(paths),planned=40)
    labels=[];sets={32:[],128:[]};split=[]
    for path in paths:
        meta=json.loads(path.with_suffix('.json').read_text());labels.append(meta['z'])
        for particles in (32,128):
            data=np.load(path.parent/f'likelihood-{particles}.npz');pooled=[]
            for r,n in enumerate(data['valid']):
                ll,_=logmean(data['raw_logp'][r,:,:n],data['logweight'][r,:n]);pooled.append(ll)
            sets[particles].append(np.stack([posterior(np.array(pooled)[:L].sum(0),prior) for L in (1,3,10)]))
            if particles==128:
                split.append(np.stack([posterior(data['round_loglik'][:L].sum(0),prior) for L in (1,3,10)]))
    labels=np.array(labels);weights=prior[labels];weights/=weights.sum()
    small=np.stack(sets[32]);large=np.stack(sets[128]);split=np.stack(split)
    hboot,qboot=bootstrap_weights(labels,prior)
    result=dict(complete=True,histories=40,selection='first two test replicates 24/25 per identity, chosen before their high-precision outcomes',lengths={})
    for j,L in enumerate((1,3,10)):
        tv=.5*np.abs(small[:,j]-large[:,j]).sum(1)
        half_tv=.5*np.abs(split[:,j,0]-split[:,j,1]).sum(1)
        a=gain_matrix(small[:,j],labels,prior,kernels['action_type_cross'])
        b=gain_matrix(large[:,j],labels,prior,kernels['action_type_cross'])
        draws=((hboot@(b-a))*qboot).sum(1)
        result['lengths'][str(L)]=dict(small_gain=float(weights@a.mean(1)),large_gain=float(weights@b.mean(1)),
            difference=float(weights@(b-a).mean(1)),difference_ci95=np.quantile(draws,[.025,.975]).tolist(),
            posterior_tv_mean=float(weights@tv),posterior_tv_max=float(tv.max()),
            large_split_tv_mean=float(weights@half_tv),large_split_tv_max=float(half_tv.max()),
            large_split_top1_agreement=float(weights@(split[:,j,0].argmax(1)==split[:,j,1].argmax(1))))
    return result


def score(root:Path,include_probes:bool=False) -> dict:
    protocol=json.loads((root/'protocol.json').read_text());prior=np.array(protocol['identity']['weights'])
    records,logs,exact,halves,metadata=history_rows(root)
    validation=records[:,1]<24;test=~validation;labels=records[test,0]
    kernels=question_kernels(root);hboot,qboot=bootstrap_weights(labels,prior)
    oracle_gains={k:gain_matrix(np.eye(20)[labels],labels,prior,kernels[f'{k}_cross']) for k in ('action_type','full_action')}
    result=dict(protocol=protocol,calibration={},curves={},precision={},probe_selection={},
        baseline=summarize_posterior(np.tile(prior,(len(labels),1)),labels,prior,kernels,hboot,qboot,oracle_gains),
        true_identity=summarize_posterior(np.eye(20)[labels],labels,prior,kernels,hboot,qboot,oracle_gains))
    rng=np.random.default_rng(20261115973)
    donor=[]
    for label in labels:
        w=prior.copy();w[label]=0;w/=w.sum();z=int(rng.choice(20,p=w))
        donor.append(int(rng.choice(np.flatnonzero(labels==z))))
    same_donor=np.arange(len(labels))
    for label in range(20):
        ids=np.flatnonzero(labels==label);same_donor[ids]=np.roll(ids,1)
    for name,data in [('likelihood',logs),('resolved_rounds',exact)]:
        result['curves'][name]={};result['curves'][name+'_calibrated']={};result['curves'][name+'_shuffled']={}
        result['curves'][name+'_same_identity']={}
        result['calibration'][name]={}
        for j,L in enumerate(LENGTHS):
            temp,curve=calibration(data[validation,j],records[validation,0],prior)
            result['calibration'][name][str(L)]=dict(temperature=temp,curve=curve)
            native=posterior(data[test,j],prior);calibrated=posterior(data[test,j],prior,temp)
            for key,p in [(name,native),(name+'_calibrated',calibrated),(name+'_shuffled',calibrated[donor]),
                          (name+'_same_identity',calibrated[same_donor])]:
                result['curves'][key][str(L)]=summarize_posterior(p,labels,prior,kernels,hboot,qboot,oracle_gains)
            a=posterior(halves[test,j,0],prior);b=posterior(halves[test,j,1],prior)
            tv=.5*np.abs(a-b).sum(1)
            result['precision'][str(L)]=dict(split_posterior_tv_mean=float(np.mean(tv)),
                split_posterior_tv_quantiles=np.quantile(tv,[.5,.9,.95,1]).tolist(),
                split_top1_agreement=float(np.mean(a.argmax(1)==b.argmax(1))))
    if include_probes:
        choices=json.loads((root/'probe-choices.json').read_text())
        meta=torch.load(root/'features'/'meta.pt',weights_only=False)
        chosen_rows=[int(torch.where((meta['y']==z)&(meta['rep']==r))[0][0]) for z,r in records[test]]
        for arm in ARMS:
            predictions=[];selection=[]
            for seed in SEEDS:
                folder=Path(choices[f'{arm}-{seed}']['folder'])
                predictions.append(np.load(folder/'predictions.npy')[:,chosen_rows])
                selection.append(json.loads((folder/'selected.json').read_text()))
            p=np.mean(predictions,axis=0);result['curves'][arm]={};result['curves'][arm+'_shuffled']={}
            result['probe_selection'][arm]=selection
            for j,L in enumerate(EVAL_LENGTHS):
                for key,pred in [(arm,p[j]),(arm+'_shuffled',p[j,donor])]:
                    result['curves'][key][str(L)]=summarize_posterior(pred,labels,prior,kernels,hboot,qboot,oracle_gains)
    inexact=[];exact_count=0
    for meta in metadata:
        data=np.load(root/'histories'/f"z{meta['z']:02d}-r{meta['replicate']:03d}"/'likelihood-32.npz')
        for r,entry in enumerate(meta['rounds']):
            if entry['exact']:exact_count+=1
            else:inexact.extend(data['ess'][r].ravel())
    result['likelihood_audit']=dict(histories=len(metadata),test_histories=int(test.sum()),rounds=len(metadata)*10,
        exact_rounds=exact_count,inexact_ess_quantiles=np.quantile(inexact,[0,.25,.5,.75,1]).tolist(),
        limitation='Finite Monte Carlo reference is not certified Bayes-optimal; exact-round-only is a public feature reference, not full posterior.',
        sources=sorted(set(m['likelihood_source_sha'] for m in metadata)))
    result['precision_extension']=precision_audit(root,prior,kernels)
    result['sources']={p:sha(p) for p in (__file__,'bench/history_identification.py','bench/history_identification_probes.py')}
    save_json(root/('summary.json' if include_probes else 'likelihood-summary.json'),result)
    for name in ['likelihood','likelihood_calibrated','resolved_rounds_calibrated']+(list(ARMS) if include_probes else []):
        for L,r in result['curves'][name].items():
            print(json.dumps(dict(arm=name,length=L,accuracy=r['identity_accuracy'],ce=r['identity_ce'],
                gain=r['action_type']['gain'],ci=r['action_type']['ci95'],recovery=r['action_type']['recovered_fraction'])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--probes',action='store_true')
    a=p.parse_args();score(a.root,a.probes)
