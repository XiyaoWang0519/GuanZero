"""Heldout, public-information probes for frozen u195450/filtered snapshots.

Regularization uses validation matches only. Test scores are paired by seed,
with two seat orientations and within-match decision captures clustered.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from bench.filtered_history_collect import ROOT,SEED,save_json
from bench.qknorm_match_ablation import sha

ALPHAS=np.array([.001,.01,.1,1,10,100,1000,10000,1e6,1e12])


def probabilities(scores: np.ndarray) -> np.ndarray:
    p=np.maximum(scores,1e-7)
    return p/p.sum(1,keepdims=True)


def brier(p: np.ndarray,y: np.ndarray) -> np.ndarray:
    return ((p-y)**2).sum(1)


def weights_for(z: np.ndarray,prior: np.ndarray) -> np.ndarray:
    counts=np.bincount(z,minlength=len(prior))
    assert (counts>0).all()
    return prior[z]/counts[z]


def fit_probe(x: np.ndarray,y: np.ndarray,split: np.ndarray,z: np.ndarray,prior: np.ndarray):
    train=split==0;val=split==1;test=split==2
    w=weights_for(z[train],prior);w=w*train.sum();wv=weights_for(z[val],prior)
    mean=np.average(x[train],axis=0,weights=w)
    std=np.sqrt(np.average((x[train]-mean)**2,axis=0,weights=w));std=np.maximum(std,1e-6)
    xx=(x-mean)/std;target=np.average(y[train],axis=0,weights=w)
    design=xx[train]*np.sqrt(w[:,None]);label=(y[train]-target)*np.sqrt(w[:,None])
    eigen,vectors=np.linalg.eigh(design.T@design);eigen=np.maximum(eigen,0)
    rhs=vectors.T@(design.T@label);vx=xx[val]@vectors;tx=xx[test]@vectors
    best=None
    for alpha in ALPHAS:
        coef=rhs/(eigen[:,None]+alpha)
        pv=probabilities(vx@coef+target);score=float(wv@brier(pv,y[val]))
        if best is None or score<best[0]:
            best=(score,float(alpha),probabilities(tx@coef+target))
    return best[2],dict(alpha=best[1],validation_brier=best[0],features=x.shape[1],
        train=train.sum().item(),validation=val.sum().item(),test=test.sum().item())


def cluster_matrix(values: np.ndarray,z: np.ndarray,rep: np.ndarray) -> np.ndarray:
    return np.array([[values[(z==zz)&(rep==rr)].mean() for rr in range(24,32)] for zz in range(20)])


def interval(values: np.ndarray,z: np.ndarray,rep: np.ndarray,prior: np.ndarray,coverage=.95) -> dict:
    a=cluster_matrix(values,z,rep);assert np.isfinite(a).all()
    rng=np.random.default_rng(SEED+701);draw=rng.integers(0,8,(10000,20,8))
    bootstrap=(a[np.arange(20)[None,:,None],draw].mean(2)*prior).sum(1)
    tail=(1-coverage)/2
    return dict(gain=float(prior@a.mean(1)),interval=np.quantile(bootstrap,[tail,1-tail]).tolist(),coverage=coverage)


def load_data(root: Path) -> dict:
    folders=sorted((root/'collection').glob('b*-r*'));assert len(folders)==16
    parts=[]
    for folder in folders:
        complete=json.loads((folder/'complete.json').read_text());assert complete['complete']
        parts.append(dict(np.load(folder/'features.npz')))
    data={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert len(set(zip(data['opponent'],data['replicate'],data['team'])))==1280
    assert (data['target_index']>data['prefix']).all()
    return data


def analyze(root: Path) -> None:
    protocol=json.loads((root/'protocol.json').read_text());prior=np.array(protocol['identity']['weights'])
    all_data=load_data(root);mask=all_data['round']==3
    d={k:v[mask] for k,v in all_data.items()};z=d['opponent'];rep=d['replicate']
    split=np.where(rep<20,0,np.where(rep<24,1,2));test=split==2
    y=np.eye(4)[d['label']];identity=np.eye(20)[z]
    common=np.c_[d['context'],d['current_profile']]
    designs={'current':common,'raw':np.c_[common,d['raw_profile']],
        'memory':np.c_[common,d['memory_opponent']],
        'reader':np.c_[common,d['reader']],'state':np.c_[common,d['state']],
        'memory_mean':np.c_[common,d['memory_mean']],'memory_last':np.c_[common,d['memory_last']],
        'reader_delta':np.c_[common,d['reader_delta']],'state_delta':np.c_[common,d['state_delta']],
        'type_oracle':np.c_[common,identity]}
    # A matched-dimensional negative control; no test labels enter the shuffle.
    rng=np.random.default_rng(SEED+702);permuted=d['raw_profile'].copy()
    for ss in range(3):
        for threshold in (0,20,50):
            for team in (0,1):
                rows=np.flatnonzero((split==ss)&(d['threshold']==threshold)&(d['team']==team))
                permuted[rows]=permuted[rng.permutation(rows)]
    designs['shuffled_raw']=np.c_[common,permuted]
    results={};preds={};identity_preds={};fit={}
    for name,x in designs.items():
        p,meta=fit_probe(x,y,split,z,prior);pi,mi=fit_probe(x,identity,split,z,prior)
        preds[name]=p;identity_preds[name]=pi;fit[name]=dict(behavior=meta,identity=mi)
        print(json.dumps(dict(probe=name,alpha=meta['alpha'],validation_brier=meta['validation_brier'])),flush=True)
    target=y[test];zt=z[test];rt=rep[test];w=weights_for(zt,prior)
    base=brier(preds['current'],target);base_id=brier(identity_preds['current'],identity[test])
    for name,p in preds.items():
        score=brier(p,target);idi=brier(identity_preds[name],identity[test]);coverage=.9875 if name in ('raw','memory','reader','state') else .95
        results[name]=dict(brier=float(w@score),gain_vs_current=interval(base-score,zt,rt,prior,coverage),
            log_loss=float(w@(-np.log(p[np.arange(len(p)),d['label'][test]]))),
            accuracy=float(w@(p.argmax(1)==d['label'][test])),
            identity_brier=float(w@idi),identity_gain_vs_current=interval(base_id-idi,zt,rt,prior),
            by_threshold={str(t):interval((base-score)[d['threshold'][test]==t],zt[d['threshold'][test]==t],
                rt[d['threshold'][test]==t],prior) for t in (0,20,50)})
    prior_behavior=np.average(y[split==0],axis=0,weights=weights_for(z[split==0],prior))
    constant=np.tile(prior_behavior,(test.sum(),1))
    result=dict(primary_round=3,rows=int(len(z)),heldout_rows=int(test.sum()),heldout_seed_clusters=160,
        heldout_matches=320,current_context_features=common.shape[1],
        constant_brier=float(w@brier(constant,target)),probes=results,fit=fit,
        native=dict(full_round_tv_mean=float(all_data['full_round_tv'].mean()),full_round_flips=int(all_data['full_round_flip'].sum()),
            decisions=int(len(all_data['past'])),cache_error_max=float(all_data['cache_error'].max()),
            cache_tv_max=float(all_data['cache_tv'].max()),cache_error_over_1e4=int((all_data['cache_error']>1e-4).sum()),
            cache_flips=int(all_data['cache_flip'].sum()),
            reader_mean_norm=float(np.linalg.norm(d['reader'].mean(0))),
            reader_deviation_median=float(np.median(np.linalg.norm(d['reader']-d['reader'].mean(0),axis=1))),
            state_mean_norm=float(np.linalg.norm(d['state'].mean(0)))),
        source_sha=sha(__file__),protocol_sha=sha(root/'protocol.json'),
        limitation='Linear probes measure decodable correlations, not all information or causal use; synthetic balanced homogeneous-opponent teams; 3 prior rounds')
    np.savez_compressed(root/'heldout-predictions.npz',**preds,labels=target,opponent=zt,replicate=rt,
                        team=d['team'][test],threshold=d['threshold'][test])
    # Public-only identity posteriors use three previous rounds, once per match.
    first=d['threshold']==0;old={k:v[first] for k,v in d.items()};zo=old['opponent'];ro=old['replicate']
    so=np.where(ro<20,0,np.where(ro<24,1,2));po={};old_meta={}
    for name,key in (('old_raw','raw_profile'),('old_memory','memory_opponent'),('old_reader','reader')):
        if name=='old_reader':continue # Reader includes current own-hand query: not a public-only observer.
        po[name],old_meta[name]=fit_probe(old[key],np.eye(20)[zo],so,zo,prior)
    np.savez_compressed(root/'posterior.npz',**po,opponent=zo[so==2],replicate=ro[so==2],team=old['team'][so==2])
    result['old_public_posterior_fit']=old_meta
    result['old_public_identity']={name:dict(brier=float(weights_for(zo[so==2],prior)@brier(p,np.eye(20)[zo[so==2]])),
        prior_brier=float(weights_for(zo[so==2],prior)@brier(np.tile(prior,(len(p),1)),np.eye(20)[zo[so==2]])),
        accuracy=float(weights_for(zo[so==2],prior)@(p.argmax(1)==zo[so==2])),
        tv_from_prior=float(weights_for(zo[so==2],prior)@(.5*np.abs(p-prior).sum(1)))) for name,p in po.items()}
    save_json(root/'probe-summary.json',result)
    print(json.dumps(dict(primary={k:results[k]['gain_vs_current'] for k in ('raw','memory','reader','state')},
                         old_public_identity=result['old_public_identity'],native=result['native']),indent=2),flush=True)


def questions(root: Path) -> None:
    prior=np.array(json.loads((root/'protocol.json').read_text())['identity']['weights'])
    counts=np.load(root/'questions.npz')['counts'];post=np.load(root/'posterior.npz');z=post['opponent'];rep=post['replicate']
    calibrate=(counts[:,:,0]+.5)/(counts[:,:,0].sum(-1,keepdims=True)+1.5)
    heldout=counts[:,:,1]/counts[:,:,1].sum(-1,keepdims=True)
    constant=np.einsum('z,qzk->qk',prior,calibrate)
    def loss(p):return (p*p).sum(-1)-2*(p*heldout[:,z]).sum(-1)+1
    base=loss(constant[:,None]);predictions={'oracle':calibrate[:,z]}
    for name in ('old_raw','old_memory'):predictions[name]=np.einsum('hz,qzk->qhk',post[name],calibrate)
    w=weights_for(z,prior);results={};rng=np.random.default_rng(SEED+703)
    for name,p in predictions.items():
        gain=base-loss(p);boot=[]
        # Cluster histories by same dealt seed and bootstrap independent questions.
        cells=np.stack([[gain[:,(z==zz)&(rep==rr)].mean(1) for rr in range(24,32)] for zz in range(20)],axis=1)
        # cells: replicate x opponent x question
        assert cells.shape==(8,20,128)
        for _ in range(10000):
            ss=rng.integers(0,8,(20,8));qq=rng.integers(0,128,128)
            byq=(cells[ss,np.arange(20)[:,None]].mean(1)*prior[:,None]).sum(0)
            boot.append(byq[qq].mean())
        results[name]=dict(brier=float((loss(p)*w).sum(1).mean()),gain=float((gain*w).sum(1).mean()),
                           ci95=np.quantile(boot,[.025,.975]).tolist())
    # Matched calibration/evaluation differences: noise-corrected squared heterogeneity.
    a=counts[:,:,0]/32;b=counts[:,:,1]/32
    ac=a-np.einsum('z,qzk->qk',prior,a)[:,None];bc=b-np.einsum('z,qzk->qk',prior,b)[:,None]
    hetero=np.einsum('z,qzk,qzk->q',prior,ac,bc)
    result=dict(questions=128,calibration_draws=32,evaluation_draws=32,prior_brier=float((base*w).sum(1).mean()),
        predictions=results,mean_cross_half_behavior_heterogeneity=float(hetero.mean()),
        source_sha=sha(__file__),limitation='fresh identical opening deals, three public action categories; calibration counts noisy; not all strategic differences')
    save_json(root/'question-summary.json',result);print(json.dumps(result,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['analyze','questions']);p.add_argument('--root',type=Path,default=ROOT)
    a=p.parse_args();globals()[a.command](a.root)

if __name__=='__main__':main()
