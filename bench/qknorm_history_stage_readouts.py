"""Matched-width standalone stage readouts, companion to conditional diagnostics."""
from __future__ import annotations
import json
import numpy as np
from bench.qknorm_history_stages import ROOT, POOL, STAGES, ridge_predict, seed_permutation, brier_probabilities
from bench.qknorm_match_summary import ENDPOINTS
from bench.public_history_analysis import cluster_weights


def analyze() -> None:
    summary={}
    with np.load(POOL/'raw/questions/public_answers.npz') as q:
        pi=(q['counts'][:,:,0]+.5)/33.5
        freq=q['counts'][:,:,1]/32
    for arm in ENDPOINTS:
        waves=[]
        for w in range(16):
            with np.load(ROOT/f'wave-{w:02d}'/f'{arm}.npz') as f:
                waves.append({k:f[k] for k in f.files})
        d={k:np.concatenate([w[k] for w in waves]) for k in waves[0]}
        train=d['wave']<8; test=~train
        stages={key:d[key] for key in STAGES}
        stages['reader_round']=d['context'][:,128:256]
        stages['state_round']=d['state']-d['state_delta']
        stages['query']=d['context'][:,:128]
        y=np.eye(4)[d['opponent'][train]]; ids=d['opponent'][test]; seeds=d['seed'][test]
        predictions={key:ridge_predict(x[train],y,x[test]) for key,x in stages.items()}
        prior=np.broadcast_to(y.mean(0),(int(test.sum()),4))
        predictions['prior']=prior
        cp=seed_permutation(d['seed'][train],16102721)
        tp=seed_permutation(seeds,16102722)
        for key in ('memory_mean','memory_opponent','memory_last','reader','state'):
            x=stages[key]
            predictions[key+'_shuffle']=ridge_predict(x[train][cp],y,x[test][tp])
        scores={key:brier_probabilities(p,pi,freq,ids) for key,p in predictions.items()}
        unique=np.unique(seeds);groups=np.array([np.flatnonzero(seeds==s) for s in unique])
        weights=cluster_weights(ids[groups[:,0]],10000,np.random.default_rng(16102723))
        qw=np.random.default_rng(16102724).multinomial(256,np.full(256,1/256),size=10000)/256
        stats={}
        for key,p in predictions.items():
            stats[key]=dict(accuracy=float(np.mean(p.argmax(1)==ids)),
                identity_brier=float(np.mean(np.sum((p-np.eye(4)[ids])**2,1))),
                question_brier=float(scores[key].mean()))
        comparisons={}
        for name,left,right in [('memory_opponent_over_prior','memory_opponent','prior'),
            ('memory_mean_over_prior','memory_mean','prior'),('memory_last_over_prior','memory_last','prior'),
            ('reader_over_round','reader','reader_round'),('state_over_round','state','state_round'),
            ('reader_delta_over_prior','reader_delta','prior'),('state_delta_over_prior','state_delta','prior'),
            ('raw_over_prior','raw_profile','prior')]+[(k+'_over_shuffle',k,k+'_shuffle') for k in
                ('memory_mean','memory_opponent','memory_last','reader','state')]:
            matrix=(scores[right]-scores[left])[groups].mean(1)
            draws=np.sum((weights@matrix)*qw,1)
            comparisons[name]=dict(estimate=float(matrix.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),
                ci99=np.quantile(draws,[.005,.995]).tolist())
        summary[arm]=dict(stages=stats,contrasts=comparisons)
        np.savez_compressed(ROOT/f'{arm}-standalone-predictions.npz',**predictions,ids=ids,seeds=seeds)
    (ROOT/'standalone-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':analyze()
