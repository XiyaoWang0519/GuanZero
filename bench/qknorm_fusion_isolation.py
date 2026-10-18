"""Offline factorial query/normalization isolation; frozen weights and saved contexts."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from bench.public_history_diverse import load_actor
from bench.public_history_analysis import cluster_weights
from bench.qknorm_history_stages import POOL, ROOT as STAGE_ROOT, brier_probabilities
from bench.qknorm_match_summary import ENDPOINTS

ROOT=Path('.work/qknorm-fusion-isolation-2026-10-16')


def normalization_parts(x: np.ndarray, eps: float) -> dict[str,np.ndarray]:
    mean=x.mean(1,keepdims=True)
    centered=x-mean
    scale=np.sqrt(np.mean(centered**2,axis=1,keepdims=True)+eps)
    return dict(mixed=x,centered=centered,unit=centered/scale,mean=mean,scale=scale)


def ridge(x: np.ndarray, y: np.ndarray, test: np.ndarray, floor: float=.01) -> np.ndarray:
    mean=x.mean(0);scale=np.maximum(x.std(0),floor)
    z=(x-mean)/scale;t=(test-mean)/scale
    coef=z.T@np.linalg.solve(z@z.T+len(z)*np.eye(len(z)),y-y.mean(0))
    p=np.maximum(y.mean(0)+t@coef,1e-4)
    return p/p.sum(1,keepdims=True)


def features_for_view(query: np.ndarray, reader: np.ndarray, gamma: np.ndarray,
                      beta: np.ndarray, eps: float) -> dict[str,np.ndarray]:
    p=normalization_parts(query+reader,eps)
    affine=p['unit']*gamma+beta
    # Fixed reference scaling is invertible; only removes mean, never row amplitude.
    return dict(reader=reader,mixed=p['mixed'],centered=p['centered'],unit=p['unit'],
        affine=affine,unit_with_mean=np.concatenate((p['unit'],p['mean']),1),
        unit_with_scale=np.concatenate((p['unit'],p['scale']),1),
        unit_with_both=np.concatenate((p['unit'],p['mean'],p['scale']),1),
        inverse_then_subtract=(p['unit']*p['scale']+p['mean'])-query)


def analyze() -> None:
    torch.set_num_threads(1);result={}
    with np.load(POOL/'raw/questions/public_answers.npz') as f:
        pi=(f['counts'][:,:,0]+.5)/33.5;freq=f['counts'][:,:,1]/32
    for arm in ENDPOINTS:
        arrays=[]
        for w in range(16):
            with np.load(STAGE_ROOT/f'wave-{w:02d}'/f'{arm}.npz') as f:
                arrays.append({k:f[k] for k in f.files})
        d={k:np.concatenate([x[k] for x in arrays]) for k in arrays[0]}
        train=d['wave']<8;test=~train;ids=d['opponent'][test];seeds=d['seed'][test]
        q=d['context'][:,:128];constant=np.broadcast_to(q[train].mean(0),q.shape)
        actor,meta=load_actor(Path(f'.work/qknorm-short-ppo-2026-10-16/remote/runs/{arm}/endpoint.pt'))
        assert meta['checkpoint_sha']==ENDPOINTS[arm]
        gamma=actor.attention_norm.weight.detach().numpy().astype(float)
        beta=actor.attention_norm.bias.detach().numpy().astype(float);eps=actor.attention_norm.eps
        features={};parity=0.;recovery=0.
        for setting,query in [('actual',q),('fixed_query',constant)]:
            for view,read in [('full',d['reader']),('round',d['context'][:,128:256])]:
                values=features_for_view(query,read,gamma,beta,eps)
                recovery=max(recovery,float(np.max(np.abs(values['inverse_then_subtract']-read))))
                if setting=='actual':
                    with torch.inference_mode():
                        native=actor.attention_norm(torch.tensor(query+read,dtype=torch.float32)).numpy()
                    parity=max(parity,float(np.max(np.abs(native-values['affine']))))
                for name,value in values.items():features[f'{setting}/{name}/{view}']=value
        assert parity<1e-5 and recovery<1e-10
        y=np.eye(4)[d['opponent'][train]]
        groups=np.array([np.flatnonzero(seeds==s) for s in np.unique(seeds)])
        weights=cluster_weights(ids[groups[:,0]],10000,np.random.default_rng(16102731))
        qw=np.random.default_rng(16102732).multinomial(256,np.full(256,1/256),size=10000)/256
        outcomes={};saved={}
        for floor in (.01,1e-6):
            predictions={key:ridge(x[train],y,x[test],floor) for key,x in features.items()}
            scores={key:brier_probabilities(p,pi,freq,ids) for key,p in predictions.items()}
            prior=brier_probabilities(np.full((int(test.sum()),4),.25),pi,freq,ids)
            values={}
            for key in sorted(k for k in features if k.endswith('/full')):
                partner=key[:-4]+'round'
                matrix=(scores[partner]-scores[key])[groups].mean(1)
                draws=np.sum((weights@matrix)*qw,1)
                vsprior=(prior-scores[key])[groups].mean(1)
                prior_draws=np.sum((weights@vsprior)*qw,1)
                values[key[:-5]]=dict(full_accuracy=float(np.mean(predictions[key].argmax(1)==ids)),
                    full_brier=float(scores[key].mean()),round_brier=float(scores[partner].mean()),
                    history_gain=dict(estimate=float(matrix.mean()),ci95=np.quantile(draws,[.025,.975]).tolist()),
                    full_over_prior=dict(estimate=float(vsprior.mean()),ci95=np.quantile(prior_draws,[.025,.975]).tolist()))
            # Paired diagnostics: center removal vs scale removal, and q fluctuation.
            contrasts={}
            for name,a,b in [('centering_cost','actual/mixed','actual/centered'),
                             ('row_scaling_cost','actual/centered','actual/unit'),
                             ('affine_cost','actual/unit','actual/affine'),
                             ('query_variation_cost_mixed','fixed_query/mixed','actual/mixed'),
                             ('query_variation_cost_normalized','fixed_query/affine','actual/affine')]:
                def gain(key: str) -> np.ndarray:return scores[key+'/round']-scores[key+'/full']
                matrix=(gain(a)-gain(b))[groups].mean(1)
                draws=np.sum((weights@matrix)*qw,1)
                contrasts[name]=dict(estimate=float(matrix.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())
            outcomes[str(floor)]=dict(conditions=values,contrasts=contrasts)
            if floor==.01:
                saved={key.replace('/','__'):p for key,p in predictions.items()}
        np.savez_compressed(ROOT/f'{arm}-predictions.npz',**saved,ids=ids,seeds=seeds)
        result[arm]=dict(native_ln_max_error=parity,reconstructed_reader_max_error=recovery,
                        gamma_min_abs=float(np.abs(gamma).min()),calibration=256,test=256,
                        independent_test_seeds=128,outcomes=outcomes)
    (ROOT/'factorial-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':analyze()
