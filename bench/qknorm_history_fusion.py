"""Read-only decomposition of frozen query/history fusion on saved stage features."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from bench.public_history_diverse import load_actor
from bench.public_history_analysis import cluster_weights
from bench.qknorm_history_stages import ROOT, POOL, ridge_predict, brier_probabilities
from bench.qknorm_match_summary import ENDPOINTS


def analyze() -> None:
    torch.set_num_threads(1)
    result={}
    with np.load(POOL/'raw/questions/public_answers.npz') as q:
        pi=(q['counts'][:,:,0]+.5)/33.5; freq=q['counts'][:,:,1]/32
    for arm in ENDPOINTS:
        parts=[]
        for w in range(16):
            with np.load(ROOT/f'wave-{w:02d}'/f'{arm}.npz') as f:
                parts.append({k:f[k] for k in f.files})
        d={k:np.concatenate([a[k] for a in parts]) for k in parts[0]}
        actor,record=load_actor(Path(f'.work/qknorm-short-ppo-2026-10-16/remote/runs/{arm}/endpoint.pt'))
        assert record['checkpoint_sha']==ENDPOINTS[arm]
        features={};check=0.
        query=torch.as_tensor(d['context'][:,:128],dtype=torch.float32)
        for view,reader in [('full',d['reader']),('round',d['context'][:,128:256])]:
            with torch.inference_mode():
                read=torch.as_tensor(reader,dtype=torch.float32)
                mixed=query+read
                norm=actor.attention_norm(mixed)
                residual=norm+actor.feed_forward(norm)
                state=actor.output_norm(residual)
            for name,value in [('reader',read),('mixed',mixed),('attention_norm',norm),
                               ('ff_residual',residual),('output_norm',state)]:
                features[name+'_'+view]=value.numpy().astype(np.float64)
            if view=='full':
                check=float(np.abs(state.numpy()-d['state']).max())
                assert check<1e-5
        train=d['wave']<8; test=~train;ids=d['opponent'][test];seeds=d['seed'][test]
        y=np.eye(4)[d['opponent'][train]]
        predictions={key:ridge_predict(x[train],y,x[test]) for key,x in features.items()}
        scores={key:brier_probabilities(p,pi,freq,ids) for key,p in predictions.items()}
        groups=np.array([np.flatnonzero(seeds==s) for s in np.unique(seeds)])
        weights=cluster_weights(ids[groups[:,0]],10000,np.random.default_rng(16102723))
        qw=np.random.default_rng(16102724).multinomial(256,np.full(256,1/256),size=10000)/256
        stages={}
        for name in ('reader','mixed','attention_norm','ff_residual','output_norm'):
            matrix=(scores[name+'_round']-scores[name+'_full'])[groups].mean(1)
            draws=np.sum((weights@matrix)*qw,1)
            stages[name]=dict(full_accuracy=float(np.mean(predictions[name+'_full'].argmax(1)==ids)),
                round_accuracy=float(np.mean(predictions[name+'_round'].argmax(1)==ids)),
                full_question_brier=float(scores[name+'_full'].mean()),round_question_brier=float(scores[name+'_round'].mean()),
                historical_gain=dict(estimate=float(matrix.mean()),ci95=np.quantile(draws,[.025,.975]).tolist()),
                calibration_columns_std_below_floor=int((features[name+'_full'][train].std(0)<.01).sum()))
        scales={}
        for name,value in [('query',query.numpy()),('reader',d['reader']),('reader_delta',d['reader_delta'])]:
            center=value[train].mean(0)
            rms=np.sqrt(np.mean((value[test]-center)**2,1))
            scales[name]=dict(centered_rms_median=float(np.median(rms)),centered_rms_mean=float(rms.mean()))
        result[arm]=dict(reconstruction_max_error=check,stages=stages,scales=scales)
    (ROOT/'fusion-summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':analyze()
