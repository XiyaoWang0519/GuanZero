"""Fixed-context offline history timeline; old weights, no collection or training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from bench.history_training_signal import ROOT
from bench.public_history_diverse import load_actor
from bench.qknorm_history_swap import ROOT as SWAP_ROOT, donor_bank, splice
from bench.qknorm_history_stages import POOL, ridge_predict, brier_probabilities
from bench.public_history_analysis import cluster_weights
from train.history_model import StreamBatch, DecisionInputs
from train.history_aux import next_token_loss


def extract(actor: Any, sample: dict, donor: tuple) -> dict:
    original=sample['original'];n=int((original[1]<3).sum())
    swapped=splice(original,donor,3,True)
    sb=StreamBatch.from_arrays([original,tuple(x[n:].copy() for x in original),swapped],'cpu')
    c=len(sample['cand'])
    inputs=DecisionInputs(sb,torch.arange(3),sb.lengths,
        torch.tensor(np.repeat(sample['obs'][None],3,0)),torch.full((3,),sample['seat']),
        torch.tensor(np.tile(sample['cand'],(3,1))),torch.arange(4)*c,one_decision_per_stream=True)
    memory=actor.encode_batch(sb)
    q=actor.private(inputs.obs.float())+actor.seat(inputs.seat)
    reader=actor._attend_encoded(q,memory,inputs.match_index,inputs.prefix,True)
    state=actor.decision_states(memory,inputs)
    table=actor._log_prob_table(inputs,memory,c).double();p=table.exp();p/=p.sum(1,keepdim=True)
    old=memory[0,1:n+1];mask=torch.tensor(original[0][:n,:4].argmax(1)%2!=sample['seat']%2)
    next_stats=next_token_loss(actor.next_logits(memory[0,:n]),torch.tensor(original[0][:n]),torch.ones(n,dtype=torch.bool))
    keys=actor.kv_proj(memory[:1]).chunk(2,-1)[0]
    heads,depth=actor.config.heads,actor.config.width//actor.config.heads
    qh=actor._query_heads(q[:1]);kh=actor._key_heads(keys.view(1,keys.shape[1],heads,depth))
    scores=torch.einsum('bhd,bshd->bhs',qh.double(),kh.double())/depth**.5
    weights=scores.softmax(-1);entropy=-(weights*weights.clamp_min(1e-300).log()).sum(-1)
    out=dict(memory=old[mask].mean(0).numpy(),reader=reader[0].numpy(),state=state[0].numpy(),
             round_reader=reader[1].numpy(),round_state=state[1].numpy(),
             swap_tv=float((p[0]-p[2]).abs().sum()/2),swap_flip=int(p[0].argmax()!=p[2].argmax()),
             full_round_tv=float((p[0]-p[1]).abs().sum()/2),
             next_type=float(next_stats['next_type_accuracy']),next_seat=float(next_stats['next_seat_accuracy']),
             next_loss=float(next_stats['next_loss']),attention_max=float(weights.max(-1).values.mean()),
             attention_normalized_entropy=float((entropy/np.log(memory.shape[1])).mean()),
             prior_round_attention=float(weights[:,:,1:n+1].sum(-1).mean()))
    assert all(np.isfinite(v).all() for v in out.values())
    return out


def collect(name: str) -> None:
    torch.set_num_threads(1)
    path=json.loads((ROOT/'checkpoints.json').read_text())[name]
    actor,meta=load_actor(Path(path));actor.causal_sdpa=True;actor.batched_match_attention=True
    rows=[]
    with torch.inference_mode():
        for context_arm in ('control','norm16'):
            bank=donor_bank(context_arm)
            lookup={(d['source'],d['wave'],d['env'],d['round'],d['window_start']):d for d in bank}
            for wave in range(8,12):
                folder=SWAP_ROOT/f'wave-{wave:02d}'
                records=json.loads((folder/'rows.json').read_text())
                samples=torch.load(folder/f'{context_arm}-samples.pt',weights_only=False)
                for r in (r for r in records if r['arm']==context_arm and r['round']==3):
                    d=lookup[(r['donor_source'],r['donor_wave'],r['donor_env'],r['round'],r['donor_window_start'])]
                    donor=tuple(x.copy() for x in d['arrays'])
                    if r['donor_seat_rotation']:donor[0][:,:4]=np.roll(donor[0][:,:4],1,1)
                    v=extract(actor,samples[r['index']],donor)
                    v.update(wave=wave,env=r['env'],seed=r['seed'],opponent=r['opponent'],context_arm=context_arm)
                    rows.append(v)
    assert len(rows)==256
    np.savez_compressed(ROOT/f'{name}-timeline.npz',**{k:np.asarray([r[k] for r in rows]) for k in rows[0]})
    (ROOT/f'{name}-timeline-meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(model=name,cases=len(rows),update=meta['update'])),flush=True)


def analyze() -> None:
    models=json.loads((ROOT/'checkpoints.json').read_text());result={};all_scores={}
    with np.load(POOL/'raw/questions/public_answers.npz') as f:
        pi=(f['counts'][:,:,0]+.5)/33.5;freq=f['counts'][:,:,1]/32
    for name in models:
        with np.load(ROOT/f'{name}-timeline.npz') as f:d={k:f[k] for k in f.files}
        train=d['wave']<10;test=~train;ids=d['opponent'][test];seeds=d['seed'][test]
        assert not set(d['seed'][train])&set(seeds)
        y=np.eye(4)[d['opponent'][train]];groups=np.array([np.flatnonzero(seeds==s) for s in np.unique(seeds)])
        assert len(groups)==32
        weights=cluster_weights(ids[groups[:,0]],10000,np.random.default_rng(17102751))
        qw=np.random.default_rng(17102752).multinomial(256,np.full(256,1/256),size=10000)/256
        predictions={k:ridge_predict(d[k][train],y,d[k][test]) for k in ('memory','reader','state','round_reader','round_state')}
        predictions['prior']=np.full((int(test.sum()),4),.25)
        scores={k:brier_probabilities(p,pi,freq,ids) for k,p in predictions.items()}
        values={}
        for stage,reference in [('memory','prior'),('reader','round_reader'),('state','round_state')]:
            matrix=(scores[reference]-scores[stage])[groups].mean(1);draws=np.sum((weights@matrix)*qw,1)
            values[stage]=dict(accuracy=float(np.mean(predictions[stage].argmax(1)==ids)),
                brier_gain=float(matrix.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())
        summary={k:float(np.mean(d[k])) for k in ('swap_tv','full_round_tv','next_type','next_seat','next_loss','attention_max',
                    'attention_normalized_entropy','prior_round_attention')}
        summary['swap_flips']=int(d['swap_flip'].sum())
        result[name]=dict(cases=256,calibration=128,heldout=128,test_seed_clusters=32,readouts=values,**summary)
        all_scores[name]=scores
    transitions={}
    for a,b in [('m6','m7'),('m7','pre74028'),('pre74028','fault74648'),('fault74648','fault75516'),('fault75516','fault77500'),('fault77500','early77748'),('early77748','post81344'),('pre74028','fault77500'),('fault77500','recover86800'),('pre74028','recover86800'),('recover86800','control')]:
        change={}
        for stage,reference in [('memory','prior'),('reader','round_reader'),('state','round_state')]:
            before=all_scores[a][reference]-all_scores[a][stage];after=all_scores[b][reference]-all_scores[b][stage]
            matrix=(after-before)[groups].mean(1);draws=np.sum((weights@matrix)*qw,1)
            change[stage]=dict(improvement_after_minus_before=float(matrix.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())
        transitions[a+'->'+b]=change
    (ROOT/'timeline-summary.json').write_text(json.dumps(dict(models=result,paired_transitions=transitions),indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--name');p.add_argument('--analyze',action='store_true');args=p.parse_args()
    if args.analyze:analyze()
    else:collect(args.name)
