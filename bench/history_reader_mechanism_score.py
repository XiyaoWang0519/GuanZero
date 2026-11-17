"""Score sealed reader mechanisms once on fresh whole-match clusters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from bench.filtered_history_collect import ACTOR, load_players, save_json
from bench.history_answer_probe import Reader, read_all, balanced_loss
from bench.history_answer_score import metrics
from bench.history_reader_mechanism import (
    ROOT, ARMS, HEAD_ARMS, QUERY_ARMS, SEEDS, SEED, CONTRASTS,
    DiagnosticHead, SlotHead, SlotReader, padded_batch, slot_read_all,
)
from bench.qknorm_match_ablation import sha


def bootstrap(probs, y, meta, prior, draw):
    correct=((probs>=.5)==y[None].astype(bool)).mean(0)
    tp=[];tn=[];pos=[];neg=[]
    for z in range(20):
        reps=sorted({r['replicate'] for r in meta if r['opponent']==z})
        assert len(reps)==8
        for rep in reps:
            ids=np.array([r['opponent']==z and r['replicate']==rep for r in meta])
            cells=[r for r,yes in zip(meta,ids) if yes]
            assert len(cells)==2 and {r['team'] for r in cells}=={0,1}
            assert len({r['seed'] for r in cells})==1
            tp.append((correct[ids]*y[ids]).sum(0));tn.append((correct[ids]*(1-y[ids])).sum(0))
            pos.append(y[ids].sum(0));neg.append((1-y[ids]).sum(0))
    def aggregate(a):
        x=np.array(a).reshape(20,8,-1)
        return (x[np.arange(20)[None,:,None],draw].sum(2)*prior[None,:,None]).sum(1)
    return .5*(aggregate(tp)/aggregate(pos)+aggregate(tn)/aggregate(neg)).mean(1)


def summarize(probs,y,w,meta,prior,draw):
    scores=[metrics(p,y,w) for p in probs]
    boot=bootstrap(probs,y,meta,prior,draw)
    return dict(mean={k:float(np.mean([s[k] for s in scores])) for k in
                      ('balanced_accuracy','balanced_bce','balanced_brier','accuracy')},
                seeds=scores,ci95=np.quantile(boot,[.025,.975]).tolist()),boot


def attention_stats(reader, data, ids):
    """Question-address mass and entropy, no labels used; average slots/heads."""
    entropy=[];mass=[];top=[];bos=[];qrows=[];outputs=[]
    with torch.no_grad():
        for chunk in ids.split(32):
            args=padded_batch(data,chunk)
            if isinstance(reader,Reader):
                query,memory,valid,mask,seats=args
                h=reader.heads;d=reader.depth
                q=reader.q_proj(query).unflatten(-1,(h,d))[:,:,None]
                k,v=reader.kv_proj(memory).chunk(2,-1)
                k=k.unflatten(-1,(h,d)).transpose(1,2)
                if reader.normalized:
                    q=torch.nn.functional.layer_norm(q,(d,))
                    k=torch.nn.functional.layer_norm(k,(d,))
                q=q.expand(-1,-1,12,-1)
                allowed=valid[:,None].expand(-1,12,-1)
                out=reader(query,memory,valid)[:,None]
            else:
                q,k,v,allowed=reader.attention(*args)
                d=reader.depth;mask=args[3];out=reader(*args)
            scores=(q@k.transpose(-1,-2))/d**.5
            prob=scores.masked_fill(~allowed[:,None],-torch.inf).softmax(-1)
            target=mask.repeat_interleave(6,1)[:,None].expand_as(prob)
            mass.append((prob*target).sum(-1).flatten())
            bos.append(prob[...,0].flatten())
            top.append(target.gather(-1,prob.argmax(-1,keepdim=True)).float().flatten())
            entropy.append((-(prob*prob.clamp_min(1e-30).log()).sum(-1)).flatten())
            qrows.append(q[:,:,0].flatten(1));outputs.append(out.flatten(1))
    q=torch.cat(qrows);features=torch.cat(outputs)
    # Pair consecutive matches only as a descriptive query-direction statistic.
    cos=torch.nn.functional.cosine_similarity(q[::2],q[1::2])
    return dict(target_mass=float(torch.cat(mass).mean()),target_top1=float(torch.cat(top).mean()),
                bos_mass=float(torch.cat(bos).mean()),entropy_nats=float(torch.cat(entropy).mean()),query_pair_cosine=float(cos.mean()),
                output_mean_norm=float(features.mean(0).norm()),
                output_rms_variation=float((features-features.mean(0)).square().sum(1).mean().sqrt()))


def evaluate(root: Path) -> None:
    torch.set_num_threads(1)
    choices=json.loads((root/'choices.json').read_text())
    assert not choices['test_labels_used'] and choices['dataset_sha']==sha(root/'dataset.pt')
    protocol=json.loads((root/'protocol.json').read_text())
    assert protocol['actor_sha']==sha(ACTOR)
    data=torch.load(root/'dataset.pt',weights_only=False)
    tr=torch.where(data['split']==0)[0];te=torch.where(data['split']==2)[0]
    meta=[data['metadata'][i] for i in te]
    assert len(te)==320 and all(32<=r['replicate']<40 for r in meta)
    y=data['y'][te].numpy();w=data['sample_weight'][te].numpy()
    prior=np.array(protocol['population']['weights'])
    draw=np.random.default_rng(SEED+99).integers(0,8,(5000,20,8))
    actor,_,identity=load_players(indices=[])
    original={k:v.clone() for k,v in actor.state_dict().items()}
    predictions={};results={};boots={};controls={};training={};selected={};mechanisms={}
    for arm in ARMS:
        full=[];swapped=[];train_scores=[];details=[];attn=[]
        for seed in SEEDS:
            info=choices['choices'][f'{arm}-{seed}'];folder=root/info['folder']
            assert info['checkpoint_sha']==sha(folder/'selected.pt')
            state=torch.load(folder/'selected.pt',weights_only=False)
            if arm in HEAD_ARMS:
                reader=Reader(actor,arm.startswith('norm'))
                head=DiagnosticHead(128,12,seed,arm.endswith('leaky'))
                def predict(features):
                    return head((features-state['scale']['mean'])/state['scale']['std']).sigmoid().detach().numpy()
                def features(ids,view='full'):
                    if view=='full':return read_all(reader,data,ids)
                    # The compact dataset stores only fresh-test shuffled memories.
                    with torch.no_grad():
                        return torch.cat([reader(*padded_batch(data,chunk,'shuffled')[:3]) for chunk in ids.split(32)])
            else:
                reader=SlotReader(actor,data,arm,seed);head=SlotHead(seed)
                def predict(features):return head(features).sigmoid().detach().numpy()
                def features(ids,view='full'):return slot_read_all(reader,data,ids,view)
            reader.load_state_dict(state['reader']);head.load_state_dict(state['head'])
            reader.eval().requires_grad_(False);head.eval().requires_grad_(False)
            with torch.no_grad():
                full.append(predict(features(te)));swapped.append(predict(features(te,'shuffled')))
                train_scores.append(metrics(predict(features(tr)),data['y'][tr].numpy(),data['sample_weight'][tr].numpy()))
            detail=json.loads((folder/'selected.json').read_text())
            # Directly test the selected diagnostic head's derivative to its
            # reader input, including examples with every hidden preactivation < 0.
            tf=features(tr).detach().requires_grad_(True)
            logits=head((tf-state['scale']['mean'])/state['scale']['std']) if arm in HEAD_ARMS else head(tf)
            value=balanced_loss(logits,data['y'][tr],data['sample_weight'][tr],data['prevalence'])
            value.backward()
            norms=tf.grad.flatten(1).norm(dim=1)
            detail['selected_head_input_gradient']=dict(zero_examples=int((norms==0).sum()),
                examples=len(tr),mean=float(norms.mean()),minimum=float(norms.min()),maximum=float(norms.max()))
            details.append(detail)
            if arm in HEAD_ARMS+QUERY_ARMS:attn.append(attention_stats(reader,data,te))
        predictions[arm]=np.stack(full);predictions[arm+'_shuffled']=np.stack(swapped)
        results[arm],boots[arm]=summarize(predictions[arm],y,w,meta,prior,draw)
        controls[arm],control_boot=summarize(predictions[arm+'_shuffled'],y,w,meta,prior,draw)
        delta=boots[arm]-control_boot
        controls[arm]['matched_minus_shuffled']=dict(
            mean=results[arm]['mean']['balanced_accuracy']-controls[arm]['mean']['balanced_accuracy'],
            ci95=np.quantile(delta,[.025,.975]).tolist())
        training[arm]=train_scores;selected[arm]=details
        if attn:mechanisms[arm]=attn
        print(json.dumps(dict(arm=arm,balanced_accuracy=results[arm]['mean']['balanced_accuracy'],
                              shuffled=controls[arm]['mean']['balanced_accuracy'])),flush=True)
    contrasts={}
    alpha=.05/len(CONTRASTS)
    for a,b in CONTRASTS:
        delta=boots[a]-boots[b]
        contrasts[f'{a}-minus-{b}']=dict(mean=results[a]['mean']['balanced_accuracy']-results[b]['mean']['balanced_accuracy'],
            ci95=np.quantile(delta,[.025,.975]).tolist(),
            familywise_ci=np.quantile(delta,[alpha/2,1-alpha/2]).tolist())
    baseline={}
    for norm in (False,True):
        baseline['normalized' if norm else 'native']=attention_stats(Reader(actor,norm),data,te)
    assert all(torch.equal(v,original[k]) for k,v in actor.state_dict().items())
    np.savez_compressed(root/'predictions.npz',**predictions,y=y,w=w)
    save_json(root/'summary.json',dict(results=results,controls=controls,contrasts=contrasts,
        train_scores=training,selected=selected,attention=mechanisms,baseline_attention=baseline,
        protocol_sha=sha(root/'protocol.json'),choices_sha=sha(root/'choices.json'),dataset_sha=sha(root/'dataset.pt'),
        scoring_source_sha=sha(__file__),actor_sha=sha(ACTOR),population_sha=identity['population_sha'],
        main_actor_unchanged=True,test_matches=320,test_clusters=160,bootstrap_draws=5000))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    evaluate(p.parse_args().root)
