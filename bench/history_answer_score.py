"""Evaluate sealed historical-fact diagnostic heads on heldout match clusters."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from bench.history_answer_probe import ROOT,SEED,SEEDS,ARMS,FACTS,load_players,Head,Reader,probe_features,read_all
from bench.filtered_history_collect import save_json
from bench.qknorm_match_ablation import sha


def restore_head(state: dict,width: int,targets: int):
    head=Head(width,targets,0);head.load_state_dict(state['head']);head.eval()
    def predict(x):
        with torch.no_grad():return head((x-state['scale']['mean'])/state['scale']['std']).sigmoid().numpy()
    return predict


def metrics(prob: np.ndarray,y: np.ndarray,w: np.ndarray) -> dict:
    p=np.clip(prob,1e-7,1-1e-7);correct=(p>=.5)==y.astype(bool)
    pos=w[:,None]*y;neg=w[:,None]*(1-y)
    tp=(correct*pos).sum(0)/pos.sum(0);tn=(correct*neg).sum(0)/neg.sum(0)
    bce=.5*((-np.log(p)*pos).sum(0)/pos.sum(0)+(-np.log1p(-p)*neg).sum(0)/neg.sum(0))
    brier=.5*(((1-p)**2*pos).sum(0)/pos.sum(0)+(p*p*neg).sum(0)/neg.sum(0))
    return dict(balanced_accuracy=float((.5*(tp+tn)).mean()),balanced_bce=float(bce.mean()),balanced_brier=float(brier.mean()),
        per_target_balanced_accuracy=(.5*(tp+tn)).tolist(),accuracy=float((correct*w[:,None]).sum()/w.sum()/y.shape[1]))


def bootstrap_scores(probs: np.ndarray,y: np.ndarray,metadata: list,prior: np.ndarray,draw: np.ndarray) -> np.ndarray:
    # Three optimization seeds are averaged, not counted as new test examples.
    correct=((probs>=.5)==y[None].astype(bool)).mean(0)
    groups=np.array([r['opponent']*8+r['replicate']-24 for r in metadata])
    shape=(160,y.shape[1]);tp=np.zeros(shape);tn=np.zeros(shape);pos=np.zeros(shape);neg=np.zeros(shape)
    for cell in range(160):
        ids=groups==cell;assert ids.sum()==2,'one paired team orientation per seed cluster'
        tp[cell]=(correct[ids]*y[ids]).sum(0);tn[cell]=(correct[ids]*(1-y[ids])).sum(0)
        pos[cell]=y[ids].sum(0);neg[cell]=(1-y[ids]).sum(0)
    def aggregate(a):
        a=a.reshape(20,8,-1)
        return (a[np.arange(20)[None,:,None],draw].sum(2)*prior[None,:,None]).sum(1)
    return .5*(aggregate(tp)/aggregate(pos)+aggregate(tn)/aggregate(neg)).mean(1)


def summarize_predictions(probs,y,w,metadata,prior,draw):
    scores=[metrics(p,y,w) for p in probs];boot=bootstrap_scores(probs,y,metadata,prior,draw)
    return dict(mean={k:float(np.mean([s[k] for s in scores])) for k in ('balanced_accuracy','balanced_bce','balanced_brier','accuracy')},
                seed_scores=scores,balanced_accuracy_ci95=np.quantile(boot,[.025,.975]).tolist()),boot


def aggregate_types(predicted: np.ndarray,owners: np.ndarray,relatives: np.ndarray,rows: np.ndarray,target_names: list) -> np.ndarray:
    answers=np.zeros((len(rows),len(target_names)))
    lookup={int(r):i for i,r in enumerate(rows)}
    family={name:kinds for name,kinds in FACTS}
    for k,name in enumerate(target_names):
        _,relative,fact=name.split('_',2);r=0 if relative=='1' else 1
        selected=(relatives==r)&np.isin(predicted,family[fact])
        for owner in np.unique(owners[selected]):answers[lookup[int(owner)],k]=1
    return answers


def evaluate(root: Path) -> None:
    torch.set_num_threads(1)
    choices=json.loads((root/'choices.json').read_text());token_choices=json.loads((root/'token-choices.json').read_text())
    assert choices['dataset_sha']==sha(root/'dataset.pt') and not choices['test_labels_used']
    for info in choices['choices'].values():assert info['checkpoint_sha']==sha(root/info['folder']/'selected.pt')
    data=torch.load(root/'dataset.pt',weights_only=False);test=torch.where(data['split']==2)[0];train=torch.where(data['split']==0)[0]
    y=data['y'][test].numpy();w=data['sample_weight'][test].numpy();meta=[data['metadata'][i] for i in test.tolist()]
    prior=np.array(json.loads((root/'protocol.json').read_text())['population']['weights'])
    draw=np.random.default_rng(SEED+13).integers(0,8,(5000,20,8));predictions={};results={};boots={};controls={};train_scores={}
    # All learned probes were selected on validation before any test score is computed here.
    with torch.no_grad():
        for name in ('current',*data['features']):
            x=probe_features(data,name);parts=[];shuffled=[]
            xx=x.clone()
            if name!='current':xx[:,256:]=x[torch.tensor(data['donors']),256:]
            for seed in SEEDS:
                state=torch.load(root/'probes'/f'{name}-{seed}'/'selected.pt',weights_only=False)
                predict=restore_head(state,512,data['y'].shape[1]);parts.append(predict(x[test]));shuffled.append(predict(xx[test]))
            key='pooled/'+name;predictions[key]=np.stack(parts)
            results[key],boots[key]=summarize_predictions(predictions[key],y,w,meta,prior,draw)
            controls[key]={'shuffled_feature':summarize_predictions(np.stack(shuffled),y,w,meta,prior,draw)[0]}
    actor,_,identity=load_players(indices=[])
    reader_details={}
    for arm in ARMS:
        full=[];round_only=[];shuffled=[];tr=[];details=[]
        for seed in SEEDS:
            info=choices['choices'][f'{arm}-{seed}'];folder=root/info['folder']
            state=torch.load(folder/'selected.pt',weights_only=False);reader=Reader(actor,arm.startswith('norm')).eval().requires_grad_(False)
            reader.load_state_dict(state['reader']);predict=restore_head(state,128,data['y'].shape[1])
            full.append(predict(read_all(reader,data,test)))
            round_only.append(predict(read_all(reader,data,test,'round')))
            shuffled.append(predict(read_all(reader,data,test,'shuffled')))
            tr.append(metrics(predict(read_all(reader,data,train)),data['y'][train].numpy(),data['sample_weight'][train].numpy()))
            details.append(json.loads((folder/'selected.json').read_text()))
        key='reader/'+arm;predictions[key]=np.stack(full);results[key],boots[key]=summarize_predictions(predictions[key],y,w,meta,prior,draw)
        controls[key]={name:summarize_predictions(np.stack(p),y,w,meta,prior,draw)[0] for name,p in (('round',round_only),('shuffled',shuffled))}
        train_scores[key]=tr;reader_details[arm]=details
    token_data=torch.load(root/'token-dataset.pt',weights_only=False);tt=torch.where(token_data['split']==2)[0]
    token_stats={}
    for name,x in token_data['features'].items():
        facts=[];accuracies=[];classes=[]
        for seed in SEEDS:
            folder=root/'token-probes'/f'{name}-{seed}';assert sha(folder/'selected.pt')==token_choices[f'{name}-{seed}']
            state=torch.load(folder/'selected.pt',weights_only=False);head=Head(128,11,0);head.load_state_dict(state['head']);head.eval()
            with torch.no_grad():
                predicted=head((x[tt]-state['scale']['mean'])/state['scale']['std']).argmax(1).numpy()
            truth=token_data['y'][tt].numpy();tw=token_data['sample_weight'][tt].numpy()
            accuracies.append(float(np.average(predicted==truth,weights=tw)))
            classes.append([float(np.average((predicted==truth)[truth==k],weights=tw[truth==k])) if (truth==k).any() else None for k in range(11)])
            facts.append(aggregate_types(predicted,token_data['owner'][tt].numpy(),token_data['relative'][tt].numpy(),test.numpy(),data['target_names']))
        key='token/'+name;predictions[key]=np.stack(facts);results[key],boots[key]=summarize_predictions(predictions[key],y,w,meta,prior,draw)
        token_stats[name]=dict(type_accuracy_mean=float(np.mean(accuracies)),type_accuracy_by_seed=accuracies,
                              class_accuracy_by_seed=classes,macro_type_recall=float(np.mean([v for row in classes for v in row if v is not None])),
                              heldout_class_support=np.bincount(truth,minlength=11).tolist())
    comparisons={}
    for better,base in (('reader/norm','reader/native'),('reader/native','reader/frozen'),('reader/norm','reader/norm_frozen'),
                        ('token/stream_norm','pooled/stream_norm'),('pooled/native_reader','pooled/current')):
        diff=boots[better]-boots[base]
        comparisons[f'{better} - {base}']=dict(balanced_accuracy_gain=results[better]['mean']['balanced_accuracy']-results[base]['mean']['balanced_accuracy'],
                                              ci95=np.quantile(diff,[.025,.975]).tolist())
    result=dict(actor_sha=identity['actor_sha'],data_sha=sha(root/'dataset.pt'),choices_sha=sha(root/'choices.json'),
        token_choices_sha=sha(root/'token-choices.json'),source_sha=sha(__file__),test_matches=len(test),test_seed_clusters=160,
        targets=data['target_names'],results=results,controls=controls,training_scores=train_scores,token_statistics=token_stats,
        comparisons=comparisons,reader_training_details=reader_details,
        metric='macro per-fact balanced accuracy; constant prediction gives 50%; means over three optimization seeds',
        uncertainty='5000 bootstrap draws, resample eight paired seed clusters within each of 20 fixed types; preserves both teams; optimization seeds not independent data',
        limitation='Pooled probes have explicit previous-round/seat masks; token decoding has per-token type supervision and deterministic aggregation; reader training only has match-level binary facts. These are different diagnostic readouts, not equal-capacity policy comparisons.')
    np.savez_compressed(root/'predictions.npz',**{k.replace('/','__'):p for k,p in predictions.items()},labels=y,
        opponent=np.array([r['opponent'] for r in meta]),replicate=np.array([r['replicate'] for r in meta]))
    save_json(root/'summary.json',result)
    print(json.dumps(dict(results={k:dict(v['mean'],ci95=v['balanced_accuracy_ci95']) for k,v in results.items()},comparisons=comparisons,
                         token_statistics=token_stats),indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args();evaluate(args.root)
