"""Complement pooled probes with token-wise decoding on the same frozen layers.

A mean pool is lossy. Decode public action type at each prior opponent token,
then aggregate argmax types into the exact same previous-round fact questions.
No actor weight updates, no new gameplay, and whole-match data splits retained.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from bench.history_answer_probe import ROOT,SEED,SEEDS,source_rows,load_players,encode_layers,Standardize,Head
from bench.filtered_history_collect import save_json
from bench.qknorm_match_ablation import sha

STEPS=1000


def prepare(root: Path) -> None:
    path=root/'token-protocol.json'
    assert not (root/'summary.json').exists(),'declare supplement before heldout evaluation'
    if not path.exists():save_json(path,dict(reason='mean pooling can discard decodable token information; inspect whole memory bank before interpreting layer failure',
        planned_after='pooled-probe and reader validation curves, before reading heldout scores',
        scope='six frozen layers, 3 seeds, 1000 steps, batch 256, Adam 0.001, class-balanced action-type CE',
        targets='11 public Play action types; only previous-round opponent tokens',
        aggregation='token argmax, then any token in family for each relative opponent; no tuned aggregation thresholds',
        selection='validation token CE every 100 steps; match split inherited; heldout tokens never used for selection',
        source_sha=sha(__file__)))
    if (root/'token-dataset.pt').exists():return
    torch.set_num_threads(1);data=torch.load(root/'dataset.pt',weights_only=False);actor,_,_=load_players(indices=[])
    rows=source_rows();xs={};labels=[];owners=[];which=[]
    with torch.no_grad():
        for i,row in enumerate(rows):
            s=row['sample'];a=s['original'];layers=encode_layers(actor,a)
            for rel,target in enumerate(((s['seat']+1)%4,(s['seat']+3)%4)):
                mask=(a[1]==2)&(a[2]==3)&(a[0][:,:4].argmax(1)==target);ids=np.flatnonzero(mask)
                y=a[0][mask,112:125].argmax(1);assert (y<11).all()
                labels.extend(y);owners.extend([i]*len(ids));which.extend([rel]*len(ids))
                for n,layer in enumerate(layers):
                    name='embedding' if n==0 else 'stream_norm' if n==5 else f'layer_{n}'
                    xs.setdefault(name,[]).append(layer[torch.tensor(ids+1)].clone())
    owner=torch.tensor(owners);split=data['split'][owner];w=data['sample_weight'][owner]
    # Give each match and relative opponent the same total weight, regardless of number of turns.
    group=owner*2+torch.tensor(which);w=w/torch.bincount(group)[group]
    labels=torch.tensor(labels,dtype=torch.long);tr=split==0
    prior=torch.zeros(11).scatter_add_(0,labels[tr],w[tr]);prior/=prior.sum();assert (prior>0).all()
    result=dict(features={k:torch.cat(v) for k,v in xs.items()},y=labels,owner=owner,relative=torch.tensor(which),
        split=split,sample_weight=w,prevalence=prior)
    torch.save(result,root/'token-dataset.pt')
    save_json(root/'token-data-manifest.json',dict(tokens=len(labels),split_tokens=[int((split==s).sum()) for s in (0,1,2)],
        class_training_prevalence=prior.tolist(),data_sha=sha(root/'token-dataset.pt'),source_sha=sha(__file__)))
    print(json.dumps(json.loads((root/'token-data-manifest.json').read_text())),flush=True)


def loss(logits,y,w,prior):
    values=F.cross_entropy(logits,y,reduction='none')/(11*prior[y])
    return (values*w).sum()/w.sum()


def worker(task):
    root,name,seed=task;root=Path(root);torch.set_num_threads(1)
    dest=root/'token-probes'/f'{name}-{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'selected.json').exists():return json.loads((dest/'selected.json').read_text())
    data=torch.load(root/'token-dataset.pt',weights_only=False);x=data['features'][name]
    tr=torch.where(data['split']==0)[0];va=torch.where(data['split']==1)[0]
    scale=Standardize(x[tr],data['sample_weight'][tr]);xx=scale(x);head=Head(128,11,seed)
    opt=torch.optim.Adam(head.parameters(),lr=.001);rng=torch.Generator().manual_seed(SEED+seed)
    best=float('inf');curve=[]
    for step in range(STEPS+1):
        if step%100==0:
            with torch.no_grad():
                vl=float(loss(head(xx[va]),data['y'][va],data['sample_weight'][va],data['prevalence']))
                tl=float(loss(head(xx[tr]),data['y'][tr],data['sample_weight'][tr],data['prevalence']))
            curve.append(dict(step=step,train_loss=tl,validation_loss=vl))
            if vl<best:
                best=vl;chosen=step;torch.save(dict(head=head.state_dict(),scale=scale.state_dict()),dest/'selected.pt')
        if step==STEPS:break
        ids=tr[torch.randint(len(tr),(256,),generator=rng)];opt.zero_grad(set_to_none=True)
        value=loss(head(xx[ids]),data['y'][ids],data['sample_weight'][ids],data['prevalence'])
        value.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
    record=dict(name=name,seed=seed,step=chosen,validation_loss=best,curve=curve,source_sha=sha(__file__))
    save_json(dest/'selected.json',record);return {k:record[k] for k in ('name','seed','step','validation_loss')}


def train(root: Path,workers: int):
    data=torch.load(root/'token-dataset.pt',weights_only=False);names=list(data['features']);del data
    tasks=[(str(root),n,s) for n in names for s in SEEDS]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(worker,t) for t in tasks]
        for f in as_completed(futures):print(json.dumps(f.result()),flush=True)
    save_json(root/'token-choices.json',{f'{name}-{seed}':sha(root/'token-probes'/f'{name}-{seed}'/'selected.pt') for name in names for seed in SEEDS})


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train']);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.command=='prepare':prepare(a.root)
    else:train(a.root,a.workers)

if __name__=='__main__':main()
