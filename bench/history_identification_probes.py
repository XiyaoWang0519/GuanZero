"""Independent public-history sequence probes; every gameplay weight is frozen."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import gd

from bench.history_identification import ROOT, SEED, LENGTHS
from bench.filtered_history_collect import load_players, raw_profile, save_json
from bench.history_answer_probe import encode_layers
from bench.qknorm_match_ablation import sha
from train.history_population import weights_digest

ARMS=('raw_tokens','embedding','layer_1','layer_2','layer_3','layer_4','stream_norm','native')
SEEDS=(71,72,73)
RATES=(.0003,.001)
STEPS=1200
EVAL_LENGTHS=(1,3,10)


def seal(root: Path) -> None:
    protocol=dict(arms=ARMS,seeds=SEEDS,rates=RATES,steps=STEPS,batch=16,
        primary_length=3,selection='weighted validation identity CE, including step zero; learning rate separately per seed; test labels never select',
        objective='pool-weighted identity cross entropy; identity posterior scored on independent frozen question bank for predictive utility',
        sequence_probe='token MLP with LeakyReLU, then mean/max within public round x seat, group MLP, then mean/max over rounds per seat, 64-wide head',
        native_probe='four fixed own-hand query contexts from question roots 0..3; original frozen attention reads archive memory; 512->128->20 LeakyReLU classifier',
        lengths='one head trained with uniformly sampled 1/3/10-round histories; validation primary 3; all history lengths share heldout sequences',
        features='raw public tokens; frozen embedding, four encoder layers and final norm; per-match causal encoding before assembling archived history; no opponent-private inputs',
        restart='gameplay and encoder reset normally on match end; native archive readout concatenates per-match memories for the independent diagnostic only',
        normalization='training-token channel means/std; native training-history statistics; no test calibration',
        controls='zero initialized final classifier equals original pool prior; nonzero finite gradient audit after initialization; shuffled and same-ID history donors scored later',
        source=sha(__file__))
    path=root/'probe-protocol.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError('probe protocol already sealed with different settings')
    save_json(path,protocol)


@torch.inference_mode()
def prepare(root: Path) -> None:
    seal(root);torch.set_num_threads(1)
    folder=root/'features';folder.mkdir(exist_ok=True)
    if (folder/'complete.json').exists():return
    actor,_,identity=load_players(indices=[]);before=weights_digest(actor.state_dict())
    queries=[]
    for index in range(4):
        r=Path('.work/filtered-identity-oracle-2026-11-15')/f'root-{index:03d}'
        state=gd.MatchState.deserialize((r/'state.bin').read_bytes())
        observer=json.loads((r/'result.json').read_text())['meta']['observer']
        q=actor.private(torch.tensor(np.asarray(state.observation(observer))[None]))+actor.seat(torch.tensor([observer]))
        queries.append(actor.reader_input(q))
    query=torch.cat(queries)
    features={a:[] for a in ARMS};groups=[];ends=[];labels=[];reps=[];restart_counts=[]
    started=time.monotonic()
    for i,path in enumerate(sorted((root/'histories').glob('*/public.pt'))):
        complete=json.loads((path.parent/'complete.json').read_text())
        public=torch.load(path,weights_only=False)
        by_segment=[encode_layers(actor,a) for a in public['segments']]
        tensors=[[] for _ in range(7)];group=[];prefix_ends=[];total=0
        for record in public['rounds']:
            r=record['history_round'];seg=record['segment'];lo=record['start'];hi=record['end']
            arrays=public['segments'][seg];tokens=arrays[0][lo:hi]
            assert not tokens[:,150:158].any()
            tensors[0].append(torch.tensor(tokens,dtype=torch.float32))
            for level in range(6):tensors[level+1].append(by_segment[seg][level][lo+1:hi+1].clone())
            group.extend((4*r+tokens[:,:4].argmax(1)).tolist())
            total+=hi-lo;prefix_ends.append(total)
        for name,parts in zip(ARMS[:-1],tensors):features[name].append(torch.cat(parts))
        native=[]
        for length in EVAL_LENGTHS:
            selected=public['rounds'][:length];lasts={r['segment']:r['end'] for r in selected}
            memory=torch.cat([by_segment[s][-1][:end+1] for s,end in sorted(lasts.items())])[None]
            out=actor._attend_encoded(query,memory,torch.zeros(4,dtype=torch.long),
                torch.full((4,),memory.shape[1]-1,dtype=torch.long),False)
            native.append(out.reshape(-1).clone())
        features['native'].append(torch.stack(native))
        groups.append(torch.tensor(group,dtype=torch.long));ends.append(prefix_ends)
        labels.append(complete['z']);reps.append(complete['replicate']);restart_counts.append(complete['segments']-1)
        if (i+1)%80==0:print(json.dumps(dict(features=i+1,seconds=time.monotonic()-started)),flush=True)
    assert len(labels)==640
    meta=dict(groups=groups,ends=torch.tensor(ends),y=torch.tensor(labels),rep=torch.tensor(reps),
        split=torch.tensor([0 if r<20 else 1 if r<24 else 2 for r in reps]),
        prior=torch.tensor(identity['weights'],dtype=torch.float32),restarts=restart_counts)
    torch.save(meta,folder/'meta.pt')
    for name,x in features.items():torch.save(x,folder/f'{name}.pt')
    assert weights_digest(actor.state_dict())==before
    save_json(folder/'complete.json',dict(histories=len(labels),seconds=time.monotonic()-started,
        frozen_verified=True,restarts=sum(restart_counts),source_sha=sha(__file__),
        feature_hashes={name:sha(folder/f'{name}.pt') for name in ARMS}))


class Scale(nn.Module):
    def __init__(self, x: list, train: list, native: bool):
        super().__init__()
        # Only training histories enter these statistics. All supplied history
        # lengths are available at training; no validation or test calibration.
        values=torch.cat([x[i] for i in train])
        self.register_buffer('mean',values.mean(0))
        self.register_buffer('std',values.std(0).clamp_min(1e-4))
    def forward(self,x):return ((x-self.mean)/self.std).clamp(-20,20)


def grouped_pool(values: torch.Tensor, groups: torch.Tensor, batch: int) -> tuple:
    total=batch*40;dim=values.shape[-1]
    counts=values.new_zeros(total).index_add_(0,groups,values.new_ones(len(values)))
    sums=values.new_zeros(total,dim).index_add_(0,groups,values)
    maxima=values.new_full((total,dim),-torch.inf).scatter_reduce_(0,
        groups[:,None].expand(-1,dim),values,reduce='amax',include_self=True)
    maxima=torch.where(counts[:,None]>0,maxima,torch.zeros_like(maxima))
    return torch.cat((sums/counts[:,None].clamp_min(1),maxima),-1).reshape(batch,10,4,2*dim),counts.reshape(batch,10,4)


class Probe(nn.Module):
    def __init__(self, dimension: int, prior: torch.Tensor, native: bool, seed: int):
        super().__init__();torch.manual_seed(seed);self.native=native
        if native:
            self.hidden=nn.Sequential(nn.Linear(dimension,128),nn.LeakyReLU(.1))
            self.output=nn.Linear(128,20)
        else:
            self.token=nn.Sequential(nn.Linear(dimension,64),nn.LeakyReLU(.1),nn.Linear(64,64),nn.LeakyReLU(.1))
            self.group=nn.Sequential(nn.Linear(129,64),nn.LeakyReLU(.1))
            self.hidden=nn.Sequential(nn.Linear(513,64),nn.LeakyReLU(.1))
            self.output=nn.Linear(64,20)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():self.output.bias.copy_(prior.log())
    def forward(self,x,groups=None,batch=None):
        if self.native:return self.output(self.hidden(x))
        pooled,counts=grouped_pool(self.token(x),groups,batch)
        group=self.group(torch.cat((pooled,counts.log1p()[...,None]),-1))
        valid=counts>0
        mean=(group*valid[...,None]).sum(1)/valid.sum(1)[...,None].clamp_min(1)
        maximum=group.masked_fill(~valid[...,None],-torch.inf).max(1).values
        maximum=torch.where(valid.any(1)[...,None],maximum,torch.zeros_like(maximum))
        length=valid.any(-1).sum(1).float().log1p()[:,None]
        summary=torch.cat((mean.reshape(batch,-1),maximum.reshape(batch,-1),length),-1)
        return self.output(self.hidden(summary))


def batch_forward(model,scale,x,meta,indices,lengths):
    if model.native:
        slot={1:0,3:1,10:2}
        values=torch.stack([x[int(i)][slot[int(l)]] for i,l in zip(indices,lengths)])
        return model(scale(values))
    values=[];groups=[]
    for b,(i,length) in enumerate(zip(indices,lengths)):
        i=int(i);end=int(meta['ends'][i,int(length)-1])
        values.append(x[i][:end]);groups.append(meta['groups'][i][:end]+b*40)
    return model(scale(torch.cat(values)),torch.cat(groups),len(indices))


@torch.no_grad()
def predict(model,scale,x,meta,indices,length,batch=32):
    return torch.cat([batch_forward(model,scale,x,meta,indices[i:i+batch],
        [length]*len(indices[i:i+batch])).softmax(-1) for i in range(0,len(indices),batch)])


def fit_one(task: tuple) -> dict:
    root_string,name,seed,rate=task;root=Path(root_string);torch.set_num_threads(1)
    dest=root/'probes'/f'{name}-{seed}-{rate}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'selected.json').exists():return json.loads((dest/'selected.json').read_text())
    meta=torch.load(root/'features'/'meta.pt',weights_only=False)
    x=torch.load(root/'features'/f'{name}.pt',weights_only=False)
    tr=torch.where(meta['split']==0)[0];va=torch.where(meta['split']==1)[0]
    scale=Scale(x,tr.tolist(),name=='native')
    model=Probe(x[0].shape[-1],meta['prior'],name=='native',seed)
    opt=torch.optim.AdamW(model.parameters(),lr=rate,weight_decay=.001)
    weights=meta['prior'][meta['y'][tr]];weights/=weights.sum()
    wv=meta['prior'][meta['y'][va]];wv/=wv.sum()
    rng=torch.Generator().manual_seed(SEED+seed)
    curve=[];best=float('inf');gradient_checks=[];started=time.monotonic()
    for step in range(STEPS+1):
        if step%100==0:
            model.eval()
            pv=predict(model,scale,x,meta,va,3)
            value=float((-pv[torch.arange(len(va)),meta['y'][va]].clamp_min(1e-30).log()*wv).sum())
            curve.append(dict(step=step,validation_ce=value))
            if value<best:
                best=value;chosen=step
                torch.save(dict(model=model.state_dict(),scale=scale.state_dict()),dest/'selected.pt')
        if step==STEPS:break
        model.train();ids=tr[torch.multinomial(weights,16,replacement=True,generator=rng)]
        lengths=[EVAL_LENGTHS[int(j)] for j in torch.randint(3,(16,),generator=rng)]
        opt.zero_grad(set_to_none=True)
        logits=batch_forward(model,scale,x,meta,ids,lengths)
        loss=F.cross_entropy(logits,meta['y'][ids]);loss.backward()
        if step in (1,10,100,500,1000):
            norms={k:float(p.grad.norm()) for k,p in model.named_parameters() if p.grad is not None}
            assert all(np.isfinite(v) for v in norms.values())
            gradient_checks.append(dict(step=step,norms=norms))
        nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    selected=torch.load(dest/'selected.pt',weights_only=False);model.load_state_dict(selected['model']);model.eval()
    with torch.no_grad():
        predictions=torch.stack([predict(model,scale,x,meta,torch.arange(len(x)),length) for length in EVAL_LENGTHS])
        train_ce=[]
        for j in range(3):
            p=predictions[j,tr];train_ce.append(float((-p[torch.arange(len(tr)),meta['y'][tr]].clamp_min(1e-30).log()*weights).sum()))
    np.save(dest/'predictions.npy',predictions.numpy())
    record=dict(name=name,seed=seed,rate=rate,selected_step=chosen,validation_ce=best,
        train_ce=train_ce,curve=curve,gradient_checks=gradient_checks,
        parameters=sum(p.numel() for p in model.parameters()),seconds=time.monotonic()-started,source_sha=sha(__file__))
    save_json(dest/'selected.json',record)
    return {k:record[k] for k in ('name','seed','rate','selected_step','validation_ce','seconds')}


def train(root:Path,workers:int):
    tasks=[(str(root),name,seed,rate) for name in ARMS for seed in SEEDS for rate in RATES]
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('spawn')) as pool:
        jobs=[pool.submit(fit_one,t) for t in tasks]
        for done,future in enumerate(as_completed(jobs),1):print(json.dumps(dict(done=done,total=len(tasks),**future.result())),flush=True)
    choices={}
    for name in ARMS:
        for seed in SEEDS:
            folders=[root/'probes'/f'{name}-{seed}-{rate}' for rate in RATES]
            chosen=min(folders,key=lambda p:json.loads((p/'selected.json').read_text())['validation_ce'])
            choices[f'{name}-{seed}']=dict(folder=str(chosen),sha=sha(chosen/'selected.pt'))
    save_json(root/'probe-choices.json',choices)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    if a.command=='prepare':prepare(a.root)
    else:train(a.root,a.workers)
