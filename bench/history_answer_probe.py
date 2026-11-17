"""Public-history fact retrieval, with frozen layer probes and isolated reader A/B.

Only diagnostic copies learn. No PPO, gameplay policy optimization, or mainline
checkpoint writes. Existing filtered trajectories supply public past facts.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from bench.filtered_history_collect import ROOT as SOURCE, ACTOR, load_players, save_json
from bench.qknorm_match_ablation import sha
from train.history_attention import project, finish
from train.history_model import StreamBatch, sinusoidal

ROOT=Path('.work/history-answer-2026-11-14')
SEED=20261114081
FACTS=(('bomb',(8,9,10)),('pair',(2,)),('full_house',(4,)),('straight',(5,)),('tube',(6,)),('plate',(7,)))
ARMS=('frozen','native','norm_frozen','norm')
PROBE_STEPS=1000
READER_STEPS=2000
SEEDS=(71,72,73)
RATES=(.0003,.001)


def past_facts(arrays: tuple,round_index: int,seat: int) -> tuple[np.ndarray,np.ndarray]:
    """Exact public labels and count features from the immediately previous round."""
    tokens,rounds,phases=arrays
    if tokens[:,150:158].any():raise ValueError('private flags in public history')
    labels=[];counts=[]
    for target in ((seat+1)%4,(seat+3)%4):
        mask=(rounds==round_index-1)&(phases==3)&(tokens[:,:4].argmax(1)==target)
        types=tokens[mask,112:125].sum(0,dtype=np.int64)
        assert mask.any(),'previous opponent play must be observed'
        counts.extend(types.tolist())
        labels.extend([int(types[list(kinds)].sum()>0) for _,kinds in FACTS])
    return np.array(labels,np.float32),np.array(counts,np.float32)


def source_rows(source: Path=SOURCE) -> list:
    rows=[]
    for folder in sorted((source/'collection').glob('b*-r*')):
        meta=json.loads((folder/'complete.json').read_text())
        samples=torch.load(folder/'samples.pt',map_location='cpu',weights_only=False)
        for sample in samples:
            if sample['round']!=3 or sample['threshold']!=0:continue
            cell=meta['cells'][sample['env']]
            rows.append(dict(sample=sample,**cell,folder=folder.name))
    rows.sort(key=lambda r:(r['opponent'],r['replicate'],r['team']))
    assert len(rows)==1280 and len(set((r['opponent'],r['replicate'],r['team']) for r in rows))==1280
    return rows


def seal_protocol(root: Path) -> None:
    root.mkdir(parents=True,exist_ok=True)
    path=root/'protocol.json'
    if path.exists():return
    original=json.loads((SOURCE/'protocol.json').read_text())
    assert sha(ACTOR)==original['identity']['actor_sha']
    protocol=dict(actor=str(ACTOR),actor_sha=sha(ACTOR),source=str(SOURCE),source_protocol_sha=sha(SOURCE/'protocol.json'),
        population=original['identity'],sample='one round-3 threshold-0 decision per match, all 1280 matches',
        target='each relative opponent played a combination family in immediately previous round; public actions only',
        facts=FACTS,target_selection='retain family only if both seats have training weighted prevalence in [0.05,0.95]; no validation/test selection',
        split={'train':[0,20],'validation':[20,24],'test':[24,32]},
        layers='raw counts; embedding; public encoder layers 1..4; final stream norm; native cached reader and state',
        pooling='mean previous-round play tokens separately for each relative opponent; this grants explicit public position/seat pooling to probes',
        common_context='frozen own query and state from current-round-only public prefix',
        probe=dict(architecture='common context + standardized layer feature -> 64 ReLU -> multi-label logits',
                   steps=PROBE_STEPS,batch=64,lr=.001,seeds=SEEDS),
        reader=dict(arms=ARMS,steps=READER_STEPS,batch=32,rates=RATES,seeds=SEEDS,
                    trainable='q_proj, both halves of kv_proj, out_proj, plus diagnostic head; encoder/private query/action head frozen',
                    normalization='per-head projected Q/K LayerNorm with no affine parameters; no reset of pretrained projections',
                    head='train-only standardized reader output -> 64 ReLU -> multi-label logits',
                    optimizer='Adam, no weight decay, clip global norm 1; fresh optimizer in all arms',
                    selection='validation balanced BCE chooses checkpoint every 200 steps and rate; heldout labels scored only after all choices sealed'),
        controls='current-round-only inputs and donor old history with own current suffix/query fixed; donor within split and same team, excluding paired deal',
        objective='mean per-label class-balanced BCE, with class/sample weights from training matches only',
        primary='macro balanced accuracy and balanced BCE on heldout matches, 160 paired seed clusters; report 3 training seeds separately',
        budget='CPU only, <=4 worker processes, one torch thread; 3 seeds x 2 rates x 4 arms x 2000 steps maximum, frozen probes 3 x 1000 per layer',
        limitation='fact retrieval is an artificial dense-supervision diagnostic, not strategic adaptation or playing-strength acceptance',
        sources={p:sha(p) for p in (__file__,'train/history_model.py','train/history_attention.py','bench/filtered_history_collect.py')},
        raw_hashes={str(p):sha(p) for p in sorted((SOURCE/'collection').glob('*/samples.pt'))})
    save_json(path,protocol)


def encode_layers(actor,arrays: tuple) -> list[torch.Tensor]:
    sb=StreamBatch.from_arrays([arrays],'cpu')
    embed=actor.public(sb.tokens.float())+actor.round_embedding(sb.rounds.clamp(0,actor.config.max_rounds-1))+actor.phase_embedding(sb.phases.clamp(0,3))
    x=torch.cat((actor.bos,embed),1)
    x=x+sinusoidal(x.shape[1],actor.config.width,x.device,x.dtype)
    layers=[x]
    for layer in actor.stream.layers:
        q,k,v=project(layer,x)
        x=finish(layer,x,F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=0.))
        layers.append(x)
    if actor.stream.norm is not None:x=actor.stream.norm(x)
    layers.append(actor.stream_norm(x))
    torch.testing.assert_close(layers[-1],actor.encode_batch(sb),rtol=0,atol=0)
    return [x[0] for x in layers]


def explicit_pool(encoded: torch.Tensor,arrays: tuple,round_index: int,seat: int) -> np.ndarray:
    tokens,rounds,phases=arrays
    return np.concatenate([encoded[1:][torch.tensor((rounds==round_index-1)&(phases==3)&(tokens[:,:4].argmax(1)==target))].mean(0).numpy()
                           for target in ((seat+1)%4,(seat+3)%4)])


def donor_indices(rows: list) -> np.ndarray:
    rng=np.random.default_rng(SEED);out=np.arange(len(rows))
    for lo,hi in ((0,20),(20,24),(24,32)):
        for team in (0,1):
            ids=np.array([i for i,r in enumerate(rows) if lo<=r['replicate']<hi and r['team']==team])
            order=rng.permutation(ids);out[order]=np.roll(order,1)
    assert all((rows[i]['opponent'],rows[i]['replicate'])!=(rows[j]['opponent'],rows[j]['replicate']) for i,j in enumerate(out))
    return out


def shifted_history(receiver: dict,donor: dict) -> tuple:
    a=receiver['original'];b=donor['original'];rnd=receiver['round']
    old=tuple(x[b[1]<donor['round']].copy() for x in b)
    shift=(receiver['seat']-donor['seat'])%4
    old[0][:,:4]=np.roll(old[0][:,:4],shift,axis=1)
    # All captures are round 3; donor retains authentic previous-round ordering.
    current=tuple(x[a[1]==rnd].copy() for x in a)
    return tuple(np.concatenate((x,y)) for x,y in zip(old,current))


def prepare(root: Path) -> None:
    seal_protocol(root)
    if (root/'dataset.pt').exists():return
    torch.set_num_threads(1);rows=source_rows();donors=donor_indices(rows)
    actor,_,_=load_players(indices=[]);features={};mem=[];round_mem=[];swap_mem=[];labels=[];queries=[];common=[];meta=[];errors=[]
    with torch.no_grad():
        for i,row in enumerate(rows):
            s=row['sample'];a=s['original'];lab,counts=past_facts(a,s['round'],s['seat'])
            layers=encode_layers(actor,a);memory=layers[-1]
            curr=tuple(x[a[1]==s['round']].copy() for x in a)
            cm=actor.encode_batch(StreamBatch.from_arrays([curr],'cpu'))[0]
            sm=actor.encode_batch(StreamBatch.from_arrays([shifted_history(s,rows[donors[i]]['sample'])],'cpu'))[0]
            q=torch.tensor(s['native']['query'][None]);reading=actor.reader_input(q)
            kr,vr=actor.project_keys_values(cm[None]);cr=actor._attend(reading,kr,vr,torch.ones(1,len(cm),dtype=torch.bool))
            state_round=actor.fuse(reading,cr)[0]
            k,v=actor.project_keys_values(memory[None]);reader=actor._attend(reading,k,v,torch.ones(1,len(memory),dtype=torch.bool))[0]
            errors.append(float((reader-torch.tensor(s['native']['reader'])).abs().max()))
            local={'raw':counts,'native_reader':s['native']['reader'],'native_state':s['native']['state'],
                   'reader_recomputed':reader.numpy(),'state_recomputed':actor.fuse(reading,reader[None])[0].numpy()}
            for n,layer in enumerate(layers):local['embedding' if n==0 else 'stream_norm' if n==5 else f'layer_{n}']=explicit_pool(layer,a,s['round'],s['seat'])
            for key,value in local.items():features.setdefault(key,[]).append(value)
            labels.append(lab);queries.append(q[0]);common.append(torch.cat((q[0],state_round)).numpy())
            mem.append(memory.contiguous());round_mem.append(cm.contiguous());swap_mem.append(sm.contiguous())
            meta.append({k:row[k] for k in ('opponent','replicate','team','seed')})
            if (i+1)%160==0:print(json.dumps(dict(prepared=i+1,total=len(rows))),flush=True)
    y=np.array(labels);split=np.array([0 if r['replicate']<20 else 1 if r['replicate']<24 else 2 for r in rows]);z=np.array([r['opponent'] for r in rows])
    prior=np.array(json.loads((root/'protocol.json').read_text())['population']['weights'])
    w=prior[z];w/=w[split==0].mean();prevalence=np.average(y[split==0],axis=0,weights=w[split==0])
    eligible=((prevalence.reshape(2,-1)>=.05)&(prevalence.reshape(2,-1)<=.95)).all(0)
    keep=np.flatnonzero(np.tile(eligible,2));assert len(keep)>=4,'too few nontrivial public facts'
    names=[f'opp_{relative}_{FACTS[k][0]}' for relative in (1,3) for k in range(len(FACTS))]
    data=dict(memory=mem,round_memory=round_mem,shuffled_memory=swap_mem,query=torch.stack(queries),
        common=torch.tensor(np.array(common),dtype=torch.float32),features={k:torch.tensor(np.array(v),dtype=torch.float32) for k,v in features.items()},
        y=torch.tensor(y[:,keep]),sample_weight=torch.tensor(w,dtype=torch.float32),prevalence=torch.tensor(prevalence[keep],dtype=torch.float32),
        split=torch.tensor(split),metadata=meta,donors=donors,target_names=[names[i] for i in keep])
    torch.save(data,root/'dataset.pt')
    save_json(root/'data-manifest.json',dict(rows=len(rows),split_counts=[int((split==s).sum()) for s in (0,1,2)],
        targets=data['target_names'],all_training_prevalence=dict(zip(names,prevalence.tolist())),
        explicit_raw_answers_exact=True,native_reader_recompute_error_max=max(errors),reader_errors_over_1e3=int((np.array(errors)>1e-3).sum()),
        max_tokens=max(map(len,mem)),donor_max_tokens=max(map(len,swap_mem)),
        protocol_sha=sha(root/'protocol.json'),dataset_sha=sha(root/'dataset.pt'),source_sha=sha(__file__)))
    print(json.dumps(json.loads((root/'data-manifest.json').read_text()),indent=2),flush=True)


class Standardize(nn.Module):
    def __init__(self,x: torch.Tensor,w: torch.Tensor):
        super().__init__();mean=(x*w[:,None]).sum(0)/w.sum()
        std=((x-mean).square()*w[:,None]).sum(0)/w.sum()
        self.register_buffer('mean',mean);self.register_buffer('std',std.sqrt().clamp_min(1e-3))
    def forward(self,x):return (x-self.mean)/self.std


class Head(nn.Module):
    def __init__(self,width: int,targets: int,seed: int):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed);self.net=nn.Sequential(nn.Linear(width,64),nn.ReLU(),nn.Linear(64,targets))
    def forward(self,x):return self.net(x)


def balanced_loss(logits,y,w,prevalence):
    balance=torch.where(y>0,.5/prevalence,.5/(1-prevalence))
    per=F.binary_cross_entropy_with_logits(logits,y,reduction='none')*balance
    return (per.mean(1)*w).sum()/w.sum()


def probe_features(data: dict,name: str) -> torch.Tensor:
    c=data['common'];feature=data['features'][name] if name!='current' else torch.zeros(len(c),256)
    assert feature.shape[1]<=256
    return torch.cat((c,F.pad(feature,(0,256-feature.shape[1]))),dim=1)


def train_probe(root: Path,name: str,seed: int) -> dict:
    torch.set_num_threads(1);data=torch.load(root/'dataset.pt',weights_only=False,map_location='cpu')
    dest=root/'probes'/f'{name}-{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'selected.json').exists():return json.loads((dest/'selected.json').read_text())
    x=probe_features(data,name);tr=torch.where(data['split']==0)[0];va=torch.where(data['split']==1)[0]
    scale=Standardize(x[tr],data['sample_weight'][tr]);xx=scale(x);head=Head(xx.shape[1],data['y'].shape[1],seed)
    opt=torch.optim.Adam(head.parameters(),lr=.001,weight_decay=0);rng=torch.Generator().manual_seed(seed+SEED)
    best=float('inf');curve=[]
    for step in range(PROBE_STEPS+1):
        if step%100==0:
            with torch.no_grad():
                vl=float(balanced_loss(head(xx[va]),data['y'][va],data['sample_weight'][va],data['prevalence']))
                tl=float(balanced_loss(head(xx[tr]),data['y'][tr],data['sample_weight'][tr],data['prevalence']))
            curve.append(dict(step=step,train_loss=tl,validation_loss=vl))
            if vl<best:
                best=vl;chosen=step;torch.save(dict(head=head.state_dict(),scale=scale.state_dict()),dest/'selected.pt')
        if step==PROBE_STEPS:break
        ids=tr[torch.randint(len(tr),(64,),generator=rng)];opt.zero_grad(set_to_none=True)
        loss=balanced_loss(head(xx[ids]),data['y'][ids],data['sample_weight'][ids],data['prevalence'])
        loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
    out=dict(kind='probe',name=name,seed=seed,step=chosen,validation_loss=best,curve=curve,source_sha=sha(__file__))
    save_json(dest/'selected.json',out);return {k:out[k] for k in ('kind','name','seed','step','validation_loss')}


class Reader(nn.Module):
    """Only the original attention projections; no encoder/query/action weights."""
    def __init__(self,actor,normalized: bool):
        super().__init__();self.q_proj=copy.deepcopy(actor.q_proj);self.kv_proj=copy.deepcopy(actor.kv_proj);self.out_proj=copy.deepcopy(actor.out_proj)
        self.heads=actor.config.heads;self.depth=actor.config.width//self.heads;self.normalized=normalized
        self.requires_grad_(True)
    def forward(self,query,memory,allowed):
        q=self.q_proj(query).unflatten(-1,(self.heads,self.depth))
        k,v=self.kv_proj(memory).chunk(2,-1);k=k.unflatten(-1,(self.heads,self.depth));v=v.unflatten(-1,(self.heads,self.depth))
        if self.normalized:q=F.layer_norm(q,(self.depth,));k=F.layer_norm(k,(self.depth,))
        with sdpa_kernel(SDPBackend.MATH):
            out=F.scaled_dot_product_attention(q[:,:,None],k.transpose(1,2),v.transpose(1,2),attn_mask=allowed[:,None,None])
        return self.out_proj(out[:,:,0].flatten(1))


def memory_batch(data: dict,ids: torch.Tensor,view: str='full'):
    source=data[{'full':'memory','round':'round_memory','shuffled':'shuffled_memory'}[view]]
    values=[source[int(i)] for i in ids];lengths=torch.tensor([len(v) for v in values])
    memory=nn.utils.rnn.pad_sequence(values,batch_first=True)
    return data['query'][ids],memory,torch.arange(memory.shape[1])[None]<lengths[:,None]


def read_all(reader,data,ids,view='full'):
    with torch.no_grad():return torch.cat([reader(*memory_batch(data,chunk,view)) for chunk in ids.split(32)])


def grad_stats(reader):
    def norm(p):return 0. if p.grad is None else float(p.grad.norm())
    w=reader.q_proj.weight.shape[0];g=reader.kv_proj.weight.grad
    return dict(q=norm(reader.q_proj.weight),k=0. if g is None else float(g[:w].norm()),
                v=0. if g is None else float(g[w:].norm()),out=norm(reader.out_proj.weight))


def train_reader(root: Path,arm: str,seed: int,rate: float) -> dict:
    torch.set_num_threads(1);data=torch.load(root/'dataset.pt',weights_only=False,map_location='cpu')
    dest=root/'readers'/f'{arm}-{seed}-{rate:g}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'selected.json').exists():return json.loads((dest/'selected.json').read_text())
    actor,_,_=load_players(indices=[]);original={k:v.clone() for k,v in actor.state_dict().items()}
    reader=Reader(actor,arm in ('norm','norm_frozen'));frozen=arm in ('frozen','norm_frozen')
    reader.requires_grad_(not frozen);before={k:v.clone() for k,v in reader.state_dict().items()}
    tr=torch.where(data['split']==0)[0];va=torch.where(data['split']==1)[0]
    initial=read_all(reader,data,tr);scale=Standardize(initial,data['sample_weight'][tr]);head=Head(128,data['y'].shape[1],seed)
    params=[p for p in list(reader.parameters())+list(head.parameters()) if p.requires_grad]
    opt=torch.optim.Adam(params,lr=rate,weight_decay=0);rng=torch.Generator().manual_seed(seed+SEED)
    if frozen:
        cached=torch.empty(len(data['y']),128);cached[tr]=initial;cached[va]=read_all(reader,data,va)
    best=float('inf');curve=[];gradients=[];now=time.monotonic()
    for step in range(READER_STEPS+1):
        if step%200==0:
            features=initial if step==0 else cached[tr] if frozen else read_all(reader,data,tr)
            vf=cached[va] if frozen else read_all(reader,data,va)
            with torch.no_grad():
                vl=float(balanced_loss(head(scale(vf)),data['y'][va],data['sample_weight'][va],data['prevalence']))
                tl=float(balanced_loss(head(scale(features)),data['y'][tr],data['sample_weight'][tr],data['prevalence']))
            curve.append(dict(step=step,train_loss=tl,validation_loss=vl))
            if vl<best:
                best=vl;chosen=step;torch.save(dict(reader=reader.state_dict(),head=head.state_dict(),scale=scale.state_dict()),dest/'selected.pt')
        if step==READER_STEPS:break
        ids=tr[torch.randint(len(tr),(32,),generator=rng)];opt.zero_grad(set_to_none=True)
        features=cached[ids] if frozen else reader(*memory_batch(data,ids))
        loss=balanced_loss(head(scale(features)),data['y'][ids],data['sample_weight'][ids],data['prevalence'])
        if not torch.isfinite(loss):raise FloatingPointError((arm,seed,rate,step,'nonfinite loss'))
        loss.backward()
        if step<10 or step%100==0:gradients.append(dict(step=step,**grad_stats(reader)))
        nn.utils.clip_grad_norm_(params,1.);opt.step()
    selected=torch.load(dest/'selected.pt',weights_only=False)
    changes={k:float((selected['reader'][k]-v).norm()/v.norm().clamp_min(1e-12)) for k,v in before.items()}
    for k,v in actor.state_dict().items():assert torch.equal(v,original[k]),'original actor changed'
    out=dict(kind='reader',arm=arm,seed=seed,rate=rate,step=chosen,validation_loss=best,curve=curve,
        sampled_gradients=gradients,relative_parameter_changes=changes,seconds=time.monotonic()-now,source_sha=sha(__file__),
        original_actor_unchanged=True,trainable=[n for n,p in reader.named_parameters() if p.requires_grad])
    save_json(dest/'selected.json',out);return {k:out[k] for k in ('kind','arm','seed','rate','step','validation_loss','seconds')}


def run_task(task):
    kind,root,*args=task
    return train_probe(Path(root),*args) if kind=='probe' else train_reader(Path(root),*args)


def train(root: Path,workers: int) -> None:
    data=torch.load(root/'dataset.pt',weights_only=False);names=['current',*data['features']];del data
    tasks=[('probe',str(root),n,s) for n in names for s in SEEDS]
    tasks += [('reader',str(root),a,s,lr) for a in ARMS for s in SEEDS for lr in RATES]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(run_task,t) for t in tasks]
        for i,f in enumerate(as_completed(futures)):print(json.dumps(dict(f.result(),completed=i+1,total=len(tasks))),flush=True)
    choices={}
    for arm in ARMS:
        for seed in SEEDS:
            candidates=[root/'readers'/f'{arm}-{seed}-{r:g}' for r in RATES]
            folder=min(candidates,key=lambda f:json.loads((f/'selected.json').read_text())['validation_loss'])
            choices[f'{arm}-{seed}']=dict(folder=str(folder.relative_to(root)),checkpoint_sha=sha(folder/'selected.pt'))
    save_json(root/'choices.json',dict(choices=choices,dataset_sha=sha(root/'dataset.pt'),protocol_sha=sha(root/'protocol.json'),
                                     source_sha=sha(__file__),test_labels_used=False))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','train']);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if a.command=='prepare':prepare(a.root)
    else:train(a.root,a.workers)

if __name__=='__main__':main()
