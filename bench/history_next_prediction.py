"""Frozen u195450 next-opponent prediction: reader structure x parameter budget.

Only offline diagnostic modules learn. No actor/PPO/optimizer checkpoint writes.
Public history is grouped automatically, without a question-specific answer mask.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time

import gd
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from bench.filtered_history_collect import (
    ACTOR, ROOT as SOURCE, Collector, DiagnosticStore, load_players, raw_profile, save_json,
)
from bench.history_answer_probe import Standardize, Reader
from bench.qknorm_match_ablation import sha
from train.history_model import StreamBatch
from train.history_rollout import SequenceRolloutBuffer

ROOT=Path('.work/history-next-prediction-2026-11-14')
SEED=20261114411
SEEDS=(71,72,73)
RATES=(.0003,.001)
STEPS=2000
PRIMARY=('original_small','original_large','improved_small','improved_large')
ARMS=PRIMARY+('current_extra','identity','raw','native')


def seal(root: Path) -> None:
    root.mkdir(parents=True,exist_ok=True)
    if (root/'protocol.json').exists():return
    identity=json.loads((SOURCE/'protocol.json').read_text())['identity']
    assert sha(ACTOR)==identity['actor_sha']
    save_json(root/'protocol.json',dict(
        identity=identity,seed=SEED,
        scope='local CPU <=4 workers x 1 torch thread; frozen u195450 encoder/private network/gameplay; no PPO, mainline writes, remote jobs or new compute rental',
        split={'train':'old reps 0..19, 800 matches','validation':'old reps 20..23, 160 matches',
               'test':'new seed base 20261114411, reps 40..47, 320 matches; all prior tests excluded'},
        samples='round 3 threshold 0/20/50; three decisions per match where a subsequent opponent Play exists',
        target='11 public action types of the first subsequent opponent Play in this round, after our observed chosen action; no opponent private inputs',
        history='only completed rounds 0..2; frozen causal encoding; all public seats/phases; current round never enters the old-history branch',
        common='native own private query, current-round-only frozen state, own chosen action, own seat, current length/round/threshold, current public profile',
        representation='all 12 relative-seat x previous-round groups plus BOS; public group IDs and log token counts available to BOTH architectures; no target answer position supplied',
        original='linear QKV attention over all old tokens; 8 heads; internal dimension 128 small / 256 large; output 128',
        improved='token MLP 128->64/128 LeakyReLU->128; mean AND max within each of 13 public groups; 256->128 merge; attention over groups with internal dimension 32/128, 8 heads; output 128',
        fairness='matched reader parameter budgets: 67840 vs 67872 small, 133760 both large; same 128-d output and downstream head; common fresh initialization rule, Q/K normalization, train-only input statistics; original means original attention architecture, not a claim of preserving checkpoint reader weights',
        controls='current-only baseline; equal downstream residual head with zero history (current_extra); privileged identity diagnostic; raw public profile; frozen pretrained old-history attention; fixed same-seat/threshold donor history at test',
        prediction_head='frozen per-seed current-only baseline + zero-initialized residual 256->64 LeakyReLU->11; train-calibrated history BatchNorm; branch may change only copied diagnostic modules',
        base_head='standardized current context ->128 LeakyReLU ->11, initialized to train class prior',
        optimization=dict(seeds=SEEDS,rates=RATES,steps=STEPS,batch=32,optimizer='AdamW weight_decay=0.001, clip norm 1',
                          loss='pool/match-weighted multiclass cross entropy, no class rebalance'),
        selection='weighted validation Brier every 200 steps, including zero, then rate per seed; all choices sealed before test scoring',
        metrics='11-class Brier primary; log loss, accuracy and coarse pass/single/bomb/other secondary; each match equal within type',
        comparisons='four models versus current_extra and versus their own shuffled-input predictions; size effects within each architecture; architecture effects within each size; 12 prespecified contrasts with simultaneous alpha=0.05/12 bootstrap intervals',
        uncertainty='5000 paired match-seed cluster bootstraps, stratified by 20 fixed opponent types, preserve two team orientations and all captures; optimizer seeds averaged, not additional data',
        limitation='controlled homogeneous-opponent teams, three prior rounds, fixed known pool; next-action prediction not strategy adaptation or strength; expanding only reader capacity',
        sources={p:sha(p) for p in (__file__,'bench/filtered_history_collect.py','train/history_model.py','train/history_attention.py')},
    ))


def collect_block(task: tuple) -> dict:
    root_string,block=task;root=Path(root_string);torch.set_num_threads(1)
    folder=root/'collection'/f'b{block:02d}-r40';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return json.loads((folder/'complete.json').read_text())
    actor,opponents,identity=load_players(indices=list(range(block*5,(block+1)*5)))
    cells=[(z,r,t) for z in range(block*5,(block+1)*5) for r in range(40,48) for t in (0,1)]
    seeds=[int(np.random.default_rng(np.random.SeedSequence([SEED,z,r])).integers(1,2**63-1)) for z,r,_ in cells]
    env=gd.VecEnv(num_envs=80,num_threads=1,seed=71,log_public_actions=True,log_env_limit=80)
    env.reset(match_seeds=seeds)
    def assignment(e,m):
        z,r,t=cells[e];return [0 if s%2==t else z+1 for s in range(4)]
    c=Collector(env,actor,DiagnosticStore(),SequenceRolloutBuffer(),device='cpu',kv_cache=True,
                seeds=seeds,seat_policy=assignment,resolve_policy=opponents.__getitem__)
    c.started=True;now=time.monotonic()
    with torch.inference_mode():
        for step in range(3000):
            c.collect(1)
            if c.done.all():break
        else:raise RuntimeError('incomplete four-round collection')
    c.tap.save(folder/'public.npz');torch.save(c.samples,folder/'samples.pt')
    assert len(c.tap.meta)>0
    out=dict(block=block,matches=80,steps=step+1,seconds=time.monotonic()-now,
             actor_sha=identity['actor_sha'],population_sha=identity['population_sha'],
             cells=[dict(opponent=z,replicate=r,team=t,seed=s,rounds=int(c.finished_rounds[e]))
                    for e,((z,r,t),s) in enumerate(zip(cells,seeds))])
    save_json(folder/'complete.json',out)
    return {k:out[k] for k in ('block','matches','steps','seconds')}


def future_target(meta: np.ndarray, actions: np.ndarray, prefix: int, rnd: int, seat: int):
    """Only the target reads future events; features are constructed separately."""
    ids=np.flatnonzero((np.arange(len(meta))>prefix)&(meta[:,1]==rnd)&
                       (meta[:,3]%2!=seat%2)&(meta[:,5]==3))
    if not len(ids):return None
    i=int(ids[0]);kind=int(actions[i,108:121].argmax())
    assert kind<11
    return kind,i


def history_groups(arrays: tuple, observer: int) -> torch.Tensor:
    tokens,rounds,phases=arrays
    if tokens[:,150:158].any():raise ValueError('private exchange flags in public history')
    if not ((rounds>=0)&(rounds<3)).all():raise ValueError('only completed rounds may enter history')
    relative=(tokens[:,:4].argmax(1)-observer)%4
    return torch.tensor(np.r_[12,(2-rounds)*4+relative],dtype=torch.long)


def prepare(root: Path) -> None:
    if (root/'dataset.pt').exists():return
    torch.set_num_threads(1);actor,_,_=load_players(indices=[])
    old_reader=Reader(actor,False).eval().requires_grad_(False)
    rows=[];excluded=[]
    for source in (SOURCE,root):
        for folder in sorted((source/'collection').glob('b*-r*')):
            manifest=json.loads((folder/'complete.json').read_text())
            public=np.load(folder/'public.npz');pm=public['public_meta'];pa=public['public_actions']
            for s in torch.load(folder/'samples.pt',weights_only=False):
                if s['round']!=3:continue
                cell=manifest['cells'][s['env']]
                if source==SOURCE and cell['replicate']>=24:continue
                assert cell['rounds']>=4
                m=pm[pm[:,0]==s['env']];a=pa[pm[:,0]==s['env']]
                assert int(m[s['prefix'],1])==3 and int(m[s['prefix'],3])==s['seat']
                target=future_target(m,a,s['prefix'],3,s['seat'])
                if target is None:
                    excluded.append(dict(**cell,threshold=s['threshold'],reason='no subsequent same-round opponent Play'))
                    continue
                rows.append(dict(sample=s,**cell,label=target[0],target_index=target[1],folder=str(folder)))
    rows.sort(key=lambda r:(r['opponent'],r['replicate'],r['team'],r['sample']['threshold']))
    memories=[];history_arrays=[];history_keys={};hist_ids=[];group_ids=[]
    xs=[];queries=[];native=[];profiles=[];metadata=[];labels=[];seats=[]
    with torch.no_grad():
        for n,r in enumerate(rows):
            s=r['sample'];a=s['original'];old=tuple(x[a[1]<3].copy() for x in a)
            cur=tuple(x[a[1]==3].copy() for x in a)
            key=(r['opponent'],r['replicate'],r['team'])
            if key not in history_keys:
                history_keys[key]=len(memories);history_arrays.append(old)
                memories.append(actor.encode_batch(StreamBatch.from_arrays([old],'cpu'))[0].clone())
            h=history_keys[key]
            for x,y in zip(old,history_arrays[h]):np.testing.assert_array_equal(x,y)
            groups=history_groups(old,s['seat']);q=torch.tensor(s['native']['query'])
            cm=actor.encode_batch(StreamBatch.from_arrays([cur],'cpu'))
            k,v=actor.project_keys_values(cm);reading=actor.reader_input(q[None])
            cr=actor._attend(reading,k,v,torch.ones(1,cm.shape[1],dtype=torch.bool))
            current=actor.fuse(reading,cr)[0]
            context=np.r_[q.numpy(),current.numpy(),s['cand'][s['choice']],np.eye(4)[s['seat']],
                          len(cur[0]),3,s['threshold'],raw_profile(cur,s['seat']%2)]
            nf=old_reader(q[None],memories[h][None],torch.ones(1,len(memories[h]),dtype=torch.bool))[0]
            hist_ids.append(h);group_ids.append(groups);xs.append(context);queries.append(q);native.append(nf)
            profiles.append(raw_profile(old,s['seat']%2));labels.append(r['label']);seats.append(s['seat'])
            metadata.append(dict(**{k:r[k] for k in ('opponent','replicate','team','seed','target_index','folder')},
                                 threshold=s['threshold'],prefix=s['prefix'],seat=s['seat']))
            if (n+1)%480==0:print(json.dumps(dict(prepared=n+1,total=len(rows))),flush=True)
    split=torch.tensor([0 if r['replicate']<20 else 1 if r['replicate']<24 else 2 for r in rows])
    prior=np.array(json.loads((root/'protocol.json').read_text())['identity']['weights'])
    hid=np.array(hist_ids);counts=np.bincount(hid)
    w=prior[[r['opponent'] for r in rows]]/counts[hid];w/=w[split.numpy()==0].mean()
    train_hist=np.unique(hid[split.numpy()==0]);token_bank=torch.cat([memories[h] for h in train_hist])
    memory_scale=Standardize(token_bank,torch.ones(len(token_bank))).state_dict()
    donors=np.arange(len(rows));rng=np.random.default_rng(SEED+9)
    for sp in (0,1,2):
        for seat in range(4):
            for threshold in (0,20,50):
                ids=np.array([i for i,r in enumerate(metadata) if split[i]==sp and r['seat']==seat and r['threshold']==threshold])
                order=rng.permutation(ids);donors[order]=np.roll(order,1)
    assert all(rows[i]['seed']!=rows[j]['seed'] and seats[i]==seats[j] for i,j in enumerate(donors))
    data=dict(memory=memories,history_id=torch.tensor(hist_ids),groups=group_ids,
        x=torch.tensor(np.array(xs),dtype=torch.float32),query=torch.stack(queries),native=torch.stack(native),
        raw=torch.tensor(np.array(profiles),dtype=torch.float32),y=torch.tensor(labels),seats=torch.tensor(seats),
        split=split,sample_weight=torch.tensor(w,dtype=torch.float32),metadata=metadata,donors=torch.tensor(donors),
        memory_scale=memory_scale)
    train=split==0;freq=torch.zeros(11).scatter_add_(0,data['y'][train],data['sample_weight'][train])
    data['prevalence']=(freq+.5)/(freq.sum()+5.5)
    assert len(memories)==1280 and len(train_hist)==800
    splits=[{r['seed'] for r,yes in zip(metadata,split==s) if yes} for s in range(3)]
    assert all(splits[a].isdisjoint(splits[b]) for a,b in ((0,1),(0,2),(1,2)))
    torch.save(data,root/'dataset.pt')
    save_json(root/'data-manifest.json',dict(rows=len(rows),matches=len(memories),
        split_rows=[int((split==s).sum()) for s in range(3)],split_seed_clusters=list(map(len,splits)),
        context_width=data['x'].shape[1],max_old_tokens=max(map(len,memories)),
        training_class_counts=torch.bincount(data['y'][train],minlength=11).tolist(),
        excluded=excluded,dataset_sha=sha(root/'dataset.pt'),source_sha=sha(__file__),
        actor_sha=sha(ACTOR),raw_samples={str(p):sha(p) for p in (root/'collection').glob('*/samples.pt')}))
    print((root/'data-manifest.json').read_text(),flush=True)


def batch(data: dict, ids: torch.Tensor, view: str='full') -> tuple:
    donor=data['donors'][ids] if view=='shuffled' else ids
    values=[data['memory'][data['history_id'][i]] for i in donor]
    memory=nn.utils.rnn.pad_sequence(values,batch_first=True)
    group=nn.utils.rnn.pad_sequence([data['groups'][i] for i in donor],batch_first=True,padding_value=13)
    valid=group!=13
    return data['query'][ids],memory,valid,group


def group_pool(values: torch.Tensor, groups: torch.Tensor, valid: torch.Tensor):
    """Public groups only; padding belongs to a discarded fourteenth bucket."""
    indices=groups[:,:,None].expand_as(values)
    sums=values.new_zeros(len(values),14,values.shape[-1]).scatter_add(1,indices,values)
    maxima=values.new_full((len(values),14,values.shape[-1]),-torch.inf).scatter_reduce(
        1,indices,values,reduce='amax',include_self=True)
    counts=values.new_zeros(len(values),14).scatter_add(1,groups,valid.float())
    sums=sums[:,:13];maxima=maxima[:,:13];counts=counts[:,:13]
    means=sums/counts.clamp_min(1)[:,:,None]
    maxima=torch.where(counts[:,:,None]>0,maxima,0.)
    return means,maxima,counts


class PredictorReader(nn.Module):
    def __init__(self, data: dict, arm: str, seed: int):
        super().__init__();self.arm=arm
        tr=data['split']==0
        self.query_scale=Standardize(data['query'][tr],data['sample_weight'][tr])
        self.register_buffer('memory_mean',data['memory_scale']['mean'])
        self.register_buffer('memory_std',data['memory_scale']['std'])
        large=arm.endswith('large');self.improved=arm.startswith('improved')
        self.width=(128 if large else 32) if self.improved else (256 if large else 128)
        self.heads=8;self.depth=self.width//8
        with torch.random.fork_rng():
            torch.manual_seed(seed+100)
            self.address=nn.Embedding(13,128)
            self.count=nn.Linear(1,128,bias=False)
            if self.improved:
                hidden=128 if large else 64
                self.phi=nn.Sequential(nn.Linear(128,hidden),nn.LeakyReLU(.1),nn.Linear(hidden,128))
                self.merge=nn.Linear(256,128)
            self.q_proj=nn.Linear(128,self.width)
            self.kv_proj=nn.Linear(128,2*self.width)
            self.out_proj=nn.Linear(self.width,128)

    def forward(self,query,memory,valid,groups):
        x=(memory-self.memory_mean)/self.memory_std
        counts=x.new_zeros(len(x),14).scatter_add(1,groups,valid.float())[:,:13]
        address=self.address.weight[None]+self.count(counts.log1p()[:,:,None])
        if self.improved:
            means,maxima,_=group_pool(self.phi(x),groups,valid)
            x=self.merge(torch.cat((means,maxima),-1))+address
            allowed=counts>0
        else:
            x=x+address.gather(1,groups.clamp_max(12)[:,:,None].expand(-1,-1,128))
            allowed=valid
        q=self.q_proj(self.query_scale(query)).reshape(-1,8,self.depth)
        k,v=self.kv_proj(x).chunk(2,-1)
        k=k.reshape(len(x),-1,8,self.depth).transpose(1,2)
        v=v.reshape(len(x),-1,8,self.depth).transpose(1,2)
        q=F.layer_norm(q,(self.depth,));k=F.layer_norm(k,(self.depth,))
        with sdpa_kernel(SDPBackend.MATH):
            out=F.scaled_dot_product_attention(q[:,:,None],k,v,attn_mask=allowed[:,None,None])
        return self.out_proj(out[:,:,0].flatten(1))


class CurrentModel(nn.Module):
    def __init__(self, data: dict, seed: int):
        super().__init__();tr=data['split']==0
        self.scale=Standardize(data['x'][tr],data['sample_weight'][tr])
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.hidden=nn.Linear(data['x'].shape[1],128);self.output=nn.Linear(128,11)
            nn.init.zeros_(self.output.weight)
            with torch.no_grad():self.output.bias.copy_(data['prevalence'].log())
    def encode(self,x):return F.leaky_relu(self.hidden(self.scale(x)),.1)
    def forward(self,x):return self.output(self.encode(x))


class ResidualHead(nn.Module):
    def __init__(self, embedding: torch.Tensor, weights: torch.Tensor, seed: int):
        super().__init__();self.context_scale=Standardize(embedding,weights)
        self.norm=nn.BatchNorm1d(128,affine=False)
        with torch.random.fork_rng():
            torch.manual_seed(seed+200)
            self.hidden=nn.Linear(256,64);self.output=nn.Linear(64,11)
            nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def calibrate(self,x):
        self.norm.running_mean.copy_(x.mean(0));self.norm.running_var.copy_(x.var(0,unbiased=False))
    def forward(self,context,history):
        x=torch.cat((self.context_scale(context),self.norm(history)),-1)
        return self.output(F.leaky_relu(self.hidden(x),.1))


def loss(logits,y,w):return (F.cross_entropy(logits,y,reduction='none')*w).sum()/w.sum()


def brier(logits,y,w):
    p=logits.softmax(-1);target=F.one_hot(y,11)
    return float(((p-target).square().sum(-1)*w).sum()/w.sum())


def train_base(task: tuple) -> dict:
    root,seed,rate=task;root=Path(root);torch.set_num_threads(1)
    dest=root/'base'/f'{seed}-{rate:g}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'selected.json').exists():return json.loads((dest/'selected.json').read_text())
    data=torch.load(root/'dataset.pt',weights_only=False);tr=torch.where(data['split']==0)[0];va=torch.where(data['split']==1)[0]
    model=CurrentModel(data,seed);opt=torch.optim.AdamW(model.parameters(),lr=rate,weight_decay=.001)
    rng=torch.Generator().manual_seed(SEED+seed);curve=[];best=float('inf');now=time.monotonic()
    for step in range(STEPS+1):
        if step%200==0:
            with torch.no_grad():
                tl=model(data['x'][tr]);vl=model(data['x'][va])
                score=brier(vl,data['y'][va],data['sample_weight'][va])
                curve.append(dict(step=step,train_brier=brier(tl,data['y'][tr],data['sample_weight'][tr]),
                                  validation_brier=score,train_loss=float(loss(tl,data['y'][tr],data['sample_weight'][tr]))))
            if score<best:
                best=score;chosen=step;torch.save(model.state_dict(),dest/'selected.pt')
        if step==STEPS:break
        ids=tr[torch.randint(len(tr),(32,),generator=rng)];opt.zero_grad(set_to_none=True)
        value=loss(model(data['x'][ids]),data['y'][ids],data['sample_weight'][ids]);value.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    out=dict(arm='base',seed=seed,rate=rate,step=chosen,validation_brier=best,curve=curve,seconds=time.monotonic()-now,
             source_sha=sha(__file__),parameters=sum(p.numel() for p in model.parameters()))
    save_json(dest/'selected.json',out)
    return {k:out[k] for k in ('arm','seed','rate','step','validation_brier','seconds')}


def base_folder(root: Path, seed: int) -> Path:
    return min((root/'base'/f'{seed}-{r:g}' for r in RATES),
               key=lambda f:json.loads((f/'selected.json').read_text())['validation_brier'])


def make_branch(data,arm,seed,actor):
    if arm in PRIMARY:return PredictorReader(data,arm,seed)
    if arm=='native':return Reader(actor,False).eval().requires_grad_(False)
    if arm=='identity':
        with torch.random.fork_rng():
            torch.manual_seed(seed+100);return nn.Embedding(20,128)
    if arm=='raw':
        tr=data['split']==0;return Standardize(data['raw'][tr],data['sample_weight'][tr])
    return nn.Identity()


def branch_features(reader,data,ids,arm,view='full'):
    if arm in PRIMARY:return reader(*batch(data,ids,view))
    if arm=='native':return reader(*batch(data,ids,view)[:3])
    if arm=='identity':return reader(torch.tensor([data['metadata'][i]['opponent'] for i in ids]))
    if arm=='raw':return reader(data['raw'][data['donors'][ids] if view=='shuffled' else ids])
    return torch.zeros(len(ids),128)


def read_all(reader,data,ids,arm,view='full'):
    with torch.no_grad():return torch.cat([branch_features(reader,data,c,arm,view) for c in ids.split(64)])


def train_branch(task: tuple) -> dict:
    root,arm,seed,rate=task;root=Path(root);torch.set_num_threads(1)
    dest=root/'runs'/f'{arm}-{seed}-{rate:g}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'selected.json').exists():return json.loads((dest/'selected.json').read_text())
    now=time.monotonic();data=torch.load(root/'dataset.pt',weights_only=False)
    tr=torch.where(data['split']==0)[0];va=torch.where(data['split']==1)[0]
    actor,_,_=load_players(indices=[]);original={k:v.clone() for k,v in actor.state_dict().items()}
    base=CurrentModel(data,seed);base.load_state_dict(torch.load(base_folder(root,seed)/'selected.pt',weights_only=True))
    base.eval().requires_grad_(False)
    with torch.no_grad():context=base.encode(data['x']);base_logits=base.output(context)
    reader=make_branch(data,arm,seed,actor);head=ResidualHead(context[tr],data['sample_weight'][tr],seed)
    params=[p for p in list(reader.parameters())+list(head.parameters()) if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=rate,weight_decay=.001);rng=torch.Generator().manual_seed(SEED+seed)
    cache=None
    if arm in ('raw','native','current_extra'):
        cache=torch.zeros(len(data['y']),128);cache[tr]=read_all(reader,data,tr,arm);cache[va]=read_all(reader,data,va,arm)
    best=float('inf');curve=[];gradients=[]
    for step in range(STEPS+1):
        if step%200==0:
            reader.eval();head.eval()
            tf=cache[tr] if cache is not None else read_all(reader,data,tr,arm)
            vf=cache[va] if cache is not None else read_all(reader,data,va,arm)
            with torch.no_grad():
                head.calibrate(tf)
                tl=base_logits[tr]+head(context[tr],tf);vl=base_logits[va]+head(context[va],vf)
                score=brier(vl,data['y'][va],data['sample_weight'][va])
                curve.append(dict(step=step,train_brier=brier(tl,data['y'][tr],data['sample_weight'][tr]),
                                  validation_brier=score,train_loss=float(loss(tl,data['y'][tr],data['sample_weight'][tr]))))
            if score<best:
                best=score;chosen=step;torch.save(dict(reader=reader.state_dict(),head=head.state_dict()),dest/'selected.pt')
        if step==STEPS:break
        reader.train();head.train();opt.zero_grad(set_to_none=True)
        ids=tr[torch.randint(len(tr),(32,),generator=rng)]
        f=cache[ids] if cache is not None else branch_features(reader,data,ids,arm)
        if f.requires_grad:f.retain_grad()
        value=loss(base_logits[ids]+head(context[ids],f),data['y'][ids],data['sample_weight'][ids])
        if not torch.isfinite(value):raise FloatingPointError((arm,seed,rate,step))
        value.backward()
        if step<10 or step%100==0:
            gs={n:float(p.grad.norm()) for n,p in reader.named_parameters() if p.grad is not None and n.endswith('weight')}
            gradients.append(dict(step=step,feature_gradient=float(f.grad.norm()) if f.requires_grad else None,weights=gs))
        nn.utils.clip_grad_norm_(params,1.);opt.step()
    assert all(torch.equal(v,original[k]) for k,v in actor.state_dict().items())
    out=dict(arm=arm,seed=seed,rate=rate,step=chosen,validation_brier=best,curve=curve,gradients=gradients,
             seconds=time.monotonic()-now,reader_parameters=sum(p.numel() for p in reader.parameters() if p.requires_grad),
             trainable_parameters=sum(p.numel() for p in params),source_sha=sha(__file__),original_actor_unchanged=True,
             base_checkpoint_sha=sha(base_folder(root,seed)/'selected.pt'))
    save_json(dest/'selected.json',out)
    return {k:out[k] for k in ('arm','seed','rate','step','validation_brier','seconds')}


def train(root: Path, workers: int) -> None:
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        tasks=[pool.submit(train_base,(str(root),s,r)) for s in SEEDS for r in RATES]
        for f in as_completed(tasks):print(json.dumps(f.result()),flush=True)
    # Base choices are sealed before fitting residual branches.
    save_json(root/'base-choices.json',{str(s):dict(folder=str(base_folder(root,s).relative_to(root)),
        checkpoint_sha=sha(base_folder(root,s)/'selected.pt')) for s in SEEDS})
    tasks=[(str(root),a,s,r) for a in ARMS for s in SEEDS for r in RATES]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for i,f in enumerate(as_completed([pool.submit(train_branch,t) for t in tasks])):
            print(json.dumps(dict(f.result(),completed=i+1,total=len(tasks))),flush=True)
    choices={}
    for arm in ARMS:
        for seed in SEEDS:
            folder=min((root/'runs'/f'{arm}-{seed}-{r:g}' for r in RATES),
                       key=lambda f:json.loads((f/'selected.json').read_text())['validation_brier'])
            choices[f'{arm}-{seed}']=dict(folder=str(folder.relative_to(root)),checkpoint_sha=sha(folder/'selected.pt'))
    save_json(root/'choices.json',dict(choices=choices,dataset_sha=sha(root/'dataset.pt'),
        protocol_sha=sha(root/'protocol.json'),base_choices_sha=sha(root/'base-choices.json'),test_labels_used=False,
        source_sha=sha(__file__)))


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('command',choices=['seal','collect','prepare','train'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();seal(a.root)
    if a.command=='collect':
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
            for f in as_completed([pool.submit(collect_block,(str(a.root),b)) for b in range(4)]):
                print(json.dumps(f.result()),flush=True)
    elif a.command=='prepare':prepare(a.root)
    elif a.command=='train':train(a.root,a.workers)


if __name__=='__main__':main()
