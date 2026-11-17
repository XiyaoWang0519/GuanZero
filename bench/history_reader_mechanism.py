"""Isolated history-reader mechanism experiment; never updates a gameplay actor.

Fresh test matches, diagnostic-head activation control, question/query controls,
and matched nonlinear-before/after-pooling controls. See the sealed protocol.
"""
from __future__ import annotations

import argparse
import copy
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
    ACTOR, ROOT as SOURCE, Collector, DiagnosticStore, load_players, save_json,
)
from bench.history_answer_probe import (
    ROOT as OLD, FACTS, SEEDS, RATES, Reader, Standardize, balanced_loss,
    encode_layers, grad_stats, memory_batch, past_facts, read_all,
    shifted_history, source_rows,
)
from bench.qknorm_match_ablation import sha
from train.history_model import StreamBatch
from train.history_rollout import SequenceRolloutBuffer

ROOT = Path('.work/history-reader-mechanism-2026-11-14')
SEED = 20261114211
STEPS = 2000
HEAD_ARMS = ('native_relu', 'native_leaky', 'norm_relu', 'norm_leaky')
QUERY_ARMS = ('actual_all', 'mean_all', 'centered_all', 'task_all',
              'actual_target', 'task_target')
POOL_ARMS = ('late_mean', 'early_mean', 'early_max')
ARMS = HEAD_ARMS + QUERY_ARMS + POOL_ARMS
CONTRASTS = (
    ('native_leaky', 'native_relu'), ('norm_leaky', 'norm_relu'),
    ('mean_all', 'actual_all'), ('centered_all', 'actual_all'),
    ('task_all', 'actual_all'), ('actual_target', 'actual_all'),
    ('task_target', 'actual_target'), ('early_mean', 'late_mean'),
    ('early_max', 'early_mean'),
)


def seal(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if (root/'protocol.json').exists():
        return
    old = json.loads((OLD/'protocol.json').read_text())
    save_json(root/'protocol.json', dict(
        actor_sha=sha(ACTOR), population=old['population'], source=str(SOURCE),
        old_dataset_sha=sha(OLD/'dataset.pt'), seed=SEED,
        split={'train': 'original reps 0..19, 800 matches',
               'validation': 'original reps 20..23, 160 matches',
               'test': 'fresh SEED=20261114211, reps 32..39, 320 matches'},
        excluded='old reps 24..31 never enter this experiment',
        target=old['target'], target_names=old['facts'],
        selection='validation balanced BCE, every 200 steps including zero; select rate per seed; seal before test evaluation',
        head_control='same original reader, fixed initial train standardizer and 128->64->12 head; replace only ReLU by LeakyReLU(0.1)',
        query_control='12 slots in every arm, common 140->64 LeakyReLU->1 question head; Q/K per-head LayerNorm in all six arms',
        queries={'actual': 'original private query repeated for all questions',
                 'mean': 'train-only own-seat mean query repeated',
                 'centered': 'train-only global featurewise standardized private query repeated',
                 'task': '24 trainable public target-seat x family queries, initialized N(0,1); no answer input'},
        mask='all valid tokens or explicit previous-round target-opponent Play tokens; public seat/round/phase only',
        pool_control='same parameter count and initial weights: 128->64 LeakyReLU->128 phi before/after masked mean; early max separately',
        pool_supervision='only 12 match-level binary facts, no per-token type supervision',
        stable_head='training-batch feature normalization, recalibrated on all 800 training matches at each checkpoint; no heldout statistics',
        optimization=dict(steps=STEPS, batch=32, rates=RATES, seeds=SEEDS,
                          optimizer='Adam no decay; gradient clip 1', arms=ARMS),
        contrasts=CONTRASTS,
        inference='5000 stratified paired seed-cluster bootstraps; three optimizer seeds averaged, not counted as matches; 95% descriptive and familywise 99.444% contrast intervals',
        scope='CPU only, <=4 processes x 1 torch thread, isolated learned readers/heads; frozen encoder/private query; no PPO or mainline writes',
        limitations='dense public fact retrieval, not strength or strategic adaptation; task queries/masks grant explicit question/address information',
        sources={p: sha(p) for p in (__file__, 'bench/history_answer_probe.py',
                                   'bench/filtered_history_collect.py', 'train/history_model.py')},
    ))


def collect_block(task: tuple) -> dict:
    root_string, block = task
    root = Path(root_string)
    torch.set_num_threads(1)
    folder = root/'collection'/f'b{block:02d}-r32'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists():
        return json.loads((folder/'complete.json').read_text())
    indices = list(range(block*5, (block+1)*5))
    actor, opponents, identity = load_players(indices=indices)
    cells = [(z, rep, team) for z in indices for rep in range(32, 40) for team in (0, 1)]
    seeds = [int(np.random.default_rng(np.random.SeedSequence([SEED,z,rep])).integers(1,2**63-1))
             for z,rep,_ in cells]
    env = gd.VecEnv(num_envs=len(cells), num_threads=1, seed=71,
                    log_public_actions=True, log_env_limit=len(cells))
    env.reset(match_seeds=seeds)
    def assignment(e, match):
        z, _, team = cells[e]
        return [0 if s%2 == team else z+1 for s in range(4)]
    collector = Collector(env, actor, DiagnosticStore(), SequenceRolloutBuffer(),
                          device='cpu', kv_cache=True, seeds=seeds,
                          seat_policy=assignment, resolve_policy=opponents.__getitem__)
    collector.started = True
    started = time.monotonic()
    with torch.inference_mode():
        for step in range(3000):
            collector.collect(1)
            if collector.done.all():
                break
        else:
            raise RuntimeError('four rounds incomplete; no censoring allowed')
    collector.tap.save(folder/'public.npz')
    torch.save(collector.samples, folder/'samples.pt')
    assert len(collector.tap.meta) > 0
    result = dict(block=block, matches=len(cells), steps=step+1,
                  seconds=time.monotonic()-started,
                  actor_sha=identity['actor_sha'], population_sha=identity['population_sha'],
                  cells=[dict(opponent=z, replicate=r, team=t, seed=s,
                              rounds=int(collector.finished_rounds[e]))
                         for e, ((z,r,t),s) in enumerate(zip(cells,seeds))])
    save_json(folder/'complete.json', result)
    return {k: result[k] for k in ('block','matches','steps','seconds')}


def target_mask(arrays: tuple, seat: int, round_index: int=3) -> torch.Tensor:
    """Two relative-opponent masks. BOS excluded. No action-type access."""
    tokens, rounds, phases = arrays
    return torch.tensor(np.stack([
        np.r_[False, (rounds==round_index-1) & (phases==3) &
              (tokens[:,:4].argmax(1)==(seat+rel)%4)] for rel in (1,3)
    ]))


def prepare(root: Path) -> None:
    if (root/'dataset.pt').exists():
        return
    torch.set_num_threads(1)
    old = torch.load(OLD/'dataset.pt', weights_only=False)
    rows = [r for r in source_rows() if r['replicate']<24]
    fresh = []
    for folder in sorted((root/'collection').glob('b*-r32')):
        meta = json.loads((folder/'complete.json').read_text())
        for s in torch.load(folder/'samples.pt', weights_only=False):
            if s['round']==3 and s['threshold']==0:
                fresh.append(dict(sample=s, **meta['cells'][s['env']]))
    fresh.sort(key=lambda r:(r['opponent'],r['replicate'],r['team']))
    # VecEnv keeps earlier-finished lanes advancing until the slowest lane
    # finishes. Every selected prefix is still the first round-3 decision.
    assert len(fresh)==320 and all(r['rounds']>=4 for r in fresh)
    assert len({(r['opponent'],r['replicate'],r['team']) for r in fresh})==320
    assert set(r['seed'] for r in rows).isdisjoint(r['seed'] for r in fresh)
    rows += fresh
    old_lookup = {(r['opponent'],r['replicate'],r['team']):i for i,r in enumerate(old['metadata'])}
    actor,_,_ = load_players(indices=[])
    memories=[]; masks=[]; query=[]; labels=[]; seats=[]; raw=[]; metadata=[]
    with torch.no_grad():
        for row in rows:
            s=row['sample']; a=s['original']
            y,counts=past_facts(a,3,s['seat'])
            key=(row['opponent'],row['replicate'],row['team'])
            if row['replicate']<24:
                i=old_lookup[key]; memory=old['memory'][i]
                np.testing.assert_array_equal(y,old['y'][i].numpy())
            else:
                memory=encode_layers(actor,a)[-1]
            memories.append(memory.clone()); masks.append(target_mask(a,s['seat']))
            query.append(torch.tensor(s['native']['query'])); labels.append(y); raw.append(counts)
            seats.append(s['seat']); metadata.append({k:row[k] for k in ('opponent','replicate','team','seed')})
    split=torch.tensor([0 if r['replicate']<20 else 1 if r['replicate']<24 else 2 for r in rows])
    prior=np.array(json.loads((root/'protocol.json').read_text())['population']['weights'])
    weight=prior[[r['opponent'] for r in rows]]; weight/=weight[split.numpy()==0].mean()
    data=dict(memory=memories, masks=masks, query=torch.stack(query), seats=torch.tensor(seats),
              y=torch.tensor(np.array(labels)), raw=torch.tensor(np.array(raw)), split=split,
              sample_weight=torch.tensor(weight,dtype=torch.float32), metadata=metadata,
              prevalence=old['prevalence'], target_names=old['target_names'])
    # Shuffled raw history is re-encoded; receiver private query/current suffix stay fixed.
    rng=np.random.default_rng(SEED+1); donors=np.arange(1280)
    for s in (0,1,2):
        for team in (0,1):
            ids=np.array([i for i,r in enumerate(rows) if split[i]==s and r['team']==team])
            order=rng.permutation(ids); donors[order]=np.roll(order,1)
    assert all(rows[i]['seed']!=rows[j]['seed'] for i,j in enumerate(donors))
    data['donors']=donors
    data['shuffled_memory']=[]; data['shuffled_masks']=[]
    with torch.no_grad():
        for i in range(960,1280):
            s=rows[i]['sample']; a=shifted_history(s,rows[donors[i]]['sample'])
            data['shuffled_memory'].append(actor.encode_batch(StreamBatch.from_arrays([a],'cpu'))[0])
            data['shuffled_masks'].append(target_mask(a,s['seat']))
    torch.save(data,root/'dataset.pt')
    save_json(root/'data-manifest.json',dict(rows=1280,split_counts=[int((split==i).sum()) for i in range(3)],
        fresh_seeds=len(set(r['seed'] for r in fresh)), targets=data['target_names'],
        max_tokens=max(map(len,memories)), actor_sha=sha(ACTOR), dataset_sha=sha(root/'dataset.pt'),
        raw_samples={str(p):sha(p) for p in sorted((root/'collection').glob('*/samples.pt'))}))
    print(json.dumps(json.loads((root/'data-manifest.json').read_text())),flush=True)


class DiagnosticHead(nn.Module):
    def __init__(self, width: int, targets: int, seed: int, leaky: bool):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.first=nn.Linear(width,64)
            self.last=nn.Linear(64,targets)
        self.leaky=leaky

    def forward(self,x):
        h=self.first(x)
        return self.last(F.leaky_relu(h,.1) if self.leaky else F.relu(h))


def padded_batch(data: dict, ids: torch.Tensor, view: str='full') -> tuple:
    values=[data['memory'][int(i)] if view=='full' else data['shuffled_memory'][int(i)-960] for i in ids]
    masks=[data['masks'][int(i)] if view=='full' else data['shuffled_masks'][int(i)-960] for i in ids]
    memory=nn.utils.rnn.pad_sequence(values,batch_first=True)
    valid=torch.arange(memory.shape[1])[None]<torch.tensor([len(x) for x in values])[:,None]
    mask=nn.utils.rnn.pad_sequence([m.T for m in masks],batch_first=True).transpose(1,2)
    assert mask.any(-1).all()
    return data['query'][ids],memory,valid,mask,data['seats'][ids]


class SlotReader(nn.Module):
    def __init__(self, actor, data: dict, arm: str, seed: int):
        super().__init__()
        self.arm=arm
        tr=data['split']==0
        self.register_buffer('query_mean',torch.stack([data['query'][tr & (data['seats']==s)].mean(0) for s in range(4)]))
        self.query_scale=Standardize(data['query'][tr],data['sample_weight'][tr])
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            if arm in POOL_ARMS:
                selected=torch.cat([m[mask.any(0)] for m,mask,yes in zip(data['memory'],data['masks'],tr) if yes])
                self.memory_scale=Standardize(selected,torch.ones(len(selected)))
                self.phi=nn.Sequential(nn.Linear(128,64),nn.LeakyReLU(.1),nn.Linear(64,128))
            else:
                self.q_proj=copy.deepcopy(actor.q_proj)
                self.kv_proj=copy.deepcopy(actor.kv_proj)
                self.out_proj=copy.deepcopy(actor.out_proj)
                self.heads=actor.config.heads;self.depth=128//self.heads
                if arm.startswith('task'):
                    self.task_query=nn.Parameter(torch.randn(24,128))
        self.requires_grad_(True)

    def queries(self, query, seats):
        mode=self.arm.split('_')[0]
        if mode=='task':
            targets=(seats[:,None]+torch.tensor([1,3])[None])%4
            indices=targets[:,:,None]*6+torch.arange(6)[None,None]
            return self.task_query[indices.flatten(1)]
        if mode=='mean':query=self.query_mean[seats]
        elif mode=='centered':query=self.query_scale(query)
        return query[:,None].expand(-1,12,-1)

    def attention(self, query, memory, valid, mask, seats):
        q=self.q_proj(self.queries(query,seats)).unflatten(-1,(self.heads,self.depth)).transpose(1,2)
        k,v=self.kv_proj(memory).chunk(2,-1)
        k=k.unflatten(-1,(self.heads,self.depth)).transpose(1,2)
        v=v.unflatten(-1,(self.heads,self.depth)).transpose(1,2)
        q=F.layer_norm(q,(self.depth,));k=F.layer_norm(k,(self.depth,))
        allowed=mask.repeat_interleave(6,1) if self.arm.endswith('_target') else valid[:,None].expand(-1,12,-1)
        return q,k,v,allowed

    def forward(self, query, memory, valid, mask, seats):
        if self.arm in POOL_ARMS:
            x=self.memory_scale(memory)
            weights=mask.float()/mask.sum(-1,keepdim=True)
            if self.arm=='late_mean':
                out=self.phi(torch.bmm(weights,x))
            else:
                values=self.phi(x)
                if self.arm=='early_mean':out=torch.bmm(weights,values)
                else:out=values[:,None].masked_fill(~mask[:,:,:,None],-torch.inf).amax(2)
            return out.repeat_interleave(6,1)
        q,k,v,allowed=self.attention(query,memory,valid,mask,seats)
        with sdpa_kernel(SDPBackend.MATH):
            out=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed[:,None])
        return self.out_proj(out.transpose(1,2).flatten(2))


class SlotHead(nn.Module):
    """Shared question head, with train-only recalibration after reader updates."""
    def __init__(self, seed: int):
        super().__init__()
        self.norm=nn.BatchNorm1d(128,affine=False)
        self.head=DiagnosticHead(140,1,seed,True)
        self.register_buffer('questions',torch.eye(12))

    def calibrate(self, features: torch.Tensor) -> None:
        x=features.flatten(0,1)
        self.norm.running_mean.copy_(x.mean(0))
        self.norm.running_var.copy_(x.var(0,unbiased=False))

    def preactivation(self,features):
        x=self.norm(features.flatten(0,1)).reshape_as(features)
        q=self.questions[None].expand(len(x),-1,-1)
        return torch.cat((x,q),dim=-1)

    def forward(self,features):
        return self.head(self.preactivation(features)).squeeze(-1)


def slot_read_all(reader, data, ids, view='full'):
    with torch.no_grad():
        return torch.cat([reader(*padded_batch(data,chunk,view)) for chunk in ids.split(32)])


def train_one(task: tuple) -> dict:
    root_string,arm,seed,rate=task;root=Path(root_string)
    torch.set_num_threads(1)
    dest=root/'runs'/f'{arm}-{seed}-{rate:g}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'selected.json').exists():
        r=json.loads((dest/'selected.json').read_text())
        return {k:r[k] for k in ('arm','seed','rate','step','validation_loss','seconds')}
    started=time.monotonic()
    data=torch.load(root/'dataset.pt',weights_only=False)
    tr=torch.where(data['split']==0)[0];va=torch.where(data['split']==1)[0]
    actor,_,_=load_players(indices=[])
    original={k:v.clone() for k,v in actor.state_dict().items()}
    phase_a=arm in HEAD_ARMS
    if phase_a:
        reader=Reader(actor,arm.startswith('norm'))
        initial=read_all(reader,data,tr)
        scale=Standardize(initial,data['sample_weight'][tr])
        head=DiagnosticHead(128,12,seed,arm.endswith('leaky'))
        def read(ids):return read_all(reader,data,ids)
        def predict(x):return head(scale(x))
    else:
        reader=SlotReader(actor,data,arm,seed);head=SlotHead(seed)
        def read(ids):return slot_read_all(reader,data,ids)
        def predict(x):return head(x)
    before={k:v.clone() for k,v in reader.state_dict().items()}
    params=list(reader.parameters())+list(head.parameters())
    opt=torch.optim.Adam(params,lr=rate,weight_decay=0)
    rng=torch.Generator().manual_seed(seed+20261114081)  # same minibatches as preceding experiment
    best=float('inf');curve=[];gradients=[]
    for step in range(STEPS+1):
        if step%200==0:
            reader.eval();head.eval()
            tf=initial if phase_a and step==0 else read(tr);vf=read(va)
            with torch.no_grad():
                if not phase_a:head.calibrate(tf)
                tl=float(balanced_loss(predict(tf),data['y'][tr],data['sample_weight'][tr],data['prevalence']))
                vl=float(balanced_loss(predict(vf),data['y'][va],data['sample_weight'][va],data['prevalence']))
                pre=head.first(scale(tf)) if phase_a else head.head.first(head.preactivation(tf))
                curve.append(dict(step=step,train_loss=tl,validation_loss=vl,
                                  all_nonpositive_fraction=float((pre<=0).all(-1).float().mean()),
                                  positive_fraction=float((pre>0).float().mean())))
            if vl<best:
                best=vl;chosen=step
                state=dict(reader=reader.state_dict(),head=head.state_dict())
                if phase_a:state['scale']=scale.state_dict()
                torch.save(state,dest/'selected.pt')
        if step==STEPS:break
        reader.train();head.train();opt.zero_grad(set_to_none=True)
        ids=tr[torch.randint(len(tr),(32,),generator=rng)]
        features=reader(*memory_batch(data,ids)) if phase_a else reader(*padded_batch(data,ids))
        features.retain_grad()
        logits=predict(features)
        loss=balanced_loss(logits,data['y'][ids],data['sample_weight'][ids],data['prevalence'])
        if not torch.isfinite(loss):raise FloatingPointError((arm,seed,rate,step))
        loss.backward()
        if step<10 or step%100==0:
            g=grad_stats(reader) if hasattr(reader,'q_proj') else {'phi':float(reader.phi[0].weight.grad.norm())}
            if hasattr(reader,'task_query'):g['task_query']=float(reader.task_query.grad.norm())
            gradients.append(dict(step=step,feature_gradient=float(features.grad.norm()),**g))
        nn.utils.clip_grad_norm_(params,1.);opt.step()
    chosen_state=torch.load(dest/'selected.pt',weights_only=False)
    changes={k:float((chosen_state['reader'][k]-v).norm()/v.norm().clamp_min(1e-12))
             for k,v in before.items() if k in dict(reader.named_parameters())}
    assert all(torch.equal(v,original[k]) for k,v in actor.state_dict().items()),'frozen actor changed'
    out=dict(arm=arm,seed=seed,rate=rate,step=chosen,validation_loss=best,curve=curve,
             gradients=gradients,relative_parameter_changes=changes,
             parameters=sum(p.numel() for p in params),seconds=time.monotonic()-started,
             original_actor_unchanged=True,source_sha=sha(__file__))
    save_json(dest/'selected.json',out)
    return {k:out[k] for k in ('arm','seed','rate','step','validation_loss','seconds')}


def train(root: Path, workers: int) -> None:
    tasks=[(str(root),arm,seed,rate) for arm in ARMS for seed in SEEDS for rate in RATES]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        for i,f in enumerate(as_completed([pool.submit(train_one,t) for t in tasks])):
            print(json.dumps(dict(f.result(),completed=i+1,total=len(tasks))),flush=True)
    choices={}
    for arm in ARMS:
        for seed in SEEDS:
            folders=[root/'runs'/f'{arm}-{seed}-{rate:g}' for rate in RATES]
            folder=min(folders,key=lambda f:json.loads((f/'selected.json').read_text())['validation_loss'])
            choices[f'{arm}-{seed}']=dict(folder=str(folder.relative_to(root)),checkpoint_sha=sha(folder/'selected.pt'))
    save_json(root/'choices.json',dict(choices=choices,dataset_sha=sha(root/'dataset.pt'),
                                     protocol_sha=sha(root/'protocol.json'),test_labels_used=False))


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('command',choices=['collect','prepare','train','seal'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();seal(a.root)
    if a.command=='collect':
        with ProcessPoolExecutor(max_workers=a.workers,mp_context=mp.get_context('spawn')) as pool:
            for f in as_completed([pool.submit(collect_block,(str(a.root),b)) for b in range(4)]):
                print(json.dumps(f.result()),flush=True)
    elif a.command=='prepare':prepare(a.root)
    elif a.command=='train':train(a.root,a.workers)


if __name__=='__main__':main()
