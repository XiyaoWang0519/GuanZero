"""Frozen u195450/current-pool history diagnostics, isolated from training.

Collect balanced four-round matches and public prediction labels. Actor weights
are frozen; only the observer's own obs/candidates and public prefixes enter
feature taps. Actual opponent identities are labels, never public features.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import gd
import numpy as np
import torch

from bench.public_history_diverse import question_bank
from bench.public_history_probe import PublicCollector
from bench.qknorm_match_ablation import sha
from train.history_model import DecisionInputs, StreamBatch, load_history_checkpoint
from train.history_population import snapshot_weights, weights_digest
from train.history_rollout import MatchEventStore, SequenceRolloutBuffer

ROOT = Path('.work/filtered-history-2026-11-12')
ACTOR = Path('.work/danlm-u195450-2026-11-12/u195450.pt')
SEED = 20261112071
REPLICATES = 32
CAPTURES = ((1,0),(3,0),(3,20),(3,50))


class DiagnosticStore(MatchEventStore):
    """Use the event API so PublicTap observes the authoritative raw stream."""


def save_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def load_players(root=ROOT, indices=None):
    actor, _, ap = load_history_checkpoint(ACTOR,'cpu')
    actor.eval().requires_grad_(False);actor.causal_sdpa=True
    payload=torch.load(root/'population-source.pt',map_location='cpu',weights_only=False)
    pop=payload['population']
    recent=sorted(pop['snapshots'])[-pop['recent']:]
    ids=sorted(set(recent)|set(pop['archive']))
    weights=np.array([(1-pop['archive_share'])/len(recent)*(i in recent)
                      +pop['archive_share']/len(pop['archive'])*(i in pop['archive']) for i in ids])
    assert np.isclose(weights.sum(),1)
    opponents={}; records=[]
    for z,i in enumerate(ids):
        record=pop['snapshots'][i]
        assert weights_digest(record['model'])==record['metadata']['sha256']
        records.append(dict(record['metadata'],index=z,pool_weight=float(weights[z]),
                            archive=i in pop['archive'],recent=i in recent))
        if indices is None or z in indices:
            a=copy.deepcopy(actor)
            missing,extra=a.load_state_dict(record['model'],strict=False)
            assert not extra and set(missing)<=a.auxiliary_parameter_names()
            opponents[z+1]=a.eval().requires_grad_(False)
    return actor,opponents,dict(actor_update=ap['progress']['updates'],population_update=payload['progress']['updates'],
        actor_sha=sha(ACTOR),population_sha=sha(root/'population-source.pt'),opponents=records,weights=weights.tolist(),
        model_config=asdict(actor.config),source=payload['run_identity']['source']['source_sha256'])


def prepare(root=ROOT):
    torch.set_num_threads(1)
    _,_,identity=load_players(root,indices=[])
    assert identity['actor_update']==195450
    n=len(identity['opponents'])
    protocol=dict(identity=identity,seed=SEED,replicates=REPLICATES,matches=n*REPLICATES*2,
        rounds_per_match=4,captures=CAPTURES,split={'train':[0,20],'validation':[20,24],'test':[24,32]},
        pairing='same seed for the two team orientations; independent seeds per type/replicate; seed-cluster inference',
        pool='all eligible archive plus recent snapshots, weighted by conditional snapshot sampling probability',
        assignment='two opponent seats share one frozen type; partner is u195450; this is not the mixed-seat training distribution',
        questions=128,question_draws=64,question_calibration_draws=32,
        stage_features='raw public profile; old encoder pools; full/round reader; full/round state; deltas',
        common_context='private query, current-round state, chosen action, own seat and current public event count; no opponent private input',
        prediction='first subsequent public opponent play category in the same round, after observer chosen action',
        probes='fixed-size ridge; regularization selected on validation; same train/validation/test matches for all stages',
        value=dict(roots=12,candidate_cap=4,selection_worlds=32,evaluation_worlds=64,
                   max_round_branches=12*n*4*96,chunk_worlds=4,
                   task='own-hand/public midgame roots from neutral u195450; hidden redraws uniform; type assigned at score node; fixed continuation',
                   primary='PPO team round seat_return; choices sealed before independent evaluation',
                   limitation='finite candidates and one-step improvement; not full-match optimal information value'),
        workers=4,torch_threads=1,scope='CPU frozen actors and statistical probes; no PPO, checkpoint edits, remote compute, or main-job changes',
        multiple_comparisons='primary raw/memory/reader/state Brier gains use 98.75% seed-cluster intervals; descriptive extras retain 95%',
        sources={p:sha(p) for p in (__file__,'train/history_model.py','train/history_inference.py',
                'bench/public_history_probe.py','bench/qknorm_match_ablation.py')})
    save_json(root/'protocol.json',protocol)
    print(json.dumps({'population_update':identity['population_update'],'opponents':n,'matches':protocol['matches'],
                      'value_branch_cap':protocol['value']['max_round_branches']},indent=2),flush=True)


def raw_profile(arrays,team):
    """128 public-only features, counts conditional on lead/response and seat."""
    tokens,_,phases=arrays
    if len(tokens)==0:return np.zeros(128,np.float64)
    assert not tokens[:,150:158].any()
    counts=np.zeros((2,2,13),np.float64);left=np.zeros((2,6,4),np.float64)
    top=False;passes=0;remaining=np.full(4,27);holder=-1
    oldround=-1
    for token,rnd,phase in zip(tokens,arrays[1],phases):
        if rnd!=oldround:top=False;passes=0;remaining[:]=27;holder=-1;oldround=rnd
        seat=int(token[:4].argmax());kind=int(token[112:125].argmax());cards=int(token[158:].argmax())
        if phase!=3:continue
        if seat%2!=team:
            which=0 if seat==(team+1)%4 else 1
            counts[which,int(top),kind]+=1
            cat=0 if kind==0 else 1 if kind==1 else 2 if kind in (8,9,10) else 3
            bucket=int(np.searchsorted([1,2,5,10,18],remaining[seat],side='left'))
            left[which,bucket,cat]+=1
        remaining[seat]=cards
        if kind==0:
            passes+=1
            if passes>=sum(remaining[s]>0 for s in range(4) if s!=holder):top=False;passes=0;holder=-1
        else:top=True;passes=0;holder=seat
    a=(counts+.5)/(counts.sum(-1,keepdims=True)+6.5)
    b=(left+.5)/(left.sum(-1,keepdims=True)+2)
    feature=np.r_[a.ravel(),b.ravel(),np.log1p(counts.sum(-1)).ravel(),np.log1p(left.sum(-1)).ravel()]
    return np.pad(feature,(0,128-len(feature)))


def stage_features(actor,sample):
    arrays=sample['original'];rnd=sample['round'];n=int((arrays[1]<rnd).sum())
    assert n>0 and np.all(arrays[1][n:]==rnd)
    current=tuple(a[n:].copy() for a in arrays)
    streams=StreamBatch.from_arrays([arrays,current],'cpu');c=len(sample['cand'])
    inputs=DecisionInputs(streams,torch.arange(2),streams.lengths,
        torch.tensor(np.repeat(sample['obs'][None],2,axis=0)),torch.full((2,),sample['seat']),
        torch.tensor(np.tile(sample['cand'],(2,1))),torch.arange(3)*c,one_decision_per_stream=True)
    with torch.inference_mode():
        encoded=actor.encode_batch(streams);query=actor.private_query(inputs);reading=actor.reader_input(query)
        reader=actor._attend_encoded(reading,encoded,inputs.match_index,inputs.prefix,True)
        state=actor.fuse(reading,reader)
        torch.testing.assert_close(state,actor.decision_states(encoded,inputs),rtol=0,atol=0)
        logp=actor._log_prob_table(inputs,encoded,c)
        error=float((logp[0]-torch.tensor(sample['cached_logp'])).abs().max())
        # Native cached features below are authoritative. Full recomputation can
        # amplify tiny FP32 encoder differences in saturated Q/K attention.
        p=logp.double().exp();p/=p.sum(1,keepdim=True)
        old=encoded[0,1:n+1]
        mask=torch.tensor((arrays[0][:n,:4].argmax(1)%2!=sample['seat']%2)&(arrays[2][:n]==3))
        assert mask.any()
        features={'query':query[0],'memory_mean':old.mean(0),'memory_opponent':old[mask].mean(0),
                  'memory_last':old[-1],'reader':reader[0],'reader_round':reader[1],
                  'reader_delta':reader[0]-reader[1],'state':state[0],'state_round':state[1],
                  'state_delta':state[0]-state[1]}
        out={k:v.numpy().astype(np.float64) for k,v in features.items()}
        native=sample['native']
        out.update({k:v.astype(np.float64) for k,v in native.items()})
        out['reader_delta']=out['reader']-out['reader_round']
        out['state_delta']=out['state']-out['state_round']
        native_p=torch.tensor(sample['cached_logp']).double().exp();native_p/=native_p.sum()
        cache_tv=float((native_p-p[0]).abs().sum()/2)
        cache_flip=int(native_p.argmax()!=p[0].argmax())
        p[0]=native_p
    out['raw_profile']=raw_profile(tuple(a[:n] for a in arrays),sample['seat']%2)
    out['current_profile']=raw_profile(current,sample['seat']%2)
    out['context']=np.r_[out['query'],out['state_round'],sample['cand'][sample['choice']],
                         np.eye(4)[sample['seat']],len(current[0]),sample['round'],sample['threshold']]
    out.update(past=n,current=len(current[0]),cache_error=error,cache_tv=cache_tv,cache_flip=cache_flip,
               full_round_tv=float((p[0]-p[1]).abs().sum()/2),
               full_round_flip=int(p[0].argmax()!=p[1].argmax()))
    return out


class Collector(PublicCollector):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.samples=[];self.captured=set();self.finished_rounds=np.zeros(len(self.done),int)
    def _drain(self,stats):
        super()._drain(stats)
        for r in self.results:
            if int(r.match_id)==0:
                self.finished_rounds[int(r.env_id)]=max(self.finished_rounds[int(r.env_id)],int(r.round_index)+1)
                if int(r.round_index)>=3:self.done[int(r.env_id)]=True
    def _infer_group(self,group,fields,r,stats,mark):
        result=super()._infer_group(group,fields,r,stats,mark)
        if group.identity!=0:return result
        picked=[]
        for j,i in enumerate(group.rows):
            if r.match_id[i]!=0 or r.counts[i]<=1:continue
            stream=group.streams[int(fields[0][j])];arrays=stream.arrays();rnd=int(r.round_index[i])
            current=int(((arrays[1]==rnd)&(arrays[2]==3)).sum())
            for rr,threshold in CAPTURES:
                key=(int(r.env_id[i]),rr,threshold)
                if rnd==rr and current>=threshold and key not in self.captured:
                    self.captured.add(key);picked.append((j,i,threshold))
        if picked:
            streams,encoded=self._encode_streams(group,self.actor,fields[1],r.prefix)
            inputs=DecisionInputs(streams,fields[0],fields[1],fields[2],fields[3],fields[4],fields[5],fields[6],
                                  one_decision_per_stream=group.one_decision_per_stream)
            table=self.actor._log_prob_table(inputs,encoded,group.max_candidates)
            query=self.actor.private_query(inputs);reading=self.actor.reader_input(query)
            reader=self.actor._attend_encoded(reading,encoded,inputs.match_index,inputs.prefix,group.one_decision_per_stream)
            states=self.actor.fuse(reading,reader)
            torch.testing.assert_close(states,self.actor.decision_states(encoded,inputs),rtol=0,atol=0)
            for j,i,threshold in picked:
                n=int(r.prefix[i]);lo,hi=map(int,r.offsets[i:i+2]);stream=group.streams[int(fields[0][j])]
                arrays=tuple(a[:n].copy() for a in stream.arrays());past=int((arrays[1]<r.round_index[i]).sum())
                old=encoded[int(fields[0][j]),1:past+1]
                mask=torch.tensor((arrays[0][:past,:4].argmax(1)%2!=r.seat[i]%2)&(arrays[2][:past]==3))
                native=dict(query=query[j],reader=reader[j],state=states[j],memory_mean=old.mean(0),
                            memory_opponent=old[mask].mean(0),memory_last=old[-1])
                self.samples.append(dict(env=int(r.env_id[i]),round=int(r.round_index[i]),threshold=threshold,
                    seat=int(r.seat[i]),prefix=n,original=arrays,native={k:v.numpy().copy() for k,v in native.items()},
                    obs=r.obs[i].copy(),cand=r.cand[lo:hi].copy(),choice=int(result[0][j]),
                    cached_logp=table[j,:hi-lo].numpy().copy()))
        return result


def collect_block(task):
    block,rep_start,root_string=task;root=Path(root_string);torch.set_num_threads(1)
    n=len(json.loads((root/'protocol.json').read_text())['identity']['opponents'])
    indices=list(range(block*5,min((block+1)*5,n)))
    folder=root/'collection'/f'b{block:02d}-r{rep_start:02d}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return json.loads((folder/'complete.json').read_text())
    actor,opponents,identity=load_players(root,indices)
    cells=[(z,rep,team) for z in indices for rep in range(rep_start,min(rep_start+8,REPLICATES)) for team in (0,1)]
    seeds=[int(np.random.default_rng(np.random.SeedSequence([SEED,z,rep])).integers(1,2**63-1)) for z,rep,_ in cells]
    env=gd.VecEnv(num_envs=len(cells),num_threads=1,seed=71,log_public_actions=True,log_env_limit=len(cells))
    env.reset(match_seeds=seeds)
    def assignment(e,m):
        z,rep,team=cells[e];return [0 if s%2==team else z+1 for s in range(4)]
    collector=Collector(env,actor,DiagnosticStore(),SequenceRolloutBuffer(),device='cpu',
        kv_cache=True,seeds=seeds,seat_policy=assignment,resolve_policy=opponents.__getitem__)
    collector.started=True;started=time.monotonic()
    with torch.inference_mode():
        for step in range(3000):
            collector.collect(1)
            if collector.done.all():break
        else:raise RuntimeError('four-round collection incomplete; no censoring')
    collector.tap.save(folder/'public.npz')
    torch.save(collector.samples,folder/'samples.pt')
    meta=np.asarray(collector.tap.meta);actions=np.asarray(collector.tap.actions)
    assert len(meta)>0 and meta.ndim==2, 'public event tap must observe collection'
    features=[];counts=[]
    for sample in collector.samples:
        e=sample['env'];z,rep,team=cells[e]
        m=meta[meta[:,0]==e];a=actions[meta[:,0]==e];start=sample['prefix']
        assert int(m[start,3])==sample['seat'] and int(m[start,1])==sample['round']
        future=np.flatnonzero((np.arange(len(m))>start)&(m[:,1]==sample['round'])&(m[:,3]%2!=team)&(m[:,5]==3))
        if not len(future):continue
        target=int(future[0]);kind=int(a[target,108:121].argmax())
        row=stage_features(actor,sample)
        row.update(env=e,opponent=z,replicate=rep,team=team,seed=seeds[e],round=sample['round'],
            threshold=sample['threshold'],label=0 if kind==0 else 1 if kind==1 else 2 if kind in (8,9,10) else 3,
            prefix=start,target_index=target)
        features.append(row)
    np.savez_compressed(folder/'features.npz',**{k:np.asarray([r[k] for r in features]) for k in features[0]})
    # Save raw inputs for independent recomputation and the current-history perturbation controls.
    torch.save(collector.samples,folder/'samples.pt')
    manifest=dict(complete=True,block=block,rep_start=rep_start,matches=len(cells),features=len(features),
        cells=[dict(opponent=z,replicate=rep,team=team,seed=seed,rounds=int(collector.finished_rounds[e]))
               for e,((z,rep,team),seed) in enumerate(zip(cells,seeds))],
        steps=step+1,seconds=time.monotonic()-started,public_events=len(meta),
        actor_sha=identity['actor_sha'],population_sha=identity['population_sha'],source_sha=sha(__file__),
        protocol_sha=sha(root/'protocol.json'),max_cache_error=max(r['cache_error'] for r in features))
    save_json(folder/'complete.json',manifest)
    return {k:manifest[k] for k in ('block','rep_start','matches','features','seconds','steps')}


def collect_all(root,workers):
    n=len(json.loads((root/'protocol.json').read_text())['identity']['opponents'])
    tasks=[(b,r,str(root)) for b in range((n+4)//5) for r in range(0,REPLICATES,8)]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(collect_block,t) for t in tasks]
        for f in as_completed(futures):print(json.dumps(f.result()),flush=True)


def questions(root):
    torch.set_num_threads(1);_,opponents,_=load_players(root)
    labels,counts=question_bank(opponents,128,64,SEED+1,SEED+2)
    np.savez_compressed(root/'questions.npz',labels=labels,counts=counts)
    save_json(root/'questions-complete.json',dict(questions=128,draws=64,source_sha=sha(__file__),
        calibration=32,evaluation=32,protocol_sha=sha(root/'protocol.json')))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','collect','questions','block'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--block',type=int,default=0);p.add_argument('--rep-start',type=int,default=0)
    args=p.parse_args()
    if args.command=='prepare':prepare(args.root)
    elif args.command=='collect':collect_all(args.root,args.workers)
    elif args.command=='block':print(collect_block((args.block,args.rep_start,str(args.root))),flush=True)
    else:questions(args.root)

if __name__=='__main__':main()
