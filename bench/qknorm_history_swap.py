"""Frozen same-state equal-length past-content probe; no optimizer or training."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import bench.qknorm_match_ablation as matches
from bench.public_history_probe import PublicCollector
from bench.public_history_diverse import load_actor, load_selected
from bench.qknorm_match_summary import ENDPOINTS
from train.history_model import DecisionInputs, StreamBatch

ROOT = Path('.work/qknorm-history-swap-2026-10-16')
POOL = Path('.work/public-history-diverse-2026-10-16')


def splice(original, donor, current_round, keep_metadata=False):
    """Replace only old events; keep total/current lengths and absolute positions."""
    t,r,p = original
    dt,dr,dp = donor
    n = int(np.count_nonzero(r < current_round))
    assert n > 0 and np.all(r[:n] < current_round) and np.all(r[n:] == current_round)
    assert len(dt) >= n and np.all(dr < current_round)
    out = [x.copy() for x in original]
    out[0][:n] = dt[-n:]
    if not keep_metadata:
        out[1][:n],out[2][:n] = dr[-n:],dp[-n:]
    assert all(np.array_equal(a[n:],b[n:]) for a,b in zip(original,out))
    assert len(out[0]) == len(t) and not out[0][:,150:158].any()
    return tuple(out)


class CaptureCollector(PublicCollector):
    last = None
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.samples=[]
        self.seen=set()
        CaptureCollector.last=self

    def _infer_group(self,group,fields,r,stats,mark):
        if group.identity==0:
            selected=[]
            for local,i in enumerate(group.rows):
                key=(int(r.env_id[i]),int(r.round_index[i]))
                if r.match_id[i]==0 and key[1] in (1,3,7) and r.counts[i]>1 and key not in self.seen:
                    self.seen.add(key)
                    selected.append((local,i,key))
            if selected:
                streams,encoded=self._encode_streams(group,self.actor,fields[1],r.prefix)
                inputs=DecisionInputs(streams,fields[0],fields[1],fields[2],fields[3],fields[4],fields[5],
                                      fields[6],one_decision_per_stream=group.one_decision_per_stream)
                table=self.actor._log_prob_table(inputs,encoded,group.max_candidates)
                for local,i,key in selected:
                    stream=group.streams[int(fields[0][local])]
                    n=int(r.prefix[i]); a,b=int(r.offsets[i]),int(r.offsets[i+1])
                    self.samples.append(dict(env=key[0],round=key[1],seat=int(r.seat[i]),
                        original=tuple(x[:n].copy() for x in stream.arrays()),
                        obs=r.obs[i].copy(),cand=r.cand[a:b].copy(),
                        cached_logp=table[local,:b-a].cpu().numpy().copy()))
        return super()._infer_group(group,fields,r,stats,mark)


def donor_bank(arm):
    bank=[]
    sources=[(name,wave) for name in ('diverse','close') for wave in list(range(8))+list(range(12,16))]
    for name,wave in sources:
        root=POOL if name=='diverse' else Path('.work/public-history-probe-2026-10-16')
        with np.load(root/'raw'/f'wave-{wave:02d}'/f'{arm}-public.npz') as raw:
            meta,actions=raw['public_meta'],raw['public_actions']
            for e in range(32):
                take=meta[:,0]==e
                m,a=meta[take],actions[take]
                t=np.zeros((len(m),186),np.uint8)
                t[np.arange(len(m)),m[:,3]]=1
                t[:,4:150]=a
                t[np.arange(len(m)),158+m[:,4]]=1
                for k in (1,3,7):
                    # Any k consecutive completed rounds in an independent archived
                    # match may supply public content. Choose longest by token count
                    # only, before evaluating policy responses; rebase round labels.
                    windows=[]
                    for start in range(int(m[:,1].max())+2-k):
                        mask=(m[:,1]>=start)&(m[:,1]<start+k)
                        windows.append((int(mask.sum()),start,mask))
                    if not windows:
                        continue
                    _,start,past=max(windows,key=lambda x:(x[0],-x[1]))
                    bank.append(dict(source=name,wave=wave,env=e,round=k,window_start=start,opponent=(e//2)%4+(4 if name=='close' else 0),
                                     arrays=(t[past],m[past,1].astype(np.int64)-start,m[past,5].astype(np.int64))))
    return bank


def measure(actor,sample,donor):
    original=sample['original']; k=sample['round']
    natural=splice(original,donor,k)
    fixed=splice(original,donor,k,True)
    sb=StreamBatch.from_arrays([original,tuple(x.copy() for x in original),natural,fixed],'cpu')
    c=len(sample['cand'])
    inputs=DecisionInputs(sb,torch.arange(4),torch.full((4,),len(original[0])),
        torch.as_tensor(np.repeat(sample['obs'][None],4,axis=0)),
        torch.full((4,),sample['seat']),torch.as_tensor(np.tile(sample['cand'],(4,1))),
        torch.arange(5)*c,one_decision_per_stream=True)
    table=actor._log_prob_table(inputs,None,c).double()
    assert torch.isfinite(table).all()
    p=table.exp(); p=p/p.sum(1,keepdim=True)
    out=dict(copy_max_logp=float((table[0]-table[1]).abs().max()),
             cache_max_logp=float((table[0]-torch.tensor(sample['cached_logp'])).abs().max()),
             choices=c,prefix=len(original[0]),past=int((original[1]<k).sum()))
    for i,name in ((2,'swap'),(3,'fixed_metadata')):
        out[name]=dict(tv=float((p[0]-p[i]).abs().sum()/2),
            kl=float((p[0]*(p[0].log()-p[i].log())).sum()),
            flip=int(p[0].argmax()!=p[i].argmax()),
            max_probability_change=float((p[0]-p[i]).abs().max()))
    assert out['copy_max_logp']<1e-7 and out['cache_max_logp']<1e-4
    return out


def run(wave):
    torch.set_num_threads(1)
    folder=ROOT/f'wave-{wave:02d}'; folder.mkdir(exist_ok=False)
    opponents,opponent_meta=load_selected(POOL)
    seeds=np.random.default_rng(16102671).integers(1,2**63-1,size=(16,16))
    schedule=np.repeat(seeds[wave],2).tolist()
    original_class=matches.EvalCollector
    matches.EvalCollector=CaptureCollector
    rows=[]
    metadata={}
    try:
        for arm in ENDPOINTS:
            path=Path(f'.work/qknorm-short-ppo-2026-10-16/remote/runs/{arm}/endpoint.pt')
            actor,identity=load_actor(path)
            assert identity['checkpoint_sha']==ENDPOINTS[arm]
            metadata[arm]=identity
            result=matches.run_wave(actor,opponents,schedule,'cpu','full',20000)
            reference=json.loads((POOL/'raw'/f'wave-{wave:02d}'/f'{arm}-matches.json').read_text())
            assert result['records']==reference['records'],'capture changed trajectory'
            samples=CaptureCollector.last.samples
            torch.save(samples,folder/f'{arm}-samples.pt')
            bank=donor_bank(arm)
            rng=np.random.default_rng([16102711,wave,list(ENDPOINTS).index(arm)])
            with torch.inference_mode():
                for index,s in enumerate(samples):
                    n=int((s['original'][1]<s['round']).sum())
                    eligible=[d for d in bank if d['round']==s['round'] and len(d['arrays'][0])>=n]
                    assert eligible,'no equal-length archived donor; do not drop sample'
                    d=eligible[int(rng.integers(len(eligible)))]
                    donor=tuple(x.copy() for x in d['arrays'])
                    rotation=int(d['env']%2 != s['env']%2)
                    if rotation:
                        donor[0][:,:4]=np.roll(donor[0][:,:4],1,axis=1)
                    v=measure(actor,s,donor)
                    v.update(arm=arm,wave=wave,index=index,env=s['env'],round=s['round'],
                        seed=schedule[s['env']],opponent=(s['env']//2)%4,
                        donor_wave=d['wave'],donor_env=d['env'],donor_opponent=d['opponent'],
                        donor_window_start=d['window_start'],donor_seat_rotation=rotation,donor_source=d['source'])
                    rows.append(v)
            print(json.dumps(dict(wave=wave,arm=arm,samples=len(samples),full_replay=True)),flush=True)
    finally:
        matches.EvalCollector=original_class
    (folder/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
    (folder/'complete.json').write_text(json.dumps(dict(endpoints=metadata,opponents=opponent_meta,
        runner_sha=matches.sha(__file__),model_sha=matches.sha(__import__('train.history_model',fromlist=['x']).__file__),
        full_replay=True,torch=torch.__version__))+'\n')


def summarize():
    rows=[]
    for w in range(8,12):
        rows.extend(json.loads((ROOT/f'wave-{w:02d}'/'rows.json').read_text()))
    result={}
    for arm in ENDPOINTS:
        rs=[r for r in rows if r['arm']==arm]
        seeds=sorted({r['seed'] for r in rs})
        draws=np.random.default_rng(16102712).integers(len(seeds),size=(10000,len(seeds)))
        out=dict(n=len(rs),matches=len({(r['seed'],r['env']%2) for r in rs}),seed_clusters=len(seeds),
                 by_round={k:sum(r['round']==k for r in rs) for k in (1,3,7)},
                 copy_max_logp=max(r['copy_max_logp'] for r in rs),
                 cache_max_logp=max(r['cache_max_logp'] for r in rs))
        for condition in ('swap','fixed_metadata'):
            stats={}
            for metric in ('tv','flip','max_probability_change'):
                means=np.array([np.mean([r[condition][metric] for r in rs if r['seed']==s]) for s in seeds])
                stats[metric]=dict(seed_mean=float(means.mean()),ci95=np.quantile(means[draws].mean(1),[.025,.975]).tolist(),
                                  row_mean=float(np.mean([r[condition][metric] for r in rs])))
            values=np.array([r[condition]['tv'] for r in rs])
            stats.update(tv_p50=float(np.median(values)),tv_p95=float(np.quantile(values,.95)),
                         tv_max=float(values.max()),flips=sum(r[condition]['flip'] for r in rs),
                         rows_tv_gt_001=int((values>.01).sum()),rows_tv_gt_01=int((values>.1).sum()),
                         by_round={k:dict(n=sum(r['round']==k for r in rs),
                             tv=float(np.mean([r[condition]['tv'] for r in rs if r['round']==k])),
                             flips=sum(r[condition]['flip'] for r in rs if r['round']==k)) for k in (1,3,7)})
            out[condition]=stats
        result[arm]=out
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wave',type=int);p.add_argument('--summary',action='store_true')
    args=p.parse_args()
    if args.summary:summarize()
    else:run(args.wave)
