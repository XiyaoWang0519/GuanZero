"""Frozen native-reader intervention and descriptive attention audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from bench.filtered_history_collect import ROOT,load_players,save_json
from bench.filtered_history_analyze import load_data,weights_for
from bench.qknorm_match_ablation import sha
from train.history_model import StreamBatch,DecisionInputs


def main(root: Path) -> None:
    torch.set_num_threads(1);actor,_,identity=load_players(root,indices=[])
    before={k:v.clone() for k,v in actor.state_dict().items()}
    data=load_data(root);train=(data['replicate']<20)&(data['round']==3)
    mean=np.average(data['reader'][train],axis=0,weights=weights_for(data['opponent'][train],np.array(identity['weights'])))
    mean=torch.tensor(mean[None],dtype=torch.float32)
    records=[];maxp=[];margins=[];past=[];query=[];readers=[];fail=[]
    for folder in sorted((root/'collection').glob('b*-r24')):
        for sample in torch.load(folder/'samples.pt',weights_only=False):
            if sample['round']!=3:continue
            native=sample['native'];q=torch.tensor(native['query'][None]);r=torch.tensor(native['reader'][None])
            candidates=torch.tensor(sample['cand']);n=len(candidates);off=torch.tensor([0,n])
            with torch.inference_mode():
                state=actor.fuse(actor.reader_input(q),r)
                patched=actor.fuse(actor.reader_input(q),mean)
                logits=actor.candidate_logits(state,candidates,off)
                constant=actor.candidate_logits(patched,candidates,off)
                p=logits.softmax(-1).double();pc=constant.softmax(-1).double()
                cached=torch.tensor(sample['cached_logp']).double().exp()
                assert float((p-cached).abs().sum()/2)<1e-4,'native state must reproduce cached policy'
                records.append(dict(tv=float((p-pc).abs().sum()/2),flip=int(p.argmax()!=pc.argmax()),
                    threshold=sample['threshold'],env=sample['env'],folder=folder.name,
                    state_change=float((state-patched).norm()),reader_change=float((r-mean).norm())))
                # Recomputed encoder only for descriptive attention, not native intervention.
                arrays=sample['original'];sb=StreamBatch.from_arrays([arrays],'cpu');enc=actor.encode_batch(sb)
                k,v=actor.project_keys_values(enc);qq=actor.project_query(actor.reader_input(q))[0]
                scores=torch.einsum('hd,shd->hs',qq.double(),k[0].reshape(-1,8,16).double())/4
                prob=scores.softmax(-1);top=scores.topk(2,dim=1)
                maxp.append(prob.max(1).values.numpy());margins.append((top.values[:,0]-top.values[:,1]).numpy())
                boundary=int((arrays[1]<sample['round']).sum())
                past.append(((top.indices[:,0]>0)&(top.indices[:,0]<=boundary)).numpy());query.append(qq.numpy());readers.append(r.numpy()[0])
                mi=torch.zeros(1,dtype=torch.long);pr=sb.lengths
                recompute=actor._attend_encoded(actor.reader_input(q),enc,mi,pr,True)
                if float((recompute-r).abs().max())>1e-3:fail.append(dict(folder=folder.name,env=sample['env'],threshold=sample['threshold'],
                    reader_error=float((recompute-r).abs().max())))
    queries=np.asarray(query);unit=queries/np.linalg.norm(queries,axis=2,keepdims=True)
    directions=queries.mean(0);directions/=np.linalg.norm(directions,axis=1,keepdims=True)
    cosine=(unit*directions).sum(2)
    result=dict(heldout_decisions=len(records),mean_reader_source='training matches only, round 3, actual conditional snapshot weights',
        constant_reader=dict(tv_mean=float(np.mean([r['tv'] for r in records])),tv_median=float(np.median([r['tv'] for r in records])),
            tv_q95=float(np.quantile([r['tv'] for r in records],.95)),tv_max=max(r['tv'] for r in records),
            changed_actions=sum(r['flip'] for r in records)),
        private_attention=dict(onehot_fraction_by_head=(np.array(maxp)>=1-1e-12).mean(0).tolist(),
            minimum_margin_by_head=np.array(margins).min(0).tolist(),old_token_fraction_by_head=np.array(past).mean(0).tolist(),
            median_query_cosine_to_mean_by_head=np.median(cosine,axis=0).tolist(),recomputed_reader_mismatches=fail),
        reader_mean_norm=float(mean.norm()),source_sha=sha(__file__),actor_sha=identity['actor_sha'],
        limitation='Layer intervention tests policy sensitivity, not playing strength; descriptive attention uses recomputed memory and flags numerical deviations')
    for k,v in actor.state_dict().items():assert torch.equal(v,before[k]),'frozen parameters changed'
    save_json(root/'mechanism-summary.json',result)
    save_json(root/'mechanism-records.json',records);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args();main(args.root)
