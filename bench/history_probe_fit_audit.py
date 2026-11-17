"""Repeat three representative diagnostic fits to measure final train loss.

Original validation-selected models/results remain untouched. This checks
optimization versus generalization; it never changes gameplay model weights.
"""
from pathlib import Path
import json
import torch
from torch import nn
from torch.nn import functional as F
from bench.history_identification_probes import ROOT,SEED,STEPS,EVAL_LENGTHS,Scale,Probe,batch_forward,predict
from bench.filtered_history_collect import save_json
from bench.qknorm_match_ablation import sha


def run(root:Path=ROOT):
    torch.set_num_threads(1);meta=torch.load(root/'features/meta.pt',weights_only=False)
    tr=torch.where(meta['split']==0)[0];va=torch.where(meta['split']==1)[0]
    weights=meta['prior'][meta['y'][tr]];weights/=weights.sum()
    wv=meta['prior'][meta['y'][va]];wv/=wv.sum();results={}
    for name in ('raw_tokens','stream_norm','native'):
        x=torch.load(root/'features'/f'{name}.pt',weights_only=False)
        scale=Scale(x,tr.tolist(),name=='native');model=Probe(x[0].shape[-1],meta['prior'],name=='native',71)
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
        rng=torch.Generator().manual_seed(SEED+71);curve=[]
        for step in range(STEPS+1):
            if step%200==0:
                model.eval()
                with torch.no_grad():
                    p=predict(model,scale,x,meta,tr,3);q=predict(model,scale,x,meta,va,3)
                    train=float(-weights@p[torch.arange(len(tr)),meta['y'][tr]].clamp_min(1e-30).log())
                    validation=float(-wv@q[torch.arange(len(va)),meta['y'][va]].clamp_min(1e-30).log())
                curve.append(dict(step=step,train_ce=train,validation_ce=validation))
            if step==STEPS:break
            model.train();ids=tr[torch.multinomial(weights,16,replacement=True,generator=rng)]
            lengths=[EVAL_LENGTHS[int(j)] for j in torch.randint(3,(16,),generator=rng)]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(batch_forward(model,scale,x,meta,ids,lengths),meta['y'][ids]).backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        results[name]=curve
        print(json.dumps(dict(arm=name,final=curve[-1])),flush=True)
    save_json(root/'fit-audit.json',dict(scope='three repeated independent heads, train/validation only; no test selection or replacement of original results',
        results=results,source_sha=sha(__file__)))


if __name__=='__main__':run()
