"""Fresh heldout scoring of sealed next-opponent prediction experiments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from bench.filtered_history_collect import ACTOR,load_players,save_json
from bench.history_next_prediction import (
    ROOT,SEED,SEEDS,PRIMARY,ARMS,CurrentModel,ResidualHead,make_branch,read_all,
)
from bench.qknorm_match_ablation import sha


def row_metrics(p: np.ndarray,y: np.ndarray) -> dict:
    target=np.eye(11)[y]
    coarse=np.stack((p[...,0],p[...,1],p[...,8:11].sum(-1),p[...,2:8].sum(-1)),-1)
    cy=np.where(y==0,0,np.where(y==1,1,np.where(y>=8,2,3)))
    return dict(brier=((p-target)**2).sum(-1),
                log_loss=-np.log(np.maximum(np.take_along_axis(p,y[None,:,None],axis=-1)[...,0],1e-9)),
                accuracy=(p.argmax(-1)==y),coarse_brier=((coarse-np.eye(4)[cy])**2).sum(-1),
                coarse_accuracy=(coarse.argmax(-1)==cy))


def clusters(values: np.ndarray,meta: list) -> np.ndarray:
    """Each paired seed: average captures per match, then the two orientations."""
    out=np.empty((20,8))
    for z in range(20):
        for j,rep in enumerate(range(40,48)):
            parts=[];seeds=[]
            for team in (0,1):
                ids=np.array([r['opponent']==z and r['replicate']==rep and r['team']==team for r in meta])
                assert ids.any(),'missing whole heldout match'
                parts.append(values[ids].mean())
                seeds.append({r['seed'] for r,yes in zip(meta,ids) if yes})
            assert len(seeds[0])==1 and seeds[0]==seeds[1]
            out[z,j]=np.mean(parts)
    return out


def draw_scores(cell: np.ndarray,prior: np.ndarray,draw: np.ndarray):
    return (cell[np.arange(20)[None,:,None],draw].mean(2)*prior[None]).sum(1)


def summarize(p,y,meta,prior,draw):
    values=row_metrics(p,y);result={};all_cells={}
    for name,v in values.items():
        per_seed=np.stack([clusters(x,meta) for x in v]);cell=per_seed.mean(0)
        samples=draw_scores(cell,prior,draw)
        result[name]=dict(mean=float((cell.mean(1)*prior).sum()),
            seeds=(per_seed.mean(2)*prior[None]).sum(1).tolist(),ci95=np.quantile(samples,[.025,.975]).tolist())
        all_cells[name]=cell
    return result,all_cells


def paired_gain(reference,candidate,prior,draw,adjusted=False):
    cell=reference-candidate;values=draw_scores(cell,prior,draw)
    out=dict(mean=float((cell.mean(1)*prior).sum()),ci95=np.quantile(values,[.025,.975]).tolist())
    if adjusted:out['simultaneous_ci']=np.quantile(values,[.05/24,1-.05/24]).tolist()
    return out


def evaluate(root: Path) -> None:
    torch.set_num_threads(1)
    choices=json.loads((root/'choices.json').read_text())
    base_choices=json.loads((root/'base-choices.json').read_text())
    assert choices['dataset_sha']==sha(root/'dataset.pt') and not choices['test_labels_used']
    assert choices['base_choices_sha']==sha(root/'base-choices.json')
    protocol=json.loads((root/'protocol.json').read_text());data=torch.load(root/'dataset.pt',weights_only=False)
    assert protocol['identity']['actor_sha']==sha(ACTOR)
    tr=torch.where(data['split']==0)[0];te=torch.where(data['split']==2)[0]
    meta=[data['metadata'][i] for i in te];y=data['y'][te].numpy()
    prior=np.array(protocol['identity']['weights']);draw=np.random.default_rng(SEED+101).integers(0,8,(5000,20,8))
    actor,_,identity=load_players(indices=[]);before={k:v.clone() for k,v in actor.state_dict().items()}
    full={a:[] for a in ('base',*ARMS)};shuffled={a:[] for a in ARMS};selected={a:[] for a in ARMS};train_scores={a:[] for a in ARMS}
    same_donors=None;same={a:[] for a in PRIMARY}
    if (root/'same-type-donors.pt').exists():same_donors=torch.load(root/'same-type-donors.pt',weights_only=True)
    for seed in SEEDS:
        info=base_choices[str(seed)];folder=root/info['folder']
        assert info['checkpoint_sha']==sha(folder/'selected.pt')
        base=CurrentModel(data,seed);base.load_state_dict(torch.load(folder/'selected.pt',weights_only=True))
        base.eval().requires_grad_(False)
        with torch.no_grad():context=base.encode(data['x']);base_logits=base.output(context)
        full['base'].append(base_logits[te].softmax(-1).numpy())
        for arm in ARMS:
            info=choices['choices'][f'{arm}-{seed}'];folder=root/info['folder']
            assert info['checkpoint_sha']==sha(folder/'selected.pt')
            state=torch.load(folder/'selected.pt',weights_only=False)
            reader=make_branch(data,arm,seed,actor);head=ResidualHead(context[tr],data['sample_weight'][tr],seed)
            reader.load_state_dict(state['reader']);head.load_state_dict(state['head'])
            reader.eval().requires_grad_(False);head.eval().requires_grad_(False)
            def predict(ids,view='full'):
                with torch.no_grad():
                    f=read_all(reader,data,ids,arm,view)
                    return (base_logits[ids]+head(context[ids],f)).softmax(-1).numpy()
            full[arm].append(predict(te));shuffled[arm].append(predict(te,'shuffled'))
            if arm in PRIMARY and same_donors is not None:
                original_donors=data['donors'];data['donors']=same_donors
                same[arm].append(predict(te,'shuffled'));data['donors']=original_donors
            tp=predict(tr);ty=data['y'][tr].numpy();tw=data['sample_weight'][tr].numpy()
            train_scores[arm].append(float(np.average(((tp-np.eye(11)[ty])**2).sum(-1),weights=tw)))
            detail=json.loads((folder/'selected.json').read_text())
            # Inspect the selected head's derivative without updating anything.
            tf=read_all(reader,data,tr,arm).detach().requires_grad_(True)
            logits=base_logits[tr]+head(context[tr],tf)
            loss=torch.nn.functional.cross_entropy(logits,data['y'][tr]);loss.backward()
            norms=tf.grad.norm(dim=-1)
            detail['selected_head_input_gradient']=dict(zero_examples=int((norms==0).sum()),
                                                        examples=len(tr),mean=float(norms.mean()))
            selected[arm].append(detail)
        print(json.dumps(dict(scored_seed=seed)),flush=True)
    predictions={k:np.stack(v) for k,v in full.items()}
    predictions['frequency_prior']=np.broadcast_to(data['prevalence'].numpy(),(3,len(te),11)).copy()
    predictions.update({k+'_shuffled':np.stack(v) for k,v in shuffled.items()})
    if same_donors is not None:predictions.update({k+'_same_type':np.stack(v) for k,v in same.items()})
    results={};cells={}
    for name,p in predictions.items():results[name],cells[name]=summarize(p,y,meta,prior,draw)
    comparisons={}
    for arm in PRIMARY:
        comparisons[arm+'_vs_current_extra']=paired_gain(cells['current_extra']['brier'],cells[arm]['brier'],prior,draw,True)
        comparisons[arm+'_matched_vs_shuffled']=paired_gain(cells[arm+'_shuffled']['brier'],cells[arm]['brier'],prior,draw,True)
    for reference,candidate in (('original_small','original_large'),('improved_small','improved_large'),
                                ('original_small','improved_small'),('original_large','improved_large')):
        comparisons[candidate+'_vs_'+reference]=paired_gain(cells[reference]['brier'],cells[candidate]['brier'],prior,draw,True)
    secondary={arm:paired_gain(cells['base']['brier'],cells[arm]['brier'],prior,draw) for arm in ARMS}
    if same_donors is not None:
        secondary.update({a+'_matched_vs_same_type':paired_gain(cells[a+'_same_type']['brier'],cells[a]['brier'],prior,draw) for a in PRIMARY})
    by_threshold={}
    for threshold in (0,20,50):
        ids=np.array([m['threshold']==threshold for m in meta]);w=data['sample_weight'][te].numpy()[ids]
        by_threshold[str(threshold)]={a:float(np.average(row_metrics(p[:,ids],y[ids])['brier'].mean(0),weights=w)) for a,p in predictions.items()}
    assert all(torch.equal(v,before[k]) for k,v in actor.state_dict().items())
    np.savez_compressed(root/'predictions.npz',**predictions,y=y)
    save_json(root/'summary.json',dict(results=results,comparisons=comparisons,secondary=secondary,
        by_threshold=by_threshold,train_brier=train_scores,selected=selected,
        test_rows=len(te),test_matches=320,test_clusters=160,actor_unchanged=True,identity=identity,
        protocol_sha=sha(root/'protocol.json'),choices_sha=sha(root/'choices.json'),dataset_sha=sha(root/'dataset.pt'),
        scoring_source_sha=sha(__file__),supplementary_controls=(root/'same-type-protocol.json').exists()))
    for arm in ('base',*ARMS):
        print(json.dumps(dict(arm=arm,brier=results[arm]['brier']['mean'],
                              gain_vs_base=0 if arm=='base' else secondary[arm]['mean'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    evaluate(p.parse_args().root)
