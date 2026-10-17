"""Offline training-history audit and backward-only PPO route checks; never steps weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from bench.qknorm_mechanism import unpack
from bench.public_history_diverse import load_actor
from bench.qknorm_match_ablation import sha
from train.history_model import HistoryActor, segment_log_softmax
from train.history_ppo import HistoryPPOConfig
from train.history_aux import next_token_loss, outcome_loss

ROOT=Path('.work/history-training-audit-2026-10-17')
FIXTURE=Path('.work/stack-fixture-2026-10-14/rank1-fixture.pt')


def policy_surrogate(logp: torch.Tensor, old: torch.Tensor, behaviour: torch.Tensor,
                     advantage: torch.Tensor, cfg: Any, moments: tuple | None) -> torch.Tensor:
    if moments is not None:advantage=(advantage-moments[0])/(moments[1]+1e-8)
    elif cfg.normalize_advantages and len(advantage)>1:
        advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
    advantage=advantage*(old-behaviour).exp().clamp(max=cfg.behaviour_weight_cap)
    ratio=(logp-old).exp()
    return -torch.minimum(ratio*advantage,ratio.clamp(1-cfg.clip,1+cfg.clip)*advantage)


def visible_rounds(inputs: Any) -> torch.Tensor:
    """Last-visible round is a conservative lower bound, not future-label lookup."""
    index=(inputs.prefix-1).clamp_min(0)
    rounds=inputs.streams.rounds[inputs.match_index,index]
    return torch.where(inputs.prefix>0,rounds,torch.zeros_like(rounds))


def audit() -> None:
    checkpoints=json.loads((ROOT/'checkpoints.json').read_text());metadata={}
    for name,path in checkpoints.items():
        p=torch.load(path,map_location='cpu',weights_only=False)
        cfg=p['config'];pool=p['population'];snapshots=sorted(pool['snapshots'])
        recent=snapshots[-pool['recent']:]
        metadata[name]=dict(path=path,sha=sha(path),update=p['progress']['updates'],
            model_config=p['model_config'],
            config={k:cfg.get(k) for k in ('snapshot_updates','population_recent','snapshot_probability',
                'population_archive_every','population_archive_share','habit_view','window','gamma','gae_lambda',
                'entropy','aux_coef','response_coef')},
            eligible_recent_updates=[pool['snapshots'][i]['metadata']['update'] for i in recent],
            resident_updates=[pool['snapshots'][i]['metadata']['update'] for i in snapshots],
            archive=pool.get('archive',[]),population_next_id=pool['next_id'])
    f=torch.load(FIXTURE,map_location='cpu',weights_only=False);inputs=unpack(f['batch']['inputs'])
    bounds=visible_rounds(inputs);past=[];fractions=[];anypast=[]
    for i in range(inputs.decisions):
        stream=int(inputs.match_index[i]);n=int(inputs.prefix[i]);bound=int(bounds[i])
        old=int((inputs.streams.rounds[stream,:n]<bound).sum())
        past.append(old);fractions.append(old/max(n,1));anypast.append(old>0)
    fixture=dict(sha=sha(FIXTURE),decisions=inputs.decisions,streams=len(inputs.streams.lengths),
        prefix_min=int(inputs.prefix.min()),prefix_max=int(inputs.prefix.max()),
        lower_bound_rounds=sorted(set(bounds.tolist())),rows_with_definitely_prior_rounds=sum(anypast),
        definitely_old_tokens_range=[min(past),max(past)],mean_definitely_old_fraction=float(np.mean(fractions)),
        note='fixture stores prefix but not current round index; use last-visible round lower bound; never inspect future labels')
    actual={}
    for arm in ('control','norm16'):
        root=Path(f'.work/qknorm-short-ppo-2026-10-16/remote/runs/{arm}')
        events=[json.loads(s) for s in (root/'population.jsonl').read_text().splitlines()]
        assignments=[e for e in events if e['event']=='assignment']
        keys=[(e.get('rollout_session'),e['env'],e['match']) for e in assignments]
        assert len(keys)==len(set(keys))
        snaps={e['identity']:e['update'] for e in events if e['event']=='snapshot'}
        initial=torch.load(checkpoints[arm],map_location='cpu',weights_only=False)
        for i,p in initial['population']['snapshots'].items():snaps[int(i)]=p['metadata']['update']
        del initial
        parent=torch.load('.work/qknorm-mechanism-2026-10-16/remote/qk-both/500M-update-094488.pt',map_location='cpu',weights_only=False)
        for i,p in parent['population']['snapshots'].items():snaps[int(i)]=p['metadata']['update']
        del parent
        alllearner=sum(all(s==0 for s in e['seats']) for e in assignments)
        ages=[e['version']-snaps[s] for e in assignments for s in e['seats'] if s and s in snaps]
        metrics=[json.loads(s) for s in (root/'metrics.jsonl').read_text().splitlines()]
        summary={}
        for k in ('mean_prefix','max_prefix','learn_mean_prefix','learn_max_prefix','encoder_grad_norm','actor_grad_norm',
                  'policy_loss','next_type_accuracy','next_seat_accuracy'):
            a=np.array([m[k] for m in metrics],float);summary[k]=dict(mean=float(a.mean()),min=float(a.min()),max=float(a.max()))
        actual[arm]=dict(assignments=len(assignments),unique_assignment_per_session_match=True,
            all_current_policy_matches=alllearner,all_current_policy_fraction=alllearner/len(assignments),
            known_snapshot_seat_ages=dict(n=len(ages),min=min(ages),max=max(ages),mean=float(np.mean(ages))),
            metrics=summary,updates=len(metrics))
    remote=json.loads(Path('.work/m8-trigger-2026-10-15/remote-evidence.json').read_text())
    historical={}
    for rank,items in remote['metrics'].items():
        if isinstance(items,dict):
            historical[rank]={'keys':list(items)};continue
        historical[rank]=dict(rows=len(items),keys=list(items[0]) if items else [])
    (ROOT/'audit.json').write_text(json.dumps(dict(checkpoints=metadata,fixture=fixture,
        actual_short_run=actual,historical_metric_structure=historical),indent=2)+'\n')
    print(json.dumps(dict(fixture=fixture,actual_short_run=actual),indent=2))


def parameter_group(name: str) -> str:
    if name=='bos' or name.startswith(('public.','round_embedding.','phase_embedding.','stream.','stream_norm.')):return 'encoder'
    if name.startswith(('private.','seat.')):return 'private'
    if name.startswith(('q_proj.','query_norm.','kv_proj.','out_proj.','qk_temperature.')):return 'reader'
    if name.startswith(('next_head.','belief_head.','outcome_head.','response_head.','response_bridge.')):return 'aux_heads'
    return 'policy_fusion'


def backward_case(name: str, fx: dict, actor: Any, mode: str) -> dict:
    actor.requires_grad_(True).train();actor.causal_sdpa=True;actor.batched_match_attention=True
    before={k:v.detach().clone() for k,v in actor.state_dict().items()}
    inputs=unpack(fx['batch']['inputs']);fields=fx['batch']['fields']
    cfg=HistoryPPOConfig.from_payload(fx['config'])
    encoded=[];projected=[]
    hook=actor.public.register_forward_hook(lambda module,args,out:projected.append(out))
    with sdpa_kernel(SDPBackend.MATH):
        state=actor.decision_states(None,inputs,encoded)
        hook.remove()
        logits,response=actor.candidate_outputs(state,inputs.cand,inputs.offsets,predict=True,rows=inputs.rows)
        all_logp=segment_log_softmax(logits,inputs.rows,inputs.decisions)
        logp=all_logp[fx['batch']['chosen']]
        old=fields['logp'] if mode=='recorded' else logp.detach()
        behaviour=fields['behaviour_logp'] if mode=='recorded' else logp.detach()
        per_row=policy_surrogate(logp,old,behaviour,fields['advantage'],cfg,fx['advantage_moments'])
        losses={'ppo':per_row.mean()}
        next_parts=[]
        for memory,group in encoded:
            ids=group.matches if group is not None else torch.arange(len(inputs.streams.lengths))
            count=(memory.shape[1]-1)
            tokens=inputs.streams.tokens[ids,:count]
            local=group.local_match if group is not None else inputs.match_index
            rows=group.rows if group is not None else torch.arange(inputs.decisions)
            cited=torch.zeros(len(ids),dtype=torch.long).scatter_reduce(0,local,inputs.prefix[rows],reduce='amax')
            valid=torch.arange(count)[None]<cited[:,None]
            next_parts.append((actor.next_logits(memory[:,:count]).reshape(-1,actor.next_logits(memory[:,:1]).shape[-1]),
                               tokens.reshape(-1,tokens.shape[-1]),valid.reshape(-1)))
        losses['next_weighted']=cfg.aux_coef*next_token_loss(torch.cat([x[0] for x in next_parts]),
            torch.cat([x[1] for x in next_parts]),torch.cat([x[2] for x in next_parts]))['next_loss']
        losses['outcome_weighted']=cfg.aux_coef*outcome_loss(actor.outcome_value(state),fields['outcome_target'])['outcome_loss']
        losses['response_weighted']=cfg.response_coef*F.cross_entropy(response[fx['batch']['chosen']],fields['response_target'])
        named=list(actor.named_parameters());params=[p for _,p in named];results={};vectors={}
        for lossname,loss in losses.items():
            gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            totals={k:0. for k in ('encoder','private','reader','policy_fusion','aux_heads')};special={}
            for (n,p),g in zip(named,gs):
                if g is None:continue
                assert torch.isfinite(g).all(),(name,lossname,n)
                totals[parameter_group(n)]+=float(g.double().square().sum())
                if n=='q_proj.weight':special['q_weight_norm']=float(g.double().norm())
                if n=='kv_proj.weight':special.update(k_weight_norm=float(g[:128].double().norm()),v_weight_norm=float(g[128:].double().norm()))
            results[lossname]=dict(loss=float(loss.detach()),gradient_norm={k:v**.5 for k,v in totals.items()},**special)
            vectors[lossname]=[g.detach().clone() if g is not None else torch.zeros_like(p) for g,p in zip(gs,params)]
        bounds=visible_rounds(inputs)
        selected=[];chosen_round={}
        for m in range(len(inputs.streams.lengths)):
            ids=torch.nonzero((inputs.match_index==m)&(bounds>=1)).flatten()
            if len(ids):
                i=int(ids[inputs.prefix[ids].argmax()]);selected.append(i);chosen_round[m]=int(bounds[i])
        assert selected
        past_objective=per_row[torch.tensor(selected)].mean()
        taps=[x[0] for x in encoded]+projected
        grads=torch.autograd.grad(past_objective,taps,retain_graph=False,allow_unused=True)
        fractions={}
        for kind,part,bos in [('encoded',grads[:len(encoded)],True),('projected',grads[len(encoded):],False)]:
            assert len(part)==len(encoded)
            old_energy=visible_energy=future_energy=0.
            for (memory,group),g in zip(encoded,part):
                if g is None:continue
                ids=group.matches.tolist() if group is not None else list(range(len(inputs.streams.lengths)))
                for j,m in enumerate(ids):
                    if m not in chosen_round:continue
                    target=next(i for i in selected if int(inputs.match_index[i])==m);prefix=int(inputs.prefix[target])
                    start=1 if bos else 0;count=min(len(g[j])-start,len(inputs.streams.rounds[m]))
                    sq=g[j,start:start+count].double().square().sum(-1)
                    positions=torch.arange(count);visible=positions<prefix
                    oldmask=visible&(inputs.streams.rounds[m,:count]<chosen_round[m])
                    old_energy+=float(sq[oldmask].sum());visible_energy+=float(sq[visible].sum());future_energy+=float(sq[~visible].sum())
            fractions[kind]=dict(old_gradient_norm=old_energy**.5,visible_gradient_norm=visible_energy**.5,
                old_squared_gradient_fraction=old_energy/max(visible_energy,1e-300),future_squared_gradient=future_energy)
        cosines={}
        ppo=vectors['ppo']
        for lossname,v in vectors.items():
            if lossname=='ppo':continue
            a=[g.flatten().double() for (n,_),g in zip(named,ppo) if parameter_group(n)=='encoder']
            b=[g.flatten().double() for (n,_),g in zip(named,v) if parameter_group(n)=='encoder']
            aa=torch.cat(a);bb=torch.cat(b)
            cosines[lossname]=float(torch.dot(aa,bb)/(aa.norm()*bb.norm()).clamp_min(1e-300))
    assert all(torch.equal(v,before[k]) for k,v in actor.state_dict().items())
    assert all(p.grad is None for p in actor.parameters())
    return dict(name=name,mode=mode,losses=results,selected_late_rows=len(selected),old_token_signal=fractions,
        encoder_gradient_cosine_with_ppo=cosines,parameters_unchanged=True,
        clipped_fraction=float(((logp-old).exp().sub(1).abs()>cfg.clip).float().mean()))


def gradients() -> None:
    torch.set_num_threads(1);f=torch.load(FIXTURE,map_location='cpu',weights_only=False)
    cfg=HistoryPPOConfig.from_payload(f['config']);actor=HistoryActor(cfg.policy_config());actor.load_state_dict(f['actor'],strict=True)
    result={'recorded_fixture':backward_case('recorded_fixture',f,actor,'recorded')}
    for arm in ('control','norm16'):
        actor,_=load_actor(Path(json.loads((ROOT/'checkpoints.json').read_text())[arm]))
        result[arm]=backward_case(arm,f,actor,'reanchored_path_check')
    (ROOT/'gradient-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',action='store_true');args=p.parse_args()
    if args.audit:audit()
    else:gradients()
