"""Bounded tensor-only QK norm-16 CUDA/full-actor-loss diagnostic.

No HistoryTrainer constructor/run/learn, collection, checkpoint writing or training loop.
The production minibatch_loss method receives an adapter for an already recorded batch.
One disposable Adam step per independent case validates migrated state consumption.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
from dataclasses import replace
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from bench.qknorm_mechanism import load_actor, sha, unpack
from bench.qknorm_plasticity import NAMES, difference, factors, migrate, scale_slices
from train import history_ppo as ppo
from train.history_model import HistoryCritic
from train.history_rollout import TrainingBatch


def move(value,device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,list):return [move(v,device) for v in value]
    return value


def plain_forward(actor,inputs,math=False):
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH) if math else nullcontext():
        return actor.candidate_log_probs(inputs).detach().double().cpu()


@contextmanager
def recorded_labels(fields):
    previous=(ppo.opponent_response_labels,ppo.round_outcomes)
    ppo.opponent_response_labels=lambda *a,**kw:fields["response_target"]
    ppo.round_outcomes=lambda *a,**kw:fields["outcome_target"]
    try:yield
    finally:ppo.opponent_response_labels,ppo.round_outcomes=previous


def production_terms(actor,critic,payload,fx,inputs,fields):
    """Invoke the owning objective, with no trainer lifecycle or rollout allocation."""
    cfg=ppo.HistoryPPOConfig.from_payload(payload["config"])
    chosen=fx["batch"]["chosen"].to(inputs.obs.device)
    batch=TrainingBatch(inputs,chosen,fields)
    trainer=object.__new__(ppo.HistoryTrainer)
    trainer.actor,trainer.critic=actor,critic
    trainer.config,trainer.progress=cfg,copy.deepcopy(payload["progress"])
    trainer.device=inputs.obs.device
    trainer.buffer=SimpleNamespace(training_batch=lambda *a,**kw:batch)
    trainer.store=None
    trainer.learn_mark=lambda *a:None
    trainer.advantage_moments=fx["advantage_moments"]
    with recorded_labels(fields):
        return trainer.minibatch_loss(np.arange(inputs.decisions))


def clipped(actor,limit):
    before={n:p.grad.detach().clone() for n,p in actor.named_parameters() if p.grad is not None}
    norm64=float(torch.stack([g.double().square().sum() for g in before.values()]).sum().sqrt())
    native=float(torch.nn.utils.clip_grad_norm_(actor.parameters(),limit))
    assert np.isfinite(native) and abs(native-norm64)<=1e-4*max(1.,norm64)
    coefficient=min(1.,limit/(native+1e-6))
    for n,p in actor.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
            torch.testing.assert_close(p.grad,before[n]*coefficient,atol=1e-7,rtol=1e-6)
    post=float(torch.stack([p.grad.double().square().sum() for p in actor.parameters()
                            if p.grad is not None]).sum().sqrt())
    assert post<=limit*(1+1e-5)
    return dict(norm64_before=norm64,native_norm_before=native,coefficient=coefficient,norm64_after=post)


def loss_case(actor,critic,payload,fx,inputs,fields,math=False):
    actor.zero_grad(set_to_none=True)
    trace=[]
    original=F.scaled_dot_product_attention
    def spy(q,k,v,*args,**kwargs):
        out=original(q,k,v,*args,**kwargs)
        if kwargs.get("attn_mask") is not None:
            trace.append(dict(shape=list(q.shape),autograd_node=type(out.grad_fn).__name__))
        return out
    F.scaled_dot_product_attention=spy
    try:
        with sdpa_kernel(SDPBackend.MATH) if math else nullcontext():
            terms=production_terms(actor,critic,payload,fx,inputs,fields)
            expected=(terms["policy_loss"]-payload["config"]["entropy"]*terms["entropy"]+
                      payload["config"]["response_coef"]*terms["response_loss"]+
                      terms["aux_coefficient"]*(terms["next_loss"]+terms["belief_loss"]+terms["outcome_loss"]))
            torch.testing.assert_close(terms["policy_total"],expected,atol=1e-6,rtol=1e-6)
            assert all(torch.isfinite(v).all() for v in terms.values())
            terms["policy_total"].backward()
    finally:F.scaled_dot_product_attention=original
    gradients={n:p.grad.detach().clone() for n,p in actor.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(g).all() for g in gradients.values())
    clip=clipped(actor,payload["config"]["grad_clip"])
    # Explicit active-clipping stress: same full gradient vector, scaled to norm 20.
    factor=20/max(clip["norm64_before"],1e-30)
    for n,p in actor.named_parameters():
        if n in gradients:p.grad=gradients[n]*factor
    stress=clipped(actor,payload["config"]["grad_clip"])
    assert .49<stress["coefficient"]<.51
    for n,p in actor.named_parameters():
        if n in gradients:p.grad=gradients[n]*clip["coefficient"]
    record=dict(terms={k:float(v.detach()) for k,v in terms.items()},clip=clip,
                active_clipping_stress=stress,private_sdpa=trace,
                gradient_norms={n:float(g.double().norm()) for n,g in gradients.items()})
    return record,gradients


def gradient_compare(base,other,actor,scale):
    mapped={n:g.clone() for n,g in other.items()}
    scale_slices(actor,mapped,scale)
    difference64=torch.stack([(mapped[n].double()-g.double()).square().sum() for n,g in base.items()]).sum().sqrt()
    norm64=torch.stack([g.double().square().sum() for g in base.values()]).sum().sqrt()
    relative=float(difference64/norm64.clamp_min(1e-30))
    assert relative<1e-3,relative
    return dict(coordinate_mapped_global_relative_error=relative)


def adam_once(actor,state):
    optimizer=torch.optim.Adam(actor.parameters(),lr=state["param_groups"][0]["lr"])
    optimizer.load_state_dict(copy.deepcopy(state))
    old=[float(optimizer.state[p]["step"]) for p in actor.parameters() if p.grad is not None]
    optimizer.step()
    new=[float(optimizer.state[p]["step"]) for p in actor.parameters() if p.grad is not None]
    assert all(b==a+1 for a,b in zip(old,new))
    assert all(torch.isfinite(p).all() for p in actor.parameters())
    assert all(torch.isfinite(v).all() for st in optimizer.state.values() for v in st.values() if isinstance(v,torch.Tensor))
    return dict(all_parameters_and_optimizer_finite=True,steps_incremented_once=True)


def snapshot_gate(base,migrated,payload,device):
    from test_history_snapshot_batch import Step,merged,reference
    policies={0:base}; changed={0:migrated}
    snapshot_factors={}
    for i,entry in enumerate(payload["population"]["snapshots"].values(),start=1):
        a=copy.deepcopy(base)
        missing,unexpected=a.load_state_dict(entry["model"],strict=False)
        assert not unexpected and set(missing)==a.auxiliary_parameter_names()
        b=copy.deepcopy(a);scale=factors(b,16.)
        scale_slices(b,dict(b.named_parameters()),scale)
        policies[i],changed[i]=a,b
        snapshot_factors[str(i)]={s:v.cpu().tolist() for s,v in scale.items()}
    step=Step([(i,1+i%3,1) for i in policies],seed=20261016,max_candidates=12,longest=96,shorter_prefix=True)
    with torch.no_grad():
        groups,layout,fields,original=merged(step,policies,device)
        expected=reference(step,policies,groups,device)
        _,_,_,scaled=merged(step,changed,device)
        target=reference(step,changed,groups,device)
    pre=float((original-expected).abs().max());post=float((scaled-target).abs().max())
    delta=float((original-scaled).abs().max())
    assert max(pre,post,delta)<=1e-4,(pre,post,delta)
    return dict(identities=len(policies),frozen_snapshots=len(policies)-1,
                max_merged_vs_reference_before=pre,max_merged_vs_reference_after=post,
                max_migration_logp_delta=delta,own_head_scale_factors=snapshot_factors)


def graph_gate(actor,device):
    if device=="cpu":return dict(skipped="CPU precheck")
    from test_history_snapshot_batch import Step
    from train.history_cuda_graphs import PrivateDecisionGraphs
    step=Step([(0,3,3)],seed=20261017,max_candidates=8,longest=48)
    inputs=step.inputs(step.groups()[0],device)
    actor.batched_private_attention=True
    with torch.no_grad():
        encoded=actor.encode_batch(inputs.streams)
        expected=actor.candidate_log_probs(inputs,encoded)
        graph=PrivateDecisionGraphs(actor,max_bytes=256<<20,admit_after=1,warmup=2)
        first=graph.log_probs(inputs,encoded)
        second=graph.log_probs(inputs,encoded)
        delta=max(float((expected-first).abs().max()),float((expected-second).abs().max()))
        metrics=graph.metrics()
    assert delta<=1e-4 and metrics["captures"]>=1 and metrics["hits"]>=1,(delta,metrics)
    return dict(max_logp_error=delta,metrics=metrics)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--fixture",type=Path,required=True)
    p.add_argument("--batches",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(20261016)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    base,payload=load_actor(args.checkpoint)
    for name in ("train/history_model.py","train/history_ppo.py","train/history_aux.py", "train/history_snapshot_batch.py"):
        source=Path(inspect.getfile(type(base))).parents[1]/name
        assert sha(source)==payload["run_identity"]["source"]["files"][name],name
    raw=torch.load(args.fixture,map_location="cpu",weights_only=False)
    assert sha(args.checkpoint)=="a4bed2bfb4c3b8c1382a8ed5604e8e797307a37ea2088523dffb1243de6a58bd"
    assert sha(args.fixture)=="9c3d1018ac52453852536ff5ef497eb1dbcbd553f02e199b30c9570e6c152cd8"
    before={n:v.clone() for n,v in base.state_dict().items()}
    scale=factors(base,16.);candidate=copy.deepcopy(base)
    scale_slices(candidate,dict(candidate.named_parameters()),scale)
    migrated=migrate(candidate,payload["optimizer"]["actor"],scale,"transport")
    for n,v in candidate.state_dict().items():
        if n not in NAMES:assert torch.equal(v,before[n])
        elif n.startswith("kv_"):assert torch.equal(v[128:],before[n][128:])
    base=base.to(args.device).train();candidate=candidate.to(args.device).train()
    critic=HistoryCritic(base.config).to(args.device);critic.load_state_dict(payload["critic"])
    critic.requires_grad_(False)
    inputs=unpack(move(raw["batch"]["inputs"],args.device))
    fields=move(raw["batch"]["fields"],args.device)
    checkpoint_hash=sha(args.checkpoint)
    result=dict(checkpoint_sha256=checkpoint_hash,torch=torch.__version__,device=args.device,
        gpu=torch.cuda.get_device_name(0) if args.device=="cuda" else None,
        source_sha256=sha(Path(__file__)),forward={},loss_cases={},source_checkpoint_unchanged=False)
    def save():
        (args.output/"gate.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    datasets={"fixture":inputs}
    for path in sorted(args.batches.glob("fresh-*.pt")):
        datasets[path.stem]=unpack(move(torch.load(path,map_location="cpu",weights_only=False),args.device))
    for n,i in datasets.items():
        a,b=plain_forward(base,i),plain_forward(candidate,i)
        d=difference(a,b,replace(i,offsets=i.offsets.cpu()))
        assert d["max_abs_logp"]<=1e-4 and d["tv"]["max"]<=1e-4 and d["argmax_flips"]==0,d
        result["forward"][n]=d
    save()
    chosen=raw["batch"]["chosen"]
    old=plain_forward(base,inputs)[chosen].to(args.device).float()
    for context in ("stored","ratio_one","ratio_clipped"):
        backend_reference=None
        row={k:v.clone() for k,v in fields.items()}
        if context!="stored":
            shift=0 if context=="ratio_one" else torch.where(row["advantage"]>=0,.3,-.3)
            row["logp"]=old-shift;row["behaviour_logp"]=row["logp"].clone()
        for path,math in (("grouped_default",False),("grouped_math",True),("scalar_default",False)):
            print(context,path,flush=True)
            a,b=copy.deepcopy(base),copy.deepcopy(candidate)
            a.batched_match_attention=b.batched_match_attention=path!="scalar_default"
            ra,ga=loss_case(a,critic,payload,raw,inputs,row,math)
            rb,gb=loss_case(b,critic,payload,raw,inputs,row,math)
            diff=gradient_compare(ga,gb,b,{s:v.to(args.device) for s,v in scale.items()})
            if path=="grouped_default":backend_reference=(ga,gb)
            else:
                unit={s:torch.ones_like(v,device=args.device) for s,v in scale.items()}
                diff["original_vs_grouped_default"]=gradient_compare(backend_reference[0],ga,a,unit)
                diff["migrated_vs_grouped_default"]=gradient_compare(backend_reference[1],gb,b,unit)
            loss_delta=abs(ra["terms"]["policy_total"]-rb["terms"]["policy_total"])
            assert loss_delta<=1e-4+1e-5*abs(ra["terms"]["policy_total"]),loss_delta
            step_a=adam_once(a,payload["optimizer"]["actor"])
            step_b=adam_once(b,migrated)
            result["loss_cases"][context+"/"+path]=dict(original=ra,migrated=rb,
                 gradient_compare=diff,loss_delta=loss_delta,adam_original=step_a,adam_migrated=step_b)
            save()
    result["snapshot_gate"]=snapshot_gate(base,candidate,payload,args.device);save()
    result["private_graph"]=graph_gate(copy.deepcopy(candidate),args.device)
    assert all(torch.equal(v.detach().cpu(),before[n]) for n,v in base.state_dict().items())
    assert sha(args.checkpoint)==checkpoint_hash
    result["source_checkpoint_unchanged"]=True
    result["passed"]=True
    save();print("GATE PASSED",flush=True)


if __name__=="__main__":main()
