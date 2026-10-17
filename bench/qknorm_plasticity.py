"""Offline Q/K rescaling and one-step Adam counterfactuals on disposable CPU copies.

Requires the matching QK-norm branch first on PYTHONPATH, as qknorm_mechanism.py.
No collection, trainer, checkpoint writing, or repeated optimization is performed.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import inspect
import json
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from bench.qknorm_mechanism import load_actor, policy, sha, stats, unpack
from train.history_aux import belief_loss, outcome_loss

NAMES = ("q_proj.weight", "q_proj.bias", "kv_proj.weight", "kv_proj.bias")


def blocks(actor, values: dict) -> dict:
    width, heads = actor.config.width, actor.config.heads
    result = {}
    for side, prefix in (("q", "q_proj"), ("k", "kv_proj")):
        w, b = values[prefix+".weight"][:width], values[prefix+".bias"][:width]
        result[side] = torch.cat((w.reshape(heads, -1), b.reshape(heads, -1)), dim=1).double()
    return result


def factors(actor, target: float) -> dict:
    b = blocks(actor, dict(actor.named_parameters()))
    return {s: (target / v.norm(dim=1)).clamp(max=1).float() for s,v in b.items()}


def scale_slices(actor, values: dict, scale: dict, power: int = 1, zero: bool = False) -> None:
    depth, width = actor.config.width // actor.config.heads, actor.config.width
    with torch.no_grad():
        for side,prefix in (("q", "q_proj"), ("k", "kv_proj")):
            f = scale[side].repeat_interleave(depth)
            for suffix in ("weight", "bias"):
                v = values[prefix+"."+suffix][:width]
                if zero:
                    v.zero_()
                else:
                    v.mul_(f.pow(power).reshape((-1,)+(1,)*(v.ndim-1)))


def forward(actor, inputs):
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        state = actor.decision_states(None, inputs)
        return policy(actor,state,inputs)


def rollout_forwards(actor, inputs):
    """Same fixed prefixes through scalar-private and independent-private CPU paths."""
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        encoded=actor.encode_batch(inputs.streams)
        previous=(actor.batched_match_attention,actor.batched_private_attention)
        try:
            actor.batched_match_attention=False
            dense=replace(inputs,match_groups=None)
            scalar=policy(actor,actor.decision_states(encoded,dense),dense)
            # 16 evenly spaced rows keep the independent-stream control bounded.
            pick=torch.linspace(0,inputs.decisions-1,min(16,inputs.decisions)).long()
            matches=inputs.match_index[pick]
            candidates=[inputs.cand[inputs.offsets[i]:inputs.offsets[i+1]] for i in pick.tolist()]
            counts=torch.tensor([len(c) for c in candidates])
            streams=replace(inputs.streams,tokens=inputs.streams.tokens[matches],
                rounds=inputs.streams.rounds[matches],phases=inputs.streams.phases[matches],
                lengths=inputs.streams.lengths[matches])
            independent=replace(inputs,obs=inputs.obs[pick],seat=inputs.seat[pick],prefix=inputs.prefix[pick],
                match_index=torch.arange(len(pick)),cand=torch.cat(candidates),
                offsets=torch.cat((torch.zeros(1,dtype=torch.long),counts.cumsum(0))),
                candidate_rows=None,one_decision_per_stream=True,match_groups=None,
                match_rank=None,match_slots=0,streams=streams)
            actor.batched_private_attention=True
            result=policy(actor,actor.decision_states(encoded[matches],independent),independent)
            return {"scalar_private":(scalar,dense),"independent_private":(result,independent)}
        finally:
            actor.batched_match_attention,actor.batched_private_attention=previous


def difference(base, other, inputs) -> dict:
    # Re-normalize in float64 so tiny FP32 log-softmax errors do not create negative KL.
    a, b = [], []
    flips = 0
    for lo,hi in zip(inputs.offsets[:-1].tolist(),inputs.offsets[1:].tolist()):
        a.append(base[lo:hi].log_softmax(0));b.append(other[lo:hi].log_softmax(0))
        flips += int(base[lo:hi].argmax()!=other[lo:hi].argmax())
    kl = torch.stack([(x.exp()*(x-y)).sum() for x,y in zip(a,b)])
    tv = torch.stack([.5*(x.exp()-y.exp()).abs().sum() for x,y in zip(a,b)])
    return dict(kl=stats(kl),tv=stats(tv),max_abs_logp=float((base-other).abs().max()),
                max_probability_delta=float((base.exp()-other.exp()).abs().max()),
                argmax_flips=flips,rows=inputs.decisions)


def gradients(actor, inputs, fields, objective: str) -> dict:
    actor.zero_grad(set_to_none=True)
    with sdpa_kernel(SDPBackend.MATH):
        state = actor.decision_states(None,inputs)
        if objective=="outcome":
            loss = outcome_loss(actor.outcome_value(state),fields["outcome_target"])["outcome_loss"]
        elif objective=="belief":
            loss = belief_loss(actor.belief_logits(state),fields["hidden"])["belief_loss"]
        else:
            logp = policy(actor,state,inputs)
            loss = (logp.exp()*logp).sum()/inputs.decisions  # negative policy entropy
        loss.backward()
    grads = {n:p.grad.detach().clone() for n,p in actor.named_parameters() if p.grad is not None}
    assert all(torch.isfinite(g).all() for g in grads.values())
    return dict(loss=float(loss.detach()), grads=grads)


def directions(actor, grads) -> dict:
    theta, gradient = blocks(actor,dict(actor.named_parameters())),blocks(actor,grads)
    report = {}
    for side,t in theta.items():
        g = gradient[side]
        norm = t.norm(dim=1)
        radial = (t*g).sum(dim=1)/norm
        tangent = g-t*((t*g).sum(dim=1)/norm.square())[:,None]
        tn = tangent.norm(dim=1)
        report[side] = dict(theta_norm=norm.tolist(), tangent_gradient=tn.tolist(),
            sgd_direction_rate=(tn/norm).tolist(), intrinsic_direction_gradient=(tn*norm).tolist(),
            radial_fraction=(radial.abs()/g.norm(dim=1).clamp_min(1e-300)).tolist())
    return report


def map_states(actor, saved):
    names = [n for n,p in actor.named_parameters()]
    assert len(saved["param_groups"])==1
    ids = saved["param_groups"][0]["params"]
    assert len(names)==len(ids)
    result = dict(zip(names,ids))
    for n,p in actor.named_parameters():
        if result[n] in saved["state"]:
            assert saved["state"][result[n]]["exp_avg"].shape==p.shape
    return result


def migrate(actor, saved, scale, mode):
    saved = copy.deepcopy(saved)
    ids = map_states(actor,saved)
    moments = {key:{n:saved["state"][ids[n]][key] for n in NAMES}
               for key in ("exp_avg","exp_avg_sq")}
    if mode=="transport":
        scale_slices(actor,moments["exp_avg"],scale,-1)
        scale_slices(actor,moments["exp_avg_sq"],scale,-2)
    elif mode=="reset_slice_old_step":
        for m in moments.values():scale_slices(actor,m,scale,zero=True)
    elif mode=="reset_whole_parameter":
        # Deliberately unsafe counterexample: also clears V and its shared clock.
        for n in NAMES:
            state = saved["state"][ids[n]]
            state["exp_avg"].zero_();state["exp_avg_sq"].zero_();state["step"].zero_()
    else:
        assert mode=="keep"
    return saved


def simulate(actor, saved, grads, reference, scale, mode, grad_clip):
    a = copy.deepcopy(actor)
    migrated = migrate(a,saved,scale,mode)
    ids=map_states(a,saved)
    width=a.config.width
    # Verify state surgery preserves every unrelated moment and V slice.
    for n,i in ids.items():
        if i not in saved["state"]:continue
        old,new=saved["state"][i],migrated["state"][i]
        if mode!="reset_whole_parameter":
            assert torch.equal(old["step"],new["step"])
            for key in ("exp_avg","exp_avg_sq"):
                if n not in NAMES:assert torch.equal(old[key],new[key])
                elif n.startswith("kv_"):assert torch.equal(old[key][width:],new[key][width:])
    opt=torch.optim.Adam(a.parameters(),lr=saved["param_groups"][0]["lr"])
    opt.load_state_dict(migrated)
    # Same non-QK cotangent isolates the QK intervention from tiny forward rounding effects.
    for n,p in a.named_parameters():
        if n in reference:p.grad=reference[n].clone().mul_(.1)
        if n in NAMES:
            if p.grad is None:p.grad=torch.zeros_like(p)
            p.grad[:width]=grads[n][:width]*.1
    raw_norm=float(torch.nn.utils.clip_grad_norm_(a.parameters(),grad_clip))
    eps=saved["param_groups"][0]["eps"]
    eps_fraction={n:float((migrated["state"][ids[n]]["exp_avg_sq"][:width].sqrt()<eps).float().mean())
                  for n in NAMES}
    before={n:p.detach().clone() for n,p in a.named_parameters()}
    opt.step()  # one independent diagnostic step on this disposable copy only
    delta={n:p.detach()-before[n] for n,p in a.named_parameters()}
    theta=blocks(a,before);change=blocks(a,delta)
    angular={}
    for side,t in theta.items():
        d=change[side]
        tang=d-t*((d*t).sum(dim=1)/t.square().sum(dim=1))[:,None]
        after=t+d
        u,v=t/t.norm(dim=1)[:,None],after/after.norm(dim=1)[:,None]
        angle=2*torch.asin(((u-v).norm(dim=1)/2).clamp(max=1))
        angular[side]=dict(delta_norm=d.norm(dim=1).tolist(),
            relative_tangent_step=(tang.norm(dim=1)/t.norm(dim=1)).tolist(),angle_radians=angle.tolist())
    return a,dict(raw_grad_norm=raw_norm,clip_coefficient=min(1.,grad_clip/(raw_norm+1e-6)),
        angular=angular,epsilon_dominant_fraction_before_step=eps_fraction),delta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--batches",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--targets",type=float,nargs="+",default=[16.,8.,4.,2.309401])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(20261016)
    actor,payload=load_actor(args.checkpoint)
    assert actor.config.qk_norm and actor.config.query_norm
    model_path=Path(inspect.getfile(type(actor)))
    assert sha(model_path)==payload["run_identity"]["source"]["files"]["train/history_model.py"]
    baseline={n:p.detach().clone() for n,p in actor.named_parameters()}
    fx=torch.load(args.fixture,map_location="cpu",weights_only=False)
    previous_audit=json.loads((args.batches/"mechanism.json").read_text())
    assert sha(args.fixture)==previous_audit["datasets"]["original_fixture"]["sha256"]
    batches={"fixture":unpack(fx["batch"]["inputs"])}
    for p in sorted(args.batches.glob("fresh-*.pt")):
        assert sha(p)==previous_audit["datasets"][p.stem.replace("-","_")]["sha256"]
        batches[p.stem]=unpack(torch.load(p,map_location="cpu",weights_only=False))
    assert len(batches)==4
    base_forward={n:forward(actor,i) for n,i in batches.items()}
    base_rollout={n:rollout_forwards(actor,i) for n,i in batches.items()}
    objectives=("outcome","belief","negative_entropy")
    base_grad={o:gradients(actor,batches["fixture"],fx["batch"]["fields"],o) for o in objectives}
    saved=copy.deepcopy(payload["optimizer"]["actor"])
    snapshot=copy.deepcopy(saved)
    ids=map_states(actor,saved)
    report=dict(checkpoint_sha256=sha(args.checkpoint),fixture_sha256=sha(args.fixture),
        torch=torch.__version__,device="cpu",backend="math",probe_source_sha256=sha(Path(__file__)),
        model_source_sha256=sha(Path(inspect.getfile(type(actor)))),
        gates=dict(max_abs_logp=1e-4,max_tv=1e-4,max_kl=1e-6,argmax_flips=0),
        optimizer_group={k:v for k,v in saved["param_groups"][0].items() if k!="params"},
        optimizer_steps={n:float(saved["state"][ids[n]]["step"]) for n in NAMES},
        base_directions={o:directions(actor,v["grads"]) for o,v in base_grad.items()},candidates={})
    # Native Adam reference step, before any rescaling.
    ones={s:torch.ones(actor.config.heads) for s in ("q","k")}
    native,base_step,base_delta=simulate(actor,saved,base_grad["outcome"]["grads"],
             base_grad["outcome"]["grads"],ones,"keep",payload["config"]["grad_clip"])
    report["baseline_step"]=base_step
    step_forward={n:forward(native,i) for n,i in batches.items()}
    for target in args.targets:
        print("target",target,flush=True)
        candidate=copy.deepcopy(actor);scale=factors(candidate,target)
        scale_slices(candidate,dict(candidate.named_parameters()),scale)
        width=actor.config.width
        for n,p in candidate.named_parameters():
            if n not in NAMES:assert torch.equal(p,baseline[n])
            elif n.startswith("kv_"):assert torch.equal(p[width:],baseline[n][width:])
        c=dict(scale={s:v.tolist() for s,v in scale.items()},forward={},directions={},steps={})
        for name,inputs in batches.items():
            c["forward"][name]=difference(base_forward[name],forward(candidate,inputs),inputs)
            for path,(logp,rows) in rollout_forwards(candidate,inputs).items():
                c["forward"][name+"/"+path]=difference(base_rollout[name][path][0],logp,rows)
        c["forward_pass"]=all(v["max_abs_logp"]<=1e-4 and v["tv"]["max"]<=1e-4
             and v["kl"]["max"]<=1e-6 and v["argmax_flips"]==0 for v in c["forward"].values())
        new_grad={o:gradients(candidate,batches["fixture"],fx["batch"]["fields"],o) for o in objectives}
        for o,v in new_grad.items():
            c["directions"][o]=directions(candidate,v["grads"])
            c["directions"][o]["loss"]=v["loss"]
            g0,g1=blocks(actor,base_grad[o]["grads"]),blocks(candidate,v["grads"])
            c["directions"][o]["covariance_relative_error"]={s:
               ((g1[s]*scale[s][:,None]-g0[s]).norm(dim=1)/g0[s].norm(dim=1).clamp_min(1e-300)).tolist()
               for s in ("q","k")}
        for mode in ("keep","transport","reset_slice_old_step","reset_whole_parameter"):
            stepped,r,delta=simulate(candidate,saved,new_grad["outcome"]["grads"],
                         base_grad["outcome"]["grads"],scale,mode,payload["config"]["grad_clip"])
            r["v_update_equal_to_baseline"]=all(torch.equal(delta[n][width:],base_delta[n][width:])
                                  for n in ("kv_proj.weight","kv_proj.bias"))
            r["forward_vs_baseline_step"]={n:difference(step_forward[n],forward(stepped,i),i)
                                           for n,i in batches.items()}
            c["steps"][mode]=r
        report["candidates"][str(target)]=c
        (args.output/"plasticity.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    assert all(torch.equal(p,baseline[n]) for n,p in actor.named_parameters())
    for i,st in saved["state"].items():
        assert all(torch.equal(v,snapshot["state"][i][k]) for k,v in st.items())
    assert sha(args.checkpoint)==report["checkpoint_sha256"]
    print("complete; checkpoint, base weights and original optimizer unchanged",flush=True)


if __name__=="__main__":main()
