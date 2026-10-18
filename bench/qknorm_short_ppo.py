"""Isolated, capped full-PPO continuation; no orchestrator or automatic resubmission.

One arm per process restores the same trusted qk-both parent. Initial frozen
population weights stay unchanged. Only norm16's learner Q/K and moments migrate.
Fixed CPU probes do not supply data or labels to the actual PPO learner.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from bench.qknorm_mechanism import capture, interventions, sha, spectrum, unpack
from bench.qknorm_plasticity import NAMES, blocks, factors, migrate, scale_slices
from train.history_model import save_history_checkpoint
from train.history_population import snapshot_weights, weights_digest
from train.history_ppo import HistoryPPOConfig, HistoryTrainer


def optimizer_fingerprint(state):
    digest=hashlib.sha256()
    for i,values in sorted(state["state"].items()):
        for name,value in sorted(values.items()):
            digest.update(f"{i}/{name}".encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    digest.update(json.dumps(state["param_groups"],sort_keys=True).encode())
    return digest.hexdigest()


def migrate_learner(trainer,parent):
    """Before first collect/cache use; retain historical opponent/partner weights."""
    assert trainer.collector.actor is trainer.actor and trainer.rollout_actor is trainer.actor
    before={n:p.detach().clone() for n,p in trainer.actor.named_parameters()}
    critic_before=weights_digest(trainer.critic.state_dict())
    pool_before={i:weights_digest(snapshot_weights(a)) for i,a in trainer.population.models.items()}
    cpu=copy.deepcopy(trainer.actor).cpu()
    scale=factors(cpu,16.)
    new_state=migrate(cpu,parent["optimizer"]["actor"],scale,"transport")
    scale_slices(trainer.actor,dict(trainer.actor.named_parameters()),
                 {s:v.to(next(trainer.actor.parameters()).device) for s,v in scale.items()})
    trainer.actor_optimizer.load_state_dict(new_state)
    width=trainer.actor.config.width
    for n,p in trainer.actor.named_parameters():
        if n not in NAMES:assert torch.equal(p,before[n])
        elif n.startswith("kv_"):assert torch.equal(p[width:],before[n][width:])
    assert weights_digest(trainer.critic.state_dict())==critic_before
    assert {i:weights_digest(snapshot_weights(a)) for i,a in trainer.population.models.items()}==pool_before
    return dict(target=16.,scale={s:v.tolist() for s,v in scale.items()},
                adam="m/a, v/a^2; retain step/lr/eps",initial_population_unchanged=True)


def finite_state(trainer):
    tensors=[p.detach() for p in (*trainer.actor.parameters(),*trainer.critic.parameters())]
    for optimizer in (trainer.actor_optimizer,trainer.critic_optimizer):
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value,torch.Tensor):
                    if value.device.type=="cpu":assert torch.isfinite(value).all()
                    else:tensors.append(value)
    assert bool(torch.stack([torch.isfinite(v).all() for v in tensors]).all()),"nonfinite weights/Adam state"


@contextmanager
def watch_clipping(trainer,records):
    original=torch.nn.utils.clip_grad_norm_
    actor_first=next(trainer.actor.parameters())
    def spy(parameters,max_norm,*args,**kwargs):
        parameters=list(parameters)
        actor=bool(parameters and parameters[0] is actor_first)
        before=None
        if actor:
            before=float(torch.stack([p.grad.detach().double().square().sum()
                      for p in parameters if p.grad is not None]).sum().sqrt())
            assert np.isfinite(before)
        result=original(parameters,max_norm,*args,**kwargs)
        if actor:
            native=float(result)
            assert np.isfinite(native) and abs(native-before)<=1e-4*max(1.,before)
            after=float(torch.stack([p.grad.detach().double().square().sum()
                      for p in parameters if p.grad is not None]).sum().sqrt())
            assert after<=float(max_norm)*(1+1e-5)
            records.append(dict(before=before,native=native,after=after,
                                clipped=native>float(max_norm)))
        return result
    torch.nn.utils.clip_grad_norm_=spy
    try:yield
    finally:torch.nn.utils.clip_grad_norm_=original


def probe(actor,datasets,meta):
    threads=torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        a=copy.deepcopy(actor).cpu().eval()
        caps={name:capture(a,inputs) for name,inputs in datasets.items()}
        calibration=torch.tensor(meta["datasets"]["fresh_control"]["calibration_rows"])
        mean=caps["fresh-control"]["reader"][calibration].mean(0,keepdim=True)
        output={"spectrum":spectrum(a),"datasets":{}}
        for name,inputs in datasets.items():
            cap=caps[name]
            rows=(torch.arange(inputs.decisions) if name=="fixture" else
                  torch.tensor(meta["datasets"][name.replace("-","_")]["test_rows"]))
            output["datasets"][name]=dict(heads=cap["heads"],
                interventions=interventions(a,inputs,cap,mean,rows))
        return output
    finally:torch.set_num_threads(threads)


def net_direction_change(actor,before):
    old=blocks(actor,before)
    now=blocks(actor,dict(actor.named_parameters()))
    result={}
    for side,value in old.items():
        delta=now[side]-value
        tangent=delta-value*((delta*value).sum(1)/value.square().sum(1))[:,None]
        result[side]=dict(norm=now[side].norm(dim=1).tolist(),
            relative_net_tangent_update=(tangent.norm(dim=1)/value.norm(dim=1)).tolist())
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm",choices=["control","norm16"],required=True)
    parser.add_argument("--resume",type=Path,required=True)
    parser.add_argument("--fixture",type=Path,required=True)
    parser.add_argument("--batches",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--updates",type=int,default=120)
    args=parser.parse_args()
    assert 1<=args.updates<=120,"short-run hard cap is 120 updates"
    assert not args.output.exists(),"new isolated output directory required"
    assert "experiments" in args.output.resolve().parts,"output must be isolated under experiments"
    assert sha(args.resume)=="a4bed2bfb4c3b8c1382a8ed5604e8e797307a37ea2088523dffb1243de6a58bd"
    parent=torch.load(args.resume,map_location="cpu",weights_only=False)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision("highest")
    start_update=int(parent["progress"]["updates"])
    trainer=HistoryTrainer(HistoryPPOConfig(updates=start_update+args.updates),args.output,
        device="cuda",resume=args.resume,allow_source_change=False,
        resume_overrides={"profile_collection":False,"profile_learn":False})
    assert trainer.config.num_envs==128 and trainer.config.steps_per_update==64
    assert trainer.config.epochs==2 and trainer.config.minibatch_matches==8
    assert trainer.actor.config.query_norm and trainer.actor.config.qk_norm
    initial_pool={i:weights_digest(snapshot_weights(a)) for i,a in trainer.population.models.items()}
    rng_before=dict(torch=sha_tensor(torch.get_rng_state()),sampler=sha_tensor(trainer.generator.get_state()),
                    population=copy.deepcopy(trainer.population.rng.bit_generator.state))
    critic_optim_before=optimizer_fingerprint(trainer.critic_optimizer.state_dict())
    migration=migrate_learner(trainer,parent) if args.arm=="norm16" else None
    assert optimizer_fingerprint(trainer.critic_optimizer.state_dict())==critic_optim_before
    assert rng_before["torch"]==sha_tensor(torch.get_rng_state())
    assert rng_before["sampler"]==sha_tensor(trainer.generator.get_state())
    assert rng_before["population"]==trainer.population.rng.bit_generator.state
    receipt=dict(arm=args.arm,parent_sha256=sha(args.resume),updates=args.updates,world_size=1,
        parent_progress=parent["progress"],torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        wrapper_sha256=sha(Path(__file__)),source=trainer.run_identity,
        initial_population=initial_pool,rng=rng_before,migration=migration,
        config=vars(trainer.config),shared_initial_pool=True,automatic_continuation=False)
    (args.output/"experiment.json").write_text(json.dumps(receipt,indent=2)+"\n")
    fixture=torch.load(args.fixture,map_location="cpu",weights_only=False)
    meta=json.loads((args.batches/"mechanism.json").read_text())
    assert sha(args.fixture)==meta["datasets"]["original_fixture"]["sha256"]
    datasets={"fixture":unpack(fixture["batch"]["inputs"])}
    for p in sorted(args.batches.glob("fresh-*.pt")):
        assert sha(p)==meta["datasets"][p.stem.replace("-","_")]["sha256"]
        datasets[p.stem]=unpack(torch.load(p,map_location="cpu",weights_only=False))
    assert len(datasets)==4
    points=sorted({0,min(30,args.updates),min(60,args.updates),args.updates})
    endpoints={}
    def measure(index):
        print("probe",args.arm,index,flush=True)
        record=probe(trainer.actor,datasets,meta)
        record.update(index=index,progress=copy.deepcopy(trainer.progress))
        (args.output/f"probe-{index:03d}.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    measure(0)
    before_decisions=int(trainer.progress["decisions"])
    began=time.monotonic()
    for index in range(1,args.updates+1):
        records=[]
        old={n:p.detach().clone() for n,p in trainer.actor.named_parameters() if n in NAMES}
        with watch_clipping(trainer,records):line=trainer.update()
        finite_state(trainer)
        assert records and len(records)==line["minibatches"]
        wanted=("update","decisions","step_decisions","update_samples","minibatches","policy_loss",
            "value_loss","entropy","approx_kl","clip_fraction","actor_grad_norm","critic_grad_norm",
            "aux_loss","response_loss","next_loss","belief_loss","outcome_loss","round_gain",
            "cuda_peak_allocated_bytes","cuda_peak_reserved_bytes","cuda_allocation_retries",
            "collect_seconds","learn_seconds","mean_prefix","max_prefix")
        out={k:line.get(k) for k in wanted}
        out.update(index=index,arm=args.arm,all_parameters_optimizer_finite=True,
                   clipping=records,direction=net_direction_change(trainer.actor,old),
                   effective_scale=(trainer.actor.qk_temperature().detach()*4).cpu().tolist())
        with (args.output/"diagnostic.jsonl").open("a") as stream:
            stream.write(json.dumps(out,allow_nan=False)+"\n")
        if index%10==0:print(args.arm,index,"loss",out["policy_loss"],"grad",out["actor_grad_norm"],flush=True)
        if index in points:measure(index)
    payload=trainer.payload()
    payload["isolated_diagnostic"]=dict(experiment="qknorm-short-ppo-2026-10-16",arm=args.arm,
        parent_sha256=receipt["parent_sha256"],migration=migration,updates=args.updates,
        not_mainline=True,no_automatic_continuation=True)
    path=trainer.output/"endpoint.pt"
    save_history_checkpoint(path,payload)
    complete=dict(completed=True,arm=args.arm,updates=args.updates,
        actual_decisions=int(trainer.progress["decisions"])-before_decisions,
        elapsed_seconds=time.monotonic()-began,endpoint_sha256=sha(path),
        final_progress=trainer.progress,probe_points=points)
    (args.output/"complete.json").write_text(json.dumps(complete,indent=2)+"\n")
    assert sha(args.resume)==receipt["parent_sha256"]
    print("COMPLETE",json.dumps(complete),flush=True)


def sha_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


if __name__=="__main__":main()
