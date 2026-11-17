"""Public-reconstructed choice opportunities, with a new untouched test cohort.

This diagnostic follows the generic token-probe negative result. It uses only
u195450 as a frozen reference, not the 20 candidate opponent weight sets.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch

from bench.history_identification import ROOT,SEED,collect_one,initialize,ResidualSampler,replay
from bench.filtered_history_collect import load_players,save_json
from bench.history_identification_probes import EVAL_LENGTHS
from bench.history_identification_score import question_kernels,gain_matrix,bootstrap_weights,summarize_posterior
from bench.qknorm_match_ablation import sha
from train.history_model import StreamBatch,DecisionInputs
from train.history_population import weights_digest

_ACTOR=None


def init_features():
    global _ACTOR
    torch.set_num_threads(1);_ACTOR=load_players(indices=[])[0]


def seal(root):
    path=root/'opportunity-protocol.json'
    record=dict(reason='generic raw-token and all frozen-layer probes fail to generalize; test whether explicit public choice opportunities make identity learnable',
        scope='post-hoc diagnostic design, prospective fresh test replicates 32..39; original train 0..19 and validation 20..23; no reuse of test 24..31 for choosing this head',
        lengths=EVAL_LENGTHS,
        eligibility='only completed rounds where both opponent hands are uniquely reconstructed from public events; omit ambiguous rounds for all compared feature arms',
        features='choice-only 11-type counts; opportunity-conditioned 6x11 residual counts; u195450 final-action-layer score feature h(chosen)-sum_a pi(a)h(a)',
        reference='only frozen u195450 policy; no candidate pool weights used to construct input features',
        private_boundary='replay uses public.pt and public-constrained residual sampler; audit-private.pt is never read',
        model='weighted ridge to 20 identity indicators, prior residual, train-only standardization; alpha selected by validation identity Brier including prior-only option',
        control='same resolved-round availability/count/length in all arms; predictions scored on same independent question bank; fresh histories whole-sequence split',
        source=sha(__file__))
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(record)):raise ValueError('supplement protocol changed')
    save_json(path,record)


@torch.inference_mode()
def feature_one(task):
    root_string,z,rep=task;root=Path(root_string);folder=root/'histories'/f'z{z:02d}-r{rep:03d}'
    if (folder/'opportunity.npz').exists():return dict(z=z,rep=rep,cached=True)
    public=torch.load(folder/'public.pt',weights_only=False);actor=_ACTOR
    before=weights_digest(actor.state_dict());rng=np.random.default_rng(int(sha(folder/'public.pt')[:16],16))
    encoded={s:actor.encode_batch(StreamBatch.from_arrays([a],'cpu')) for s,a in enumerate(public['segments'])}
    sb={s:StreamBatch.from_arrays([a],'cpu') for s,a in enumerate(public['segments'])}
    # choice counts 11; conditioned type residuals 6x11; final-layer score 128;
    # count denominators 6 plus resolved-round flags.
    choice=np.zeros((10,11));residual=np.zeros((10,6,11));fisher=np.zeros((10,actor.config.fusion_width))
    count=np.zeros((10,6));resolved=np.zeros(10,dtype=bool)
    for r,meta in enumerate(public['rounds']):
        exact=sum(c>0 for c in meta['left'])==1 or meta['left'][1]==meta['left'][3]==0
        if not exact:continue
        arrays=public['segments'][meta['segment']];sampler=ResidualSampler(meta,arrays);frames=None
        for _ in range(100000):
            initial,_=sampler.draw(rng)
            if initial is not None:frames=replay(meta,arrays,initial)
            if frames is not None:break
        if frames is None:raise RuntimeError('failed public resolved-round replay')
        resolved[r]=True
        n=len(frames);offsets=np.r_[0,np.cumsum([len(d['cand']) for d in frames])]
        inputs=DecisionInputs(sb[meta['segment']],torch.zeros(n,dtype=torch.long),
            torch.tensor([d['prefix'] for d in frames]),torch.tensor(np.stack([d['obs'] for d in frames])),
            torch.tensor([d['seat'] for d in frames]),torch.tensor(np.concatenate([d['cand'] for d in frames])),
            torch.from_numpy(offsets),one_decision_per_stream=False)
        state=actor.decision_states(encoded[meta['segment']],inputs)
        features=actor.fusion[0](torch.cat((state[inputs.rows],actor.action_tower(inputs.cand.float())),-1))
        logits=actor.fusion[1](features).squeeze(-1)
        torch.testing.assert_close(logits,actor.candidate_logits(state,inputs.cand,inputs.offsets),rtol=0,atol=0)
        for i,d in enumerate(frames):
            lo,hi=offsets[i:i+2];p=logits[lo:hi].double().softmax(0).numpy();h=features[lo:hi].double().numpy()
            kinds=d['cand'][:,108:119].argmax(1);chosen=int(d['choices'][0]);typ=int(kinds[chosen])
            assert (kinds[d['choices']]==typ).all()
            # If multiple canonical candidates share the observed public
            # action, average their score features by conditional probability.
            observed=p[d['choices']];observed/=observed.sum()
            gradient=observed@h[d['choices']]-p@h
            cards=int(d['obs'][:108].sum());bucket=0 if cards>18 else 1 if cards>8 else 2
            responding=bool(np.any(kinds==0));context=3*int(responding)+bucket
            actual=np.eye(11)[typ];expected=np.bincount(kinds,weights=p,minlength=11)
            choice[r]+=actual;residual[r,context]+=actual-expected
            fisher[r]+=gradient;count[r,context]+=1
    assert before==weights_digest(actor.state_dict())
    np.savez_compressed(folder/'opportunity.npz',choice=choice,residual=residual,fisher=fisher,count=count,resolved=resolved)
    return dict(z=z,rep=rep,resolved=int(resolved.sum()))


def probabilities(scores):
    p=np.maximum(scores,1e-6);return p/p.sum(1,keepdims=True)


def ridge(x,y,split,prior):
    tr=split==0;va=split==1;te=split==2;w=prior[y[tr]];w/=w.sum();wv=prior[y[va]];wv/=wv.sum()
    mean=w@x[tr];std=np.sqrt(w@((x[tr]-mean)**2)).clip(1e-5);xx=(x-mean)/std
    design=xx[tr]*np.sqrt(w[:,None]);targets=(np.eye(20)[y[tr]]-prior)*np.sqrt(w[:,None])
    eigen,vectors=np.linalg.eigh(design.T@design);rhs=vectors.T@(design.T@targets)
    best=float(wv@((prior-np.eye(20)[y[va]])**2).sum(1));alpha=None;p=np.tile(prior,(te.sum(),1));curve=[]
    for a in (1e-5,1e-4,.001,.01,.1,1.,10.,100.,1000.):
        coefficient=vectors@(rhs/(np.maximum(eigen,0)[:,None]+a))
        pv=probabilities(xx[va]@coefficient+prior)
        value=float(wv@((pv-np.eye(20)[y[va]])**2).sum(1));curve.append(dict(alpha=a,validation_brier=value))
        if value<best:best=value;alpha=a;p=probabilities(xx[te]@coefficient+prior)
    return p,dict(alpha=alpha,validation_brier=best,curve=curve,dimensions=x.shape[1])


def score(root):
    prior=np.array(json.loads((root/'protocol.json').read_text())['identity']['weights'])
    rows=[(z,r) for z in range(20) for r in list(range(24))+list(range(32,40))]
    y=np.array([z for z,r in rows]);split=np.array([0 if r<20 else 1 if r<24 else 2 for z,r in rows])
    loaded=[dict(np.load(root/'histories'/f'z{z:02d}-r{r:03d}'/'opportunity.npz')) for z,r in rows]
    labels=y[split==2];kernels=question_kernels(root);hboot,qboot=bootstrap_weights(labels,prior)
    oracle={k:gain_matrix(np.eye(20)[labels],labels,prior,kernels[f'{k}_cross']) for k in ('action_type','full_action')}
    result=dict(protocol=json.loads((root/'opportunity-protocol.json').read_text()),curves={},fits={})
    saved_predictions={}
    for name in ('choice_only','opportunity','fisher','combined'):
        result['curves'][name]={};result['fits'][name]={}
        for length in EVAL_LENGTHS:
            features=[]
            for d in loaded:
                count=d['count'][:length].sum(0);n=count.sum()
                common=np.r_[count/np.maximum(n,1),np.log1p(n),d['resolved'][:length].mean()]
                chosen=d['choice'][:length].sum(0)/np.maximum(n,1)
                residual=(d['residual'][:length].sum(0)/np.sqrt(count[:,None]+1)).ravel()
                fisher=d['fisher'][:length].sum(0)/np.sqrt(n+1)
                extra={'choice_only':chosen,'opportunity':np.r_[chosen,residual],
                       'fisher':fisher,'combined':np.r_[chosen,residual,fisher]}[name]
                features.append(np.r_[common,extra])
            p,fit=ridge(np.asarray(features),y,split,prior)
            saved_predictions[(name,length)]=p
            result['fits'][name][str(length)]=fit
            result['curves'][name][str(length)]=summarize_posterior(p,labels,prior,kernels,hboot,qboot,oracle)
            gain=gain_matrix(p,labels,prior,kernels['action_type_cross'])
            draws=((hboot@gain)*qboot).sum(1)
            result['curves'][name][str(length)]['action_type']['ci_familywise_12']=np.quantile(draws,[.05/24,1-.05/24]).tolist()
            print(json.dumps(dict(arm=name,length=length,alpha=fit['alpha'],**result['curves'][name][str(length)])),flush=True)
    result['paired_vs_choice_only']={}
    for name in ('opportunity','fisher','combined'):
        result['paired_vs_choice_only'][name]={}
        for length in EVAL_LENGTHS:
            delta=gain_matrix(saved_predictions[(name,length)],labels,prior,kernels['action_type_cross'])-gain_matrix(
                saved_predictions[('choice_only',length)],labels,prior,kernels['action_type_cross'])
            draws=((hboot@delta)*qboot).sum(1);w=prior[labels];w/=w.sum()
            result['paired_vs_choice_only'][name][str(length)]=dict(gain=float(w@delta.mean(1)),
                ci95=np.quantile(draws,[.025,.975]).tolist(),ci_familywise_12=np.quantile(draws,[.05/24,1-.05/24]).tolist())
    result['analysis_source_sha']=sha(__file__)
    save_json(root/'opportunity-summary.json',result)


def run(args):
    if args.command!='score':seal(args.root)
    if args.command=='fresh':
        tasks=[(str(args.root),z,r,10) for z in range(20) for r in range(32,40)];worker=collect_one;init=initialize
    elif args.command=='features':
        tasks=[(str(args.root),z,r) for z in range(20) for r in list(range(24))+list(range(32,40))];worker=feature_one;init=init_features
    else:score(args.root);return
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=init) as pool:
        jobs=[pool.submit(worker,t) for t in tasks]
        for done,future in enumerate(as_completed(jobs),1):
            value=future.result()
            if done%40==0:print(json.dumps(dict(done=done,total=len(tasks),last=value)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['fresh','features','score'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--workers',type=int,default=2)
    run(p.parse_args())
