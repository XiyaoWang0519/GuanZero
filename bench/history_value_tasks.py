"""Public-prefix mid/endgame task generation and frozen heterogeneous rollouts.

At the score node hidden hands are uniformly redrawn subject to the public
state; opponent type is assigned there. This is a controlled diagnostic,
not a conditional sampler for natural matches with a persistent earlier type.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
from pathlib import Path
from typing import Any

import gd
import numpy as np
import torch

from eval.history_policy import apply_and_observe
from train.history_inference import BatchedHistoryCache, ForkedHistoryCache
from train.history_model import PublicStream, StreamBatch, DecisionInputs
from bench.qknorm_match_ablation import MatchSampler, inverse_cdf


class Listener:
    def __init__(self, stream: PublicStream): self.stream=stream
    def observe(self,event: Any) -> None: self.stream.append(event)


def stream_from(arrays: tuple) -> PublicStream:
    stream=PublicStream()
    for token,round_index,phase in zip(*arrays):
        stream.append_token(token,int(round_index),int(phase))
    return stream


def decision_probabilities(actor: Any,state: Any,arrays: tuple) -> np.ndarray:
    engine=gd.Engine();legal=engine.legal_actions(state);seat=int(state.to_move)
    sb=StreamBatch.from_arrays([arrays],'cpu')
    inputs=DecisionInputs(sb,torch.zeros(1,dtype=torch.long),sb.lengths,
        torch.tensor(np.asarray(state.observation(seat))[None]),torch.tensor([seat]),
        torch.tensor(np.asarray([engine.encode_action(a,state,seat) for a in legal])),
        torch.tensor([0,len(legal)]),one_decision_per_stream=True)
    with torch.inference_mode():table=actor._log_prob_table(inputs,None,len(legal))[0].double().numpy()
    p=np.exp(table);p/=p.sum()
    if not np.isfinite(p).all():raise ValueError('nonfinite actor probabilities')
    return p


def candidates_for(legal: list,prob: np.ndarray,roster_prob: np.ndarray,cap: int=8) -> np.ndarray:
    """Public/own-hand semantic diversity, never payoff- or hidden-hand-ranked."""
    selected=[]
    def add(a: int) -> None:
        if len(selected)<cap and int(a) not in selected:selected.append(int(a))
    add(int(prob.argmax()))
    passes=[i for i,a in enumerate(legal) if a.is_pass]
    bombs=[i for i,a in enumerate(legal) if str(a.type) in ('Bomb','StraightFlush','JokerBomb')]
    plain=[i for i,a in enumerate(legal) if not a.is_pass and i not in bombs]
    if passes:add(passes[0])
    for group in (plain,bombs):
        if group:add(max(group,key=lambda i:prob[i]))
    if plain:add(max(plain,key=lambda i:(len(legal[i].cards),prob[i])))
    # Power keys are compared only inside one combination family.
    family=str(legal[int(prob.argmax())].type)
    same=[i for i in plain if str(legal[i].type)==family]
    if same:add(min(same,key=lambda i:legal[i].key))
    for a in roster_prob.argmax(1):add(int(a))
    for kind in sorted({str(a.type) for a in legal}):
        group=[i for i,a in enumerate(legal) if str(a.type)==kind]
        add(max(group,key=lambda i:prob[i]))
    for a in np.argsort(-prob,kind='stable'):add(int(a))
    return np.asarray(selected,np.int64)


def generate_tasks(actor: Any,opponents: dict,output: Path,seed: int,
                   source_rounds: int=64,pool_cap: int=96,discover: int=32) -> dict:
    """Frozen control generates prefixes; filter uses own hand and public counts."""
    output.mkdir(exist_ok=False)
    engine=gd.Engine();engine.auto_pass=False
    seeds=np.random.default_rng(seed).integers(1,2**63-1,size=source_rounds)
    pool=[];completed=0
    for source,deal_seed in enumerate(seeds):
        state=gd.MatchState();engine.new_match(state,int(deal_seed))
        stream=PublicStream();listener=Listener(stream);remaining=np.full(4,27)
        cache=BatchedHistoryCache(actor);taken=0;last_capture=-100
        for step in range(512):
            if state.phase==gd.Phase.RoundEnd:
                completed+=1;break
            if state.phase!=gd.Phase.Play:raise ValueError('unexpected non-play phase')
            legal=engine.legal_actions(state);seat=int(state.to_move)
            eligible=(seat==0 and 4<=remaining[0]<=14 and min(remaining[1],remaining[3])<=8
                and (remaining>0).all() and 16<=remaining.sum()<=56 and len(legal)>=3
                and taken<2 and step-last_capture>=8 and len(pool)<pool_cap)
            arrays=tuple(x.copy() for x in stream.arrays())
            with torch.inference_mode():
                metadata,memory=cache.encode([(0,0)],[stream])
                sb=DecisionInputs(metadata,torch.zeros(1,dtype=torch.long),metadata.lengths,
                    torch.tensor(np.asarray(state.observation(seat))[None]),torch.tensor([seat]),
                    torch.tensor(np.asarray([engine.encode_action(a,state,seat) for a in legal])),
                    torch.tensor([0,len(legal)]),one_decision_per_stream=True)
                table=actor._log_prob_table(sb,memory,len(legal))[0].double().numpy()
                p=np.exp(table);p/=p.sum()
                if eligible:
                    roster=np.stack([decision_probabilities(a,state,arrays) for a in opponents.values()])
                    disagreement=max(.5*np.abs(roster[i]-roster[j]).sum() for i in range(4) for j in range(i))
                    cand=candidates_for(legal,p,roster)
                    index=len(pool);folder=output/f'pool-{index:03d}';folder.mkdir()
                    (folder/'simulator-state.bin').write_bytes(state.serialize())
                    np.savez_compressed(folder/'public-inputs.npz',tokens=arrays[0],rounds=arrays[1],phases=arrays[2],
                        candidates=cand,actor_prob=p,roster_prob=roster,observation=state.observation(0))
                    record=dict(index=index,source=source,deal_seed=int(deal_seed),step=step,
                        remaining=remaining.tolist(),public_events=len(arrays[0]),open=bool(state.top_is_open),
                        legal_actions=len(legal),candidates=cand.tolist(),disagreement=float(disagreement),
                        kinds=[str(legal[a].type) for a in cand],native_choice=int(p.argmax()))
                    pool.append(record);taken+=1;last_capture=step
            choice=int(p.argmax())
            event=apply_and_observe(engine,state,legal[choice],[listener])
            remaining[event.seat]=event.cards_left
        else:raise RuntimeError('generation round cap hit; no censoring')
        if len(pool)>=pool_cap:break
    # All candidate ranking above is public/own-hand policy disagreement, not score.
    ranked=sorted(pool,key=lambda r:(-r['disagreement'],r['index']))
    selected=[];used=set()
    for record in ranked:
        if record['source'] in used:continue
        selected.append(record);used.add(record['source'])
        if len(selected)==discover:break
    if len(selected)<6:raise RuntimeError('too few independent candidate roots')
    return dict(pool=pool,discovery=selected,generation_rounds=completed,source_round_limit=source_rounds)


def load_task(folder: Path) -> tuple[Any,tuple,np.ndarray]:
    state=gd.MatchState.deserialize((folder/'simulator-state.bin').read_bytes())
    with np.load(folder/'public-inputs.npz') as raw:
        arrays=tuple(raw[k].copy() for k in ('tokens','rounds','phases'));cand=raw['candidates'].copy()
    return state,arrays,cand


def rollout(actor: Any,opponents: dict,state: Any,arrays: tuple,candidates: np.ndarray,
            seeds: np.ndarray,max_steps: int=256,use_cache: bool=True) -> np.ndarray:
    """z x candidate x hidden-world x [team round win, PPO return]."""
    if state.to_move!=0:raise ValueError('diagnostic observer must act at seat zero')
    engine=gd.Engine();engine.auto_pass=False;legal_root=engine.legal_actions(state)
    expected_obs=np.asarray(state.observation(0))
    expected_cand=np.asarray([engine.encode_action(a,state,0) for a in legal_root])
    worlds=[state.determinize_uniform(0,int(s)) for s in seeds]
    branches=[];streams=[];labels=[];sampling=[]
    root_stream=stream_from(arrays)
    for z in range(len(opponents)):
        for candidate in candidates:
            for w,seed in zip(worlds,seeds):
                branch=gd.MatchState.deserialize(w.serialize())
                if not np.array_equal(branch.observation(0),expected_obs):raise ValueError('private world altered actor observation')
                actual=engine.legal_actions(branch)
                if not np.array_equal(np.asarray([engine.encode_action(a,branch,0) for a in actual]),expected_cand):
                    raise ValueError('private world altered legal candidates')
                stream=copy.deepcopy(root_stream)
                apply_and_observe(engine,branch,actual[int(candidate)],[Listener(stream)])
                branches.append(branch);streams.append(stream);labels.append(z);sampling.append(int(seed))
    n=len(branches);sampler=MatchSampler(sampling);values=np.full((n,2),np.nan)
    policies={0:actor,**{i+1:a for i,a in enumerate(opponents.values())}}
    groups={0:list(range(n)),**{i+1:[r for r,z in enumerate(labels) if z==i] for i in range(len(opponents))}}
    mappings={k:{r:i for i,r in enumerate(rows)} for k,rows in groups.items()}
    roots={};caches={}
    with torch.inference_mode(),ExitStack() as frozen:
        if use_cache:
            for k,a in policies.items():
                roots[k]=BatchedHistoryCache(a);roots[k].prefill([(0,0)],[root_stream])
                caches[k]=ForkedHistoryCache(roots[k],(0,0),[streams[r] for r in groups[k]])
                frozen.enter_context(roots[k].frozen_weights())
        active=list(range(n))
        for step in range(max_steps):
            pending={k:[] for k in policies};legal_sets={}
            for r in active:
                branch=branches[r]
                if branch.phase==gd.Phase.RoundEnd:
                    result=engine.end_round(branch)
                    values[r]=[int(result.winning_team)==0,float(result.seat_return[0])]
                    continue
                legal=engine.legal_actions(branch)
                if len(legal)==1:
                    apply_and_observe(engine,branch,legal[0],[Listener(streams[r])]);continue
                k=0 if int(branch.to_move)%2==0 else labels[r]+1
                pending[k].append(r);legal_sets[r]=legal
            for k,rows in pending.items():
                if not rows:continue
                a=policies[k];seats=np.asarray([branches[r].to_move for r in rows])
                obs=np.stack([branches[r].observation(int(s)) for r,s in zip(rows,seats)])
                cand=[];offsets=[0]
                for r,s in zip(rows,seats):
                    cand.extend(engine.encode_action(action,branches[r],int(s)) for action in legal_sets[r]);offsets.append(len(cand))
                if use_cache:metadata,encoded=caches[k].encode([mappings[k][r] for r in rows])
                else:metadata,encoded=StreamBatch.from_streams([streams[r] for r in rows],'cpu'),None
                inputs=DecisionInputs(metadata,torch.arange(len(rows)),metadata.lengths,
                    torch.tensor(obs),torch.tensor(seats),torch.tensor(np.asarray(cand)),torch.tensor(offsets),
                    one_decision_per_stream=True)
                table=a._log_prob_table(inputs,encoded,max(len(legal_sets[r]) for r in rows))
                choice,_=inverse_cdf(table,torch.tensor(sampler.draw(rows,seats)))
                for r,c in zip(rows,choice.tolist()):
                    apply_and_observe(engine,branches[r],legal_sets[r][int(c)],[Listener(streams[r])])
            active=[r for r in active if not np.isfinite(values[r]).all()]
            if not active:return values.reshape(len(opponents),len(candidates),len(seeds),2)
    raise RuntimeError('unfinished branch; no partial result accepted')
