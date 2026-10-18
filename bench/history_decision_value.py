"""Bounded CPU information-set action-value pilot with frozen Transformers.

Run as a file with the checkpoint-matching model root first on PYTHONPATH.
Hidden worlds are simulator inputs only; choices use averages over worlds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import gd
import numpy as np
import torch

from bench.public_history_analysis import Tables, iter_matches, softmax
from bench.public_history_diverse import load_actor, load_selected
from bench.public_history_probe import PublicCollector
from bench.qknorm_match_ablation import sha
from train.history_model import DecisionInputs, StreamBatch
from train.history_rollout import MatchEventStore, SequenceRolloutBuffer

POOL = Path('.work/public-history-diverse-2026-10-16')
ENDPOINT = Path('.work/qknorm-short-ppo-2026-10-16/remote/runs/control/endpoint.pt')


def choose_actions(q: np.ndarray, posterior: np.ndarray) -> np.ndarray:
    """q[z, action, world]; integrate hidden worlds BEFORE taking argmax."""
    if q.ndim != 3 or posterior.shape[-1] != q.shape[0]:
        raise ValueError('incompatible Q/posterior shapes')
    if not np.isfinite(q).all() or not np.isfinite(posterior).all():
        raise ValueError('nonfinite input')
    if (posterior < 0).any() or not np.allclose(posterior.sum(-1), 1):
        raise ValueError('posterior must be a probability vector')
    return np.argmax(posterior @ q.mean(-1), axis=-1)


def select_candidates(round_prob: np.ndarray, history_prob: np.ndarray) -> np.ndarray:
    """Include all five greedy actor choices, then fill to five by ROUND score."""
    selected = list(dict.fromkeys([int(round_prob.argmax()), *history_prob.argmax(1).tolist()]))
    for a in np.argsort(-round_prob, kind='stable'):
        if len(selected) >= 5:
            break
        if int(a) not in selected:
            selected.append(int(a))
    return np.asarray(selected, np.int64)


def public_histories(root_index: int, tables: Tables) -> tuple[list[tuple], np.ndarray, list[dict]]:
    """Fixed archive schedule, independent of score deals; first three public rounds."""
    wave = 12 + root_index // 4
    folder = POOL / 'raw' / f'wave-{wave:02d}'
    histories, posterior, records = [], [], []
    with np.load(folder/'control-public.npz') as raw:
        meta, actions, results = (raw[k] for k in ('public_meta','public_actions','public_round_results'))
        for z in range(4):
            e = (root_index % 4)*8 + 2*z  # team zero in the original schedule
            mask = (meta[:,0] == e) & (meta[:,1] < 3)
            m, a = meta[mask], actions[mask]
            if not len(m) or set(m[:,1]) != {0,1,2} or a[:,146:].any():
                raise ValueError('incomplete or nonpublic history')
            tokens = np.zeros((len(m),186), np.uint8)
            tokens[np.arange(len(m)),m[:,3]] = 1
            tokens[:,4:150] = a
            tokens[np.arange(len(m)),158+m[:,4]] = 1
            histories.append((tokens,m[:,1].copy(),m[:,5].copy()))
            counts = np.zeros((2,4))
            # Only public events/results enter the count-table posterior.
            from bench.public_history_analysis import public_rows
            for _, lead, category, *_ in public_rows(m,a,results[results[:,0]==e],0):
                counts[lead,category] += 1
            posterior.append(softmax(tables.history_logits(0,counts)))
            records.append(dict(wave=wave,env=e,type=z,events=len(m),
                                public_sha=hashlib.sha256(tokens.tobytes()).hexdigest()))
    return histories,np.asarray(posterior),records


def root_state(seed: int) -> Any:
    engine, source, state = gd.Engine(), gd.MatchState(), gd.MatchState()
    engine.new_match(source, seed)
    deal = gd.DealSpec()
    deal.hands = [source.hand(s) for s in range(4)]
    deal.level, deal.team_levels, deal.owner, deal.leader = 0,[0,0],-1,0
    engine.set_deal(state, deal)
    return state


def root_probabilities(actor: Any, state: Any, histories: list[tuple]) -> np.ndarray:
    engine = gd.Engine()
    legal = engine.legal_actions(state)
    empty = (np.empty((0,186),np.uint8),np.empty(0,np.int64),np.empty(0,np.int64))
    streams = StreamBatch.from_arrays([empty,*histories],'cpu')
    n, c = 5,len(legal)
    inputs = DecisionInputs(streams,torch.arange(n),streams.lengths,
        torch.tensor(np.repeat(np.asarray(state.observation(0))[None],n,0)),
        torch.zeros(n,dtype=torch.long),
        torch.tensor(np.tile(np.asarray([engine.encode_action(a,state,0) for a in legal]),(n,1))),
        torch.arange(n+1)*c,one_decision_per_stream=True)
    with torch.inference_mode():
        logp = actor._log_prob_table(inputs,None,c).double().numpy()
    p = np.exp(logp); p /= p.sum(1,keepdims=True)
    return p


class RoundCollector(PublicCollector):
    """Stop inference per slot after the first round; force only its first action."""
    def __init__(self, *args: Any, root_actions: np.ndarray, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.root_actions = root_actions
        self.root_applied = np.zeros(len(root_actions),bool)

    def _drain(self, stats: Any) -> None:
        super()._drain(stats)
        for r in self.results:
            if int(r.match_id)==0 and int(r.round_index)==0:
                self.done[int(r.env_id)] = True

    def _store(self, r: Any, stats: Any) -> None:
        for i,e in enumerate(r.env_id):
            if not self.root_applied[e] and not self.done[e]:
                if r.seat[i]!=0 or r.round_index[i]!=0 or r.prefix[i]!=0:
                    raise ValueError('root must be the first play by seat zero')
                r.choices[i] = self.root_actions[e]
                self.root_applied[e] = True
        super()._store(r,stats)


def rollout_matrix(actor: Any, opponents: dict, state: Any, candidates: np.ndarray,
                   seeds: np.ndarray, max_steps: int = 512) -> tuple[np.ndarray, dict]:
    """Common deals and per-seat random draws across all types and root actions."""
    ids = sorted(opponents)
    worlds = [state.determinize_uniform(0,int(s)) for s in seeds]
    deals, labels, forced, policy_seeds = [], [], [], []
    for z in range(len(ids)):
        for a in candidates:
            for j,w in enumerate(worlds):
                d = gd.DealSpec()
                d.hands = [w.hand(s) for s in range(4)]
                d.level,d.team_levels,d.owner,d.leader = 0,[0,0],-1,0
                deals.append(d); labels.append(ids[z]); forced.append(int(a))
                policy_seeds.append(int(seeds[j]))
    n = len(deals)
    env = gd.VecEnv(num_envs=n,num_threads=1,seed=71,log_public_actions=True,log_env_limit=n)
    env.reset(deals)
    batch = env.pending()
    # Every hidden world has exactly the same legal actor information at the root.
    expected_obs = np.asarray(state.observation(0))
    expected_cand = np.asarray([gd.Engine().encode_action(a,state,0) for a in gd.Engine().legal_actions(state)])
    for i in range(n):
        a,b = map(int,batch.offsets[i:i+2])
        if int(batch.seat[i])!=0 or not np.array_equal(batch.obs[i],expected_obs) or not np.array_equal(batch.cand[a:b],expected_cand):
            raise ValueError('hidden sampling changed the root information set')
    collector = RoundCollector(env,actor,MatchEventStore(),SequenceRolloutBuffer(),
        device='cpu',kv_cache=True,seeds=policy_seeds,root_actions=np.asarray(forced),
        seat_policy=lambda e,m:[0,labels[e],0,labels[e]],resolve_policy=opponents.__getitem__)
    collector.started = True
    returns = np.full(n,np.nan)
    started = time.monotonic()
    with torch.inference_mode():
        for step in range(max_steps):
            collector.collect(1)
            for r in collector.results:
                if int(r.match_id)==0 and int(r.round_index)==0:
                    returns[int(r.env_id)] = float(r.seat_return[0])
            if collector.done.all():
                if not np.isfinite(returns).all() or not collector.root_applied.all():
                    raise ValueError('missing round return/root intervention')
                return returns.reshape(len(ids),len(candidates),len(seeds)),dict(
                    rounds=n,steps=step+1,seconds=time.monotonic()-started,root_information_equal=True)
    raise RuntimeError('round step cap reached; no partial-result censoring')


def evaluate(q_select: np.ndarray, q_test: np.ndarray, posterior: np.ndarray,
             actor_prob: np.ndarray, candidates: np.ndarray) -> dict:
    """Select only on q_select; all reported gains use the independent q_test."""
    nz = q_select.shape[0]
    prior = np.full(nz,1/nz)
    baseline = int(choose_actions(q_select,prior))
    oracle = choose_actions(q_select,np.eye(nz))
    history = choose_actions(q_select,posterior)
    actor_choice = np.asarray([int(np.flatnonzero(candidates==a)[0]) for a in actor_prob.argmax(1)])
    q = q_test.mean(-1)
    z = np.arange(nz)
    base = q[:,baseline]
    # Native actor is greedy, and its exact argmax is guaranteed inside candidates.
    gains = dict(identity=q[z,oracle]-base,posterior=q[z,history]-base,
                 actor_full=q[z,actor_choice[1:]]-base,
                 actor_round=q[:,actor_choice[0]]-base,
                 actor_history_effect=q[z,actor_choice[1:]]-q[:,actor_choice[0]])
    # This plug-in value is optimistic and is explicitly NOT the held-out result.
    qs = q_select.mean(-1)
    plugin = float(qs.max(1).mean()-qs.mean(0).max())
    return dict(baseline_action=int(candidates[baseline]),identity_actions=candidates[oracle].tolist(),
        posterior_actions=candidates[history].tolist(),actor_actions=candidates[actor_choice].tolist(),
        posterior=posterior.tolist(),posterior_accuracy=float(np.mean(posterior.argmax(1)==z)),
        selection_plugin_identity_value=plugin,
        gains={k:dict(mean=float(v.mean()),by_type=v.tolist()) for k,v in gains.items()})


def summarize(rows: list[dict]) -> dict:
    rng = np.random.default_rng(17101731)
    draws = rng.integers(len(rows),size=(10000,len(rows)))
    result = {}
    for metric in rows[0]['result']['gains']:
        values = np.asarray([r['result']['gains'][metric]['mean'] for r in rows])
        result[metric] = dict(mean=float(values.mean()),
            ci95_root_bootstrap=np.quantile(values[draws].mean(1),[.025,.975]).tolist(),by_root=values.tolist())
    return dict(roots=len(rows),gains=result,unit='PPO team round seat_return',
        uncertainty='Exploratory root-cluster bootstrap; only eight independent roots; fixed archive histories and selected Q tables; not a power guarantee',
        posterior_accuracy=float(np.mean([r['result']['posterior_accuracy'] for r in rows])),
        rounds=sum(r['selection']['rounds']+r['evaluation']['rounds'] for r in rows))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--roots',type=int,default=8,choices=range(1,9))
    p.add_argument('--selection-worlds',type=int,default=6)
    p.add_argument('--evaluation-worlds',type=int,default=12)
    args = p.parse_args()
    if args.selection_worlds<1 or args.evaluation_worlds<1:
        p.error('world counts must be positive')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    actor,identity = load_actor(ENDPOINT)
    opponents,opponent_meta = load_selected(POOL)
    import train.history_model as model
    protocol = dict(roots=args.roots,candidate_cap=5,selection_worlds=args.selection_worlds,
        evaluation_worlds=args.evaluation_worlds,max_rounds=args.roots*4*5*(args.selection_worlds+args.evaluation_worlds),
        source_sha=sha(__file__),model_source_sha=sha(model.__file__),engine_sha=sha(gd._gd_core.__file__),
        endpoint=identity,opponents=opponent_meta,torch=torch.__version__,numpy=np.__version__,
        selection_seed=17101711,evaluation_seed=17101712,root_seed=17101710,
        continuation='Round-only frozen actor on seats 0/2; type-z frozen Transformer on seats 1/3; temperature-one paired sampling',
        root_policy='Greedy native FULL with three archived public rounds versus empty ROUND; only initial action uses old history',
        task='Observe archived public rounds then independent fresh level-2 score deal; synthetic diagnostic, not natural continuous match',
        metrics='Independent score-round PPO seat_return gain vs prior-selected action; exploratory, no confirmatory gate',
        censoring='Any incomplete round aborts; no outcome-driven budget extension')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    tables = Tables()
    for match in iter_matches(POOL/'raw',range(8)):
        tables.add(match)
    tables.finish()
    root_seeds = np.random.default_rng(17101710).integers(1,2**63-1,size=args.roots)
    rows = []
    for i,seed in enumerate(root_seeds):
        state = root_state(int(seed))
        histories,post,history_meta = public_histories(i,tables)
        probs = root_probabilities(actor,state,histories)
        candidates = select_candidates(probs[0],probs[1:])
        inputs = dict(seed=int(seed),root=i,candidates=candidates.tolist(),histories=history_meta,
                      own_hand=state.hand(0),round_actor_prob=probs[0].tolist())
        (args.output/f'root-{i:02d}-inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
        qs,qt,details = None,None,[]
        for split,count in ((0,args.selection_worlds),(1,args.evaluation_worlds)):
            seeds = np.random.default_rng(np.random.SeedSequence([17101711+split,i])).integers(1,2**63-1,size=count)
            q,detail = rollout_matrix(actor,opponents,state,candidates,seeds)
            details.append(detail)
            np.savez_compressed(args.output/f'root-{i:02d}-{"selection" if split==0 else "evaluation"}.npz',returns=q,world_seeds=seeds,candidates=candidates)
            if split==0: qs=q
            else: qt=q
        result = evaluate(qs,qt,post,probs,candidates)
        row = dict(root=i,selection=details[0],evaluation=details[1],result=result)
        rows.append(row)
        (args.output/f'root-{i:02d}-result.json').write_text(json.dumps(row,indent=2)+'\n')
        print(json.dumps(dict(root=i,candidates=len(candidates),rounds=sum(d['rounds'] for d in details),gains={k:v['mean'] for k,v in result['gains'].items()})),flush=True)
    summary = summarize(rows)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'complete.json').write_text(json.dumps(dict(complete=True,rounds=summary['rounds']))+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
