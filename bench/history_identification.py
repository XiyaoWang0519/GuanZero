"""Frozen persistent-opponent histories and public, whole-round likelihoods.

Only independent diagnostic heads may learn in the companion module. Hidden
hands are reconstructed from public played cards and sampled residual holdings;
the likelihood code never receives the actual deal or its RNG seed.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
import math

import gd
import numpy as np
import torch

from bench.filtered_history_collect import load_players, ROOT as SOURCE, save_json
from bench.filtered_identity_oracle import encode_world, predict_worlds
from bench.history_value_tasks import Listener
from bench.qknorm_match_ablation import sha
from eval.history_events import apply_and_observe
from train.history_inference import BatchedHistoryCache
from train.history_model import PublicStream, StreamBatch, DecisionInputs
from train.history_population import weights_digest

ROOT = Path('.work/history-identification-2026-11-15')
SEED = 20261115817
LENGTHS = (0, 1, 3, 10)
PLAYERS = None


def initialize() -> None:
    global PLAYERS
    torch.set_num_threads(1)
    PLAYERS = load_players(SOURCE)


def decode_cards(encoded: np.ndarray) -> np.ndarray:
    """Canonical public card multiplicities, including duplicate physical cards."""
    return encoded[:54].astype(np.int64) + encoded[54:108].astype(np.int64)


def public_transfers(state) -> list:
    # This observation slice is documented as public exchange bookkeeping.
    # It contains no own hand or unseen cards. Seat 0 makes relative=absolute.
    slots = np.asarray(state.observation(0))[1439:1687].reshape(4, 62)
    return [(int(row[54:58].argmax()), int(row[58:62].argmax()), int(row[:54].argmax()))
            for row in slots if row[:54].sum()]


@torch.inference_mode()
def collect_one(task: tuple) -> dict:
    root_string, z, replicate, rounds = task
    folder = Path(root_string)/'histories'/f'z{z:02d}-r{replicate:03d}'
    if (folder/'complete.json').exists():
        return json.loads((folder/'complete.json').read_text())
    folder.mkdir(parents=True, exist_ok=True)
    actor, opponents, identity = PLAYERS
    opponent = opponents[z+1]
    before = [weights_digest(m.state_dict()) for m in (actor, opponent)]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, z, replicate]))
    engine = gd.Engine(); engine.auto_pass = False
    stream = PublicStream(); listener = Listener(stream)
    caches = [BatchedHistoryCache(actor), BatchedHistoryCache(opponent)]
    state = gd.MatchState(); engine.new_match(state, int(rng.integers(1, 2**63-1)))
    segments = []; records = []; audit = []; previous = None
    segment = 0; started = time.monotonic()
    for r in range(rounds):
        start = stream.prefix
        meta = dict(history_round=r, segment=segment, local_round=int(state.round_index),
            start=start, level=int(state.level), levels=list(state.levels),
            fails=list(state.fails), owner=int(state.owner), previous=previous,
            leader=int(state.to_move) if previous is None else -1)
        original_state = state.serialize()
        decision_audit = []
        for step in range(2048):
            if state.phase == gd.Phase.RoundEnd:
                break
            legal = engine.legal_actions(state); seat = int(state.to_move)
            if len(legal)==1:
                choice = 0
            elif state.phase != gd.Phase.Play:
                choice = int(engine.greedy(state))
            else:
                side = seat % 2; policy = actor if side==0 else opponent
                sb, memory = caches[side].encode([(0, segment)], [stream])
                p = predict_worlds(policy, sb, memory, [encode_world(engine, state)])[0]
                choice = int(rng.choice(len(p), p=p))
            if state.phase == gd.Phase.Play and seat % 2:
                # Audit-only hidden truth; never passed to the public likelihood.
                decision_audit.append((stream.prefix, np.asarray(state.observation(seat), dtype=np.uint8)))
            apply_and_observe(engine, state, legal[choice], [listener], forced=len(legal)==1)
        else:
            raise RuntimeError('round did not finish')
        arrays = stream.arrays()
        played = np.zeros((4,54), np.int64)
        for token, phase in zip(arrays[0][start:], arrays[2][start:]):
            if phase==3: played[int(token[:4].argmax())] += decode_cards(token[4:158])
        left = [len(state.hand(s)) for s in range(4)]
        assert np.array_equal(played, np.array([np.bincount(state.played(s), minlength=54) for s in range(4)]))
        meta.update(end=stream.prefix, played=played.tolist(), left=left, transfers=public_transfers(state))
        records.append(meta)
        audit.append(dict(start_state=original_state, final_hands=[state.hand(s) for s in range(4)],
                          decision_observations=decision_audit))
        result = engine.end_round(state); previous = list(result.order)
        if r+1==rounds or state.winner>=0:
            segments.append(tuple(x.copy() for x in stream.arrays()))
        if r+1<rounds:
            if state.winner>=0:
                segment += 1; previous = None
                engine.new_match(state, int(rng.integers(1,2**63-1)))
                stream = PublicStream(); listener = Listener(stream)
                caches = [BatchedHistoryCache(actor), BatchedHistoryCache(opponent)]
            else:
                engine.begin_round(state)
    assert before == [weights_digest(m.state_dict()) for m in (actor, opponent)]
    public = dict(segments=segments, rounds=records)
    torch.save(public, folder/'public.pt')
    torch.save(audit, folder/'audit-private.pt')
    record = dict(z=z, replicate=replicate, rounds=rounds, segments=len(segments),
        events=sum(len(s[0]) for s in segments), seconds=time.monotonic()-started,
        public_sha=sha(folder/'public.pt'), frozen_verified=True)
    save_json(folder/'complete.json', record)
    return record


def sample_initial(meta: dict, rng: np.random.Generator) -> tuple:
    """Uniform residual proposal and its exact duplicate-card importance ratio.

    Prior(initial deal)/q(residual allocation) is proportional to
    prod residual_count! / prod initial_count!. Ignoring this factor overweights
    deals in which a seat retained a duplicate of a card it already played.
    """
    played = np.asarray(meta['played'], np.int64)
    pool = np.repeat(np.arange(54), 2-played.sum(0))
    rng.shuffle(pool)
    residual = np.zeros((4,54), np.int64); at = 0
    for seat, count in enumerate(meta['left']):
        residual[seat] = np.bincount(pool[at:at+count], minlength=54); at += count
    initial = played+residual
    for payer, receiver, card in reversed(meta['transfers']):
        initial[receiver,card] -= 1; initial[payer,card] += 1
        if initial[receiver,card] < 0:return None, None
    if not np.all(initial.sum(1)==27) or (initial<0).any() or (initial>2).any():return None,None
    log_weight = math.log(2)*(int((residual==2).sum())-int((initial==2).sum()))
    return initial, log_weight


class ResidualSampler:
    """Uniform physical residual allocations conditioned on public bounds.

    Round-end rules leave at most two nonempty hands. A small dynamic program
    samples that hypergeometric allocation exactly, including duplicate-copy
    multiplicities. Tribute rank ceilings reduce rejection without using truth.
    Full legal/greedy replay remains authoritative for all other constraints.
    """
    def __init__(self, meta: dict, arrays: tuple):
        self.meta=meta;self.played=np.asarray(meta['played'],np.int64)
        self.cards=2-self.played.sum(0)
        self.base=self.played.copy()
        for payer,receiver,card in reversed(meta['transfers']):
            self.base[receiver,card]-=1;self.base[payer,card]+=1
        lower=np.maximum(-self.base,0);upper=np.broadcast_to(self.cards,(4,54)).copy()
        for seat,count in enumerate(meta['left']):
            if not count:lower[seat]=0;upper[seat]=0
        level=meta['level'];powers=np.array([gd.power(c//4 if c<52 else c-39,level) for c in range(54)])
        for pos in range(meta['start'],meta['end']):
            if int(arrays[2][pos])!=1:continue
            token=arrays[0][pos];seat=int(token[:4].argmax());card=int(decode_cards(token[4:158]).argmax())
            forbidden=(powers>powers[card]);forbidden[level*4+1]=False
            required=-self.base[seat,forbidden]
            lower[seat,forbidden]=required;upper[seat,forbidden]=required
            lower[seat,card]=max(lower[seat,card],1-self.base[seat,card])
        self.active=np.flatnonzero(meta['left'])
        if len(self.active)==1:
            self.single=int(self.active[0]);return
        assert len(self.active)==2
        a,b=map(int,self.active);self.a=a;self.b=b;self.n=int(meta['left'][a])
        self.options=[]
        for c,total in enumerate(self.cards):
            options=[(k,math.comb(int(total),k)) for k in range(int(total)+1)
                if lower[a,c]<=k<=upper[a,c] and lower[b,c]<=total-k<=upper[b,c]]
            if not options:raise ValueError('inconsistent public residual bounds')
            self.options.append(options)
        self.dp=np.zeros((55,self.n+1));self.dp[54,0]=1
        for c in range(53,-1,-1):
            for k,multiplicity in self.options[c]:
                if k<=self.n:self.dp[c,k:]+=multiplicity*self.dp[c+1,:self.n+1-k]
        if self.dp[0,self.n]<=0:raise ValueError('public allocation has empty support')

    def draw(self,rng:np.random.Generator) -> tuple:
        residual=np.zeros((4,54),np.int64)
        if len(self.active)==1:residual[self.single]=self.cards
        else:
            remaining=self.n
            for c,options in enumerate(self.options):
                eligible=[(k,m*self.dp[c+1,remaining-k]) for k,m in options if k<=remaining]
                weights=np.array([w for _,w in eligible]);weights/=weights.sum()
                k=eligible[int(rng.choice(len(eligible),p=weights))][0]
                residual[self.a,c]=k;residual[self.b,c]=self.cards[c]-k;remaining-=k
            assert remaining==0
        initial=self.base+residual
        if (initial<0).any() or not np.all(initial.sum(1)==27):return None,None
        log_weight=math.log(2)*(int((residual==2).sum())-int((initial==2).sum()))
        return initial,log_weight


def replay(meta: dict, arrays: tuple, initial: np.ndarray) -> list | None:
    """Replay one coherent hidden deal through every recorded public action."""
    engine = gd.Engine(); engine.auto_pass = False
    deal = gd.DealSpec(); deal.hands = [np.repeat(np.arange(54), x).tolist() for x in initial]
    deal.level = meta['level']; deal.team_levels = meta['levels']; deal.fails = meta['fails']; deal.owner = meta['owner']
    if meta['previous'] is None:deal.leader = meta['leader']
    else:deal.prev_order = meta['previous']
    state = gd.MatchState(); engine.set_deal(state, deal)
    decisions = []
    for pos in range(meta['start'], meta['end']):
        token = arrays[0][pos]; seat = int(token[:4].argmax()); phase = int(arrays[2][pos])
        if state.phase==gd.Phase.RoundEnd or int(state.to_move)!=seat or int(state.phase)!=phase:
            return None
        legal = engine.legal_actions(state)
        encoded = np.asarray([engine.encode_action(a,state,seat) for a in legal], dtype=np.uint8)
        matches = np.flatnonzero(np.all(encoded[:,:146] == token[None,4:150],axis=1))
        if not len(matches):return None
        if phase!=3:
            choice = int(engine.greedy(state))
            if choice not in matches:return None
        else:
            choice = int(matches[0])
            if seat%2:
                decisions.append(dict(obs=np.asarray(state.observation(seat),dtype=np.uint8),
                    seat=seat, cand=encoded, prefix=pos, choices=matches))
        engine.apply(state, legal[choice])
        if len(state.hand(seat))!=int(token[158:].argmax()):return None
    if state.phase!=gd.Phase.RoundEnd:return None
    if [len(state.hand(s)) for s in range(4)]!=meta['left']:return None
    if public_transfers(state)!=[tuple(t) for t in meta['transfers']]:return None
    return decisions


@torch.inference_mode()
def trajectory_logp(actor, sb, encoded, trajectories: list[list], batch_size: int=128) -> np.ndarray:
    frames = [(i,d) for i,t in enumerate(trajectories) for d in t]
    totals = np.zeros(len(trajectories))
    for at in range(0,len(frames),batch_size):
        chunk = frames[at:at+batch_size]; n=len(chunk)
        offsets=np.r_[0,np.cumsum([len(d['cand']) for _,d in chunk])]
        inputs=DecisionInputs(sb,torch.zeros(n,dtype=torch.long),
            torch.tensor([d['prefix'] for _,d in chunk]),
            torch.from_numpy(np.stack([d['obs'] for _,d in chunk])),
            torch.tensor([d['seat'] for _,d in chunk]),
            torch.from_numpy(np.concatenate([d['cand'] for _,d in chunk])),
            torch.from_numpy(offsets),one_decision_per_stream=False)
        lp=actor.candidate_log_probs(inputs,encoded=encoded).double().numpy()
        for j,(i,d) in enumerate(chunk):
            a=lp[offsets[j]+d['choices']]; m=a.max()
            totals[i] += m + np.log(np.exp(a-m).sum())
    return totals


def logmean(logp: np.ndarray, logweight: np.ndarray) -> tuple:
    value=logp+logweight[None,:]
    m=value.max(1,keepdims=True); w=np.exp(value-m)
    total=m[:,0]+np.log(w.mean(1))
    ess=w.sum(1)**2/(w*w).sum(1)
    # Only common per-round scale is discarded; identity ratios are preserved.
    return total-total.max(), ess


@torch.inference_mode()
def likelihood_one(task: tuple) -> dict:
    root_string,z,replicate,particles=task
    folder=Path(root_string)/'histories'/f'z{z:02d}-r{replicate:03d}'
    destination=folder/f'likelihood-{particles}.npz'
    if destination.exists():return dict(z=z,replicate=replicate,cached=True)
    actor,opponents,identity=PLAYERS
    # This is the only history input: no actual initial hands or source seed.
    public=torch.load(folder/'public.pt',weights_only=False)
    prepared=[]; metadata=[];started=time.monotonic()
    for r,meta in enumerate(public['rounds']):
        exact = (sum(c>0 for c in meta['left'])==1 or (meta['left'][1]==0 and meta['left'][3]==0))
        wanted = 1 if exact else 2*particles
        # Randomness is a function of public bytes, never the true label.
        public_seed=int(sha(folder/'public.pt')[:16],16)
        rng=np.random.default_rng([SEED+19,public_seed,r,particles])
        sampler=ResidualSampler(meta,public['segments'][meta['segment']])
        trajectories=[];logweights=[];attempts=0
        while len(trajectories)<wanted and attempts<max(10000,wanted*2000):
            attempts+=1
            initial, weight=sampler.draw(rng)
            if initial is None:continue
            frames=replay(meta,public['segments'][meta['segment']],initial)
            if frames is None:continue
            trajectories.append(frames);logweights.append(weight)
        if len(trajectories)!=wanted:
            raise RuntimeError(f'public proposal failed: {z}/{replicate}/{r}: {len(trajectories)}/{attempts}')
        prepared.append((trajectories,np.asarray(logweights)))
        metadata.append(dict(exact=exact,accepted=wanted,attempts=attempts,decisions=len(trajectories[0]),
            proposal='public-bounds conditional hypergeometric'))
    logs=np.zeros((len(prepared),20,2*particles)); valid=np.zeros(len(prepared),np.int64)
    for policy_index,policy in enumerate(opponents.values()):
        for segment,arrays in enumerate(public['segments']):
            sb=StreamBatch.from_arrays([arrays],'cpu');encoded=policy.encode_batch(sb)
            for r,meta in enumerate(public['rounds']):
                if meta['segment']!=segment:continue
                trajectories,_=prepared[r]
                p=trajectory_logp(policy,sb,encoded,trajectories)
                logs[r,policy_index,:len(p)]=p;valid[r]=len(p)
    round_likelihood=np.zeros((len(prepared),2,20));ess=np.zeros((len(prepared),2,20))
    weights_array=np.zeros((len(prepared),2*particles))
    for r,(_,lw) in enumerate(prepared):
        weights_array[r,:len(lw)]=lw
        for half in range(2):
            sl=slice(0,1) if valid[r]==1 else slice(half*particles,(half+1)*particles)
            round_likelihood[r,half],ess[r,half]=logmean(logs[r,:,sl],lw[sl])
    np.savez_compressed(destination,round_loglik=round_likelihood,ess=ess,raw_logp=logs,
        logweight=weights_array,valid=valid)
    result=dict(z=z,replicate=replicate,particles=particles,seconds=time.monotonic()-started,
        rounds=metadata,minimum_ess=float(ess.min()),source_public_sha=sha(folder/'public.pt'))
    result['likelihood_source_sha']=sha(__file__)
    save_json(folder/f'likelihood-{particles}.json',result)
    return {k:result[k] for k in ('z','replicate','particles','seconds','minimum_ess')}


def seal(args) -> None:
    args.root.mkdir(parents=True,exist_ok=True)
    _,_,identity=load_players(SOURCE,indices=[])
    protocol=dict(identity=identity,seed=SEED,replicates=args.replicates,rounds=10,lengths=LENGTHS,
        split='per opponent: replicates 0..19 train, 20..23 validation, 24..31 test',
        histories='same frozen opponent at seats 1/3, u195450 at 0/2, 10 consecutive rounds; normal match restart if terminal; all public events retained',
        posterior='whole-round public likelihood; exact when opponent starting hands are determined by public completed-round cards; otherwise coherent residual-hand importance integration',
        public_only='played cards, remaining counts, public tribute ledger and board state; original private deals and RNG seeds excluded from likelihood and probes',
        duplicate_cards='proposal corrected by prod residual_count! / prod pre-tribute initial_count!',
        primary_length=3,primary_metric='independent-question Brier improvement and fraction of true-identity benefit recovered; 11 types primary, full actions secondary',
        question_bank='.work/filtered-identity-oracle-2026-11-15; fixed independent neutral-prefix questions; no new question selection',
        controls='no history, different-opponent history, same-identity independent history, true identity; known planted preference controls in tests; exact cases and independent particle halves',
        probes='raw public token sequence; embedding, encoder layers 1..4, final norm; frozen native attention output; train-only scaling and heldout validation; no main model optimization',
        scope='isolated CPU <=4 workers, frozen checkpoints and main training untouched',
        sources={p:sha(p) for p in (__file__,'train/history_model.py','cpp/src/state.cpp','cpp/src/encoder.cpp')})
    path=args.root/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(protocol)):
        raise ValueError('protocol differs; choose another directory')
    save_json(path,protocol)


def dispatch(args) -> None:
    if args.command=='collect':seal(args)
    else:assert (args.root/'protocol.json').exists()
    tasks=[(str(args.root),z,r,10 if args.command=='collect' else args.particles)
           for z in range(args.types) for r in range(args.replicate_start,args.replicates)]
    started=time.monotonic()
    with ProcessPoolExecutor(args.workers,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        jobs=[pool.submit(collect_one if args.command=='collect' else likelihood_one,t) for t in tasks]
        for done,future in enumerate(as_completed(jobs),1):
            print(json.dumps(dict(done=done,total=len(tasks),elapsed=round(time.monotonic()-started,2),**future.result())),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['collect','likelihood'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--replicates',type=int,default=32);p.add_argument('--types',type=int,default=20)
    p.add_argument('--particles',type=int,default=32)
    p.add_argument('--replicate-start',type=int,default=0)
    dispatch(p.parse_args())
