"""Local matched-deal search comparison with persisted per-deal diagnostics."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from .duplicate import bootstrap_interval, generate_deals, play_duplicate
from .policies import load_policy
from .search import SearchConfig, SearchPolicy

_BASE = _OPPONENT = _DEALS = _CONFIGS = None


def initialize(checkpoint, device, count, seed, configs):
    import torch
    global _BASE, _OPPONENT, _DEALS, _CONFIGS
    torch.set_num_threads(1)
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS required but unavailable; CPU fallback is disabled')
    _BASE, _OPPONENT = (load_policy(checkpoint, device=device) for _ in range(2))
    for policy in (_BASE, _OPPONENT):
        policy.actor.causal_sdpa = True
        policy.actor.batched_private_attention = True
    _DEALS, _CONFIGS = generate_deals(count, seed), configs
    # Warm up the device before timed deals, without touching evaluation states.
    import gd, random
    engine, state = gd.Engine(), gd.MatchState()
    engine.new_match(state, seed + 999999)
    _BASE.start_match()
    _BASE.select(engine, state, engine.legal_actions(state), random.Random(0))


def run_deal(task):
    arm, index, seed = task
    search = SearchPolicy(_BASE, SearchConfig(**_CONFIGS[arm]))
    start = time.perf_counter()
    score = play_duplicate(_DEALS[index], search, _OPPONENT, seed + index)
    return {'arm': arm, 'deal': index, 'score': asdict(score),
            'net': score.levels_per_round,
            'win_rate': ((score.first.winning_team == 0) + (score.swapped.winning_team == 1)) / 2,
            'wall_seconds': time.perf_counter() - start, 'calls': search.calls,
            'triggered': search.triggered, 'completed': search.completed,
            'fallbacks': search.fallbacks, 'overrides': search.overrides,
            'search': search.decisions}


_MIN_WORLDS = {}


def _min_worlds(arm):
    return _MIN_WORLDS.get(arm, 1)


def summarize(records, seed):
    arms = {}
    for arm in sorted({r['arm'] for r in records}):
        rows = sorted((r for r in records if r['arm'] == arm), key=lambda r:r['deal'])
        decisions = [d for r in rows for d in r['search']]
        lat = [d['elapsed_ms'] for d in decisions]
        worlds = [d['worlds'] for d in decisions]
        arms[arm] = {'deals':len(rows), 'rounds':2*len(rows),
            'net':float(np.mean([r['net'] for r in rows])),
            'net_ci':bootstrap_interval([r['net'] for r in rows], seed+173, 5000),
            'win_rate':float(np.mean([r['win_rate'] for r in rows])),
            'win_ci':bootstrap_interval([r['win_rate'] for r in rows], seed+174, 5000),
            'calls':sum(r['calls'] for r in rows),
            **{k:sum(r[k] for r in rows) for k in ('triggered','completed','fallbacks','overrides')},
            'summed_deal_seconds':sum(r['wall_seconds'] for r in rows),
            'search_ms_p50_p95_max':([float(np.quantile(lat,.5)),float(np.quantile(lat,.95)),max(lat)] if lat else []),
            'worlds_min_median_max':([min(worlds),float(np.median(worlds)),max(worlds)] if worlds else []),
            'total_worlds':sum(worlds),
            'budget_exhausted':sum(d['budget_exhausted'] for d in decisions),
            'all_action_fraction':float(np.mean([d['candidates']==d['legal_actions'] for d in decisions])) if decisions else None,
            'override_fraction':(sum(r['overrides'] for r in rows)/max(1,sum(r['completed'] for r in rows))),
            'min_worlds_met':sum(d['worlds']>=_min_worlds(arm) for d in decisions),
            'max_legal_actions':max((d['legal_actions'] for d in decisions),default=0)}
    contrasts = {}
    by_arm = {arm:{r['deal']:r['net'] for r in records if r['arm']==arm} for arm in arms}
    names = list(arms)
    for i,a in enumerate(names):
        for b in names[:i]:
            common = sorted(by_arm[a].keys() & by_arm[b].keys())
            delta = [by_arm[a][k]-by_arm[b][k] for k in common]
            if delta:
                contrasts[f'{a} minus {b}']={'deals':len(delta),'net_delta':float(np.mean(delta)),
                    'ci':bootstrap_interval(delta,seed+175,5000)}
    return {'arms':arms,'paired_contrasts':contrasts}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--configs',required=True,help='JSON mapping arm names to SearchConfig fields')
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',choices=['mps','cpu'],default='mps')
    parser.add_argument('--deals',type=int,default=256)
    parser.add_argument('--seed',type=int,default=2026100201)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--deadline-seconds',type=float,default=1800)
    args=parser.parse_args()
    if args.deals<1 or args.workers<1 or args.deadline_seconds<=0:
        parser.error('positive deals, workers and deadline required')
    configs=json.loads(Path(args.configs).read_text())
    configs={arm:asdict(SearchConfig(**config)) for arm,config in configs.items()}
    _MIN_WORLDS.update({arm:config['min_worlds'] for arm,config in configs.items()})
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    raw_path=out/'raw.jsonl'
    if raw_path.exists():
        raise ValueError('output already has raw results; use a new directory')
    import torch
    source_paths=['eval/search.py','eval/search_rollout.py','eval/search_sweep.py',
                  'eval/history_policy.py','eval/duplicate.py','train/history_model.py',
                  'python/gd/_gd_core.cpython-312-darwin.so']
    provenance={'checkpoint':str(Path(args.checkpoint).resolve()),
        'checkpoint_sha256':hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'configs':configs,'device':args.device,'workers':args.workers,'deals':args.deals,'seed':args.seed,
        'deadline_seconds':args.deadline_seconds,'torch':torch.__version__,
        'runtime':{'causal_sdpa':True,'batched_private_attention':True,'threads_per_worker':1},
        'source_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths}}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    start=time.perf_counter();records=[];stopped=False
    # Deal groups complete in every arm before another group is submitted.
    # The time limit is checked between groups, never in response to score.
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),
        initializer=initialize,initargs=(args.checkpoint,args.device,args.deals,args.seed,configs)) as pool, raw_path.open('w') as raw:
        group_size=max(1,args.workers)
        for first in range(0,args.deals,group_size):
            if time.perf_counter()-start>=args.deadline_seconds:
                stopped=True;break
            # Arm order rotates across groups to limit time/thermal ordering effects.
            names=list(configs);shift=(first//group_size)%len(names);names=names[shift:]+names[:shift]
            tasks=[(arm,k,args.seed) for k in range(first,min(first+group_size,args.deals)) for arm in names]
            futures={pool.submit(run_deal,t) for t in tasks}
            while futures:
                done,futures=wait(futures,timeout=30,return_when=FIRST_COMPLETED)
                for future in done:
                    r=future.result();records.append(r)
                    raw.write(json.dumps(r,separators=(',',':'))+'\n');raw.flush()
                progress={'records':len(records),'target':args.deals*len(configs),
                    'elapsed_seconds':round(time.perf_counter()-start,2),'group_first':first}
                (out/'progress.json').write_text(json.dumps(progress)+'\n')
                print(json.dumps(progress),flush=True)
    result={'status':'time_limit' if stopped else 'complete',
            'wall_seconds':time.perf_counter()-start,**summarize(records,args.seed)}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
