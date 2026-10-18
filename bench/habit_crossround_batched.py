"""Stage 0b on the batched rollout engine: many matches advance together and
their searches share one VecEnv / one forward per step.

Same task, planners and records as ``habit_crossround_value.play_match`` (see
that module); here a match is a generator that yields what it needs:

  ('probs', state, arrays)   one live decision: the base network's probabilities
  ('search', SearchRequest)  one paired late-round search: the Q table

and a scheduler answers all yielded requests of all live matches at once
(``bench/batched_rollout.py``). ``exact=True`` answers each ``probs`` request
with ``decision_probabilities`` and runs searches with the reference sampler,
which reproduces ``play_match`` bit for bit (the equivalence test); the
default batches ``probs`` requests and uses the counter-based sampler.

Run with the checkpoint-matching model root and the VecEnv.load engine build
first on PYTHONPATH.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Iterator

import gd
import numpy as np
import torch

from bench.batched_rollout import SearchRequest, batched_search
from bench.habit_crossround_value import (CANDIDATES, CONFIGS, MARGIN, SKIP_GAP, TARGET_ROUND, TRIGGER,
                                          WORLDS, Branch, analyze, deal_plan, sample)
from bench.habit_identifiability import feature, llr, sigmoid, tilt
from bench.history_habit_round import choose
from bench.history_habit_value import BASE, load_plain, sha
from bench.history_value_tasks import Listener, decision_probabilities
from eval.history_policy import apply_and_observe
from train.history_model import DecisionInputs, PublicStream, StreamBatch


def match_steps(axis: str, b: float, deal_seed: int, z: int, seed: int,
                candidates: int = CANDIDATES, worlds: int = WORLDS, margin: float = MARGIN,
                trigger: int = TRIGGER, target: int = TARGET_ROUND,
                skip_gap: float = SKIP_GAP) -> Iterator[tuple]:
    """Generator form of ``play_match``; returns the same record."""
    engine = gd.Engine()
    engine.auto_pass = False
    state = gd.MatchState()
    engine.new_match(state, int(deal_seed))
    live_ss, world_ss, prefix_ss = np.random.SeedSequence([seed, deal_seed]).spawn(3)
    prefix_rng = np.random.default_rng(prefix_ss)
    stream = PublicStream()
    listener = Listener(stream)
    full = 0.0
    start = time.time()

    def opponent(s, legal, strm, rng):
        p = yield ('probs', s, tuple(x.copy() for x in strm.arrays()))
        f = feature(legal, axis)
        a = sample(tilt(p, f, z * b), rng.random())
        return a, llr(p, f, b, a)

    for _ in range(target):
        for _ in range(4000):
            if state.phase == gd.Phase.RoundEnd:
                break
            legal = engine.legal_actions(state)
            if state.phase != gd.Phase.Play:
                apply_and_observe(engine, state, legal[engine.greedy(state)], [listener])
                continue
            if len(legal) == 1:
                apply_and_observe(engine, state, legal[0], [listener])
                continue
            if int(state.to_move) % 2 == 1:
                a, step = yield from opponent(state, legal, stream, prefix_rng)
                full += step
            else:
                p = yield ('probs', state, tuple(x.copy() for x in stream.arrays()))
                a = sample(p, prefix_rng.random())
            apply_and_observe(engine, state, legal[a], [listener])
        else:
            raise RuntimeError('prefix round decision cap hit')
        engine.end_round(state)
        if state.winner >= 0:
            raise RuntimeError('match ended before the target round')
        engine.begin_round(state)
    out = dict(deal_seed=int(deal_seed), z=int(z), forked=False, disagreements=0, searches=0, skipped=0,
               prefix_llr=full, prefix_seconds=time.time() - start, post_round=[], post_full=[])
    todo = [Branch(state, stream, np.random.default_rng(live_ss), np.random.default_rng(world_ss),
                   'both', 0.0, full)]
    while todo:
        t = todo.pop()
        for _ in range(4000):
            s = t.state
            if s.phase == gd.Phase.RoundEnd:
                result = engine.end_round(s)
                rec = dict(win=int(result.winning_team) == 0, ret=float(result.seat_return[0]),
                           searches=t.searches)
                for name in (('full', 'round') if t.planner == 'both' else (t.planner,)):
                    out[name] = rec
                break
            legal = engine.legal_actions(s)
            if s.phase != gd.Phase.Play:
                apply_and_observe(engine, s, legal[engine.greedy(s)], [Listener(t.stream)])
                continue
            if len(legal) == 1:
                apply_and_observe(engine, s, legal[0], [Listener(t.stream)])
                continue
            if int(s.to_move) % 2 == 1:
                a, step = yield from opponent(s, legal, t.stream, t.live)
                t.llr_round += step
                t.llr_full += step
                apply_and_observe(engine, s, legal[a], [Listener(t.stream)])
                continue
            arrays = tuple(x.copy() for x in t.stream.arrays())
            p = yield ('probs', s, arrays)
            pr, pf = sigmoid(t.llr_round), sigmoid(t.llr_full)
            late = min(len(s.hand(i)) for i in range(4)) <= trigger
            if late and abs(pf - pr) < skip_gap and t.planner == 'both':
                out['skipped'] += 1
            if not late or abs(pf - pr) < skip_gap:
                apply_and_observe(engine, s, legal[int(np.argmax(p))], [Listener(t.stream)])
                continue
            cand = np.argsort(-p, kind='stable')[:candidates]
            q = yield ('search', SearchRequest(s, arrays, cand, t.worlds.integers(1, 2**63 - 1, size=worlds),
                                               (-1, 1), axis, b))
            t.searches += 1
            picks = {}
            if t.planner in ('both', 'full'):
                picks['full'] = choose(q, np.array([1 - pf, pf]), margin)
            if t.planner in ('both', 'round'):
                picks['round'] = choose(q, np.array([1 - pr, pr]), margin)
            if t.planner == 'both':
                out['searches'] += 1
                out['post_round'].append(pr if z == 1 else 1 - pr)
                out['post_full'].append(pf if z == 1 else 1 - pf)
                if picks['full'] != picks['round']:
                    out['disagreements'] += 1
                    out['forked'] = True
                    other = t.fork('full')
                    apply_and_observe(engine, other.state, legal[int(cand[picks['full']])],
                                      [Listener(other.stream)])
                    todo.append(other)
                    t.planner = 'round'
                    apply_and_observe(engine, s, legal[int(cand[picks['round']])], [Listener(t.stream)])
                    continue
                pick = picks['full']
            else:
                pick = picks[t.planner]
            apply_and_observe(engine, s, legal[int(cand[pick])], [Listener(t.stream)])
        else:
            raise RuntimeError('target round decision cap hit')
    return out


@torch.no_grad()
def batched_probs(actor, items: list[tuple]) -> list[np.ndarray]:
    """Candidate probabilities for several (state, arrays) decisions in one forward."""
    engine = gd.Engine()
    device = actor.bos.device
    legal = [engine.legal_actions(s) for s, _ in items]
    seats = [int(s.to_move) for s, _ in items]
    sb = StreamBatch.from_arrays([a for _, a in items], device)
    cand = np.concatenate([np.asarray([engine.encode_action(a, s, seat) for a in lg], np.float32)
                           for (s, _), lg, seat in zip(items, legal, seats)])
    counts = [len(lg) for lg in legal]
    offsets = np.r_[0, np.cumsum(counts)]
    inputs = DecisionInputs(sb, torch.arange(len(items), device=device), sb.lengths,
                            torch.tensor(np.stack([np.asarray(s.observation(seat)) for (s, _), seat in zip(items, seats)]),
                                         device=device),
                            torch.tensor(seats, device=device), torch.tensor(cand, device=device),
                            torch.tensor(offsets, device=device), one_decision_per_stream=True)
    table = actor._log_prob_table(inputs, None, max(counts)).cpu().double().numpy()
    out = []
    for i, n in enumerate(counts):
        p = np.exp(table[i, :n])
        out.append(p / p.sum())
    return out


def run_matches(actor, specs: list[dict], exact: bool = False, concurrency: int = 32,
                on_done=None, max_branches: int = 4096) -> list[dict]:
    """Advance up to ``concurrency`` matches together; ``specs`` are play_match kwargs."""
    queue = list(specs)
    live: list[tuple[dict, Iterator, Any]] = []
    done = []

    def admit():
        while queue and len(live) < concurrency:
            spec = queue.pop(0)
            gen = match_steps(**{k: v for k, v in spec.items() if k not in ('config', 'index')})
            live.append((spec, gen, None, time.time()))

    def advance(entry, answer):
        spec, gen, _, started = entry
        try:
            req = gen.send(answer) if answer is not None else next(gen)
            return (spec, gen, req, started)
        except StopIteration as stop:
            rec = stop.value
            rec.update(config=spec.get('config'), index=spec.get('index'), seconds=time.time() - started)
            done.append(rec)
            if on_done:
                on_done(rec)
            return None

    admit()
    live = [e for e in (advance(e, None) for e in live) if e]
    while live or queue:
        # Answer every live match's pending request; probs first (cheap), then
        # all searches of this round together.
        probs = [i for i, e in enumerate(live) if e[2][0] == 'probs']
        if probs:
            items = [(live[i][2][1], live[i][2][2]) for i in probs]
            answers = ([decision_probabilities(actor, s, a) for s, a in items] if exact
                       else batched_probs(actor, items))
            for i, ans in zip(probs, answers):
                live[i] = advance(live[i], ans)
        else:
            searches = [i for i, e in enumerate(live) if e[2][0] == 'search']
            # Bound one VecEnv/cache batch to max_branches rollout branches.
            chunk, size = [], 0
            for i in searches + [None]:
                req = live[i][2][1] if i is not None else None
                n = 0 if req is None else len(req.z) * len(req.candidates) * len(req.seeds)
                if chunk and (req is None or size + n > max_branches):
                    qs = batched_search(actor, [live[j][2][1] for j in chunk],
                                        sampler='reference' if exact else 'fast')
                    for j, q in zip(chunk, qs):
                        live[j] = advance(live[j], q)
                    chunk, size = [], 0
                if req is not None:
                    chunk.append(i)
                    size += n
        live = [e for e in live if e]
        if len(live) < concurrency and queue:
            before = len(live)
            admit()
            live[before:] = [e for e in (advance(e, None) for e in live[before:]) if e]
    return done


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('plan', 'run', 'analyze'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deals', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cpu')
    p.add_argument('--concurrency', type=int, default=32)
    p.add_argument('--max-branches', type=int, default=4096, help='rollout branches per batched search call')
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--prefix', type=int, default=None)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.stage == 'plan':
        if args.deals < 1 or args.seed < 1:
            p.error('--deals and --seed are required')
        (args.output / 'matches').mkdir(parents=True, exist_ok=False)
        import bench.batched_rollout as br
        import train.history_model as model
        protocol = dict(base=str(BASE), base_sha=sha(BASE), source_sha=sha(__file__),
                        rollout_source_sha=sha(br.__file__), model_source_sha=sha(model.__file__),
                        engine_sha=sha(gd._gd_core.__file__), seed=args.seed, deals=args.deals,
                        configs=[list(c) for c in CONFIGS], candidates=CANDIDATES, worlds=WORLDS,
                        margin=MARGIN, trigger=TRIGGER, target_round=TARGET_ROUND, skip_gap=SKIP_GAP,
                        sampler='counter-based splitmix64 (fast)',
                        primary='full-posterior minus round-posterior planner, target-round team win (pp)')
        (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        print(json.dumps(protocol, indent=2))
        return
    protocol = json.loads((args.output / 'protocol.json').read_text())
    if protocol['source_sha'] != sha(__file__) or protocol['base_sha'] != sha(BASE):
        raise ValueError('source or checkpoint changed since plan')
    if args.stage == 'analyze':
        s = analyze(args.output, args.prefix)
        (args.output / 'summary.json').write_text(json.dumps(s, indent=2) + '\n')
        for r in s['configs']:
            print(f"{r['axis']:>5} b={r['strength']:.0f}  n {r['deals']}  full {r['full_win']:.1f}%  "
                  f"round {r['round_win']:.1f}%  diff {r['win_diff']:+.2f} "
                  f"[{r['win_diff_ci'][0]:+.2f},{r['win_diff_ci'][1]:+.2f}]  forked {r['forked']}/{r['deals']}")
        return
    actor = load_plain(BASE).to(args.device)
    seeds, zs = deal_plan(protocol['deals'], protocol['seed'])
    count = args.limit or protocol['deals']
    specs = []
    for i in range(args.start, count):
        for c, (axis, b) in enumerate(CONFIGS):
            if not (args.output / 'matches' / f'c{c}-d{i:04d}.json').exists():
                specs.append(dict(config=c, index=i, axis=axis, b=b, deal_seed=int(seeds[i]),
                                  z=int(zs[i]), seed=protocol['seed']))
    start, finished = time.time(), [0]

    def save(rec):
        path = args.output / 'matches' / f"c{rec['config']}-d{rec['index']:04d}.json"
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(rec) + '\n')
        tmp.rename(path)
        finished[0] += 1
        print(f'{finished[0]}/{len(specs)} matches, {time.time() - start:.0f}s', flush=True)

    run_matches(actor, specs, concurrency=args.concurrency, on_done=save, max_branches=args.max_branches)


if __name__ == '__main__':
    main()
