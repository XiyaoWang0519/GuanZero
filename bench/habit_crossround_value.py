"""Stage 0b of the habit attention plan: decision value of cross-round history.

docs/reports/habit-attention-ab-plan-2026-10-17.md. Two public-information
planners play the same matches against the habit pack (one z per match):

  full   weights the two styles by the ideal-observer posterior from every
         opponent action of the match so far;
  round  weights them by the posterior from this round's opponent actions only.

Their difference bounds what a FULL learner can gain over a ROUND learner in
the scored round. Rounds before TARGET_ROUND are played without search (the
observer team samples u2623 at temperature one) to build history. In the
target round the observer team plays the blueprint's greedy choice until some
seat holds <= TRIGGER cards, then every decision with two or more legal
actions runs the paired search of history_habit_round.py: one rollout table
under both styles, each planner applies its own weights. Where the two
posteriors differ by less than SKIP_GAP the planners would choose alike, so
both play the blueprint's greedy choice there instead of searching (each
planner applies this rule with its own posteriors after a fork). Shared trajectory
while the planners agree, fork at the first disagreement; each fork keeps the
same RNG state and tracks its own posteriors. Only the target round is scored.

Run with the checkpoint-matching model root first on PYTHONPATH.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import gd
import numpy as np
import torch

from bench.habit_identifiability import feature, llr, sigmoid, tilt
from bench.history_habit_round import bootstrap, choose, team_rollout
from bench.history_habit_value import BASE, load_plain, load_styled, sha
from bench.history_value_tasks import Listener, decision_probabilities
from eval.history_policy import apply_and_observe
from train.history_model import PublicStream

CANDIDATES = 4
WORLDS = 64
MARGIN = 0.02
TRIGGER = 10
TARGET_ROUND = 2        # zero-based: the third round of the match
SKIP_GAP = 0.05         # search only where |P_full(z=+1) - P_round(z=+1)| >= SKIP_GAP
WORKERS = 3
CONFIGS = (('pass', 1.0), ('bomb', 2.0))


class Branch:
    def __init__(self, state, stream, live, worlds, planner, llr_round, llr_full):
        self.state, self.stream, self.live, self.worlds = state, stream, live, worlds
        self.planner, self.llr_round, self.llr_full = planner, llr_round, llr_full
        self.searches = 0

    def fork(self, planner: str) -> 'Branch':
        b = Branch(gd.MatchState.deserialize(self.state.serialize()), copy.deepcopy(self.stream),
                   copy.deepcopy(self.live), copy.deepcopy(self.worlds), planner,
                   self.llr_round, self.llr_full)
        b.searches = self.searches
        return b


def sample(p: np.ndarray, u: float) -> int:
    return min(int(np.searchsorted(np.cumsum(p), u, side='right')), len(p) - 1)


def opponent_move(actor, engine, s, legal, stream, axis, b, z, rng) -> tuple[int, float]:
    """Styled opponent action and its exact LLR (the pack's base is ``actor``)."""
    p = decision_probabilities(actor, s, tuple(x.copy() for x in stream.arrays()))
    f = feature(legal, axis)
    a = sample(tilt(p, f, z * b), rng.random())
    return a, llr(p, f, b, a)


def play_match(actor: Any, styled: dict, axis: str, b: float, deal_seed: int, z: int,
               seed: int, candidates: int = CANDIDATES, worlds: int = WORLDS,
               margin: float = MARGIN, trigger: int = TRIGGER, target: int = TARGET_ROUND,
               skip_gap: float = SKIP_GAP) -> dict:
    """z in {-1, +1}; styled[0] is z = -1, styled[1] is z = +1."""
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
    for _ in range(target):                      # unscored history rounds
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
                a, step = opponent_move(actor, engine, state, legal, stream, axis, b, z, prefix_rng)
                full += step
            else:
                p = decision_probabilities(actor, state, tuple(x.copy() for x in stream.arrays()))
                a = sample(p, prefix_rng.random())
            apply_and_observe(engine, state, legal[a], [listener])
        else:
            raise RuntimeError('prefix round decision cap hit')
        engine.end_round(state)
        if state.winner >= 0:
            raise RuntimeError('match ended before the target round')
        engine.begin_round(state)
    prefix_seconds = time.time() - start
    out = dict(deal_seed=int(deal_seed), z=int(z), forked=False, disagreements=0, searches=0, skipped=0,
               prefix_llr=full, prefix_seconds=prefix_seconds, post_round=[], post_full=[])
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
                a, step = opponent_move(actor, engine, s, legal, t.stream, axis, b, z, t.live)
                t.llr_round += step
                t.llr_full += step
                apply_and_observe(engine, s, legal[a], [Listener(t.stream)])
                continue
            arrays = tuple(x.copy() for x in t.stream.arrays())
            p = decision_probabilities(actor, s, arrays)
            pr, pf = sigmoid(t.llr_round), sigmoid(t.llr_full)        # P(z = +1)
            late = min(len(s.hand(i)) for i in range(4)) <= trigger
            if late and abs(pf - pr) < skip_gap and t.planner == 'both':
                out['skipped'] += 1
            if not late or abs(pf - pr) < skip_gap:
                apply_and_observe(engine, s, legal[int(np.argmax(p))], [Listener(t.stream)])
                continue
            cand = np.argsort(-p, kind='stable')[:candidates]
            q = team_rollout(actor, styled, s, arrays, cand, t.worlds.integers(1, 2**63 - 1, size=worlds))
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


_MODELS: dict = {}


def init_worker(base: str) -> None:
    torch.set_num_threads(1)
    _MODELS['self'] = load_plain(Path(base))
    for axis, b in CONFIGS:
        _MODELS[(axis, b)] = load_styled(Path(base), axis, b)


def work(output: str, config: int, index: int, deal_seed: int, z: int, seed: int) -> str:
    out = Path(output) / 'matches' / f'c{config}-d{index:04d}.json'
    if out.exists():
        return str(out)
    axis, b = CONFIGS[config]
    start = time.time()
    rec = play_match(_MODELS['self'], _MODELS[(axis, b)], axis, b, deal_seed, z, seed)
    rec.update(config=config, index=index, seconds=time.time() - start)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec) + '\n')
    tmp.rename(out)
    return str(out)


def deal_plan(deals: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    seeds = np.random.default_rng(seed).integers(1, 2**63 - 1, size=deals)
    return seeds, np.tile([-1, 1], (deals + 1) // 2)[:deals]


def analyze(output: Path, prefix: int | None = None) -> dict:
    protocol = json.loads((output / 'protocol.json').read_text())
    rows = []
    for c, (axis, b) in enumerate(CONFIGS):
        recs = [json.loads(p.read_text()) for p in sorted((output / 'matches').glob(f'c{c}-d*.json'))]
        if prefix is not None:
            recs = [r for r in recs if r['index'] < prefix]
        if not recs:
            continue
        w = np.array([r['full']['win'] - r['round']['win'] for r in recs], float) * 100
        t = np.array([r['full']['ret'] - r['round']['ret'] for r in recs], float)
        pr = [x for r in recs for x in r['post_round']]
        pf = [x for r in recs for x in r['post_full']]
        rows.append(dict(axis=axis, strength=b, deals=len(recs),
                         full_win=float(np.mean([r['full']['win'] for r in recs]) * 100),
                         round_win=float(np.mean([r['round']['win'] for r in recs]) * 100),
                         win_diff=float(w.mean()), win_diff_ci=bootstrap(w, seed=1),
                         ret_diff=float(t.mean()), ret_diff_ci=bootstrap(t, seed=2),
                         forked=int(sum(r['forked'] for r in recs)),
                         searches=int(sum(r['searches'] for r in recs)),
                         skipped=int(sum(r.get('skipped', 0) for r in recs)),
                         mean_post_round=float(np.mean(pr)) if pr else None,
                         mean_post_full=float(np.mean(pf)) if pf else None,
                         seconds_mean=float(np.mean([r['seconds'] for r in recs])),
                         seconds_median=float(np.median([r['seconds'] for r in recs])),
                         forked_seconds_mean=float(np.mean([r['seconds'] for r in recs if r['forked']] or [np.nan])),
                         unforked_seconds_mean=float(np.mean([r['seconds'] for r in recs if not r['forked']] or [np.nan]))))
    return dict(protocol=protocol, units='win in percentage points; ret is PPO seat_return', configs=rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('plan', 'run', 'analyze'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deals', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=WORKERS)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--prefix', type=int, default=None, help='analyze indices below this only')
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.stage == 'plan':
        if args.deals < 1 or args.seed < 1:
            p.error('--deals and --seed are required')
        (args.output / 'matches').mkdir(parents=True, exist_ok=False)
        import train.history_model as model
        protocol = dict(base=str(BASE), base_sha=sha(BASE), source_sha=sha(__file__),
                        round_source_sha=sha(Path(__file__).with_name('history_habit_round.py')),
                        model_source_sha=sha(model.__file__), engine_sha=sha(gd._gd_core.__file__),
                        seed=args.seed, deals=args.deals, configs=[list(c) for c in CONFIGS],
                        candidates=CANDIDATES, worlds=WORLDS, margin=MARGIN, trigger=TRIGGER,
                        target_round=TARGET_ROUND, skip_gap=SKIP_GAP,
                        primary='full-posterior minus round-posterior planner, target-round team win (pp)',
                        boundary='late-round search only; ideal-observer posteriors from the known pack; '
                                 'no history reading by any network; no training')
        (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        print(json.dumps(protocol, indent=2))
        return
    protocol = json.loads((args.output / 'protocol.json').read_text())
    if protocol['source_sha'] != sha(__file__) or protocol['base_sha'] != sha(BASE):
        raise ValueError('source or checkpoint changed since plan')
    if args.stage == 'run':
        seeds, zs = deal_plan(protocol['deals'], protocol['seed'])
        count = args.limit or protocol['deals']
        jobs = [(c, i, int(seeds[i]), int(zs[i])) for i in range(args.start, count) for c in range(len(CONFIGS))]
        start = time.time()
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                                 initializer=init_worker, initargs=(str(BASE),)) as pool:
            futures = [pool.submit(work, str(args.output), *j, protocol['seed']) for j in jobs]
            for i, f in enumerate(as_completed(futures), 1):
                f.result()
                print(f'{i}/{len(jobs)} matches, {time.time() - start:.0f}s', flush=True)
        return
    s = analyze(args.output, args.prefix)
    (args.output / 'summary.json').write_text(json.dumps(s, indent=2) + '\n')
    for r in s['configs']:
        print(f"{r['axis']:>5} b={r['strength']:.0f}  n {r['deals']}  full {r['full_win']:.1f}%  round {r['round_win']:.1f}%  "
              f"diff {r['win_diff']:+.2f} [{r['win_diff_ci'][0]:+.2f},{r['win_diff_ci'][1]:+.2f}]  "
              f"forked {r['forked']}/{r['deals']}  searches {r['searches']}  post R/F {r['mean_post_round']:.3f}/{r['mean_post_full']:.3f}  "
              f"sec mean {r['seconds_mean']:.0f} (forked {r['forked_seconds_mean']:.0f}, unforked {r['unforked_seconds_mean']:.0f})")


if __name__ == '__main__':
    main()
