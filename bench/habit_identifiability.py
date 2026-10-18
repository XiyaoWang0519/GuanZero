"""Stage 0a of the habit attention plan: how fast an ideal observer identifies z.

docs/reports/habit-attention-ab-plan-2026-10-17.md. Full matches (capped at
MAX_ROUNDS rounds): the observer team (seats 0, 2) plays plain u2623 sampled at
temperature one; both opponents play the habit pack pi_z ∝ pi exp(z b f) with
one z per match. Tribute and back-tribute use the engine heuristic, as in
training. No training, frozen weights, local CPU.

The pack's policies are known, so each public opponent action a carries the
exact log-likelihood ratio LLR = log pi_+(a) - log pi_-(a) (0 for forced
moves). With a 50/50 prior the posterior of the true z after a set of events
is sigmoid(z * sum LLR). At every observer-team play decision we record that
posterior for two views: ROUND (this round's opponent events only) and FULL
(all opponent events of the match so far).

Selection rule, fixed before any run (plan, stage 0a): pick the largest b for
which, over rounds 2+ and the first half of the observer's decisions in each
round, the mean ROUND posterior of the true z is < 0.75 while the mean FULL
posterior is > 0.95. If none qualifies, the habit is visible within a round
(or never), and stage 1 needs a different habit.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

import gd
import numpy as np
import torch

from bench.history_habit_value import BASE, load_plain, sha
from bench.history_value_tasks import Listener, decision_probabilities
from eval.history_policy import apply_and_observe
from train.history_habit import STYLE_AXES
from train.history_model import PublicStream

MAX_ROUNDS = 6
MATCHES = 64
WORKERS = 3
SEED = 17102000
CONFIGS = (('pass', 1.0), ('pass', 2.0), ('pass', 4.0), ('bomb', 2.0), ('bomb', 4.0))
BOMBS = ('Bomb', 'StraightFlush', 'JokerBomb')
ROUND_MAX, FULL_MIN = 0.75, 0.95


def feature(legal: list, axis: str) -> np.ndarray:
    """f(a) per legal action; matches train.history_habit.style_feature."""
    if axis == 'pass':
        return np.array([a.is_pass for a in legal], float)
    if axis == 'bomb':
        return np.array([str(a.type) in BOMBS for a in legal], float)
    if axis == 'lead_single':
        leading = not any(a.is_pass for a in legal)
        return np.array([leading and str(a.type) == 'Single' for a in legal], float)
    raise ValueError(f'unknown axis {axis!r}')


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def tilt(p: np.ndarray, f: np.ndarray, zb: float) -> np.ndarray:
    w = p * np.exp(zb * f)
    return w / w.sum()


def llr(p: np.ndarray, f: np.ndarray, b: float, a: int) -> float:
    """log pi_+(a) - log pi_-(a) for the habit pack at this decision."""
    return float(2 * b * f[a] - math.log((p * np.exp(b * f)).sum()) + math.log((p * np.exp(-b * f)).sum()))


def play_match(actor, axis: str, b: float, z: int, seed: int) -> list[dict]:
    """One match; a row per observer-team play decision with both posteriors."""
    engine = gd.Engine()
    engine.auto_pass = False
    state = gd.MatchState()
    engine.new_match(state, int(seed))
    rng = np.random.default_rng(np.random.SeedSequence([SEED, seed]))
    stream = PublicStream()
    listener = Listener(stream)
    rows, full = [], 0.0
    for rnd in range(MAX_ROUNDS):
        current, opp_events, own = 0.0, 0, []
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
            seat = int(state.to_move)
            arrays = tuple(x.copy() for x in stream.arrays())
            p = decision_probabilities(actor, state, arrays)
            if seat % 2 == 0:
                own.append(dict(round=rnd, seat=seat, opp_events=opp_events,
                                post_round=sigmoid(z * current),
                                post_full=sigmoid(z * full)))
                q = p
            else:
                f = feature(legal, axis)
                q = tilt(p, f, z * b)
            a = min(int(np.searchsorted(np.cumsum(q), rng.random(), side='right')), len(q) - 1)
            if seat % 2 == 1:
                step = llr(p, feature(legal, axis), b, a)
                current += step
                full += step
                opp_events += 1
            apply_and_observe(engine, state, legal[a], [listener])
        else:
            raise RuntimeError('round decision cap hit')
        for i, r in enumerate(own):
            r['position'] = i / max(1, len(own))
        rows.extend(own)
        engine.end_round(state)
        if state.winner >= 0:
            break
        engine.begin_round(state)
    return rows


_ACTOR = None


def init_worker(base: str) -> None:
    global _ACTOR
    torch.set_num_threads(1)
    _ACTOR = load_plain(Path(base))


def work(output: str, config: int, match: int) -> str:
    out = Path(output) / 'matches' / f'c{config}-m{match:03d}.json'
    if out.exists():
        return str(out)
    axis, b = CONFIGS[config]
    seed = int(np.random.default_rng(SEED).integers(1, 2**63 - 1, size=MATCHES)[match])
    z = 1 if match % 2 else -1
    start = time.time()
    rows = play_match(_ACTOR, axis, b, z, seed)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(config=config, match=match, z=z, seed=seed,
                                   seconds=time.time() - start, rows=rows)) + '\n')
    tmp.rename(out)
    return str(out)


def summarize(output: Path) -> dict:
    out = []
    for c, (axis, b) in enumerate(CONFIGS):
        recs = [json.loads(p.read_text()) for p in sorted((output / 'matches').glob(f'c{c}-m*.json'))]
        rows = [r for m in recs for r in m['rows']]
        if not rows:
            continue
        gate = [r for r in rows if r['round'] >= 1 and r['position'] < .5]
        by_round = {}
        for k in range(MAX_ROUNDS):
            rr = [r for r in rows if r['round'] == k]
            if rr:
                by_round[k] = dict(n=len(rr), round_view=float(np.mean([r['post_round'] for r in rr])),
                                   full_view=float(np.mean([r['post_full'] for r in rr])))
        first = [r for r in rows if r['round'] == 0]
        halves = {}
        for name, lo, hi in (('first_half', 0, .5), ('second_half', .5, 1.01)):
            rr = [r for r in rows if r['round'] >= 1 and lo <= r['position'] < hi]
            halves[name] = dict(round_view=float(np.mean([r['post_round'] for r in rr])),
                                full_view=float(np.mean([r['post_full'] for r in rr])))
        gr, gf = float(np.mean([r['post_round'] for r in gate])), float(np.mean([r['post_full'] for r in gate]))
        out.append(dict(axis=axis, strength=b, matches=len(recs), decisions=len(rows),
                        gate_decisions=len(gate), gate_round_view=gr, gate_full_view=gf,
                        qualifies=bool(gr < ROUND_MAX and gf > FULL_MIN),
                        round1_late_round_view=float(np.mean([r['post_round'] for r in first if r['position'] >= .5])) if first else None,
                        rounds_2plus=halves, by_round=by_round))
    chosen = {}
    for axis in sorted({r['axis'] for r in out}):
        ok = [r for r in out if r['axis'] == axis and r['qualifies']]
        chosen[axis] = max(ok, key=lambda r: r['strength'])['strength'] if ok else None
    return dict(rule=f'largest b with mean ROUND < {ROUND_MAX} and FULL > {FULL_MIN} '
                     'over rounds 2+, first half of observer decisions', chosen=chosen, configs=out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('run', 'analyze'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=WORKERS)
    p.add_argument('--matches', type=int, default=MATCHES)
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.stage == 'run':
        (args.output / 'matches').mkdir(parents=True, exist_ok=True)
        proto = args.output / 'protocol.json'
        if not proto.exists():
            import train.history_model as model
            proto.write_text(json.dumps(dict(
                base=str(BASE), base_sha=sha(BASE), source_sha=sha(__file__),
                model_source_sha=sha(model.__file__), seed=SEED, matches=MATCHES,
                max_rounds=MAX_ROUNDS, configs=[list(c) for c in CONFIGS],
                rule=dict(round_max=ROUND_MAX, full_min=FULL_MIN)), indent=2) + '\n')
        elif json.loads(proto.read_text())['source_sha'] != sha(__file__):
            raise ValueError('source changed since the protocol was written')
        jobs = [(c, m) for m in range(args.matches) for c in range(len(CONFIGS))]
        start = time.time()
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                                 initializer=init_worker, initargs=(str(BASE),)) as pool:
            futures = [pool.submit(work, str(args.output), *j) for j in jobs]
            for i, f in enumerate(as_completed(futures), 1):
                f.result()
                if i % 20 == 0 or i == len(jobs):
                    print(f'{i}/{len(jobs)} matches, {time.time() - start:.0f}s', flush=True)
        return
    s = summarize(args.output)
    (args.output / 'summary.json').write_text(json.dumps(s, indent=2) + '\n')
    for r in s['configs']:
        h = r['rounds_2plus']
        print(f"{r['axis']:>5} b={r['strength']:.0f}  gate: ROUND {r['gate_round_view']:.3f} FULL {r['gate_full_view']:.3f} "
              f"{'QUALIFIES' if r['qualifies'] else '-'}  | rounds2+ late: ROUND {h['second_half']['round_view']:.3f} "
              f"FULL {h['second_half']['full_view']:.3f} | round1 late ROUND {r['round1_late_round_view']:.3f}  ({r['matches']} matches)")
    print('chosen:', s['chosen'])


if __name__ == '__main__':
    main()
