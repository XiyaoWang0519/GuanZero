"""Whole-round value of knowing a planted opponent habit (local CPU, frozen models).

Follow-up to history_habit_value.py, which measured one changed decision. Here
decisions of the observer team (seats 0 and 2) with two or more legal actions
run a small rollout search once some seat holds at most TRIGGER cards (the
announced count; before that the team plays the blueprint's greedy choice);
opponents (seats 1 and 3) are the Sept 30
habit pack, u2623 with pi_z ∝ pi exp(z b f), one z per deal.

Two planners share one search at each node. Each evaluates the blueprint's
top-C candidates on the same 2W uniform hidden worlds with rollouts under both
z: the oracle planner scores candidates under the true z only, the type-blind
planner averages over both z (the 50/50 prior). Both keep the blueprint choice
unless another candidate's mean round-win estimate beats it by MARGIN. Every
continuation, live and in rollouts, samples at temperature one.

While the two planners agree, the deal has one trajectory and the difference
is exactly zero. At the first disagreement the deal forks; each fork keeps the
same live RNG state and continues under its own planner only. The reported
quantity is oracle minus blind per deal, averaged over deals.

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

from bench.history_habit_value import BASE, load_plain, load_styled, sha
from bench.history_value_tasks import Listener, decision_probabilities, stream_from
from bench.qknorm_match_ablation import MatchSampler, inverse_cdf
from eval.history_policy import apply_and_observe
from train.history_inference import BatchedHistoryCache, ForkedHistoryCache
from train.history_model import DecisionInputs, PublicStream, StreamBatch

TRIGGER = 10         # search once min public hand count <= TRIGGER
CANDIDATES = 4
WORLDS = 64          # 2W in the module docstring: worlds per search, both z on each
MARGIN = 0.02        # round-win probability a candidate must gain over the blueprint choice
WORKERS = 10
CONFIGS = (('pass', 4.0), ('lead_single', 1.0))


def team_rollout(actor: Any, opponents: dict, state: Any, arrays: tuple,
                 candidates: np.ndarray, seeds: np.ndarray, max_steps: int = 512) -> np.ndarray:
    """[type, candidate, world, (team round win, PPO return)] from the mover's seat.

    Generalises history_value_tasks.rollout to either seat of the observer team:
    seats of the mover's parity play ``actor``, the other two the type's model.
    """
    seat = int(state.to_move)
    team = seat % 2
    engine = gd.Engine()
    engine.auto_pass = False
    expected_obs = np.asarray(state.observation(seat))
    root_legal = engine.legal_actions(state)
    expected_cand = np.asarray([engine.encode_action(a, state, seat) for a in root_legal])
    worlds = [state.determinize_uniform(seat, int(s)) for s in seeds]
    root_stream = stream_from(arrays)
    branches, streams, labels, sampling = [], [], [], []
    for z in range(len(opponents)):
        for candidate in candidates:
            for w, s in zip(worlds, seeds):
                branch = gd.MatchState.deserialize(w.serialize())
                if not np.array_equal(branch.observation(seat), expected_obs):
                    raise ValueError('private world altered the mover observation')
                legal = engine.legal_actions(branch)
                if not np.array_equal(np.asarray([engine.encode_action(a, branch, seat) for a in legal]),
                                      expected_cand):
                    raise ValueError('private world altered the legal candidates')
                stream = copy.deepcopy(root_stream)
                apply_and_observe(engine, branch, legal[int(candidate)], [Listener(stream)])
                branches.append(branch)
                streams.append(stream)
                labels.append(z)
                sampling.append(int(s))
    n = len(branches)
    sampler = MatchSampler(sampling)
    values = np.full((n, 2), np.nan)
    policies = {0: actor, **{i + 1: a for i, a in enumerate(opponents.values())}}
    groups = {0: list(range(n)), **{i + 1: [r for r, z in enumerate(labels) if z == i]
                                    for i in range(len(opponents))}}
    mappings = {k: {r: i for i, r in enumerate(rows)} for k, rows in groups.items()}
    with torch.inference_mode():
        caches = {}
        for k, a in policies.items():
            root = BatchedHistoryCache(a)
            root.prefill([(0, 0)], [root_stream])
            caches[k] = ForkedHistoryCache(root, (0, 0), [streams[r] for r in groups[k]])
        active = list(range(n))
        for _ in range(max_steps):
            pending = {k: [] for k in policies}
            legal_sets = {}
            for r in active:
                branch = branches[r]
                if branch.phase == gd.Phase.RoundEnd:
                    result = engine.end_round(branch)
                    values[r] = [int(result.winning_team) == team, float(result.seat_return[seat])]
                    continue
                legal = engine.legal_actions(branch)
                if len(legal) == 1:
                    apply_and_observe(engine, branch, legal[0], [Listener(streams[r])])
                    continue
                k = 0 if int(branch.to_move) % 2 == team else labels[r] + 1
                pending[k].append(r)
                legal_sets[r] = legal
            for k, rows in pending.items():
                if not rows:
                    continue
                seats = np.asarray([branches[r].to_move for r in rows])
                obs = np.stack([branches[r].observation(int(s)) for r, s in zip(rows, seats)])
                cand, offsets = [], [0]
                for r, s in zip(rows, seats):
                    cand.extend(engine.encode_action(a, branches[r], int(s)) for a in legal_sets[r])
                    offsets.append(len(cand))
                metadata, encoded = caches[k].encode([mappings[k][r] for r in rows])
                inputs = DecisionInputs(metadata, torch.arange(len(rows)), metadata.lengths,
                                        torch.tensor(obs), torch.tensor(seats),
                                        torch.tensor(np.asarray(cand)), torch.tensor(offsets),
                                        one_decision_per_stream=True)
                table = policies[k]._log_prob_table(inputs, encoded,
                                                    max(len(legal_sets[r]) for r in rows))
                choice, _ = inverse_cdf(table, torch.tensor(sampler.draw(rows, seats)))
                for r, c in zip(rows, choice.tolist()):
                    apply_and_observe(engine, branches[r], legal_sets[r][int(c)], [Listener(streams[r])])
            active = [r for r in active if not np.isfinite(values[r]).all()]
            if not active:
                return values.reshape(len(opponents), len(candidates), len(seeds), 2)
    raise RuntimeError('unfinished rollout branch; no partial result accepted')


def choose(q: np.ndarray, weights: np.ndarray, margin: float) -> int:
    """Candidate index (0 = blueprint choice) from q [type, candidate, world, 2].

    Score is the weighted mean round win; return breaks exact ties. A candidate
    replaces the blueprint choice only when its win estimate is ``margin`` higher.
    """
    win = np.tensordot(weights, q[..., 0].mean(2), 1)
    ret = np.tensordot(weights, q[..., 1].mean(2), 1)
    best = max(range(len(win)), key=lambda i: (win[i], ret[i]))
    return best if win[best] >= win[0] + margin else 0


class Trajectory:
    def __init__(self, state: Any, stream: PublicStream, live: np.random.Generator,
                 worlds: np.random.Generator, planner: str) -> None:
        self.state, self.stream, self.live, self.worlds, self.planner = state, stream, live, worlds, planner
        self.searches = 0

    def fork(self, planner: str) -> 'Trajectory':
        t = Trajectory(gd.MatchState.deserialize(self.state.serialize()), copy.deepcopy(self.stream),
                       copy.deepcopy(self.live), copy.deepcopy(self.worlds), planner)
        t.searches = self.searches
        return t


def play_deal(actor: Any, styled: dict, deal_seed: int, z: int, candidates: int, worlds: int,
              margin: float, seed: int, trigger: int = TRIGGER) -> dict:
    """One round from a fresh match; z indexes ``styled`` (0: z=-1, 1: z=+1)."""
    engine = gd.Engine()
    engine.auto_pass = False
    state = gd.MatchState()
    engine.new_match(state, int(deal_seed))
    ss = np.random.SeedSequence([seed, deal_seed])
    live_ss, world_ss = ss.spawn(2)
    todo = [Trajectory(state, PublicStream(), np.random.default_rng(live_ss),
                       np.random.default_rng(world_ss), 'both')]
    out = dict(deal_seed=int(deal_seed), z=int(z), forked=False, disagreements=0, searches=0,
               fork_step=None)
    true_weight = np.eye(2)[z]
    blind_weight = np.full(2, .5)
    while todo:
        t = todo.pop()
        for step in range(1024):
            s = t.state
            if s.phase == gd.Phase.RoundEnd:
                result = engine.end_round(s)
                team = dict(win=int(result.winning_team) == 0, ret=float(result.seat_return[0]),
                            searches=t.searches)
                for name in (('oracle', 'blind') if t.planner == 'both' else (t.planner,)):
                    out[name] = team
                break
            if s.phase != gd.Phase.Play:
                raise ValueError('unexpected non-play phase')
            legal = engine.legal_actions(s)
            seat = int(s.to_move)
            if len(legal) == 1:
                apply_and_observe(engine, s, legal[0], [Listener(t.stream)])
                continue
            arrays = tuple(x.copy() for x in t.stream.arrays())
            if seat % 2 == 1:
                p = decision_probabilities(styled[z], s, arrays)
                choice = min(int(np.searchsorted(np.cumsum(p), t.live.random(), side='right')), len(p) - 1)
                apply_and_observe(engine, s, legal[choice], [Listener(t.stream)])
                continue
            p = decision_probabilities(actor, s, arrays)
            if min(len(s.hand(i)) for i in range(4)) > trigger:
                apply_and_observe(engine, s, legal[int(np.argmax(p))], [Listener(t.stream)])
                continue
            cand = np.argsort(-p, kind='stable')[:candidates]
            seeds = t.worlds.integers(1, 2**63 - 1, size=worlds)
            types = {0: styled[0], 1: styled[1]} if t.planner != 'oracle' else {0: styled[z]}
            q = team_rollout(actor, types, s, arrays, cand, seeds)
            t.searches += 1
            if t.planner == 'oracle':
                picks = {'oracle': choose(q, np.ones(1), margin)}
            else:
                picks = {'blind': choose(q, blind_weight, margin)}
                if t.planner == 'both':
                    picks['oracle'] = choose(q, true_weight, margin)
            if t.planner == 'both':
                out['searches'] += 1
                if picks['oracle'] != picks['blind']:
                    out['disagreements'] += 1
                    out['forked'] = True
                    out['fork_step'] = step
                    other = t.fork('oracle')
                    apply_and_observe(engine, other.state, legal[int(cand[picks['oracle']])],
                                      [Listener(other.stream)])
                    todo.append(other)
                    t.planner = 'blind'
                    apply_and_observe(engine, s, legal[int(cand[picks['blind']])], [Listener(t.stream)])
                    continue
            pick = picks[t.planner if t.planner != 'both' else 'blind']
            apply_and_observe(engine, s, legal[int(cand[pick])], [Listener(t.stream)])
        else:
            raise RuntimeError('live round step cap hit')
    return out


_MODELS: dict = {}


def init_worker(base: str) -> None:
    torch.set_num_threads(1)
    _MODELS['self'] = load_plain(Path(base))
    for axis, b in CONFIGS:
        _MODELS[(axis, b)] = load_styled(Path(base), axis, b)


def work(output: str, config: int, index: int, deal_seed: int, z: int, worlds: int,
         seed: int, trigger: int) -> str:
    out = Path(output) / 'deals' / f'c{config}-d{index:04d}.json'
    if out.exists():
        return str(out)
    start = time.time()
    record = play_deal(_MODELS['self'], _MODELS[CONFIGS[config]], deal_seed, z, CANDIDATES,
                       worlds, MARGIN, seed, trigger)
    record.update(config=config, index=index, seconds=time.time() - start)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(record) + '\n')
    tmp.rename(out)
    return str(out)


def deal_plan(deals: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    seeds = rng.integers(1, 2**63 - 1, size=deals)
    z = np.tile([0, 1], (deals + 1) // 2)[:deals]   # balanced z, same deals for both configs
    return seeds, z


def bootstrap(x: np.ndarray, n: int = 10000, seed: int = 0) -> list[float]:
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    return np.percentile(x[idx].mean(1), [2.5, 97.5]).tolist()


def analyze(output: Path) -> dict:
    protocol = json.loads((output / 'protocol.json').read_text())
    rows = []
    for c, (axis, b) in enumerate(CONFIGS):
        recs = [json.loads(p.read_text()) for p in sorted((output / 'deals').glob(f'c{c}-d*.json'))]
        recs = [r for r in recs if r['index'] < protocol['deals']]
        if not recs:
            continue
        win = np.array([r['oracle']['win'] - r['blind']['win'] for r in recs], float) * 100
        ret = np.array([r['oracle']['ret'] - r['blind']['ret'] for r in recs], float)
        rows.append(dict(
            axis=axis, strength=b, deals=len(recs),
            oracle_win=float(np.mean([r['oracle']['win'] for r in recs]) * 100),
            blind_win=float(np.mean([r['blind']['win'] for r in recs]) * 100),
            win_diff=float(win.mean()), win_diff_ci=bootstrap(win, seed=1),
            ret_diff=float(ret.mean()), ret_diff_ci=bootstrap(ret, seed=2),
            forked=int(sum(r['forked'] for r in recs)),
            searches=int(sum(r['searches'] for r in recs)),
            disagreement_rate=float(sum(r['disagreements'] for r in recs) / max(1, sum(r['searches'] for r in recs))),
            by_z={str(z): float(win[[r['z'] == z for r in recs]].mean()) for z in (0, 1)},
            seconds=float(sum(r['seconds'] for r in recs))))
    return dict(protocol=protocol, units='win in percentage points; ret is PPO seat_return', configs=rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('plan', 'run', 'analyze'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deals', type=int, default=0, help='deals per config (plan stage)')
    p.add_argument('--worlds', type=int, default=WORLDS)
    p.add_argument('--seed', type=int, default=0, help='deal/live/world seed (plan stage)')
    p.add_argument('--trigger', type=int, default=TRIGGER)
    p.add_argument('--workers', type=int, default=WORKERS)
    p.add_argument('--limit', type=int, default=0, help='first N deals per config only (pilot)')
    p.add_argument('--start', type=int, default=0, help='first deal index (split one plan across machines)')
    args = p.parse_args()
    torch.set_num_threads(1)
    if args.stage == 'plan':
        if args.deals < 1 or args.seed < 1:
            p.error('--deals and --seed are required')
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'deals').mkdir()
        import train.history_model as model
        protocol = dict(
            base=str(BASE), base_sha=sha(BASE), source_sha=sha(__file__),
            value_source_sha=sha(Path(__file__).with_name('history_habit_value.py')),
            model_source_sha=sha(model.__file__), engine_sha=sha(gd._gd_core.__file__),
            seed=args.seed, trigger=args.trigger, deals=args.deals, configs=[list(c) for c in CONFIGS],
            candidates=CANDIDATES, worlds=args.worlds, margin=MARGIN,
            primary='oracle minus type-blind round team win (pp), deal bootstrap 95% CI',
            boundary=f'Search only once some seat holds <= {args.trigger} cards. Both planners know the habit family and strength; only oracle knows z. '
                     'Single fresh round per deal, level 2; uniform hidden worlds; '
                     'no history reading; no training; no budget extension')
        (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        print(json.dumps(protocol, indent=2))
        return
    protocol = json.loads((args.output / 'protocol.json').read_text())
    if protocol['source_sha'] != sha(__file__) or protocol['base_sha'] != sha(BASE):
        raise ValueError('source or checkpoint changed since plan')
    if args.stage == 'run':
        seeds, zs = deal_plan(protocol['deals'], protocol['seed'])
        count = args.limit or protocol['deals']
        # Interleave configs so a partial run covers both.
        jobs = [(c, i, int(seeds[i]), int(zs[i])) for i in range(args.start, count) for c in range(len(CONFIGS))]
        start = time.time()
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'),
                                 initializer=init_worker, initargs=(str(BASE),)) as pool:
            futures = [pool.submit(work, str(args.output), *j, protocol['worlds'], protocol['seed'],
                                   protocol['trigger']) for j in jobs]
            for i, f in enumerate(as_completed(futures), 1):
                f.result()
                if i % 20 == 0 or i == len(jobs):
                    print(f'{i}/{len(jobs)} deals, {time.time() - start:.0f}s', flush=True)
        return
    summary = analyze(args.output)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for r in summary['configs']:
        print(f"{r['axis']:>11} b={r['strength']:.0f}  deals {r['deals']}  oracle {r['oracle_win']:.2f}%  "
              f"blind {r['blind_win']:.2f}%  diff {r['win_diff']:+.2f} pp "
              f"[{r['win_diff_ci'][0]:+.2f},{r['win_diff_ci'][1]:+.2f}]  forked {r['forked']}  "
              f"disagree {100 * r['disagreement_rate']:.1f}% of {r['searches']} searches")


if __name__ == '__main__':
    main()
