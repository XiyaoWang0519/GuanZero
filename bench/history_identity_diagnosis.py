"""Frozen CPU diagnosis: Q selection noise versus opponent-specific responses.

Reuses all six selected roots, old data as explicitly reused fitting data,
then fresh 1024-world estimates and sealed policies before fresh 2048 evaluation.
No new root selection, historical-input test, training, or production change.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from bench import history_value_curriculum as previous
from bench import history_value_tasks as tasks

BASE = previous.BASE
OLD = Path('.work/history-value-curriculum-2026-10-17')
SEED = 17101790
COUNTS = {'estimate': 1024, 'evaluation': 2048}
ROOTS = 6
WORKERS = 4
CHUNK = 8
BUDGET = 368640
LAMBDAS = (0., .25, .5, .75, 1.)


def save(path: Path, data: dict) -> None:
    BASE.save_json(path, data)


def seeds(stage: str, root: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([SEED, list(COUNTS).index(stage), root])).integers(
        1, 2**63-1, size=COUNTS[stage])


def sources() -> dict:
    return {**previous.sources(), __file__: BASE.sha(__file__)}


def old_q(root: int) -> np.ndarray:
    folder = OLD / 'selected' / f'case-{root:02d}'
    return np.concatenate([np.load(folder / stage / 'all.npz')['returns']
                           for stage in ('estimate-a', 'estimate-b', 'confirm')], axis=2)


def contrasts(q: np.ndarray, actions: list[int], reference: int) -> dict[str, np.ndarray]:
    """Exact decomposition: identity gain = matching gain + generic action change."""
    values = q[np.arange(len(q)), actions]
    mixture = q[:, actions].mean(1)
    baseline = q[:, reference]
    return {'gain': values-baseline, 'matching': values-mixture,
            'generic': mixture-baseline}


def policies(q: np.ndarray) -> dict:
    means = q[..., 0].mean(2)
    blind = means.mean(0)
    result = {f'shrink-{lam:g}': ((1-lam)*blind[None] + lam*means).argmax(1).tolist()
              for lam in LAMBDAS}
    result['prior'] = int(blind.argmax())
    return result


def seal_policies(root: int, fresh: np.ndarray) -> dict:
    old = old_q(root)
    legacy = json.loads((OLD/'selected'/f'case-{root:02d}'/'sealed-choices.json').read_text())['choices']
    pooled = np.concatenate([old, fresh], axis=2)
    final = policies(pooled)
    result = {'pooled2048': {'reference': final['prior'], 'actions': final['shrink-1']},
              'legacy512': {'reference': legacy['prior'], 'actions': legacy['identity']}}
    for lam in LAMBDAS:
        result[f'shrink-{lam:g}'] = {'reference': final['prior'], 'actions': final[f'shrink-{lam:g}']}
    for label, q in [('old1024', old), ('fresh1024', fresh)]:
        p = policies(q)
        result[label] = {'reference': p['prior'], 'actions': p['shrink-1']}
    for n in (64, 128, 256, 512):
        p = policies(fresh[:, :, :n])
        result[f'fresh{n}'] = {'reference': p['prior'], 'actions': p['shrink-1']}
    return result


def seed_validation() -> dict:
    used = set()
    for root in range(8):
        for stage in ('selection', 'evaluation'):
            used.update(map(int, np.load(BASE.PILOT/f'root-{root:02d}-{stage}.npz')['world_seeds']))
        for folder in (previous.old.PREVIOUS, Path('.work/history-decision-precision-2026-10-17')):
            for stage in ('estimate-a', 'estimate-b', 'evaluation'):
                used.update(map(int, np.load(folder/f'root-{root:02d}'/stage/'all.npz')['world_seeds']))
    used.update(map(int, np.random.default_rng(previous.SEED+1).integers(1, 2**63-1, size=64)))
    for stage in previous.STAGES:
        for root in range(32 if stage.startswith('screen') else 6):
            used.update(map(int, previous.seeds(stage, root)))
    n = len(used)
    for stage in COUNTS:
        for root in range(ROOTS):
            ss = seeds(stage, root); fresh = set(map(int, ss))
            if len(fresh) != len(ss) or used.intersection(fresh):
                raise ValueError('new seed collision')
            used.update(fresh)
    return dict(prior_or_reserved_unique=n, new_unique=len(used)-n, disjoint=True)


def check(output: Path, require_seal: bool = False) -> None:
    protocol = json.loads((output/'protocol.json').read_text())
    previous.check_choices(OLD)
    if protocol['sources'] != sources():
        raise ValueError('source identity changed')
    for name, sha in protocol['old_artifacts'].items():
        if BASE.sha(name) != sha:
            raise ValueError('old data changed')
    if require_seal:
        seal = json.loads((output/'estimation-complete.json').read_text())
        for name, sha in seal['choices'].items():
            if BASE.sha(output/name) != sha:
                raise ValueError('choice seal changed')


def prepare(output: Path) -> None:
    previous.check_choices(OLD)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = {}
    for name in ('protocol.json', 'selection.json', 'estimation-complete.json', 'confirm-summary.json', 'complete.json'):
        path = OLD/name; artifacts[str(path)] = BASE.sha(path)
    total_candidates = 0
    for root in range(ROOTS):
        old_folder = OLD/'selected'/f'case-{root:02d}'
        meta = json.loads((old_folder/'task.json').read_text())
        generated = OLD/'generated'/f"pool-{meta['index']:03d}"
        paths = [old_folder/'task.json', old_folder/'sealed-choices.json', generated/'public-inputs.npz',
                 generated/'simulator-state.bin']
        paths += [old_folder/stage/'all.npz' for stage in ('estimate-a', 'estimate-b', 'confirm')]
        for path in paths: artifacts[str(path)] = BASE.sha(path)
        total_candidates += len(meta['candidates'])
        (output/f'case-{root:02d}').mkdir()
    if total_candidates != 30 or 4*total_candidates*sum(COUNTS.values()) != BUDGET:
        raise ValueError('task or budget changed')
    protocol = dict(seed=SEED, roots=ROOTS, candidates=30, worlds=COUNTS, score_branch_budget=BUDGET,
        workers=WORKERS, torch_threads=1, engine_threads=1, chunk_worlds=CHUNK,
        primary='Round team win probability; PPO return secondary',
        old_data='Old256+256+512 reused for fitting and diagnosis only, not new independent evidence',
        order='All fresh1024 estimates, pooled2048 and all choices sealed, then independent2048 evaluation',
        policies='Legacy512; old1024; fresh64/128/256/512/1024; pooled2048; shrink lambda0/.25/.5/.75/1, all prespecified, no tuning on evaluation',
        primary_comparisons=['pooled2048 identity vs its blind prior', 'same identity vs same marginal type-independent action mixture'],
        decomposition='gain = matching + generic; legacy and refined policies independently scored',
        uncertainty='Fixed roots paired-world bootstrap 10000; simultaneous 97.5% CIs for two primary comparisons; other intervals descriptive',
        interaction='Type x action difference-in-differences; additive-null paired centered bootstrap diagnostic; not equal to value of identity',
        value_upper='95% simultaneous empirical-Bernstein upper bound on candidate-limited oracle identity value, relative to sealed blind reference; conservative, not a global game bound',
        practical_threshold=.02, no_root_filtering=True, no_training=True, cpu_only=True,
        sources=sources(), old_artifacts=artifacts)
    save(output/'protocol.json', protocol)
    save(output/'seed-validation.json', seed_validation())
    snap = output/'source-snapshot'; snap.mkdir()
    for n, path in enumerate(protocol['sources']):
        if path.endswith('.py'): shutil.copyfile(path, snap/f'{n:02d}-{Path(path).name}')
    check(output)


def worker(path: str, root: int, stage: str) -> dict:
    actor, opponents = previous.old._MODELS
    output = Path(path); folder = output/f'case-{root:02d}'/stage; folder.mkdir()
    meta = json.loads((OLD/'selected'/f'case-{root:02d}'/'task.json').read_text())
    state, arrays, candidates = tasks.load_task(OLD/'generated'/f"pool-{meta['index']:03d}")
    ss = seeds(stage, root); parts = []; start = time.monotonic()
    with (folder/'worker.log').open('x') as log, redirect_stdout(log):
        for offset in range(0, len(ss), CHUNK):
            subset = ss[offset:offset+CHUNK]
            q = tasks.rollout(actor, opponents, state, arrays, candidates, subset)
            np.savez_compressed(folder/f'chunk-{offset//CHUNK:03d}.npz', returns=q, world_seeds=subset, candidates=candidates)
            parts.append(q)
            print(json.dumps(dict(worlds_done=offset+len(subset), total=len(ss))), flush=True)
    q = np.concatenate(parts, axis=2)
    np.savez_compressed(folder/'all.npz', returns=q, world_seeds=ss, candidates=candidates)
    save(folder/'complete.json', dict(rounds=int(np.prod(q.shape[:3])), seconds=time.monotonic()-start))
    return dict(root=root, stage=stage, complete=True)


def dispatch(output: Path, stage: str) -> None:
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context('spawn'),
                             initializer=previous.old.init_worker) as pool:
        futures = [pool.submit(worker, str(output), root, stage) for root in range(ROOTS)]
        for f in as_completed(futures): print(json.dumps(f.result()), flush=True)
    save(output/f'{stage}-timing.json', dict(wall_seconds=time.monotonic()-started))


def estimate(output: Path) -> None:
    prepare(output); dispatch(output, 'estimate')
    seals = {}; rows = []
    for root in range(ROOTS):
        folder = output/f'case-{root:02d}'
        q = np.load(folder/'estimate/all.npz')['returns']
        old = old_q(root)
        choices = seal_policies(root, q)
        file = folder/'sealed-choices.json'; save(file, choices)
        seals[str(file.relative_to(output))] = BASE.sha(file)
        row = BASE.stability_metrics(old[..., 0].mean(2), q[..., 0].mean(2))
        save(folder/'stability.json', row); rows.append(row)
    save(output/'estimation-complete.json', dict(choices=seals, stability=rows))
    check(output, True)


def additive(q: np.ndarray) -> np.ndarray:
    return q.mean(0, keepdims=True)+q.mean(1, keepdims=True)-q.mean()


def oracle_value(q: np.ndarray) -> float:
    return float(q.max(1).mean()-q.mean(0).max())


def value_bound(q: np.ndarray, reference: int, comparisons: int, delta: float = .05) -> dict:
    """Simultaneous bounded paired-difference UCB; range of differences is two."""
    win = q[..., 0]; n = win.shape[2]
    difference = win-win[:, reference:reference+1]
    variance = difference.var(2, ddof=1)
    log = np.log(2*comparisons/delta)
    radius = np.sqrt(2*variance*log/n)+14*log/(3*(n-1))
    upper = np.minimum(1, difference.mean(2)+radius)
    upper[:, reference] = 0
    return dict(upper=float(upper.max(1).mean()), mean_gaps=difference.mean(2).tolist(),
                simultaneous_upper_gaps=upper.tolist(), delta=delta, comparisons=comparisons)


def summarize(output: Path) -> dict:
    root_gains = []; root_metrics = []; tables = []
    for root in range(ROOTS):
        folder = output/f'case-{root:02d}'
        choices = json.loads((folder/'sealed-choices.json').read_text())
        q = np.load(folder/'evaluation/all.npz')['returns']
        table = q[..., 0].mean(2); tables.append(table)
        gains = {name: contrasts(q, p['actions'], p['reference']) for name, p in choices.items()}
        root_gains.append(gains)
        # Rank reversal using point estimates is descriptive, not a test.
        mean = table.mean(0); interaction = table-additive(table)
        old_folder = OLD/'selected'/f'case-{root:02d}'
        legacy = choices['legacy512']; legacy_rows = []
        train = np.concatenate([np.load(old_folder/stage/'all.npz')['returns'] for stage in ('estimate-a', 'estimate-b')], 2)
        for z, action in enumerate(legacy['actions']):
            if action == legacy['reference']: continue
            diff = q[z, action, :, 0]-q[z, legacy['reference'], :, 0]
            estimate_gap = (train[z, action, :, 0]-train[z, legacy['reference'], :, 0]).mean()
            legacy_rows.append(dict(type=z, action=action, reference=legacy['reference'],
                old_estimated_gap=float(estimate_gap), new_gap=float(diff.mean()),
                paired_se=float(diff.std(ddof=1)/np.sqrt(len(diff)))))
        root_metrics.append(dict(root=root, win_q=table.tolist(), blind_best=int(mean.argmax()),
            type_best=table.argmax(1).tolist(), type_strength_span=float(np.ptp(table.mean(1))),
            interaction_rms=float(np.sqrt((interaction**2).mean())), plugin_identity_value=oracle_value(table),
            legacy_switches=legacy_rows))
    names = list(root_gains[0]); stats = {}
    draws = {name: {k: np.zeros((10000, 2)) for k in ('gain', 'matching', 'generic')} for name in names}
    rng = np.random.default_rng(SEED+10)
    null_values = np.zeros(10000); null_interactions = np.zeros(10000)
    comparisons = sum(4*(len(table[0])-1) for table in tables)
    upper_bounds = []
    for root, gains in enumerate(root_gains):
        q = np.load(output/f'case-{root:02d}'/'evaluation/all.npz')['returns']
        n = q.shape[2]
        # One set of weights per root preserves every action/type/policy pairing.
        weights = rng.multinomial(n, np.full(n, 1/n), size=10000)/n
        for name in names:
            for metric in draws[name]: draws[name][metric] += (weights @ gains[name][metric].mean(0))/ROOTS
        table = tables[root]
        # Centered paired resampling retains heteroscedasticity and covariance.
        boot = (weights @ q[..., 0].reshape(-1, n).T).reshape(-1, 4, q.shape[1])
        null = boot-table[None]+additive(table)[None]
        null_values += (null.max(2).mean(1)-null.mean(1).max(1))/ROOTS
        null_residual = null-null.mean(1, keepdims=True)-null.mean(2, keepdims=True)+null.mean((1, 2), keepdims=True)
        null_interactions += (null_residual**2).mean((1, 2))/ROOTS
        reference = json.loads((output/f'case-{root:02d}'/'sealed-choices.json').read_text())['pooled2048']['reference']
        upper_bounds.append(value_bound(q, reference, comparisons))
    for name in names:
        stats[name] = {}
        for metric in draws[name]:
            means = np.asarray([g[name][metric].mean((0, 1)) for g in root_gains])
            rr = np.random.default_rng(SEED+12).integers(ROOTS, size=(10000, ROOTS))
            stats[name][metric] = dict(mean=means.mean(0).tolist(), by_root=means.tolist(),
                world_ci95=np.quantile(draws[name][metric], [.025, .975], axis=0).T.tolist(),
                world_ci975=np.quantile(draws[name][metric], [.0125, .9875], axis=0).T.tolist(),
                root_ci95=np.quantile(means[rr].mean(1), [.025, .975], axis=0).T.tolist())
    observed_value = float(np.mean([r['plugin_identity_value'] for r in root_metrics]))
    observed_interaction = float(np.mean([r['interaction_rms']**2 for r in root_metrics]))
    return dict(policies=stats, roots=root_metrics,
        primary_positive=bool(stats['pooled2048']['gain']['world_ci975'][0][0]>0 and stats['pooled2048']['matching']['world_ci975'][0][0]>0),
        additive_null=dict(plugin_value=observed_value, null_value_mean=float(null_values.mean()),
            value_p=float((1+np.count_nonzero(null_values>=observed_value))/(1+len(null_values))),
            interaction_ms=observed_interaction, null_interaction_mean=float(null_interactions.mean()),
            interaction_p=float((1+np.count_nonzero(null_interactions>=observed_interaction))/(1+len(null_interactions))),
            scope='Approximate paired centered-bootstrap additive-null diagnostic; nonadditivity need not change the best action; plug-in identity value biased upward'),
        oracle_candidate_upper95=float(np.mean([b['upper'] for b in upper_bounds])),
        upper_bounds=upper_bounds, practical_threshold=.02,
        scope='Six previously selected roots, same finite candidate sets and frozen continuations; final2048 worlds independent of all choices; no natural-match/general opponent claim')


def evaluate(output: Path) -> None:
    check(output, True); dispatch(output, 'evaluation')
    result = summarize(output); save(output/'evaluation-summary.json', result)
    check(output, True)
    total = sum(json.loads(f.read_text())['rounds'] for f in output.glob('case-*/*/complete.json'))
    if total != BUDGET: raise ValueError('budget mismatch')
    save(output/'complete.json', dict(complete=True, score_branches=total, cap=BUDGET, cpu_only=True, no_training=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('estimate', 'evaluate'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(1)
    if args.stage == 'estimate': estimate(args.output)
    else: evaluate(args.output)


if __name__ == '__main__': main()
