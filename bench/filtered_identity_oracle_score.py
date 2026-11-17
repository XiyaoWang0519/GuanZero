"""Paired root-level inference for the frozen filtered identity audit."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from bench.filtered_identity_oracle import ROOT, STAGES, save_json


def interval(values: np.ndarray, bootstrap: np.ndarray) -> dict:
    draws = values[bootstrap].mean(1)
    return dict(mean=float(values.mean()), ci95=np.quantile(draws, [.025, .975]).tolist())


def summarize(root: Path, repetitions: int = 5000) -> dict:
    protocol = json.loads((root/'protocol.json').read_text())
    completion = json.loads((root/'complete.json').read_text())
    rows = [json.loads((root/f'root-{i:03d}'/'result.json').read_text()) for i in range(protocol['roots'])]
    assert len(rows) == completion['roots']
    assert all(r['frozen_weights_verified'] and r['observer_information_verified'] for r in rows)
    assert completion['actor_sha_after'] == protocol['identity']['actor_sha']
    assert completion['population_sha_after'] == protocol['identity']['population_sha']
    rng = np.random.default_rng(20261115731)
    selections = {s:np.asarray([i for i,r in enumerate(rows) if r['meta']['stage']==s]) for s in STAGES}
    selections['all'] = np.arange(len(rows))
    summaries = {}
    for name, selected in selections.items():
        if name == 'all':
            boot = np.concatenate([rng.choice(ids, size=(repetitions, len(ids))) for s,ids in selections.items() if s!='all'], axis=1)
        else:
            boot = rng.choice(selected, size=(repetitions, len(selected)))
        current = {}
        def stat(values):
            values = np.asarray(values)
            return dict(mean=float(values[selected].mean()),
                        ci95=np.quantile(values[boot].mean(1), [.025, .975]).tolist())
        for label in ('full_action', 'action_type'):
            part = {}
            for kind in ('native', 'conditional_world_mean', 'marginal'):
                keys = rows[0]['metrics'][label][kind]
                part[kind] = {k:stat([r['metrics'][label][kind][k] for r in rows]) for k in keys}
            part['convergence'] = {n:{k:stat([r['metrics'][label]['convergence'][n][k] for r in rows])
                for k in rows[0]['metrics'][label]['convergence'][n]}
                for n in rows[0]['metrics'][label]['convergence']}
            # Paired ratio interval keeps the same resampled roots in numerator
            # and denominator; it is not a ratio of unrelated CI endpoints.
            num = np.array([r['metrics'][label]['marginal']['cross_gain'] for r in rows])
            den = np.array([r['metrics'][label]['conditional_world_mean']['brier_gain'] for r in rows])
            part['retained_fraction'] = dict(mean=float(num[selected].mean()/den[selected].mean()),
                ci95=np.quantile(num[boot].mean(1)/den[boot].mean(1), [.025, .975]).tolist())
            current[label] = part
        current['controls'] = {key:{k:stat([r['controls'][key][k] for r in rows])
            for k in rows[0]['controls'][key]} for key in ('positive_full', 'positive_type')}
        current['roots'] = len(selected)
        summaries[name] = current
    return dict(protocol=protocol, completion=completion, summaries=summaries,
        checks=dict(frozen_weights=True, observer_information=True,
            duplicate_forward_max_abs=max(r['controls']['duplicate_forward_max_abs'] for r in rows),
            batch_vs_serial_max_tv=max(r['controls']['batch_vs_serial_max_tv'] for r in rows),
            duplicate_identity_gain_max=max(r['controls']['duplicate_identity_gain'] for r in rows)),
        totals=dict(roots=len(rows), forward_decisions=sum(r['forward_decisions'] for r in rows),
            source_attempts=sum(r['meta']['attempt']+1 for r in rows),
            public_events_range=[min(r['meta']['public_events'] for r in rows), max(r['meta']['public_events'] for r in rows)],
            native_forced_roots=sum(r['legal_candidates'][0]==1 for r in rows),
            hidden_forced_fraction=float(np.mean([np.mean(np.asarray(r['legal_candidates'][1:])==1) for r in rows])),
            root_seconds_sum=sum(r['seconds'] for r in rows)),
        root_results=rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = summarize(args.root)
    save_json(args.output or args.root/'summary.json', result)
    print(json.dumps({k:result[k] for k in ('checks', 'totals')}, indent=2))
    for stage, row in result['summaries'].items():
        print(json.dumps(dict(stage=stage, roots=row['roots'], metrics={k:dict(
            full=row[k]['conditional_world_mean']['brier_gain'],
            marginal=row[k]['marginal']['cross_gain'],
            heldout=row[k]['marginal']['heldout_gain']) for k in ('full_action','action_type')})))
