"""Prepare, execute on an owned pod, and evaluate the T7 connection experiment.

prepare: local source/manifest/evaluation freeze only; no provider operations.
workload: three arms on a pre-provisioned Linux pod, under its existing guard.
evaluate: local CPU inference on downloaded fixed endpoints; never trains.
No subcommand creates a pod or reads provider credentials.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from infra.history_artifacts import ROOT, engine_digest, pack_source, sha256, source_identity
from infra.history_pilot import write_json

SEEDS = (2026092801, 2026092802, 2026092803)
MODES = {'A': 'none', 'B': 'auxiliary', 'C': 'explicit'}
TESTS = [f'tests/test_history_{name}.py' for name in
         ('model', 'rollout', 'ppo', 'population', 'inference', 'eval', 'response',
          'devices', 'response_experiment')]


def prepare(root: Path, learner_device: str = 'cuda', rollout_device: str | None = None) -> dict:
    """Freeze three independent, bounded run kits; approval remains pending."""
    from eval.duplicate import generate_deals
    from train.history_ppo import HistoryPPOConfig
    from train.history_response import RESPONSE_SCHEMA

    if learner_device not in ('cpu', 'cuda') or rollout_device not in (None, 'cpu', 'cuda'):
        raise ValueError('explicit cpu/cuda placement required')
    if root.exists():
        raise ValueError('refusing to overwrite an existing experiment')
    # Reuse only the evaluator identities, never old training data or weights.
    old = json.loads((ROOT / '.work/runpod-history-batch-2026-09-26/evaluation/freeze.json').read_text())
    for baseline in old['baselines']:
        if sha256(Path(baseline['path'])) != baseline['sha256']:
            raise ValueError('evaluation baseline digest mismatch')
    root.mkdir(parents=True)
    config = HistoryPPOConfig(width=64, layers=2, heads=4, num_envs=32, num_threads=2,
                              steps_per_update=64, torch_threads=2, epochs=2,
                              minibatch_matches=2, causal_sdpa=True, rollout_kv_cache=True,
                              snapshot_probability=0.5, population_recent=4,
                              updates=1000000, checkpoint_updates=50, response_coef=0.1,
                              rollout_device=rollout_device)
    dev_seed = 2026092851
    deals = generate_deals(256, dev_seed)
    development = dict(seed=dev_seed, policy_seed=2026092852,
                       deals=[{k:getattr(d,k) for k in
                               ('hands','level','team_levels','owner','leader')} for d in deals],
                       match_seeds=[2026092900+i for i in range(64)])
    campaign = dict(status='prepared; paid execution requires confirmed campaign budget',
                    seeds=list(SEEDS), arms=MODES, primary_comparison='C minus B',
                    secondary_comparisons=['B minus A', 'C minus A'],
                    total_cap_usd=6.0, per_allocation_cap_usd=1.75,
                    primary_budget='4800 trainer seconds per arm; three arms on the same pod',
                    endpoint='final.pt at the fixed time cap; never select by scores',
                    decision='report all paired seed effects and uncertainty; no automatic promotion',
                    final_test='existing sealed final test remains unopened', kits=[])
    for seed in SEEDS:
        kit = root / f'seed-{seed}'
        payload, evaluation = kit/'payload', kit/'evaluation'
        for directory in (payload, evaluation, kit/'results', kit/'local'):
            directory.mkdir(parents=True)
        write_json(evaluation/'development.json', development)
        freeze = dict(engine_digest=engine_digest(), baselines=old['baselines'],
                       development=dict(file='development.json',sha256=sha256(evaluation/'development.json')),
                       final_test=old['final_test'], selection=campaign['endpoint'])
        write_json(evaluation/'freeze.json', freeze)
        shutil.copy2(ROOT/'infra/history_setup.sh', payload/'setup.sh')
        (payload/'run.sh').write_text(
            '#!/usr/bin/env bash\nset -euo pipefail\ncd /workspace/GuanZero\n'
            ': "${POD_DEADLINE_EPOCH:?owned provider deadline required}"\n'
            'export PYTHONPATH="$PWD/python:$PWD/oracle:$PWD"\n'
            'export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NVIDIA_TF32_OVERRIDE=0\n'
            'exec python -m infra.watchdog --budget-usd 1.75 --hourly-rate-usd 0.90 '
            '--max-hours 1.75 --grace-seconds 30 --log-file /workspace/results/watchdog.jsonl -- '
            'python -m infra.history_response_experiment workload '
            '--manifest /workspace/payload/run-manifest.json --output /workspace/results\n')
        source = pack_source(kit/'source.tar.gz')
        base = asdict(config)
        base['seed'] = seed
        manifest = dict(id=f'history-response-{seed}',status='prepared; approval pending',
                         prepared_at=datetime.now(timezone.utc).isoformat(),
                         source=source,engine_digest=engine_digest(),config=base,
                         arms=MODES,training_seconds=4800,training_device=learner_device,
                         prediction=dict(schema=RESPONSE_SCHEMA,classes=23,
                             bridge='detached probabilities minus uniform, zero-initialized linear bridge',
                             provenance='only actually executed own-lineage actions and public responses',
                             target_metadata_in_actor_inputs=False),
                         evaluation=dict(freeze_sha256=sha256(evaluation/'freeze.json'),
                                         endpoint='final.pt',deals=256,full_match_pairs=64),
                         compute=dict(proposed_gpu='NVIDIA RTX PRO 4500 Blackwell',cloud='SECURE',
                                      minimum_usable_cpus=16,precision='FP32; TF32 disabled',
                                      note='CPU game engine; learner and rollout placement frozen from same-host measurements'),
                         lifecycle=dict(proposed_budget_usd=1.75,max_hours_from_create=1.9,
                                        max_hourly_usd=0.90,campaign_cap_usd=6.0,
                                        artifact_destination=str(kit/'local/results')),
                         payload_files={p.name:sha256(p) for p in payload.iterdir()})
        write_json(kit/'run-manifest.json', manifest)
        shutil.copy2(kit/'run-manifest.json', payload/'run-manifest.json')
        campaign['kits'].append(str(kit))
    write_json(root/'campaign.json', campaign)
    return campaign


def training_device(manifest: dict) -> str:
    """Use the explicit measured placement, with no silent device fallback."""
    import torch

    device = manifest.get('training_device')
    if device not in ('cpu', 'cuda'):
        raise ValueError('this experiment requires explicit cpu/cuda training')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA training requested but CUDA is unavailable')
    return device


def arm(manifest: dict, name: str, output: Path) -> None:
    import torch
    from train.history_model import save_history_checkpoint
    from train.history_ppo import HistoryPPOConfig, HistoryTrainer
    from infra.history_pilot import health

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    values = dict(manifest['config'], response_mode=manifest['arms'][name])
    device = training_device(manifest)
    trainer = HistoryTrainer(HistoryPPOConfig(**values), output/'train', device=device)
    if any(p.device.type != device for module in (trainer.actor, trainer.critic)
           for p in module.parameters()):
        raise RuntimeError('actor or critic did not move to the selected device')
    write_json(output/'device.json', dict(device=str(trainer.device),
               actor_device=str(next(trainer.actor.parameters()).device),
               critic_device=str(next(trainer.critic.parameters()).device),
               sampler_device=str(trainer.generator.device),
               rollout_device=str(trainer.rollout_device),
               gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None, torch=torch.__version__,
               cuda=torch.version.cuda, tf32=False))
    write_json(output/'resolved-config.json', values)
    def snapshot(path):
        payload = trainer.payload()
        payload['optimizer'] = {}
        payload.pop('population', None)
        save_history_checkpoint(path, payload)
    snapshot(output/'initial.pt')
    interrupted = []
    signal.signal(signal.SIGTERM, lambda *_: interrupted.append('SIGTERM'))
    signal.signal(signal.SIGINT, lambda *_: interrupted.append('SIGINT'))
    start = time.monotonic()
    seconds = manifest['training_seconds']
    if time.time() + seconds > float(os.environ['POD_DEADLINE_EPOCH']) - 300:
        raise RuntimeError('insufficient remaining allocation for the predeclared training budget')
    last_save = start
    recent = []
    reason = 'time_limit'
    while time.monotonic()-start < seconds and not interrupted:
        tick = time.monotonic()
        line = trainer.update()
        recent.append(line)
        recent = recent[-120:]
        check = health(recent)
        check.update(epoch=time.time(),elapsed_seconds=time.monotonic()-start,
                     update_seconds=time.monotonic()-tick,device=str(trainer.device),
                     gpu_allocated_bytes=torch.cuda.memory_allocated() if device=='cuda' else 0,
                     gpu_reserved_bytes=torch.cuda.memory_reserved() if device=='cuda' else 0)
        write_json(output/'health.json', check)
        if not check['healthy'] or check['update_seconds'] > 150:
            trainer.save()
            raise RuntimeError(f'arm {name} failed the health/throughput gate')
        update = trainer.progress['updates']
        if update % 50 == 0 or time.monotonic()-last_save >= 300:
            trainer.save()
            snapshot(output/f'update-{update:06d}.pt')
            last_save = time.monotonic()
    if interrupted:
        reason = interrupted[-1]
    trainer.save()
    snapshot(output/'final.pt')
    write_json(output/'receipt.json', dict(complete=not interrupted,reason=reason,
               elapsed_seconds=time.monotonic()-start,progress=trainer.progress,
               final_sha256=sha256(output/'final.pt')))
    if interrupted:
        raise RuntimeError(f'arm {name} interrupted')


def workload(manifest_path: Path, output: Path) -> None:
    from infra.cpu_budget import host_facts, usable_cpus

    if sys.platform != 'linux' or 'POD_DEADLINE_EPOCH' not in os.environ:
        raise RuntimeError('research training requires an owned Linux pod with a provider deadline')
    def interrupted(signum, frame):
        raise RuntimeError(f'workload interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    plan = json.loads(manifest_path.read_text())
    training_device(plan)
    if source_identity()['source_sha256'] != plan['source']['source_sha256']:
        raise ValueError('frozen source mismatch')
    if usable_cpus() < plan['compute']['minimum_usable_cpus']:
        raise RuntimeError('insufficient effective CPU quota for the three-arm comparison')
    for name,digest in plan['payload_files'].items():
        if sha256(manifest_path.parent/name) != digest:
            raise ValueError('payload digest mismatch')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'host-facts.json',host_facts())
    write_json(output/'health.json',dict(healthy=True,phase='tests'))
    with (output/'tests.log').open('w') as log:
        subprocess.run([sys.executable,'-m','pytest','-q',*TESTS],stdout=log,stderr=subprocess.STDOUT,
                       check=True,timeout=150)
    children = {}
    try:
        for name in MODES:
            with (output/f'arm-{name}.log').open('w') as log:
                children[name] = subprocess.Popen([sys.executable,'-m',__spec__.name,'arm',
                    '--manifest',str(manifest_path),'--output',str(output/f'arm-{name}'),
                    '--arm',name],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while any(p.poll() is None for p in children.values()):
            if time.time() > float(os.environ['POD_DEADLINE_EPOCH'])-240:
                raise RuntimeError('provider cleanup deadline')
            for name,child in children.items():
                if child.poll() not in (None,0):
                    raise RuntimeError(f'arm {name} exited with {child.returncode}')
                path = output/f'arm-{name}/health.json'
                if path.exists():
                    health = json.loads(path.read_text())
                    if not health['healthy'] or (child.poll() is None and time.time()-path.stat().st_mtime > 170):
                        raise RuntimeError(f'arm {name} health/staleness failure')
            write_json(output/'health.json',dict(healthy=True,phase='training',epoch=time.time()))
            time.sleep(10)
        for name,child in children.items():
            receipt = json.loads((output/f'arm-{name}/receipt.json').read_text())
            if child.returncode or not receipt['complete']:
                raise RuntimeError(f'arm {name} incomplete')
        write_json(output/'receipt.json',dict(complete=True,arms=list(MODES),claim='evaluation pending'))
    finally:
        for child in children.values():
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
        for child in children.values():
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()


def evaluate(root: Path) -> None:
    """Evaluate only the fixed endpoints, then paired seeds and whole deals."""
    import numpy as np
    import torch
    from eval.history_frozen import evaluate as frozen_evaluate

    torch.set_num_threads(2)
    campaign = json.loads((root/'campaign.json').read_text())
    reports = {}
    for kit_name in campaign['kits']:
        kit = Path(kit_name)
        manifest = json.loads((kit/'run-manifest.json').read_text())
        seed = manifest['config']['seed']
        if sha256(kit/'evaluation/freeze.json') != manifest['evaluation']['freeze_sha256']:
            raise ValueError('evaluation freeze changed after preparation')
        reports[seed] = {}
        for name in MODES:
            directory = kit/f'local/results/arm-{name}'
            receipt = json.loads((directory/'receipt.json').read_text())
            checkpoint = directory/'final.pt'
            if not receipt['complete'] or sha256(checkpoint) != receipt['final_sha256']:
                raise ValueError('fixed endpoint incomplete or modified')
            reports[seed][name] = frozen_evaluate(kit/'evaluation/freeze.json',checkpoint,
                                                 kit/f'results/endpoint-{name}.json')['reports']
    summary = dict(primary='C minus B',seed_count=len(reports),comparisons={},
                   caveat='three training seeds; exploratory uncertainty, no automatic promotion')
    rng = np.random.default_rng(2026092853)
    for high,low in [('C','B'),('B','A'),('C','A')]:
        comparisons = {}
        for baseline in next(iter(reports.values()))['A']:
            d = np.array([np.array(r[high][baseline]['duplicates']['pair_scores'])-
                          np.array(r[low][baseline]['duplicates']['pair_scores']) for r in reports.values()])
            # Paired seed resampling, and one common deal resample across seeds.
            match_d = np.array([
                np.array([p['win_rate'] for p in r[high][baseline]['full_matches']['pairs']])-
                np.array([p['win_rate'] for p in r[low][baseline]['full_matches']['pairs']])
                for r in reports.values()])
            boot, match_boot = [], []
            for _ in range(10000):
                seed_ids = rng.integers(0,len(d),len(d))
                deal_ids = rng.integers(0,d.shape[1],d.shape[1])
                boot.append(d[seed_ids][:,deal_ids].mean())
                match_ids = rng.integers(0,match_d.shape[1],match_d.shape[1])
                match_boot.append(match_d[seed_ids][:,match_ids].mean())
            comparisons[baseline] = dict(mean_delta=float(d.mean()),
                seed_deltas={str(s):float(v.mean()) for s,v in zip(reports,d)},
                paired_seed_and_deal_95_ci=np.quantile(boot,[.025,.975]).tolist(),
                full_match_mean_delta=float(match_d.mean()),
                full_match_seed_deltas={str(s):float(v.mean()) for s,v in zip(reports,match_d)},
                paired_seed_and_match_95_ci=np.quantile(match_boot,[.025,.975]).tolist(),
                full_match_win_rates={str(s):{name:r[name][baseline]['full_matches']['win_rate']
                                             for name in MODES} for s,r in reports.items()})
        summary['comparisons'][f'{high}-{low}'] = comparisons
    write_json(root/'comparison.json',summary)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','workload','arm','evaluate'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--arm',choices=list(MODES))
    parser.add_argument('--learner-device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--rollout-device',choices=['cpu','cuda'])
    args=parser.parse_args(argv)
    if args.command=='prepare':
        prepare(args.output.resolve(),args.learner_device,args.rollout_device)
    elif args.command=='evaluate':
        evaluate(args.output.resolve())
    else:
        if args.manifest is None:
            parser.error('--manifest is required')
        if args.command=='workload':
            workload(args.manifest,args.output)
        else:
            if sys.platform!='linux' or 'POD_DEADLINE_EPOCH' not in os.environ:
                raise RuntimeError('research arm requires an owned Linux pod')
            if args.arm is None:
                parser.error('--arm is required')
            arm(json.loads(args.manifest.read_text()),args.arm,args.output)


if __name__=='__main__':
    main()
