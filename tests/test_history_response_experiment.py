"""Experiment preparation cannot allocate resources or start local research."""
import json
from pathlib import Path

import pytest

from infra import history_response_experiment as experiment
from infra.history_artifacts import sha256


def test_prepare_freezes_matched_arms_and_three_distinct_seeds(tmp_path, monkeypatch):
    # Avoid packing the whole checkout in this unit test. The preparation
    # operation itself has no provider client/provisioning path.
    def pack(destination):
        destination.write_bytes(b'test source archive')
        return {'source_sha256':'test-source','archive_sha256':sha256(destination)}
    monkeypatch.setattr(experiment,'pack_source',pack)
    source = tmp_path/'source'
    setup = source/'infra/history_setup.sh'
    setup.parent.mkdir(parents=True)
    setup.write_text('#!/bin/sh\nexit 0\n')
    checkpoint = source/'baseline.pt'
    checkpoint.write_bytes(b'fixture evaluator checkpoint')
    freeze = source/'.work/runpod-history-batch-2026-09-26/evaluation/freeze.json'
    freeze.parent.mkdir(parents=True)
    freeze.write_text(json.dumps({'baselines':[{'path':str(checkpoint),'sha256':sha256(checkpoint)}],
                                 'final_test':{'opened':False,'commitment_sha256':'fixture'}}))
    monkeypatch.setattr(experiment,'ROOT',source)
    root = tmp_path/'campaign'
    campaign = experiment.prepare(root)
    assert len(set(campaign['seeds'])) == 3
    assert campaign['primary_comparison'] == 'C minus B'
    assert len(campaign['kits']) == 3
    deal_hashes = set()
    for kit in map(Path,campaign['kits']):
        manifest = json.loads((kit/'run-manifest.json').read_text())
        assert manifest['arms'] == {'A':'none','B':'auxiliary','C':'explicit'}
        assert manifest['config']['epochs'] == 2
        assert manifest['training_seconds'] == 4800
        assert manifest['training_device'] == 'cuda'
        assert manifest['lifecycle']['proposed_budget_usd'] <= 2
        assert manifest['lifecycle']['max_hours_from_create'] < 2
        assert not (kit/'approval.json').exists()
        assert not (kit/'pod.json').exists()
        assert sha256(kit/'evaluation/freeze.json') == manifest['evaluation']['freeze_sha256']
        dev = json.loads((kit/'evaluation/development.json').read_text())
        assert len(dev['deals']) == 256 and len(dev['match_seeds']) == 64
        deal_hashes.add(sha256(kit/'evaluation/development.json'))
    assert len(deal_hashes) == 1
    with pytest.raises(ValueError,match='overwrite'):
        experiment.prepare(root)


def test_workload_refuses_local_research(tmp_path,monkeypatch):
    monkeypatch.setattr(experiment.sys,'platform','darwin')
    with pytest.raises(RuntimeError,match='owned Linux pod'):
        experiment.workload(tmp_path/'missing.json',tmp_path)


def test_training_device_never_falls_back_to_cpu(monkeypatch):
    import torch

    for manifest in ({}, {'training_device':'auto'}):
        with pytest.raises(ValueError,match='explicit cpu/cuda'):
            experiment.training_device(manifest)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    assert experiment.training_device({'training_device':'cpu'}) == 'cpu'
    with pytest.raises(RuntimeError,match='CUDA is unavailable'):
        experiment.training_device({'training_device':'cuda'})
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    assert experiment.training_device({'training_device':'cuda'}) == 'cuda'
