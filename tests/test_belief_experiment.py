"""Data, split, statistical and completion contracts for the larger belief gate."""
import json

import numpy as np
import pytest
import torch

from train.belief_experiment import evaluate, load_dataset, paired_improvement, run, split_rounds
from train.buffer import Decision
from train.logs import TOKEN_DIM, save_round
from train.tribute_data import engine_source_digest


def make_dataset(path):
    path.mkdir()
    for index in range(20):
        hidden = np.zeros((3, 54), dtype=np.uint8)
        hidden[index % 3, index % 54] = 1
        obs = np.zeros(1849, dtype=np.uint8)
        obs[108 + index % 54] = 1
        for rel in range(3):
            obs[648 + 28 * rel + int(rel == index % 3)] = 1
        tokens = np.zeros((3, TOKEN_DIM), dtype=np.uint8)
        tokens[np.arange(3), np.arange(3)] = 1
        decisions = [Decision(obs, np.zeros(154), hidden, seat=i, phase=3, prefix=i)
                     for i in range(3)]
        save_round(path / f"round-{index:08d}.npz", decisions, list(tokens), f"match-{index}")
    metadata = {"status": "complete", "purpose": "architecture_probe", "learner_updates": 0,
                "engine_source_sha256": engine_source_digest(), "action_mode": "canonical",
                "tribute_policy": "heuristic", "sampling_margin": 0, "collected_rounds": 20,
                "seed": 901, "training_seed": 7, "stage": "dmc", "play_mode": "fp32_argmax",
                "checkpoint_id": "a" * 64, "requested_rounds": 20,
                "collected_decisions": 60, "match_groups": 20}
    (path / "provenance.json").write_text(json.dumps(metadata))


def test_split_is_stable_and_match_disjoint(tmp_path):
    make_dataset(tmp_path / "data")
    rounds, provenance, digest = load_dataset(tmp_path / "data")
    split = split_rounds(rounds, 123)
    assert [len(split[k]) for k in ("train", "validation", "test")] == [14, 3, 3]
    sets = [{r['group'] for r in v} for v in split.values()]
    assert len(set.union(*sets)) == 20
    assert all(not a & b for i, a in enumerate(sets) for b in sets[i+1:])
    repeat = split_rounds(list(reversed(rounds)), 123)
    assert all({r['group'] for r in split[k]} == {r['group'] for r in repeat[k]} for k in split)
    assert len(digest) == 64


@pytest.mark.parametrize("field,value", [("status", "incomplete"), ("purpose", "evaluation_only"),
    ("engine_source_sha256", "bad"), ("collected_rounds", 19), ("sampling_margin", 1),
    ("checkpoint_id", "bad"), ("training_seed", 901), ("match_groups", 19),
    ("collected_decisions", 2), ("stage", "a2")])
def test_dataset_rejects_wrong_provenance(tmp_path, field, value):
    path = tmp_path / "data"
    make_dataset(path)
    p = path / "provenance.json"
    d = json.loads(p.read_text()); d[field] = value; p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        load_dataset(path)


@pytest.mark.parametrize("mutation", ["negative_prefix", "hidden_range", "private_token", "wrong_seat", "unseen", "relative_seat"])
def test_dataset_rejects_corrupt_supervision(tmp_path, mutation):
    path = tmp_path / "data"
    make_dataset(path)
    file = path / "round-00000000.npz"
    with np.load(file, allow_pickle=False) as loaded:
        d = {k: loaded[k].copy() for k in loaded.files}
    if mutation == "negative_prefix": d['prefix'][0] = -1
    elif mutation == "hidden_range": d['hidden'][0, 0, 0] = 3
    elif mutation == "private_token": d['tokens'][0, 150] = 1
    elif mutation == "wrong_seat": d['seat'][1] = 3
    elif mutation == "unseen": d['hidden'][0, 0, 1] = 1
    elif mutation == "relative_seat": d['hidden'] = d['hidden'][:, ::-1].copy()
    np.savez(file, **d)
    with pytest.raises(ValueError):
        load_dataset(path)


def test_paired_bootstrap_clusters_whole_matches_not_decisions():
    ref = {'log_loss': .7, 'matches': {'a': {'log_loss': .8, 'decisions': 10000},
                                      'b': {'log_loss': .4, 'decisions': 1}}}
    candidate = {'log_loss': .5, 'matches': {'a': {'log_loss': .5, 'decisions': 10000},
                                            'b': {'log_loss': .5, 'decisions': 1}}}
    paired = paired_improvement(ref, candidate, 9, 1000)
    assert paired['mean_match_log_loss_improvement'] == pytest.approx(.1)
    assert paired['micro_log_loss_improvement'] == pytest.approx(.2)
    assert paired['bootstrap_95_ci'] == pytest.approx([-.1, .3])
    candidate['matches']['b']['decisions'] = 2
    with pytest.raises(ValueError, match='decisions'):
        paired_improvement(ref, candidate, 9)


def test_experiment_selects_validation_checkpoint_and_keeps_diagnostics_only(tmp_path):
    data, out = tmp_path / 'data', tmp_path / 'out'
    make_dataset(data)
    report = run(data, out, seeds=(31,), steps=2, min_steps=1, validation_interval=1,
                 batch_size=2, width=16, layers=1, threads=1, max_seconds=30)
    assert report['status'] == 'complete' and report['rl_enabled'] is False
    assert report['gate'] == 'v2_not_yet_justified'
    assert len(report['runs']) == 1
    for name, metrics in report['runs'][0]['models'].items():
        assert metrics['validation_log_loss'] == min(x['validation_log_loss'] for x in metrics['learning_curve'])
        payload = torch.load(out / f'{name}-s31.pt', weights_only=False)
        assert payload['purpose'] == 'belief_probe_only'
        assert payload['selected_step'] == metrics['selected_step']
        assert len(metrics['test']['matches']) == 3
    with pytest.raises(FileExistsError):
        run(data, out)


def test_experiment_timeout_is_not_success_and_seed_overlap_rejected(tmp_path):
    make_dataset(tmp_path / 'data')
    with pytest.raises(TimeoutError):
        run(tmp_path / 'data', tmp_path / 'timeout', max_seconds=1e-12)
    assert json.loads((tmp_path / 'timeout/report.json').read_text())['status'] == 'incomplete'
    with pytest.raises(ValueError, match='seeds'):
        run(tmp_path / 'data', tmp_path / 'overlap', seeds=(7,))
