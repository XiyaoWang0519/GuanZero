"""Data, split, statistical and completion contracts for the larger belief gate."""
import json

import numpy as np
import pytest
import torch

from train.belief_experiment import (dataset_bytes, evaluate, load_dataset, paired_improvement,
                                     round_bin, run, split_rounds)
from train.belief_model import matched_models
from train.belief_probe import examples
from train.buffer import Decision
from train.logs import TOKEN_DIM, save_round
from train.tribute_data import engine_source_digest


def make_dataset(path, *, count=20, rounds_per_match=1, styled=False):
    """Twenty schema-1 rounds by default; `styled` writes schema-2 metadata.

    With `rounds_per_match` above one the rounds of a match carry increasing
    match-relative round indices, which is what the round-bin breakdown needs.
    """
    path.mkdir()
    for index in range(count):
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
        match = index // rounds_per_match
        meta = None
        if styled:
            meta = {"match_id": match, "round_index": index % rounds_per_match, "env_id": 0,
                    "styles": np.zeros((4, 3), dtype=np.float32),
                    "seat_driver": [0, 1, 0, 1], "styled": True,
                    "style_region": "heldout" if match % 3 == 0 else "train"}
        save_round(path / f"round-{index:08d}.npz", decisions, list(tokens),
                   f"match-{match}", meta)
    metadata = {"status": "complete", "purpose": "architecture_probe", "learner_updates": 0,
                "engine_source_sha256": engine_source_digest(), "action_mode": "canonical",
                "tribute_policy": "heuristic", "sampling_margin": 0, "collected_rounds": count,
                "seed": 901, "training_seed": 7, "stage": "dmc", "play_mode": "fp32_argmax",
                "checkpoint_id": "a" * 64, "requested_rounds": count,
                "collected_decisions": 3 * count,
                "match_groups": -(-count // rounds_per_match)}
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


def test_round_bins_cover_every_index_and_schema_one_is_unknown():
    assert [round_bin(i) for i in range(8)] == ["0", "1", "2-3", "2-3", "4-5", "4-5", "6+", "6+"]
    assert round_bin(-1) == "unknown" and round_bin(400) == "6+"


def test_schema_one_logs_fall_into_unknown_cells(tmp_path):
    make_dataset(tmp_path / "data")
    rounds, _, _ = load_dataset(tmp_path / "data")
    torch.manual_seed(0)
    result = evaluate(matched_models(1849, 16, 1)["v1"], examples(rounds), "cpu", 8)
    assert set(result["cells"]) == {"round_bin:unknown", "style_region:unknown",
                                    "target_driver:unknown"}
    assert result["cells"]["round_bin:unknown"]["decisions"] == 60
    # Driver cells score each of the three predicted seats separately.
    assert result["cells"]["target_driver:unknown"]["decisions"] == 180
    assert set(result["cells"]["round_bin:unknown"]["matches"]) == {f"match-{i}" for i in range(20)}


def test_styled_breakdowns_cover_round_bins_regions_and_drivers(tmp_path):
    data, out = tmp_path / "data", tmp_path / "out"
    make_dataset(data, count=160, rounds_per_match=8, styled=True)
    report = run(data, out, seeds=(31,), steps=2, min_steps=1, validation_interval=1,
                 batch_size=4, width=16, layers=1, threads=1, max_seconds=300,
                 cell_bootstrap_samples=64)
    assert report["status"] == "complete"
    run_result = report["runs"][0]
    for name in ("v1", "v2", "no_history"):
        cells = run_result["models"][name]["test"]["cells"]
        assert {c.split(":", 1)[1] for c in cells if c.startswith("round_bin:")} == {
            "0", "1", "2-3", "4-5", "6+"}
        assert {c.split(":", 1)[1] for c in cells if c.startswith("target_driver:")} == {
            "bot", "policy"}
        regions = {c.split(":", 1)[1] for c in cells if c.startswith("style_region:")}
        assert regions and regions <= {"train", "heldout"}
        decisions = run_result["models"][name]["test"]["decisions"]
        assert sum(v["decisions"] for k, v in cells.items() if k.startswith("round_bin:")) == decisions
        assert sum(v["decisions"] for k, v in cells.items()
                   if k.startswith("target_driver:")) == 3 * decisions
        # Per-cell match tables are consumed by the pairing and not published.
        assert all("matches" not in v and v["test_matches"] >= 1 for v in cells.values())
    for key in ("v2_vs_v1", "v2_vs_no_history"):
        paired = run_result[key]["cells"]
        assert set(paired) == set(run_result["models"]["v2"]["test"]["cells"])
        assert all(len(v["bootstrap_95_ci"]) == 2 and v["test_matches"] >= 1
                   for v in paired.values())


def test_decisions_per_round_keeps_whole_rounds_and_is_deterministic(tmp_path):
    path = tmp_path / "data"
    make_dataset(path, styled=True)
    full, _, digest = load_dataset(path)
    capped, _, capped_digest = load_dataset(path, decisions_per_round=2)
    again, _, _ = load_dataset(path, decisions_per_round=2)
    assert len(capped) == len(full) == 20 and digest == capped_digest
    assert all(len(r["obs"]) == 2 for r in capped)
    assert all(r["obs"].dtype == np.uint8 for r in capped)
    # Whole rounds and their full public token streams survive the cap.
    assert all(len(r["tokens"]) == len(f["tokens"]) for r, f in zip(capped, full))
    assert all(np.array_equal(a["obs"], b["obs"]) for a, b in zip(capped, again))
    assert 0 < dataset_bytes(capped) < dataset_bytes(full)
    with pytest.raises(ValueError, match="decisions_per_round"):
        load_dataset(path, decisions_per_round=-1)


def test_stop_reason_distinguishes_plateau_from_the_step_cap(tmp_path):
    data = tmp_path / "data"
    make_dataset(data)
    capped = run(data, tmp_path / "cap", seeds=(31,), steps=2, min_steps=2,
                 validation_interval=1, batch_size=2, width=16, layers=1, threads=1,
                 max_seconds=120, cell_bootstrap_samples=32)
    assert capped["dataset_bytes"] > 0 and capped["loaded_decisions"] == 60
    for metrics in capped["runs"][0]["models"].values():
        assert metrics["stop_reason"] == "step_cap" and metrics["early_stopped"] is False
        assert metrics["best_validation_step"] == metrics["selected_step"]
        assert metrics["validation_evaluations"] == len(metrics["learning_curve"]) == 2
    plateau = run(data, tmp_path / "plateau", seeds=(31,), steps=200, min_steps=1,
                  validation_interval=1, patience=1, batch_size=2, width=16, layers=1,
                  threads=1, max_seconds=300, cell_bootstrap_samples=32)
    for metrics in plateau["runs"][0]["models"].values():
        assert (metrics["stop_reason"] == "plateau") == metrics["early_stopped"]
        assert metrics["steps_run"] <= 200


def test_validation_decisions_caps_selection_but_never_the_test_set(tmp_path):
    data = tmp_path / "data"
    make_dataset(data)
    report = run(data, tmp_path / "out", seeds=(31,), steps=2, min_steps=2,
                 validation_interval=1, batch_size=2, width=16, layers=1, threads=1,
                 max_seconds=120, cell_bootstrap_samples=32, validation_decisions=4)
    splits = report["splits"]
    assert splits["validation"]["decisions"] == 9 and splits["validation"]["decisions_used"] == 4
    assert splits["test"]["decisions_used"] == splits["test"]["decisions"] == 9
    assert report["runs"][0]["models"]["v2"]["test"]["decisions"] == 9
