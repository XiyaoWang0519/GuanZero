"""Frozen-policy collection provenance, privacy boundary, and bounded setup."""
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import eval.collect_belief as collection
from eval.policies import model_digest
from train.ckpt import load_checkpoint, save_checkpoint
from train.model import GuandanModel, ModelConfig
from train.tribute_data import engine_source_digest


@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(5)
    config = ModelConfig(state_width=8, state_layers=1, action_width=8,
                         action_layers=1, fusion_width=8, fusion_layers=1)
    model = GuandanModel(config)
    path = tmp_path / "frozen.pt"
    save_checkpoint(path, {"model_config": asdict(config), "model": model.state_dict(),
        "optimizer": {}, "config": {"seed": 7, "action_mode": "canonical"},
        "progress": {}, "rng": {}})
    return path


def test_architecture_probe_has_truthful_identity_counts_and_causal_prefixes(checkpoint, tmp_path):
    original = checkpoint.read_bytes()
    output = tmp_path / "probe"
    report = collection.collect_belief(checkpoint, output, rounds=5, num_envs=2,
        seed=103, max_seconds=30, purpose="architecture_probe")
    assert report == json.loads((output / "provenance.json").read_text())
    assert report["status"] == "complete"
    assert report["purpose"] == "architecture_probe"
    assert report["stage"] == "dmc"
    assert report["action_mode"] == "canonical"
    assert report["tribute_policy"] == "heuristic"
    assert report["sampling_margin"] == 0
    assert report["learner_updates"] == 0
    assert report["round_quotas"] == report["completed_per_env"] == [3, 2]
    assert report["checkpoint_id"] == model_digest(load_checkpoint(checkpoint)["model"])
    assert report["engine_source_sha256"] == engine_source_digest()
    extension = Path(importlib.import_module("gd._gd_core").__file__)
    with extension.open("rb") as stream:
        assert report["engine_sha256"] == hashlib.file_digest(stream, "sha256").hexdigest()
    assert report["collector_sha256"] == hashlib.sha256(Path(collection.__file__).read_bytes()).hexdigest()
    groups, decisions = set(), 0
    rounds = sorted(output.glob("round-*.npz"))
    assert len(rounds) == report["collected_rounds"] == 5
    for path in rounds:
        with np.load(path, allow_pickle=False) as record:
            assert set(record.files) == {"schema_version", "group", "obs", "hidden", "seat", "prefix", "tokens"}
            groups.add(str(record["group"]))
            assert str(record["group"]).startswith("architecture_probe:103:")
            decisions += len(record["obs"])
            prefixes = record["prefix"]
            assert np.all(np.diff(prefixes) > 0)
            assert np.all(prefixes < len(record["tokens"]))
            np.testing.assert_array_equal(record["tokens"][prefixes, :4].argmax(1), record["seat"])
            assert np.all(record["tokens"][:, 150:158] == 0)
            np.testing.assert_array_equal(record["hidden"].sum(1),
                record["obs"][:, 108:162] + record["obs"][:, 162:216])
    assert len(groups) == report["match_groups"] >= 2
    assert decisions == report["collected_decisions"]
    assert checkpoint.read_bytes() == original


@pytest.mark.parametrize("mode", ["full", "a2", "unknown_seed"])
def test_rejects_sources_that_would_misrepresent_collection(checkpoint, tmp_path, mode):
    payload = load_checkpoint(checkpoint)
    if mode == "full":
        payload["config"]["action_mode"] = "full"
    elif mode == "a2":
        payload.update(stage="a2", tribute_policy="learned", base_checkpoint_id="0" * 64)
    else:
        payload["config"].pop("seed")
    save_checkpoint(checkpoint, payload)
    output = tmp_path / mode
    with pytest.raises(ValueError, match="canonical DMC|training seed"):
        collection.collect_belief(checkpoint, output, rounds=2, num_envs=2, seed=103)
    report = json.loads((output / "provenance.json").read_text())
    assert report["status"] == "incomplete"
    assert report["collected_rounds"] == 0
    assert report["error_type"] == "ValueError"
    assert not list(output.glob("round-*.npz"))


def test_deadline_includes_model_loading(checkpoint, tmp_path, monkeypatch):
    now = [0.0]
    load_policy = collection.load_policy

    def slow_load(*args, **kwargs):
        policy = load_policy(*args, **kwargs)
        now[0] = 11.0
        return policy

    monkeypatch.setattr(collection.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(collection, "load_policy", slow_load)
    output = tmp_path / "bounded"
    with pytest.raises(TimeoutError):
        collection.collect_belief(checkpoint, output, rounds=2, max_seconds=10)
    report = json.loads((output / "provenance.json").read_text())
    assert report["status"] == "incomplete"
    assert report["error_type"] == "TimeoutError"
    assert report["elapsed_seconds"] == 11
    assert report["collected_rounds"] == 0


def test_hidden_supervision_cannot_influence_actions(checkpoint, tmp_path, monkeypatch):
    original = tmp_path / "original"
    changed = tmp_path / "changed"
    kwargs = {"rounds": 2, "num_envs": 2, "seed": 104, "max_seconds": 30}
    collection.collect_belief(checkpoint, original, **kwargs)
    vec_env = collection.gd.VecEnv

    class BatchWithoutLabels:
        def __init__(self, batch):
            self.batch = batch

        def __getattr__(self, name):
            if name == "hidden_counts":
                return np.zeros_like(self.batch.hidden_counts)
            return getattr(self.batch, name)

    class EnvWithoutLabels:
        def __init__(self, *args, **kwargs):
            self.env = vec_env(*args, **kwargs)

        def pending(self):
            return BatchWithoutLabels(self.env.pending())

        def __getattr__(self, name):
            return getattr(self.env, name)

    monkeypatch.setattr(collection.gd, "VecEnv", EnvWithoutLabels)
    collection.collect_belief(checkpoint, changed, **kwargs)
    for path in original.glob("round-*.npz"):
        with np.load(path, allow_pickle=False) as before, np.load(changed / path.name, allow_pickle=False) as after:
            for key in before.files:
                if key == "hidden":
                    assert before[key].any() and not after[key].any()
                else:
                    np.testing.assert_array_equal(before[key], after[key])


def test_failed_source_load_is_marked_incomplete(tmp_path):
    output = tmp_path / "failed"
    with pytest.raises(FileNotFoundError):
        collection.collect_belief(tmp_path / "missing.pt", output, rounds=2)
    report = json.loads((output / "provenance.json").read_text())
    assert report["status"] == "incomplete" and report["error_type"] == "FileNotFoundError"


def test_cli_accepts_architecture_purpose(checkpoint, tmp_path, capsys):
    collection.main(["--checkpoint", str(checkpoint), "--output", str(tmp_path / "cli"),
        "--rounds", "2", "--num-envs", "2", "--seed", "105", "--purpose", "architecture_probe"])
    assert json.loads(capsys.readouterr().out)["purpose"] == "architecture_probe"
