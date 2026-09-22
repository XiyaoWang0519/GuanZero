"""Stage B critic dataset: splits, returns, schema and coverage (task B1)."""
from dataclasses import asdict
import json
from pathlib import Path

import gd
import numpy as np
import pytest
import torch

import eval.collect_critic as critic
from train.ckpt import save_checkpoint
from train.model import GuandanModel, ModelConfig


def tiny_checkpoint(path: Path, updates: int = 0) -> Path:
    torch.manual_seed(5)
    config = ModelConfig(state_width=8, state_layers=1, action_width=8,
                         action_layers=1, fusion_width=8, fusion_layers=1)
    model = GuandanModel(config)
    save_checkpoint(path, {"model_config": asdict(config), "model": model.state_dict(),
        "optimizer": {}, "config": {"seed": 7, "action_mode": "canonical"},
        "progress": {"updates": updates}, "rng": {}})
    return path


@pytest.fixture(scope="module")
def collection(tmp_path_factory):
    root = tmp_path_factory.mktemp("critic")
    checkpoint = tiny_checkpoint(root / "frozen.pt")
    output = root / "data"
    manifest = critic.collect_critic(checkpoint, output, rounds=48, num_envs=6, seed=311,
                                     max_seconds=120, shard_rounds=5,
                                     split_fractions=(0.5, 0.25, 0.25))
    shards = {split: [critic.load_shard(output / s["path"]) for s in manifest["shards"]
                      if s["path"].startswith(f"{split}/")] for split in critic.SPLITS}
    return {"checkpoint": checkpoint, "output": output, "manifest": manifest, "shards": shards}


def rounds_of(shard: dict):
    offsets = shard["round_offsets"]
    for r in range(len(offsets) - 1):
        yield r, slice(int(offsets[r]), int(offsets[r + 1]))


def test_manifest_is_complete_and_records_checkpoint_digest(collection):
    manifest = collection["manifest"]
    assert manifest == json.loads((collection["output"] / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["schema_version"] == critic.SCHEMA_VERSION
    assert manifest["collected_rounds"] == 48
    assert manifest["checkpoint_file_sha256"] == critic.file_sha256(collection["checkpoint"])
    assert len(manifest["checkpoint_id"]) == 64
    assert manifest["learner_updates"] == 0
    for entry in manifest["shards"]:
        assert critic.file_sha256(collection["output"] / entry["path"]) == entry["sha256"]


def test_no_match_appears_in_two_splits(collection):
    seed = collection["manifest"]["seed"]
    fractions = tuple(collection["manifest"]["split_fractions"][s] for s in critic.SPLITS)
    matches = {}
    for split, shards in collection["shards"].items():
        matches[split] = set()
        for shard in shards:
            assert str(shard["split"]) == split
            for env_id, match_id in zip(shard["env_id"], shard["match_id"]):
                matches[split].add((int(env_id), int(match_id)))
                assert critic.split_of(seed, int(env_id), int(match_id), fractions) == split
    assert all(matches[s] for s in critic.SPLITS), "every split should be populated"
    for a in critic.SPLITS:
        for b in critic.SPLITS:
            if a < b:
                assert not matches[a] & matches[b]
    # Round identities are unique across the whole dataset.
    keys = [(int(e), int(m), int(r)) for shards in collection["shards"].values()
            for shard in shards
            for e, m, r in zip(shard["env_id"], shard["match_id"], shard["round_index"])]
    assert len(keys) == len(set(keys)) == 48


def test_split_is_deterministic_under_seed():
    fractions = (0.8, 0.1, 0.1)
    first = [critic.split_of(9, e, m, fractions) for e in range(20) for m in range(20)]
    assert first == [critic.split_of(9, e, m, fractions) for e in range(20) for m in range(20)]
    assert first != [critic.split_of(10, e, m, fractions) for e in range(20) for m in range(20)]
    counts = {s: first.count(s) for s in critic.SPLITS}
    assert counts["train"] > counts["val"] and counts["train"] > counts["test"]


def test_team_return_matches_engine_round_result(collection):
    for shards in collection["shards"].values():
        for shard in shards:
            for r, rows in rounds_of(shard):
                seat_return = shard["seat_return"][r]
                order, gain, winner = shard["order"][r], int(shard["gain"][r]), int(shard["winning_team"][r])
                # Partners share a return and the teams' returns are opposite.
                assert seat_return[0] == seat_return[2] == -seat_return[1] == -seat_return[3]
                # Magnitude from the finishing order: the partner of the first
                # finisher in second, third or last place gives 3, 2 or 1.
                first = int(order[0])
                partner_place = [int(s) for s in order].index((first + 2) % 4)
                assert gain == {1: 3, 2: 2, 3: 1}[partner_place]
                assert winner == first % 2
                for seat in range(4):
                    value = int(seat_return[seat])
                    if value == 0:
                        # Only a level-A round won by Banker and Dweller is zeroed.
                        assert int(shard["round_level"][r]) == 12 and gain == 1
                    else:
                        assert value == (gain if seat % 2 == winner else -gain)
                np.testing.assert_array_equal(shard["team_return"][rows],
                                              seat_return[shard["seat"][rows]])


def test_decision_fields_are_consistent(collection):
    play = int(gd.Phase.Play)
    for shards in collection["shards"].values():
        for shard in shards:
            obs = shard["obs"]
            assert obs.shape[1] == gd.OBS_DIM and obs.max() <= 1
            assert shard["hidden"].shape == (len(obs), 3, 54)
            # Hidden rows partition the cards the actor cannot see.
            np.testing.assert_array_equal(shard["hidden"].sum(1),
                                          obs[:, 108:162] + obs[:, 162:216])
            np.testing.assert_array_equal(shard["stage"], critic.stage_bins(obs))
            is_play = shard["phase"] == play
            assert np.isnan(shard["q_chosen"][~is_play]).all()
            assert np.isfinite(shard["q_chosen"][is_play]).all()
            # Argmax self-play: the chosen action is the best-Q candidate.
            np.testing.assert_array_equal(shard["q_chosen"][is_play], shard["q_max"][is_play])
            assert (shard["num_candidates"] >= 1).all()
            for _, rows in rounds_of(shard):
                assert len(np.unique(shard["seat"][rows])) >= 2


def test_stage_bin_counts_are_reported(collection):
    manifest = collection["manifest"]
    summary = manifest["summary"]
    assert summary == critic.summarize(collection["output"])
    for split in critic.SPLITS:
        stages = np.zeros(3, dtype=np.int64)
        decisions = 0
        for shard in collection["shards"][split]:
            stages += np.bincount(shard["stage"], minlength=3)
            decisions += len(shard["stage"])
        reported = summary["splits"][split]
        assert [reported["stage_decisions"][s] for s in critic.STAGES] == stages.tolist()
        assert reported["decisions"] == decisions
    total = summary["total"]
    assert sum(total["stage_decisions"].values()) == total["decisions"] == manifest["collected_decisions"]
    assert total["rounds"] == 48
    assert sum(total["decision_return_counts"].values()) == total["decisions"]
    assert sum(total["round_return_team0_counts"].values()) == total["rounds"]


def test_shard_schema_round_trip(tmp_path):
    rng = np.random.default_rng(3)
    writer = critic.ShardWriter(tmp_path, "val", shard_rounds=2, seed=5)
    records = []
    for index, length in enumerate((3, 4)):
        obs = rng.integers(0, 2, size=(length, gd.OBS_DIM), dtype=np.uint8)
        returns = np.array([2, -2, 2, -2], dtype=np.int8)
        seats = rng.integers(0, 4, size=length).astype(np.int8)
        records.append({
            "obs_bits": np.packbits(obs, axis=1), "obs": obs,
            "hidden": rng.integers(0, 3, size=(length, 3, 54), dtype=np.uint8),
            "phase": np.full(length, 3, dtype=np.int8), "seat": seats,
            "stage": critic.stage_bins(obs),
            "q_chosen": rng.normal(size=length).astype(np.float32),
            "q_max": rng.normal(size=length).astype(np.float32),
            "num_candidates": rng.integers(1, 40, size=length).astype(np.int32),
            "team_return": returns[seats], "env_id": index, "match_id": 10 + index,
            "round_index": 0, "round_level": 0, "winning_team": 0, "gain": 2,
            "match_winner": -1, "order": [0, 1, 2, 3], "seat_return": returns.tolist(),
        })
    assert writer.add(records[0]) is None
    entry = writer.add(records[1])
    assert entry["path"] == "val/shard-00000.npz" and entry["rounds"] == 2 and entry["decisions"] == 7
    loaded = critic.load_shard(tmp_path / entry["path"])
    assert int(loaded["schema_version"]) == critic.SCHEMA_VERSION
    assert str(loaded["split"]) == "val" and int(loaded["seed"]) == 5
    np.testing.assert_array_equal(loaded["round_offsets"], [0, 3, 7])
    for key in ("obs", "hidden", "phase", "seat", "stage", "q_chosen", "q_max",
                "num_candidates", "team_return"):
        np.testing.assert_array_equal(loaded[key], np.concatenate([r[key] for r in records]))
        assert loaded[key].dtype == records[0][key].dtype
    np.testing.assert_array_equal(loaded["match_id"], [10, 11])
    np.testing.assert_array_equal(loaded["seat_return"], [r["seat_return"] for r in records])
    # Unsupported schema versions are refused.
    bad = dict(np.load(tmp_path / entry["path"]))
    bad["schema_version"] = np.asarray(99)
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError, match="schema"):
        critic.load_shard(tmp_path / "bad.npz")


def test_collection_is_deterministic_under_seed(collection, tmp_path):
    output = tmp_path / "again"
    critic.collect_critic(collection["checkpoint"], output, rounds=48, num_envs=6, seed=311,
                          max_seconds=120, shard_rounds=5, split_fractions=(0.5, 0.25, 0.25))
    first = [s["sha256"] for s in collection["manifest"]["shards"]]
    again = json.loads((output / "manifest.json").read_text())
    assert [s["sha256"] for s in again["shards"]] == first


def test_refuses_undertrained_checkpoint_and_existing_output(collection, tmp_path):
    with pytest.raises(ValueError, match="smoke model"):
        critic.collect_critic(collection["checkpoint"], tmp_path / "x", rounds=2, num_envs=2,
                              seed=311, min_updates=30_000)
    manifest = json.loads((tmp_path / "x" / "manifest.json").read_text())
    assert manifest["status"] == "incomplete"
    with pytest.raises(FileExistsError):
        critic.collect_critic(collection["checkpoint"], collection["output"], rounds=2)
    with pytest.raises(ValueError, match="training seed"):
        critic.collect_critic(collection["checkpoint"], tmp_path / "y", rounds=2, seed=7)


def test_time_limit_flushes_consistent_partial_dataset(collection, tmp_path, monkeypatch):
    clock = iter(np.arange(0.0, 1e6, 1.0))
    monkeypatch.setattr(critic.time, "monotonic", lambda: float(next(clock)))
    manifest = critic.collect_critic(collection["checkpoint"], tmp_path / "t", rounds=10_000,
                                     num_envs=4, seed=311, max_seconds=150, shard_rounds=3)
    assert manifest["status"] == "time_limited"
    assert 0 < manifest["collected_rounds"] < 10_000
    assert manifest["summary"]["total"]["rounds"] == manifest["collected_rounds"]


def test_default_checkpoint_is_the_m1_final_path():
    path = critic.default_checkpoint()
    assert path.as_posix().endswith(".work/runpod/artifacts/pilot/final.pt")
    assert "m1-full-model" not in path.as_posix()


def test_parallel_workers_merge_without_match_leakage(collection, tmp_path):
    output = tmp_path / "parallel"
    manifest = critic.collect_parallel(collection["checkpoint"], output, workers=2, rounds=21,
                                       seed=311, num_envs=3, max_seconds=120, shard_rounds=4,
                                       split_fractions=(0.5, 0.25, 0.25))
    assert manifest["status"] == "complete" and manifest["worker_seeds"] == [311, 312]
    assert manifest["collected_rounds"] == 21 and manifest["worker_quotas"] == [11, 10]
    assert manifest == json.loads((output / "manifest.json").read_text())
    owner = {}
    for path in critic.shard_paths(output):
        shard = critic.load_shard(path)
        for env_id, match_id in zip(shard["env_id"], shard["match_id"]):
            key = (int(shard["seed"]), int(env_id), int(match_id))
            assert owner.setdefault(key, str(shard["split"])) == str(shard["split"])
    assert {seed for seed, _, _ in owner} == {311, 312}
    total = manifest["summary"]["total"]
    assert total["rounds"] == 21 and total["matches"] == len(owner)
    assert sum(total["stage_decisions"].values()) == manifest["collected_decisions"]
