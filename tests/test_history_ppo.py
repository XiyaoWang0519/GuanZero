"""STAGE_C T2 acceptance: cold-start boundary, encoder gradients, chunk carry-over,
checkpoint interruption and resume, and the CLI end to end."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from train import history_ppo
from train.ckpt import save_checkpoint
from train.history_model import load_history_checkpoint
from train.history_ppo import HistoryPPOConfig, HistoryTrainer

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (ROOT / "train" / "history_ppo.py", ROOT / "train" / "history_rollout.py")
BANNED = ("StageBPolicy", "prune_candidates", "top_k", "reference", "load_policy",
          "train.policy", "train.model.GuandanModel")


def small_config(**overrides) -> HistoryPPOConfig:
    values = dict(width=32, layers=1, heads=4, num_envs=4, steps_per_update=90, seed=3,
                  updates=1, epochs=1, minibatch_matches=2)
    values.update(overrides)
    return HistoryPPOConfig(**values)


def test_cold_start_source_boundary(tmp_path, monkeypatch):
    for path in SOURCES:
        text = path.read_text()
        for word in BANNED:
            assert word not in text, f"{path.name} mentions {word}"
    calls = []
    real = history_ppo.fresh_player

    def spy(config, seed):
        calls.append((config, seed))
        return real(config, seed)

    monkeypatch.setattr(history_ppo, "fresh_player", spy)
    trainer = HistoryTrainer(small_config(), tmp_path / "fresh")
    assert calls == [(trainer.config.policy_config(), 3)]
    manifest = json.loads((tmp_path / "fresh" / "manifest.json").read_text())
    assert manifest["init"] == "random" and manifest["teacher"] is None
    assert manifest["stage"] == "history_ppo" and manifest["token_schema"]["forced_bit"] is False
    assert len(manifest["engine_digest"]) == 64
    for name, value in history_ppo.runtime_settings(trainer.rollout_device).items():
        assert manifest["inference"][name] is value
    assert manifest["inference"]["reuse_cache_lengths"] is trainer.collector.reuse_cache_lengths
    # An old-stage checkpoint is not a resume point.
    old = tmp_path / "old.pt"
    save_checkpoint(old, {"stage": "stage_b", "model_config": {}, "model": {}, "optimizer": {},
                          "config": {}, "progress": {}, "rng": {}})
    with pytest.raises(ValueError):
        HistoryTrainer(small_config(), tmp_path / "resumed", resume=old)


def test_update_reaches_history_encoder(tmp_path):
    trainer = HistoryTrainer(small_config(), tmp_path)
    before = {n: p.detach().clone() for n, p in trainer.actor.stream.named_parameters()}
    line = trainer.update()
    assert line["update_samples"] > 0
    assert line["encoder_grad_norm"] > 0
    assert line["mean_prefix"] > 0 and line["max_prefix"] >= line["mean_prefix"]
    changed = [n for n, p in trainer.actor.stream.named_parameters()
               if not torch.equal(before[n], p)]
    assert changed and len(changed) == len(before)
    for p in list(trainer.actor.parameters()) + list(trainer.critic.parameters()):
        assert torch.isfinite(p).all()


def test_update_boundary_mid_match_keeps_full_prefix(tmp_path):
    trainer = HistoryTrainer(small_config(steps_per_update=90, updates=2), tmp_path)
    line = trainer.update()
    assert line["update_samples"] > 0
    assert line["carried_rows"] > 0, "a round should still be in progress at the boundary"
    tokens_at_boundary = {key: stream.prefix for key, stream in trainer.store.streams.items()}
    assert all(0 < n for n in tokens_at_boundary.values())
    assert not any(match for _, match in tokens_at_boundary), "matches must still be in progress"
    collected = trainer.collect()
    data = trainer.buffer.compact()
    fresh = np.flatnonzero(data["version"] == 1)
    old = np.flatnonzero(data["version"] == 0)
    assert fresh.size == collected.learner_rows and old.size == line["carried_rows"]
    for row in fresh.tolist():
        key = (int(data["env"][row]), int(data["match"][row]))
        assert int(data["prefix"][row]) >= tokens_at_boundary[key]
    assert (data["prefix"][fresh] > 0).all()
    # Rows collected after the boundary recompute against the whole stream,
    # including tokens appended before the boundary.
    with torch.no_grad():
        log_prob, _ = trainer.recompute_log_probs(fresh)
    assert np.abs(log_prob.numpy() - data["logp"][fresh]).max() < 1e-5
    # Carried rows belong to rounds that end after the boundary: their reward
    # lands on the round's last row, which is a post-boundary row.
    values = trainer.refresh_values()
    trainer.buffer.finalize(values, trainer.config.gamma, trainer.config.gae_lambda)
    done = np.flatnonzero(trainer.buffer.done)
    assert done.size and (trainer.buffer.reward[~trainer.buffer.done] == 0).all()
    expected = {(int(r.env_id), int(r.match_id), int(r.round_index), team): float(r.seat_return[team])
                for r in trainer.collector.results for team in (0, 1)}
    for row in done.tolist():
        key = (int(data["env"][row]), int(data["match"][row]), int(data["round"][row]),
               int(data["seat"][row]) % 2)
        assert trainer.buffer.reward[row] == expected[key]
    completed = set(trainer.buffer.samples.tolist())
    assert completed & set(old.tolist()), "carried rows joined a completed trajectory"
    for t in trainer.buffer.trajectories:
        if t.complete and any(r in set(old.tolist()) for r in t.rows):
            assert t.rows[-1] in set(fresh.tolist())


def test_checkpoint_interruption_and_resume(tmp_path, monkeypatch):
    output = tmp_path / "run"
    trainer = HistoryTrainer(small_config(updates=2), output)
    trainer.run()
    assert trainer.progress["updates"] == 2
    good = torch.load(output / "latest.pt", weights_only=False)
    calls = {"n": 0}
    real = torch.save

    def failing(*args, **kwargs):
        calls["n"] += 1
        raise OSError("simulated interruption during save")

    monkeypatch.setattr(torch, "save", failing)
    trainer.update()
    with pytest.raises(OSError):
        trainer.save()
    monkeypatch.setattr(torch, "save", real)
    assert calls["n"] == 1
    assert not [p for p in output.iterdir() if p.name.startswith(".latest.pt.")], \
        "no temporary file survives the failed write"
    again = torch.load(output / "latest.pt", weights_only=False)
    assert again["progress"] == good["progress"]
    assert again["progress"]["updates"] == 2
    resumed = HistoryTrainer(small_config(updates=3), output / "resumed",
                             resume=output / "latest.pt")
    assert resumed.lineage == good["lineage"]
    assert resumed.config == HistoryPPOConfig.from_payload(good["config"], updates=3)
    assert resumed.config.updates == 3 and resumed.config.seed == trainer.config.seed
    for (n, p), q in zip(resumed.actor.state_dict().items(), good["model"].values()):
        assert torch.equal(p, q), n
    assert resumed.actor_optimizer.state_dict()["state"], "optimizer moments restored"
    assert resumed.generator.get_state().equal(good["rng"]["sampler"])
    resumed.run()
    assert resumed.progress["updates"] == 3
    steps = resumed.config.num_envs * resumed.config.steps_per_update
    assert resumed.progress["decisions"] == 3 * steps == good["progress"]["decisions"] + steps
    assert resumed.progress["rounds"] >= good["progress"]["rounds"]
    for p in list(resumed.actor.parameters()) + list(resumed.critic.parameters()):
        assert torch.isfinite(p).all()
    lines = [json.loads(l) for l in (output / "resumed" / "metrics.jsonl").read_text().splitlines()]
    assert [l["update"] for l in lines] == [3]
    final = torch.load(output / "resumed" / "latest.pt", weights_only=False)
    assert final["progress"]["updates"] == 3 and final["lineage"] == good["lineage"]


def test_resume_under_other_source_only_on_request(tmp_path):
    trainer = HistoryTrainer(small_config(updates=1), tmp_path / "run")
    trainer.run()
    good = torch.load(tmp_path / "run" / "latest.pt", weights_only=False)

    def variant(name, **identity):
        payload = dict(good, run_identity=dict(good["run_identity"], **identity))
        save_checkpoint(tmp_path / name, payload)
        return tmp_path / name

    older = variant("older.pt", source=dict(good["run_identity"]["source"],
                                            source_sha256="old-source", revision="old"))
    with pytest.raises(ValueError, match="identity mismatch"):
        HistoryTrainer(small_config(updates=2), tmp_path / "strict", resume=older)
    # An engine mismatch is a changed dynamics source (the recorded digest alone
    # may differ across digest versions; the per-file hashes decide).
    other_engine = dict(good["run_identity"]["source"],
                        files=dict(good["run_identity"]["source"]["files"],
                                   **{"cpp/src/rules.cpp": "0" * 64}))
    for name, identity in (("engine.pt", dict(engine_digest="other-engine", source=other_engine)),
                           ("tokens.pt", dict(token_schema=99))):
        with pytest.raises(ValueError, match="identity mismatch"):
            HistoryTrainer(small_config(updates=2), tmp_path / name[:-3], resume=variant(name, **identity),
                           allow_source_change=True)
    moved = HistoryTrainer(small_config(updates=2), tmp_path / "moved", resume=older,
                           allow_source_change=True)
    for p, q in zip(moved.actor.state_dict().values(), good["model"].values()):
        assert torch.equal(p, q)
    [change] = moved.source_changes
    assert change["at_update"] == 1 and change["previous_source_sha256"] == "old-source"
    assert change["source_sha256"] == moved.run_identity["source"]["source_sha256"]
    manifest = json.loads((tmp_path / "moved" / "manifest.json").read_text())
    assert manifest["source_changes"] == [change]
    moved.run()
    saved = torch.load(tmp_path / "moved" / "latest.pt", weights_only=False)
    assert saved["run_identity"] == moved.run_identity and saved["source_changes"] == [change]
    # The lineage now carries the current identity: a plain resume works again.
    again = HistoryTrainer(small_config(updates=3), tmp_path / "again", resume=tmp_path / "moved" / "latest.pt")
    assert again.source_changes == [change]


def test_cli_tiny_run(tmp_path):
    output = tmp_path / "cli"
    env = dict(os.environ, PYTHONPATH="python:oracle:.")
    args = [sys.executable, "-m", "train.history_ppo", "--output", str(output), "--updates", "2",
            "--num-envs", "4", "--steps-per-update", "24", "--width", "32", "--layers", "1",
            "--seed", "1"]
    done = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["config"]["width"] == 32 and manifest["model_config"]["layers"] == 1
    assert manifest["token_schema"]["version"] == 1 and manifest["reward"]["reward"]
    lines = [json.loads(l) for l in (output / "metrics.jsonl").read_text().splitlines()]
    assert [l["update"] for l in lines] == [1, 2]
    for line in lines:
        assert line["decisions_per_sec"] > 0 and line["mean_prefix"] >= 0
        for key in ("policy_loss", "value_loss", "entropy", "mean_reward"):
            assert key in line
    actor, critic, payload = load_history_checkpoint(output / "latest.pt")
    assert payload["progress"]["updates"] == 2
    assert payload["progress"]["decisions"] == 2 * 4 * 24
    assert actor.config.width == 32 and payload["config"]["stage"] == "history_ppo"
    assert all(torch.isfinite(p).all() for p in actor.parameters())


def test_window_control_passthrough_keeps_ratio_parity(tmp_path):
    trainer = HistoryTrainer(small_config(window=8), tmp_path)
    assert trainer.actor.config.window == 8
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["model_config"]["window"] == 8
    trainer.collect()
    rows = np.arange(len(trainer.buffer))
    with torch.no_grad():
        log_prob, _ = trainer.recompute_log_probs(rows)
    assert np.abs(log_prob.numpy() - trainer.buffer.compact()["logp"][rows]).max() < 1e-5
    assert trainer.learn()["encoder_grad_norm"] > 0


class _FakeTrainer:
    def __init__(self, fail=False):
        self.fail, self.stop_requested, self.calls = fail, False, 0

    def run(self):
        self.calls += 1
        sum(i * i for i in range(1000))
        if self.fail:
            raise RuntimeError("boom")


def test_cprofile_hook_is_off_by_default_and_writes_rank_profiles(tmp_path, monkeypatch):
    import pstats
    monkeypatch.delenv(history_ppo.CPROFILE_ENV, raising=False)
    trainer = _FakeTrainer()
    history_ppo.run_profiled(trainer, 3)
    assert trainer.calls == 1 and not any(tmp_path.iterdir())
    monkeypatch.setenv(history_ppo.CPROFILE_ENV, str(tmp_path / "prof"))
    history_ppo.run_profiled(trainer, 1)
    stats = pstats.Stats(str(tmp_path / "prof" / "rank-1.prof"))
    assert any(name == "run" for _, _, name in stats.stats)
    # A failing run still leaves its profile, then re-raises.
    with pytest.raises(RuntimeError, match="boom"):
        history_ppo.run_profiled(_FakeTrainer(fail=True), 0)
    assert (tmp_path / "prof" / "rank-0.prof").stat().st_size > 0


def test_cprofile_hook_profiles_the_cli_trainer(tmp_path, monkeypatch):
    import pstats
    monkeypatch.setenv(history_ppo.CPROFILE_ENV, str(tmp_path / "prof"))
    assert history_ppo.main(["--output", str(tmp_path / "run"), "--updates", "1",
                             "--num-envs", "2", "--steps-per-update", "4", "--width", "16",
                             "--layers", "1", "--heads", "2", "--snapshot-updates", "0"]) == 0
    stats = pstats.Stats(str(tmp_path / "prof" / "rank-0.prof"))
    ours = {name for path, _, name in stats.stats if path.endswith("history_ppo.py")}
    assert {"run", "update", "collect", "learn", "save"} <= ours
