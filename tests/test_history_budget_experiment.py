"""Fixed-budget screen: exact update budgets, resumable failures, no local research."""
import json
from pathlib import Path

import numpy as np
import pytest

from infra import history_budget_experiment as experiment
from infra.history_artifacts import sha256
from train.history_ppo import HistoryPPOConfig, HistoryTrainer

T7_PRESENT = (experiment.T7 / f"seed-{experiment.T7_SEEDS[0]}/run-manifest.json").exists()


def tiny(**overrides) -> dict:
    values = dict(width=32, layers=1, heads=4, num_envs=4, steps_per_update=16, seed=5,
                  epochs=1, minibatch_matches=2, response_mode="auxiliary", updates=4)
    values.update(overrides)
    return values


@pytest.fixture
def cpu_pod(monkeypatch, tmp_path):
    """Run the pod-side loop on CPU with a tiny collection shape."""
    monkeypatch.setenv("POD_DEADLINE_EPOCH", "9999999999")
    monkeypatch.setattr(experiment, "DECISIONS_PER_UPDATE", 64)
    monkeypatch.setattr(experiment, "SNAPSHOT_UPDATES", 2)
    monkeypatch.setattr(experiment, "LATEST_UPDATES", 1)

    def build(job, output, payload_dir, target=None):
        if job["kind"] == "fresh":
            goal = target or job["target_updates"]
            return HistoryTrainer(HistoryPPOConfig(**dict(job["config"], updates=goal)),
                                  output / "train"), goal
        trainer = HistoryTrainer(HistoryPPOConfig(), output / "train",
                                 resume=payload_dir / job["resume"],
                                 allow_source_change=bool(job.get("allow_source_change")))
        goal = target or trainer.progress["updates"] + job["extra_updates"]
        trainer.config.updates = goal
        return trainer, goal

    monkeypatch.setattr(experiment, "build_trainer", build)
    return tmp_path


def test_fresh_job_stops_at_exactly_the_update_budget(cpu_pod):
    job = dict(kind="fresh", config=tiny(rollout_epsilon=0.02), target_updates=4)
    receipt = experiment.run_job(job, cpu_pod / "arm", cpu_pod)
    assert receipt["complete"] and receipt["progress"]["updates"] == 4
    assert receipt["progress"]["decisions"] == 4 * 64
    assert receipt["final_sha256"] == sha256(cpu_pod / "arm/final.pt")
    names = sorted(p.name for p in (cpu_pod / "arm").glob("update-*.pt"))
    assert names == ["update-000002.pt", "update-000004.pt"]
    lines = [json.loads(l) for l in (cpu_pod / "arm/train/metrics.jsonl").read_text().splitlines()]
    assert [l["update"] for l in lines] == [1, 2, 3, 4]
    assert all(l["rollout_epsilon_pick_fraction"] is not None for l in lines)
    resolved = json.loads((cpu_pod / "arm/resolved-config.json").read_text())
    assert resolved["rollout_epsilon"] == 0.02 and resolved["entropy"] == 0.01


def test_resume_job_adds_the_declared_updates(cpu_pod):
    experiment.run_job(dict(kind="fresh", config=tiny(), target_updates=2), cpu_pod / "a", cpu_pod)
    job = dict(kind="resume", resume="a/train/latest.pt", extra_updates=2)
    receipt = experiment.run_job(job, cpu_pod / "b", cpu_pod)
    assert receipt["complete"] and receipt["progress"]["updates"] == 4
    assert receipt["resume_count"] == 1
    assert not (cpu_pod / "b/initial.pt").exists()


def test_wrong_collection_shape_fails_and_leaves_a_resumable_latest(cpu_pod, monkeypatch):
    monkeypatch.setattr(experiment, "DECISIONS_PER_UPDATE", 65)
    with pytest.raises(RuntimeError, match="different number of decisions"):
        experiment.run_job(dict(kind="fresh", config=tiny(), target_updates=3), cpu_pod / "x", cpu_pod)
    assert (cpu_pod / "x/train/latest.pt").exists()
    assert not (cpu_pod / "x/final.pt").exists()


def test_deadline_stops_before_the_budget_without_an_endpoint(cpu_pod, monkeypatch):
    monkeypatch.setenv("POD_DEADLINE_EPOCH", "1")
    with pytest.raises(RuntimeError, match="provider deadline"):
        experiment.run_job(dict(kind="fresh", config=tiny(), target_updates=3), cpu_pod / "d", cpu_pod)
    assert (cpu_pod / "d/train/latest.pt").exists() and not (cpu_pod / "d/final.pt").exists()


def test_pod_commands_refuse_local_research(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment.sys, "platform", "darwin")
    for call in (lambda: experiment.workload(tmp_path / "m.json", tmp_path),
                 lambda: experiment.preflight(tmp_path / "m.json", tmp_path / "c.json", tmp_path, 4)):
        with pytest.raises(RuntimeError, match="owned Linux pod"):
            call()


def test_paired_bootstrap_recovers_a_constant_shift():
    rng = np.random.default_rng(0)
    low = rng.normal(size=(3, 50)).tolist()
    high = (np.asarray(low) + 0.5).tolist()
    result = experiment.paired(high, low, rng, samples=200)
    assert result["mean_delta"] == pytest.approx(0.5)
    assert result["ci_95"] == pytest.approx([0.5, 0.5])
    assert result["seed_deltas"] == pytest.approx([0.5] * 3)


@pytest.mark.skipif(not T7_PRESENT, reason="T7 run kits are local artifacts")
def test_screen_arms_differ_only_by_their_declared_setting():
    configs = {name: dict(experiment.screen_config(experiment.SEEDS[0]), **override)
               for name, override in experiment.ARMS.items()}
    for name, config in configs.items():
        HistoryPPOConfig(**config)
        changed = {k for k in config if config[k] != configs["control"].get(k)}
        assert changed == set(experiment.ARMS[name])
    control = configs["control"]
    assert control["response_mode"] == "auxiliary" and control["epochs"] == 2
    assert control["entropy"] == 0.01 and control["updates"] == experiment.UPDATES
    assert control["num_envs"] * control["steps_per_update"] == experiment.DECISIONS_PER_UPDATE
    t7 = json.loads((experiment.T7 / "seed-2026092801/local/results/arm-B/resolved-config.json").read_text())
    same = {k for k in t7 if k not in ("seed", "updates", "checkpoint_updates")}
    assert all(control[k] == t7[k] for k in same)


def line(update, kl=0.01, clip=0.1):
    return dict(update=update, update_samples=100, policy_loss=0.0, value_loss=1.0, entropy=0.4,
                approx_kl=kl, clip_fraction=clip, actor_grad_norm=1.0, encoder_grad_norm=1.0,
                critic_grad_norm=1.0, rounds=update, mean_prefix=10, decisions_per_sec=1.0,
                learn_decisions_per_sec=1.0, cuda_peak_reserved_bytes=0, ratio_deviation=1.0,
                population={"decisions": {"0": 1}})


def test_gate_records_ratio_spikes_and_fails_only_on_bursts_or_real_damage():
    calm = [line(u) for u in range(1, 20)]
    assert experiment.gate(calm)["healthy"]
    one = calm + [line(20, kl=201.0)]
    check = experiment.gate(one)
    assert check["healthy"] and [s["update"] for s in check["ratio_spikes"]] == [20]
    assert experiment.gate(one + [line(u) for u in range(21, 40)])["healthy"]
    assert experiment.gate(one + [line(25, kl=0.7)])["healthy"]
    burst = one + [line(22, kl=0.7), line(24, kl=0.9)]
    assert not experiment.gate(burst)["healthy"]
    spread = calm + [line(20, kl=0.7), line(31, kl=0.7), line(42, kl=0.7)]
    assert experiment.gate(spread)["healthy"]
    assert not experiment.gate(calm + [line(20, clip=0.97)])["healthy"]
    broken = calm + [dict(line(20), entropy=float("nan"))]
    assert not experiment.gate(broken)["healthy"]
