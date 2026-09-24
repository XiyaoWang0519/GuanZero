"""infra.cpu_budget: usable CPUs from the cpuset and the cgroup quota, never
from OMP_NUM_THREADS, and the thread plan for concurrent PPO arms."""
import json
import os

import pytest

from infra import cpu_budget


@pytest.fixture
def affinity(monkeypatch):
    def set_cpus(n: int) -> None:
        monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(n)), raising=False)
    return set_cpus


def test_cgroup_v2_quota_below_the_cpuset_wins(tmp_path, affinity, monkeypatch):
    affinity(256)
    (tmp_path / "cpu.max").write_text("3200000 100000\n")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    assert cpu_budget.cgroup_quota(tmp_path) == 32.0
    assert cpu_budget.usable_cpus(tmp_path) == 32


def test_unlimited_or_missing_quota_leaves_the_cpuset(tmp_path, affinity, monkeypatch):
    affinity(64)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    assert cpu_budget.usable_cpus(tmp_path) == 64
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert cpu_budget.cgroup_quota(tmp_path) is None
    assert cpu_budget.usable_cpus(tmp_path) == 64


def test_cgroup_v1_and_fractional_quotas(tmp_path, affinity):
    affinity(64)
    v1 = tmp_path / "cpu"
    v1.mkdir()
    (v1 / "cpu.cfs_quota_us").write_text("250000\n")
    (v1 / "cpu.cfs_period_us").write_text("100000\n")
    assert cpu_budget.usable_cpus(tmp_path) == 2
    (v1 / "cpu.cfs_quota_us").write_text("50000\n")
    assert cpu_budget.usable_cpus(tmp_path) == 1
    (v1 / "cpu.cfs_quota_us").write_text("-1\n")
    assert cpu_budget.usable_cpus(tmp_path) == 64


def test_thread_plan():
    assert cpu_budget.thread_plan(64, 3) == {"cpus": 64, "arms": 3, "actors": 0,
                                             "engine_threads": 20, "torch_threads": 4}
    # Actors: 3 trainers + 12 actor processes keep a core each; 12 roll out.
    plan = cpu_budget.thread_plan(64, 3, actors=4)
    assert plan["engine_threads"] == 4 and plan["torch_threads"] == 1
    assert cpu_budget.thread_plan(2, 3)["engine_threads"] == 1
    with pytest.raises(ValueError):
        cpu_budget.thread_plan(8, 0)


def test_cli_field_and_facts(tmp_path, affinity, capsys):
    affinity(64)
    root = tmp_path / "cgroup"
    root.mkdir()
    (root / "cpu.max").write_text("2400000 100000\n")
    (root / "cpu.stat").write_text("usage_usec 10\nnr_throttled 3\nthrottled_usec 7\n")
    cpu_budget.main(["--arms", "2", "--field", "engine_threads", "--cgroup-root", str(root)])
    assert capsys.readouterr().out.strip() == "11"
    facts = tmp_path / "run" / "host-facts.json"
    cpu_budget.main(["--facts", str(facts), "--cgroup-root", str(root)])
    data = json.loads(facts.read_text())
    assert data["usable_cpus"] == 24 and data["affinity_cpus"] == 64
    assert data["cgroup_cpu_stat"]["nr_throttled"] == "3"


def test_nested_v2_group_from_proc_self_cgroup(tmp_path, affinity):
    """With the host's cgroup namespace the quota sits in a nested group."""
    affinity(256)
    group = tmp_path / "system.slice" / "docker-abc.scope"
    group.mkdir(parents=True)
    (tmp_path / "cgroup.controllers").write_text("cpu\n")
    (group / "cpu.max").write_text("3200000 100000\n")
    self_cgroup = tmp_path / "self-cgroup"
    self_cgroup.write_text("0::/system.slice/docker-abc.scope\n")
    assert cpu_budget.usable_cpus(tmp_path, self_cgroup) == 32


def test_engine_threads_are_capped():
    assert cpu_budget.thread_plan(256, 1)["engine_threads"] == cpu_budget.MAX_ENGINE_THREADS
