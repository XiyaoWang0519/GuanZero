"""Allocator cache trim (``rollout_trim_cuda_cache``, ``rollout_trim_slack_mb``).

``torch.cuda.empty_cache()`` after collect and after learn returns only free
cached blocks to the driver: allocator timing, no numeric change. On CPU the
trim is a no-op, so a run with it on must equal a run with it off bit for bit
(rows, choices, metrics, weights, sampler). The CUDA variant checks the same on
the device and that reserved memory does not grow across a trim. The slack
gate only decides whether a trim runs; its decision is tested on CPU over a
faked allocator and, on CUDA, with the same bitwise comparison.
"""
import os

import numpy as np
import pytest
import torch

from train.history_ppo import (HistoryPPOConfig, HistoryTrainer, build_parser, config_from_args,
                               parse_resume_overrides)

# Deterministic cuBLAS for the CUDA variants; set before any CUDA work in this process.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

TRIM_KEYS = {"rollout_trim_cuda_cache", "cuda_trim", "cuda_trim_seconds"}
TIMING_KEYS = {"collection_phase_seconds", "collection_profile_synchronized",
               "collection_group_phase_seconds", "collection_policy_call_rows",
               "decisions_per_sec", "collect_seconds", "learn_seconds",
               "learn_decisions_per_sec", "learn_exposures_per_sec",
               "learner_collect_decisions_per_sec", "elapsed_seconds"}
# Process-wide CUDA allocator telemetry: torch.cuda.memory_* count every live
# tensor of the process (other trainers, the cuBLAS workspace, test fixtures),
# and reserved/split/retry counters are exactly what the trim changes. None of
# them is a training result; all zero on CPU.
MEMORY_KEYS = {"cuda_allocated_bytes", "cuda_peak_allocated_bytes", "cuda_reserved_bytes",
               "cuda_peak_reserved_bytes", "cuda_inactive_split_peak_bytes",
               "cuda_allocation_retries"}
EXCLUDED_KEYS = TIMING_KEYS | MEMORY_KEYS | TRIM_KEYS
POINTS = {"after_collect", "after_learn"}


@pytest.fixture(autouse=True)
def strict_fp32():
    previous = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled(),
                torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_num_threads(previous[0])
        torch.use_deterministic_algorithms(previous[1], warn_only=previous[2])
        torch.backends.cuda.matmul.allow_tf32 = previous[3]
        torch.backends.cudnn.allow_tf32 = previous[4]


def base(**extra):
    values = dict(width=16, layers=1, heads=4, num_envs=4, steps_per_update=30, epochs=1,
                  minibatch_matches=1, snapshot_updates=1, seed=5, causal_sdpa=True,
                  rollout_kv_cache=True, snapshot_probability=1.0)
    values.update(extra)
    return values


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    return value


def record(trainer, lines) -> dict:
    """Everything the comparison needs, copied off the device, so the trainer
    can be freed before the next one runs."""
    return dict(lines=lines, choices=[c.copy() for c in trainer.collector.choice_log],
                rows={k: np.array(v, copy=True) for k, v in trainer.buffer.compact().items()},
                actor=cpu_copy(trainer.actor.state_dict()),
                critic=cpu_copy(trainer.critic.state_dict()),
                actor_optimizer=cpu_copy(trainer.actor_optimizer.state_dict()["state"]),
                critic_optimizer=cpu_copy(trainer.critic_optimizer.state_dict()["state"]),
                generator=trainer.generator.get_state().clone(),
                progress=dict(trainer.progress))


def memory(device) -> str:
    if torch.device(device).type != "cuda":
        return "cpu"
    return (f"allocated {torch.cuda.memory_allocated() / 2**20:.1f} MiB, "
            f"reserved {torch.cuda.memory_reserved() / 2**20:.1f} MiB")


def run_pair(tmp_path, device, overrides_on, overrides_off, updates=3,
             order=("on", "off"), **extra):
    """Resume one checkpoint twice (snapshot seats at once), trim on vs off.
    Each run is recorded to host copies and freed before the next starts."""
    import gc
    start = HistoryTrainer(HistoryPPOConfig(**base(**extra)), tmp_path / "start", device=device)
    for _ in range(3):
        start.update()
    checkpoint = start.save()
    del start
    overrides = dict(on=overrides_on, off=overrides_off)
    records = {}
    for name in order:
        gc.collect()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print(f"\nrun {name} (order {order}): before construction {memory(device)}")
        trainer = HistoryTrainer(HistoryPPOConfig(updates=3 + updates), tmp_path / name,
                                 device=device, resume=checkpoint,
                                 resume_overrides=parse_resume_overrides(overrides[name]))
        trainer.collector.choice_log = []
        lines = [trainer.update() for _ in range(updates)]
        for line in lines:
            print(f"  {name} update {line['update']}: end allocated "
                  f"{line['cuda_allocated_bytes'] / 2**20:.1f} MiB, peak allocated "
                  f"{line['cuda_peak_allocated_bytes'] / 2**20:.1f} MiB, end reserved "
                  f"{line['cuda_reserved_bytes'] / 2**20:.1f} MiB")
        records[name] = record(trainer, lines)
        del trainer, lines
    return records


def assert_identical(records):
    on_lines, off_lines = records["on"]["lines"], records["off"]["lines"]
    assert len(on_lines) == len(off_lines) > 0
    for on, off in zip(on_lines, off_lines):
        assert on["rollout_trim_cuda_cache"] and not off["rollout_trim_cuda_cache"]
        assert set(on["cuda_trim"]) >= POINTS and off["cuda_trim"] == {}
        assert off["cuda_trim_seconds"] == 0.0 and on["cuda_trim_seconds"] >= 0.0
        assert set(on) == set(off)
        # Every other metric (losses, entropy, KL, counters, rewards, prefixes,
        # population, cache, graph and head metrics) must be equal.
        assert ({k: v for k, v in on.items() if k not in EXCLUDED_KEYS}
                == {k: v for k, v in off.items() if k not in EXCLUDED_KEYS})
    a, b = records["on"], records["off"]
    assert len(a["choices"]) == len(b["choices"]) > 0
    for x, y in zip(a["choices"], b["choices"]):
        assert np.array_equal(x, y)
    assert a["rows"].keys() == b["rows"].keys()
    for key in a["rows"]:
        assert np.array_equal(a["rows"][key], b["rows"][key]), key
    for model in ("actor", "critic"):
        assert a[model].keys() == b[model].keys()
        for name in a[model]:
            assert torch.equal(a[model][name], b[model][name]), (model, name)
    for opt in ("actor_optimizer", "critic_optimizer"):
        assert a[opt].keys() == b[opt].keys()
        for key in a[opt]:
            for name in a[opt][key]:
                assert torch.equal(torch.as_tensor(a[opt][key][name]),
                                   torch.as_tensor(b[opt][key][name])), (opt, key, name)
    assert torch.equal(a["generator"], b["generator"])
    assert a["progress"] == {**b["progress"], "elapsed_seconds": a["progress"]["elapsed_seconds"]}


def test_default_follows_the_merged_snapshot_arm(tmp_path):
    assert HistoryPPOConfig().rollout_trim_cuda_cache is None
    parser = build_parser()
    for raw, value in (("auto", None), ("true", True), ("false", False)):
        args = parser.parse_args(["--output", "x", "--rollout-trim-cuda-cache", raw])
        assert config_from_args(args).rollout_trim_cuda_cache is value
    assert config_from_args(parser.parse_args(["--output", "x"])).rollout_trim_cuda_cache is None
    trainer = HistoryTrainer(HistoryPPOConfig(**base(batch_snapshot_policies_schedule="1:off,1:on")),
                             tmp_path / "auto")
    flags = [trainer.update()["rollout_trim_cuda_cache"] for _ in range(2)]
    assert flags == [False, True]
    trainer.config.rollout_trim_cuda_cache = False
    assert trainer.update()["cuda_trim"] == {}


def test_resume_set_switches_and_records_the_trim(tmp_path):
    assert parse_resume_overrides(["rollout_trim_cuda_cache=true"]) == {
        "rollout_trim_cuda_cache": True}
    assert parse_resume_overrides(["rollout_trim_cuda_cache=auto"]) == {
        "rollout_trim_cuda_cache": None}
    with pytest.raises(ValueError):
        parse_resume_overrides(["rollout_trim_cuda_cache=maybe"])
    trainer = HistoryTrainer(HistoryPPOConfig(**base()), tmp_path / "start")
    trainer.update()
    checkpoint = trainer.save()
    resumed = HistoryTrainer(HistoryPPOConfig(updates=3), tmp_path / "resumed", resume=checkpoint,
                             resume_overrides=parse_resume_overrides(
                                 ["batch_snapshot_policies=true",
                                  "rollout_trim_cuda_cache=true"]))
    assert resumed.config_changes[-1] == dict(
        at_update=1, changes={"batch_snapshot_policies": [False, True],
                              "rollout_trim_cuda_cache": [None, True]})
    line = resumed.update()
    assert line["rollout_trim_cuda_cache"] is True and set(line["cuda_trim"]) >= POINTS
    for record in line["cuda_trim"].values():   # CPU: a no-op with zero readings
        assert {k: v for k, v in record.items() if k != "seconds"} == dict(
            reserved_before=0, allocated_before=0, reserved_after=0, allocated_after=0)
    manifest = (tmp_path / "resumed" / "manifest.json").read_text()
    assert '"rollout_trim_cuda_cache": true' in manifest
    # A later resume of the saved config keeps it without recording a change.
    again = HistoryTrainer(HistoryPPOConfig(updates=4), tmp_path / "again",
                           resume=resumed.save(),
                           resume_overrides={"rollout_trim_cuda_cache": True})
    assert again.config_changes == resumed.config_changes


@pytest.mark.parametrize("merge", ["true", "false"])
def test_trim_leaves_training_bitwise_unchanged_on_cpu(tmp_path, merge):
    records = run_pair(tmp_path, "cpu",
                       [f"batch_snapshot_policies={merge}", "rollout_trim_cuda_cache=true"],
                       [f"batch_snapshot_policies={merge}", "rollout_trim_cuda_cache=false"])
    assert_identical(records)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocator cache trim")
@pytest.mark.parametrize("order", [("on", "off"), ("off", "on")], ids=["on-first", "off-first"])
def test_trim_leaves_training_bitwise_unchanged_on_cuda(tmp_path, order):
    records = run_pair(tmp_path, "cuda",
                       ["batch_snapshot_policies=true", "rollout_trim_cuda_cache=true"],
                       ["batch_snapshot_policies=true", "rollout_trim_cuda_cache=false"],
                       order=order, rollout_device="cuda")
    for line in records["on"]["lines"]:
        print(f"\nupdate {line['update']} trims (MiB): " + ", ".join(
            f"{point} reserved {r['reserved_before'] / 2**20:.0f}->{r['reserved_after'] / 2**20:.0f} "
            f"allocated {r['allocated_before'] / 2**20:.0f} in {r['seconds']:.4f}s"
            for point, r in line["cuda_trim"].items()))
    for name in ("on", "off"):
        print(f"{name}: " + ", ".join(
            f"u{l['update']} alloc {l['cuda_allocated_bytes']} peak {l['cuda_peak_allocated_bytes']}"
            for l in records[name]["lines"]))
    assert_identical(records)
    for line in records["on"]["lines"]:
        for record_ in line["cuda_trim"].values():
            assert record_["reserved_before"] > 0
            assert record_["reserved_after"] <= record_["reserved_before"]
            assert record_["allocated_after"] == record_["allocated_before"]


class FakeAllocator:
    """torch.cuda's memory readers, synchronize and empty_cache over fixed
    readings: empty_cache returns every cached free byte."""

    def __init__(self, monkeypatch, reserved_mb: int, allocated_mb: int, split_mb: int = 0):
        self.reserved, self.allocated = reserved_mb << 20, allocated_mb << 20
        self.split = split_mb << 20
        self.calls: list[str] = []
        monkeypatch.setattr(torch.cuda, "memory_stats", lambda device=None: {
            "inactive_split_bytes.all.current": self.split})
        monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: self.calls.append("sync"))
        monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device=None: self.reserved)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device=None: self.allocated)
        monkeypatch.setattr(torch.cuda, "empty_cache", self.empty_cache)

    def empty_cache(self):
        self.calls.append("empty_cache")
        self.reserved = self.allocated


@pytest.mark.parametrize("slack_mb, skipped", [(0, False), (39, False), (40, True), (4096, True)])
def test_slack_gate_skips_only_while_the_cached_free_bytes_fit(monkeypatch, slack_mb, skipped):
    """reserved 100 MiB, allocated 60 MiB: 40 MiB of cached free blocks. The trim
    is skipped iff that is at most the slack; 0 always trims (today's path)."""
    from types import SimpleNamespace
    fake = FakeAllocator(monkeypatch, 100, 60)
    owner = SimpleNamespace(rollout_device=torch.device("cuda"), device=torch.device("cpu"))
    trims: dict[str, dict] = {}
    HistoryTrainer.trim_cuda_cache(owner, "after_collect", trims, slack_mb)
    record = trims["after_collect"]
    assert record["seconds"] >= 0.0
    assert (record["reserved_before"], record["allocated_before"]) == (100 << 20, 60 << 20)
    assert record["allocated_after"] == 60 << 20
    if skipped:
        assert record["skipped"] is True and record["reserved_after"] == 100 << 20
        assert fake.calls == ["sync"] and fake.reserved == 100 << 20
    else:
        assert "skipped" not in record and record["reserved_after"] == 60 << 20
        assert fake.calls == ["sync", "empty_cache", "sync"]


@pytest.mark.parametrize("slack_mb, skipped", [(24, False), (25, True)])
def test_slack_gate_ignores_free_fragments_empty_cache_cannot_release(monkeypatch, slack_mb,
                                                                      skipped):
    """40 MiB cached free, 15 MiB of it fragments of partly used segments: only
    the 25 MiB empty_cache can return counts against the slack."""
    from types import SimpleNamespace
    FakeAllocator(monkeypatch, 100, 60, split_mb=15)
    owner = SimpleNamespace(rollout_device=torch.device("cuda"), device=torch.device("cpu"))
    trims: dict[str, dict] = {}
    HistoryTrainer.trim_cuda_cache(owner, "after_learn", trims, slack_mb)
    assert trims["after_learn"]["releasable_before"] == 25 << 20
    assert trims["after_learn"].get("skipped", False) is skipped


def test_slack_gates_after_collect_and_after_learn_but_never_graphs_released(tmp_path,
                                                                             monkeypatch):
    trainer = HistoryTrainer(HistoryPPOConfig(**base(batch_snapshot_policies=True,
                                                     rollout_trim_cuda_cache=True,
                                                     rollout_trim_slack_mb=512)),
                             tmp_path / "gated")
    calls, original = [], trainer.trim_cuda_cache

    def spy(point, trims, slack_mb=0):
        calls.append((point, slack_mb))
        original(point, trims, slack_mb)

    monkeypatch.setattr(trainer, "trim_cuda_cache", spy)
    monkeypatch.setattr(trainer.collector, "release_snapshot_graphs", lambda: 1)
    line = trainer.update()
    assert calls == [("graphs_released", 0), ("after_collect", 512), ("after_learn", 512)]
    assert set(line["cuda_trim"]) == {"graphs_released"} | POINTS
    for record_ in line["cuda_trim"].values():   # CPU: no device, nothing to gate
        assert "skipped" not in record_ and record_["reserved_before"] == 0
    manifest = (tmp_path / "gated" / "manifest.json").read_text()
    assert '"rollout_trim_slack_mb": 512' in manifest


def test_slack_defaults_off_and_is_a_resume_and_arm_setting(tmp_path):
    from bench.history_arms import SWITCHABLE
    from train.history_ppo import RESUME_OVERRIDES
    assert HistoryPPOConfig().rollout_trim_slack_mb == 0
    parser = build_parser()
    assert config_from_args(parser.parse_args(["--output", "x"])).rollout_trim_slack_mb == 0
    assert config_from_args(parser.parse_args(
        ["--output", "x", "--rollout-trim-slack-mb", "2048"])).rollout_trim_slack_mb == 2048
    with pytest.raises(ValueError):
        HistoryPPOConfig(rollout_trim_slack_mb=-1)
    assert "rollout_trim_slack_mb" in RESUME_OVERRIDES and "rollout_trim_slack_mb" in SWITCHABLE
    assert parse_resume_overrides(["rollout_trim_slack_mb=2048"]) == {
        "rollout_trim_slack_mb": 2048}
    trainer = HistoryTrainer(HistoryPPOConfig(**base()), tmp_path / "start")
    trainer.update()
    resumed = HistoryTrainer(HistoryPPOConfig(updates=2), tmp_path / "resumed",
                             resume=trainer.save(), resume_overrides=parse_resume_overrides(
                                 ["rollout_trim_cuda_cache=true", "rollout_trim_slack_mb=2048"]))
    assert resumed.config_changes[-1] == dict(
        at_update=1, changes={"rollout_trim_cuda_cache": [None, True],
                              "rollout_trim_slack_mb": [0, 2048]})


def test_slack_gated_trim_leaves_training_bitwise_unchanged_on_cpu(tmp_path):
    records = run_pair(tmp_path, "cpu",
                       ["batch_snapshot_policies=true", "rollout_trim_cuda_cache=true",
                        "rollout_trim_slack_mb=2048"],
                       ["batch_snapshot_policies=true", "rollout_trim_cuda_cache=false"])
    assert_identical(records)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocator cache trim")
@pytest.mark.parametrize("slack_mb", [1, 1 << 20], ids=["small-slack", "always-skip"])
def test_slack_gated_trim_leaves_training_bitwise_unchanged_on_cuda(tmp_path, slack_mb):
    records = run_pair(tmp_path, "cuda",
                       ["batch_snapshot_policies=true", "rollout_trim_cuda_cache=true",
                        f"rollout_trim_slack_mb={slack_mb}"],
                       ["batch_snapshot_policies=true", "rollout_trim_cuda_cache=false"],
                       rollout_device="cuda")
    for line in records["on"]["lines"]:
        print(f"\nupdate {line['update']} gated trims: " + ", ".join(
            f"{point} {'skipped' if r.get('skipped') else 'trimmed'} free "
            f"{(r['reserved_before'] - r['allocated_before']) / 2**20:.1f} MiB"
            for point, r in line["cuda_trim"].items()))
    assert_identical(records)
    for line in records["on"]["lines"]:
        for point in POINTS:
            record_ = line["cuda_trim"][point]
            free = record_["reserved_before"] - record_["allocated_before"]
            assert bool(record_.get("skipped")) == (free <= slack_mb << 20), point
            assert record_["reserved_after"] <= record_["reserved_before"]
            assert record_["allocated_after"] == record_["allocated_before"]
    if slack_mb == 1 << 20:
        assert all(line["cuda_trim"][point].get("skipped")
                   for line in records["on"]["lines"] for point in POINTS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA private graphs")
def test_switching_the_arm_on_releases_snapshot_graph_pools(tmp_path):
    # Off: snapshot seats capture private graphs. On: they are cleared before
    # collection and the trim right after returns their pools.
    config = HistoryPPOConfig(**base(rollout_device="cuda", rollout_private_graphs=True,
                                     rollout_batched_attention=True, num_envs=8))
    start = HistoryTrainer(config, tmp_path / "start", device="cuda")
    for _ in range(2):
        start.update()
    # A resume restarts the environments: every new match draws snapshot seats.
    trainer = HistoryTrainer(HistoryPPOConfig(updates=8), tmp_path / "switch", device="cuda",
                             resume=start.save(), resume_overrides=parse_resume_overrides(
                                 ["batch_snapshot_policies_schedule=4:off,2:on",
                                  "rollout_trim_cuda_cache=true"]))
    lines = [trainer.update() for _ in range(6)]
    had_snapshot_graphs = any(set(line["private_graphs"]) - {"0"} for line in lines[:4])
    released = [line["cuda_trim"].get("graphs_released") for line in lines[4:]]
    print(f"\nsnapshot private graphs before the switch: {had_snapshot_graphs}; "
          f"graphs_released trim: {released[0]}")
    if had_snapshot_graphs:
        assert released[0] is not None
        assert released[0]["reserved_after"] <= released[0]["reserved_before"]
    assert not set(trainer.collector.decision_graphs) - {0}
    assert released[1] is None      # nothing left to release on the next update
