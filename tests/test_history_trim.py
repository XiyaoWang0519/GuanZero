"""Allocator cache trim (``rollout_trim_cuda_cache``).

``torch.cuda.empty_cache()`` after collect and after learn returns only free
cached blocks to the driver: allocator timing, no numeric change. On CPU the
trim is a no-op, so a run with it on must equal a run with it off bit for bit
(rows, choices, metrics, weights, sampler). The CUDA variant checks the same on
the device and that reserved memory does not grow across a trim.
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
