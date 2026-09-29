"""Allocator cache trim (``rollout_trim_cuda_cache``).

``torch.cuda.empty_cache()`` after collect and after learn returns only free
cached blocks to the driver: allocator timing, no numeric change. On CPU the
trim is a no-op, so a run with it on must equal a run with it off bit for bit
(rows, choices, metrics, weights, sampler). The CUDA variant checks the same on
the device and that reserved memory does not grow across a trim.
"""
import numpy as np
import pytest
import torch

from train.history_ppo import (HistoryPPOConfig, HistoryTrainer, build_parser, config_from_args,
                               parse_resume_overrides)

TRIM_KEYS = {"rollout_trim_cuda_cache", "cuda_trim", "cuda_trim_seconds"}
TIMING_KEYS = {"collection_phase_seconds", "collection_profile_synchronized",
               "collection_group_phase_seconds", "collection_policy_call_rows",
               "decisions_per_sec", "collect_seconds", "learn_seconds",
               "learn_decisions_per_sec", "learn_exposures_per_sec",
               "learner_collect_decisions_per_sec", "elapsed_seconds",
               # device memory readings are allocator state, which the trim changes
               "cuda_reserved_bytes", "cuda_peak_reserved_bytes",
               "cuda_inactive_split_peak_bytes", "cuda_allocation_retries"} | TRIM_KEYS
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


def run_pair(tmp_path, device, overrides_on, overrides_off, updates=3, **extra):
    """Resume one checkpoint twice (snapshot seats at once), trim on vs off."""
    start = HistoryTrainer(HistoryPPOConfig(**base(**extra)), tmp_path / "start", device=device)
    for _ in range(3):
        start.update()
    checkpoint = start.save()
    trainers, lines = {}, {}
    for name, overrides in (("on", overrides_on), ("off", overrides_off)):
        trainer = HistoryTrainer(HistoryPPOConfig(updates=3 + updates), tmp_path / name,
                                 device=device, resume=checkpoint,
                                 resume_overrides=parse_resume_overrides(overrides))
        trainer.collector.choice_log = []
        lines[name] = [trainer.update() for _ in range(updates)]
        trainers[name] = trainer
    return trainers, lines


def assert_identical(trainers, lines):
    for on, off in zip(lines["on"], lines["off"]):
        assert on["rollout_trim_cuda_cache"] and not off["rollout_trim_cuda_cache"]
        assert set(on["cuda_trim"]) >= POINTS and off["cuda_trim"] == {}
        assert off["cuda_trim_seconds"] == 0.0 and on["cuda_trim_seconds"] >= 0.0
        assert ({k: v for k, v in on.items() if k not in TIMING_KEYS}
                == {k: v for k, v in off.items() if k not in TIMING_KEYS})
    a, b = trainers["on"], trainers["off"]
    assert len(a.collector.choice_log) == len(b.collector.choice_log) > 0
    for x, y in zip(a.collector.choice_log, b.collector.choice_log):
        assert np.array_equal(x, y)
    rows_a, rows_b = a.buffer.compact(), b.buffer.compact()
    assert rows_a.keys() == rows_b.keys()
    for key in rows_a:
        assert np.array_equal(rows_a[key], rows_b[key]), key
    for model in ("actor", "critic"):
        for p, q in zip(getattr(a, model).parameters(), getattr(b, model).parameters()):
            assert torch.equal(p, q)
    for opt in ("actor_optimizer", "critic_optimizer"):
        sa, sb = getattr(a, opt).state_dict()["state"], getattr(b, opt).state_dict()["state"]
        for key in sa:
            for name in sa[key]:
                assert torch.equal(sa[key][name], sb[key][name])
    assert torch.equal(a.generator.get_state(), b.generator.get_state())
    assert a.progress == {**b.progress, "elapsed_seconds": a.progress["elapsed_seconds"]}


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
    trainers, lines = run_pair(tmp_path, "cpu",
                               [f"batch_snapshot_policies={merge}", "rollout_trim_cuda_cache=true"],
                               [f"batch_snapshot_policies={merge}", "rollout_trim_cuda_cache=false"])
    assert_identical(trainers, lines)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA allocator cache trim")
def test_trim_leaves_training_bitwise_unchanged_on_cuda(tmp_path, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    trainers, lines = run_pair(tmp_path, "cuda",
                               ["batch_snapshot_policies=true", "rollout_trim_cuda_cache=true"],
                               ["batch_snapshot_policies=true", "rollout_trim_cuda_cache=false"],
                               rollout_device="cuda")
    assert_identical(trainers, lines)
    for line in lines["on"]:
        for record in line["cuda_trim"].values():
            assert record["reserved_before"] > 0
            assert record["reserved_after"] <= record["reserved_before"]
            assert record["allocated_after"] == record["allocated_before"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA private graphs")
def test_switching_the_arm_on_releases_snapshot_graph_pools(tmp_path, monkeypatch):
    # Off: snapshot seats capture private graphs. On: they are cleared before
    # collection and the trim right after returns their pools.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    pytest.importorskip("triton")
    config = HistoryPPOConfig(**base(rollout_device="cuda", rollout_private_graphs=True,
                                     rollout_batched_attention=True, num_envs=8,
                                     batch_snapshot_policies_schedule="4:off,2:on",
                                     rollout_trim_cuda_cache=True))
    trainer = HistoryTrainer(config, tmp_path / "switch", device="cuda")
    lines = [trainer.update() for _ in range(6)]
    had_snapshot_graphs = any(set(line["private_graphs"]) - {"0"} for line in lines[:4])
    released = [line["cuda_trim"].get("graphs_released") for line in lines[4:]]
    if had_snapshot_graphs:
        assert released[0] is not None
        assert released[0]["reserved_after"] <= released[0]["reserved_before"]
    assert not set(trainer.collector.decision_graphs) - {0}
    assert released[1] is None      # nothing left to release on the next update
