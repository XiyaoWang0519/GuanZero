"""CPU-testable fixture, summaries, and CLI fail-closed behavior; no CUDA claims."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from bench.paged_attention import (
    check_outputs, dense_inputs, fp64_sample, gather_sdpa, make_fixture,
    source_facts, strict_fp32, summarize,
)


def test_benchmark_fixture_preserves_strided_layout_and_independent_pages():
    q, kv, layout = make_fixture(5, 8, 16, 27, 133, device="cpu")
    assert not q.is_contiguous()
    live_ids = []
    for row, end in zip(layout.table, layout.ends):
        live_ids.extend(row[:(int(end) + 63) // 64].tolist())
    assert len(live_ids) == len(set(live_ids))
    assert 0 not in live_ids
    assert not kv[:64].any()
    with torch.no_grad():
        dense = gather_sdpa(q, kv, layout, dense_inputs(layout))
        samples = fp64_sample(q, kv, layout)
    for b, expected in samples.items():
        torch.testing.assert_close(dense[b, :, :expected.shape[1]].double(), expected,
                                   rtol=2e-5, atol=2e-6)
    counts = layout.ends - layout.starts
    candidate = dense.clone()
    for b, count in enumerate(counts.tolist()):
        candidate[b, :, count:] = 0
    result = check_outputs(candidate, dense, q, kv, layout)
    assert result["passed"] and result["max_absolute_difference"] == 0


def test_benchmark_gate_rejects_wrong_values_and_nan():
    q, kv, layout = make_fixture(2, 2, 16, 3, 65, device="cpu")
    with torch.no_grad():
        dense = gather_sdpa(q, kv, layout, dense_inputs(layout))
    for error in (1.0, torch.nan):
        bad = dense.clone()
        bad[0, 0, 0, 0] += error
        with pytest.raises(AssertionError):
            check_outputs(bad, dense, q, kv, layout)


def test_summary_retains_slowdowns_instead_of_clipping_to_gain():
    trials = [dict(backend=b, wall_ms=ms) for b, ms in
              [("torch", 1.0), ("triton", 2.0), ("triton", 2.2), ("torch", 1.2)]]
    result = summarize(trials)
    assert result["ratio_of_median_rates"] == pytest.approx(1.1 / 2.1)
    assert "component-only" in result["interpretation"]
    with pytest.raises(ValueError):
        summarize([])
    with pytest.raises(ValueError):
        summarize(trials[:1])
    with pytest.raises(ValueError):
        summarize([dict(backend=b, wall_ms=0) for b in ("torch", "triton")])


def test_precision_scope_restores_flags_after_error():
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    with pytest.raises(RuntimeError):
        with strict_fp32():
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            raise RuntimeError("test")
    assert before == (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)


def test_source_fingerprints_are_portable_and_json_serializable():
    facts = source_facts()
    json.dumps(facts, allow_nan=False)
    assert len(facts["source_sha256"]["train/history_paged_attention_kernel.py"]) == 64
    assert all(not Path(name).is_absolute() for name in facts["source_sha256"])


@pytest.mark.parametrize("module", ["bench.paged_attention", "bench.paged_attention_ppo"])
def test_cli_help_and_cpu_fail_closed_without_training_imports(module, tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", module, "--help"], cwd=root,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0 and "--output" in result.stdout
    target = tmp_path / "result"
    args = [sys.executable, "-m", module, "--device", "cpu", "--output", str(target)]
    if module.endswith("_ppo"):
        args += ["--backend", "triton"]
    result = subprocess.run(args, cwd=root, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0 and "CUDA" in result.stderr
    assert not target.exists()


def reports():
    return [dict(status="passed", backend=b, case="paged", config={"seed": 7},
                 environment={"gpu": "synthetic fixture"}, source={"sha": "same"}, warmup_updates=24,
                 summary=dict(decisions_per_wall_second=rate, learned_rows=20, max_prefix=512))
            for b, rate in [("torch", 100), ("triton", 90), ("triton", 92), ("torch", 102)]]


def test_ppo_summary_reports_negative_results():
    from bench.paged_attention_results import summarize_reports
    result = summarize_reports(reports())
    assert result["cases"]["paged"]["ratio_of_median_rates"] == pytest.approx(91/101)
    assert result["cases"]["paged"]["throughput_change_percent"] < 0
    assert "no confidence interval" in result["uncertainty"]


@pytest.mark.parametrize("fault", ["failed", "incomplete", "wrong_order", "config", "source", "environment", "nan"])
def test_ppo_summary_rejects_unmatched_or_incomplete_trials(fault):
    from bench.paged_attention_results import summarize_reports
    data = reports()
    if fault == "failed":
        data[0]["status"] = "failed"
    elif fault == "incomplete":
        data.pop()
    elif fault == "wrong_order":
        data[0], data[1] = data[1], data[0]
    elif fault in ("config", "source", "environment"):
        data[1][fault] = {"different": True}
    elif fault == "nan":
        data[1]["summary"]["decisions_per_wall_second"] = float("nan")
    with pytest.raises(ValueError):
        summarize_reports(data)
