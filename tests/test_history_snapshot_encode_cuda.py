"""CUDA gate for the merged snapshot public encode (``batch_snapshot_encode``).

Skipped without CUDA; run on a pod before any GPU use of the flag:

    PYTHONPATH=python:oracle:. python -m pytest -q -s tests/test_history_snapshot_encode_cuda.py

Production architecture (width 128, 4 layers, 8 heads), ~10 identities x ~8
streams, prefixes up to ~2,000 tokens. Reports the maximum absolute difference
of memory/KV and of snapshot log-probabilities between the merged encode and
each identity's own ``cache.encode`` (expected <= ~1e-5, tier 2), with the
Triton write-and-pack kernel and with the eager copies, and runs the collector
under the production CUDA options with the flag off and on.
"""
import pytest
import torch

from test_history_snapshot_batch import assert_same_collection, strict_fp32  # noqa: F401
from test_history_snapshot_encode import (ATOL, compare_caches, production_size_check,
                                          run_pair)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA merged encode gate")


@pytest.mark.parametrize("triton", [True, False])
def test_production_size_gate(triton):
    if triton:
        pytest.importorskip("triton")
    worst_memory, worst_logp, worst_reference = production_size_check("cuda", 2000, triton)
    assert worst_memory < ATOL and worst_logp < ATOL and worst_reference < 2e-5


def test_production_cuda_options_with_merged_encode(monkeypatch):
    # Private graphs and the Triton cache for the learner; snapshot seats use
    # the merged head and the merged encode (Triton write-and-pack).
    pytest.importorskip("triton")
    options = dict(private_graphs=True, triton_cache=True, triton_min_batch=4)
    ((plain, _), (fast, _)), (plain_tables, fast_tables) = run_pair(
        monkeypatch, "cuda", steps=60, actor_flags=dict(
            causal_sdpa=True, batched_private_attention=True, wide_private_projection=True),
        **options)
    assert_same_collection(plain, fast)
    worst = max(float((a - b).abs().max()) for a, b in zip(plain_tables, fast_tables))
    worst_cache = compare_caches(plain, fast)
    print(f"collector cuda production options: snapshot log-prob {worst:.3e}, "
          f"cache K/V/memory {worst_cache:.3e}, {fast.snapshot_encoder_metrics()}")
    assert worst < ATOL and worst_cache < ATOL
    assert fast.snapshot_encoder_metrics()["triton_launches"] > 0
    assert set(fast.decision_graphs) <= {0}
