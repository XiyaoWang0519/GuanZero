"""CPU layout/algorithm checks plus CUDA-only execution gates for the prototype.

CPU checks exercise the reference and the online recurrence, NOT Triton code
execution or compilation. CUDA cases must pass separately on the target GPU.
"""
import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from train.history_paged_attention import (
    PagedAttentionLayout, attention_reference, dense_inputs,
    experimental_paged_attention, paged_attention, require_triton, resolve_backend,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_num_threads(previous)
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def case(ends=(1, 65, 133), counts=(1, 17, 27), *, heads=2, depth=16,
         page=64, width=None, device="cpu", seed=71, scale=1.0):
    width = max(max(counts), 1) if width is None else width
    span = max(1, max(ends))
    columns = (span + page - 1) // page
    needed = [(end + page - 1) // page for end in ends]
    pages = 1 + sum(needed) + 3
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.arange(1, pages))
    table = np.full((len(ends), columns), -1, np.int64)
    offset = 0
    for i, n in enumerate(needed):
        table[i, :n] = ids[offset:offset+n]
        offset += n
    generator = torch.Generator().manual_seed(seed)
    kv = torch.randn(pages * page, 2, heads, depth, generator=generator) * scale
    kv[:page].zero_()
    # Match the noncontiguous view produced by a fused QKV projection.
    qkv = torch.randn(len(ends), width, 3, heads, depth, generator=generator) * scale
    q = qkv[:, :, 0].transpose(1, 2).to(device)
    kv = kv.to(device)
    table = torch.from_numpy(table).to(device)
    end = torch.tensor(ends, dtype=torch.int64, device=device)
    start = end - torch.tensor(counts, dtype=torch.int64, device=device)
    layout = PagedAttentionLayout.from_tensors(
        table, start, end, page_tokens=page, span=span,
        query_width=width, pool_rows=kv.shape[0])
    return q, kv, layout


def independent_oracle(q, kv, layout):
    """Per-stream FP64 dense attention, with independent index construction."""
    result = torch.zeros_like(q, dtype=torch.float64)
    for b in range(q.shape[0]):
        start, end = int(layout.starts[b]), int(layout.ends[b])
        if start == end:
            continue
        locations = [int(layout.table[b, n // layout.page_tokens]) * layout.page_tokens
                     + n % layout.page_tokens for n in range(end)]
        history = kv[locations].double()
        for t in range(end - start):
            keys = history[:start+t+1, 0].permute(1, 0, 2)
            values = history[:start+t+1, 1].permute(1, 0, 2)
            score = (keys * q[b, :, t].double()[:, None]).sum(-1) * q.shape[-1] ** -0.5
            result[b, :, t] = (score.softmax(-1)[..., None] * values).sum(1)
    return result


def online_reference(q, kv, layout, block_n=64):
    """CPU model of the kernel's online softmax; not GPU kernel execution."""
    output = torch.zeros_like(q)
    for b in range(q.shape[0]):
        start, end = int(layout.starts[b]), int(layout.ends[b])
        count = end - start
        if not count:
            continue
        query = q[b, :, :count]
        maximum = torch.full((q.shape[1], count), -torch.inf, dtype=q.dtype)
        denominator = torch.zeros_like(maximum)
        numerator = torch.zeros_like(query)
        for begin in range(0, end, block_n):
            n = torch.arange(begin, min(begin + block_n, end))
            rows = layout.table[b, n // layout.page_tokens] * layout.page_tokens + n % layout.page_tokens
            keys, values = kv[rows].unbind(1)
            scores = query @ keys.permute(1, 2, 0) * q.shape[-1] ** -0.5
            scores.masked_fill_(n[None, None] > start + torch.arange(count)[None, :, None], -torch.inf)
            next_maximum = torch.maximum(maximum, scores.max(-1).values)
            correction = (maximum - next_maximum).exp()
            probability = (scores - next_maximum[..., None]).exp()
            numerator = numerator * correction[..., None] + probability @ values.permute(1, 0, 2)
            denominator = denominator * correction + probability.sum(-1)
            maximum = next_maximum
        output[b, :, :count] = numerator / denominator[..., None]
    return output


@pytest.mark.parametrize("depth", [16, 32, 64, 128])
@pytest.mark.parametrize("page", [8, 64])
@pytest.mark.parametrize("counts", [(1, 1, 1), (1, 17, 27), (0, 0, 0)])
def test_reference_and_online_recurrence(depth, page, counts):
    q, kv, layout = case(depth=depth, page=page, counts=counts)
    with torch.no_grad():
        actual = attention_reference(q, kv, layout)
        expected = independent_oracle(q, kv, layout)
        online = online_reference(q, kv, layout)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(online.double(), expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("end", [0, 1, 63, 64, 65, 127, 128, 129, 1537, 4097])
def test_boundaries_and_long_context(end):
    q, kv, layout = case(ends=(end,), counts=(min(end, 27),))
    with torch.no_grad():
        expected = independent_oracle(q, kv, layout)
        actual = online_reference(q, kv, layout)
    torch.testing.assert_close(actual.double(), expected, rtol=2e-5, atol=2e-6)


def test_prefix_causality_and_stream_isolation():
    q, kv, layout = case(ends=(129, 133), counts=(27, 27))
    with torch.no_grad():
        before = attention_reference(q, kv, layout)
        changed = kv.clone()
        # Alter another match completely, plus future tokens in the first match.
        for b, first in [(1, 0), (0, int(layout.starts[0]) + 1)]:
            n = torch.arange(first, int(layout.ends[b]))
            rows = layout.table[b, n // layout.page_tokens] * layout.page_tokens + n % layout.page_tokens
            changed[rows] = changed[rows] * 100 + 100
        after = attention_reference(q, changed, layout)
    assert torch.equal(before[0, :, 0], after[0, :, 0])
    assert not torch.equal(before[1], after[1])


def test_poisoned_unused_storage_and_query_padding():
    q, kv, layout = case()
    with torch.no_grad():
        expected = attention_reference(q, kv, layout)
        rows, _ = dense_inputs(layout)
        live = torch.unique(rows)
        poisoned = torch.full_like(kv, torch.nan)
        poisoned[live] = kv[live]
        for b in range(q.shape[0]):
            q[b, :, int(layout.ends[b] - layout.starts[b]):] = torch.nan
        actual = attention_reference(q, poisoned, layout)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("fault", ["zero_page", "negative_page", "out_of_bounds", "negative_start",
                                   "end_before_start", "too_long", "query_overflow", "wrong_dtype"])
def test_reject_invalid_metadata(fault):
    q, kv, layout = case()
    table, starts, ends = (t.clone() for t in (layout.table, layout.starts, layout.ends))
    if fault == "zero_page": table[0, 0] = 0
    elif fault == "negative_page": table[0, 0] = -1
    elif fault == "out_of_bounds": table[0, 0] = kv.shape[0] // layout.page_tokens
    elif fault == "negative_start": starts[0] = -1
    elif fault == "end_before_start": ends[2] = starts[2] - 1
    elif fault == "too_long": ends[2] = layout.span + 1
    elif fault == "query_overflow": starts[2] = 0
    elif fault == "wrong_dtype": table = table.int()
    with pytest.raises(ValueError):
        PagedAttentionLayout.from_tensors(
            table, starts, ends, page_tokens=layout.page_tokens, span=layout.span,
            query_width=layout.query_width, pool_rows=kv.shape[0])


@pytest.mark.parametrize("field,value", [("span", 0), ("page_tokens", -1), ("query_width", True),
                                          ("span", 2**31), ("pool_rows", 65)])
def test_reject_invalid_layout_dimensions(field, value):
    q, kv, layout = case()
    values = dict(page_tokens=64, span=layout.span, query_width=layout.query_width, pool_rows=kv.shape[0])
    values[field] = value
    with pytest.raises(ValueError):
        PagedAttentionLayout.from_tensors(layout.table, layout.starts, layout.ends, **values)


def test_backend_scope_is_explicit_nested_and_exception_safe():
    assert resolve_backend(None) == "torch"
    with experimental_paged_attention("triton"):
        assert resolve_backend(None) == "triton"
        assert resolve_backend("torch") == "torch"
        with pytest.raises(RuntimeError):
            with experimental_paged_attention("torch"):
                assert resolve_backend(None) == "torch"
                raise RuntimeError("interrupted")
        assert resolve_backend(None) == "triton"
    assert resolve_backend(None) == "torch"
    with pytest.raises(ValueError):
        with experimental_paged_attention("auto"):
            pass
    assert resolve_backend(None) == "torch"


def test_fail_closed_without_backward_or_cpu_fallback():
    q, kv, layout = case()
    with pytest.raises(RuntimeError, match="rollout-only"):
        paged_attention(q, kv, layout)
    with torch.no_grad(), pytest.raises(RuntimeError, match="NVIDIA CUDA"):
        paged_attention(q, kv, layout)
    with pytest.raises(ValueError, match="FP32"):
        require_triton(torch.device("cuda"), torch.float16, 16)
    with pytest.raises(ValueError, match="depth"):
        require_triton(torch.device("cuda"), torch.float32, 8)


def test_kernel_source_requests_ieee_products():
    # Static guardrail only: this does not compile/execute Triton.
    path = Path(__file__).resolve().parents[1] / "train/history_paged_attention_kernel.py"
    tree = ast.parse(path.read_text())
    dots = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == "dot"]
    assert len(dots) == 2
    assert all(any(k.arg == "input_precision" and isinstance(k.value, ast.Constant)
                   and k.value.value == "ieee" for k in n.keywords) for n in dots)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs NVIDIA CUDA and Triton execution")
@pytest.mark.parametrize("depth", [16, 32, 64, 128])
@pytest.mark.parametrize("width,end", [(1, 1), (1, 1537), (27, 65), (27, 1537), (128, 4097)])
@pytest.mark.parametrize("scale", [1.0, 6.0])
def test_cuda_kernel_matches_fp64_and_original_gather(depth, width, end, scale):
    pytest.importorskip("triton")
    q, kv, layout = case(ends=(end, 65, 0), counts=(min(width, end), min(width, 65), 0),
                         width=width, depth=depth, device="cuda", scale=scale)
    # Quantize the same inputs once, then compute the FP64 oracle from those bits.
    cpu_layout = PagedAttentionLayout.from_tensors(
        layout.table.cpu(), layout.starts.cpu(), layout.ends.cpu(), page_tokens=layout.page_tokens,
        span=layout.span, query_width=layout.query_width, pool_rows=layout.pool_rows)
    with torch.no_grad():
        actual = paged_attention(q, kv, layout)
        reference = attention_reference(q, kv, layout)
        oracle = independent_oracle(q.cpu(), kv.cpu(), cpu_layout)
    assert torch.isfinite(actual).all()
    if scale == 1.0:
        torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(actual.cpu().double(), oracle, rtol=1e-4, atol=1e-5)
    else:
        # At high scale FP32 rounding of large logits puts the original SDPA path itself
        # 3e-5..4e-4 from FP64 on an L40S (Oct 18), above any fixed 1e-5. Gate against
        # FP64 instead: the kernel may not be worse than SDPA by more than a small factor.
        kernel_error = (actual.cpu().double() - oracle).abs().max().item()
        sdpa_error = (reference.cpu().double() - oracle).abs().max().item()
        assert kernel_error <= 4.0 * sdpa_error + 1e-5, (kernel_error, sdpa_error)
    # Repeatability and no writes to the shared pool.
    before = kv.clone()
    with torch.no_grad():
        again = paged_attention(q, kv, layout)
    assert torch.equal(actual, again)
    assert torch.equal(before, kv)
