"""Paged append attention: Triton kernel (CPU interpreter / CUDA) against today's
gather + masked SDPA reference, and the reference against a float64 loop.

Triton picks the interpreter when it is first imported, and a normal test run
imports it compiled (other modules use it). The ``interpret`` cases therefore
run in a child pytest started with ``TRITON_INTERPRET=1``
(``test_interpreter_cases_in_subprocess``); under ``TRITON_INTERPRET=1`` they
run in-process.
"""
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import train.history_paged_attention as paged
from train.history_paged_attention import (
    LaunchConfig, kernel_eligible, launch_config, paged_append_attention,
    paged_append_attention_reference, paged_append_attention_triton,
)

PAGE = 64
ROOT = Path(__file__).resolve().parents[1]


def make_case(ends, counts, *, heads=2, depth=16, page=PAGE, spare_blocks=0, scale=1.0,
              seed=0, index_dtype=torch.int64):
    """A shuffled page pool holding each stream's keys < end; page 0 is zero and every
    other pool row (past a stream's end, or in an unreferenced page) is NaN, so any
    read the reference does not make poisons the output."""
    g = torch.Generator().manual_seed(seed)
    ends = torch.tensor(ends, dtype=torch.int64)
    counts = torch.tensor(counts, dtype=torch.int64)
    assert bool((counts >= 1).all() and (counts <= ends).all())
    B, W = len(ends), int(counts.max())
    need = (ends + page - 1) // page
    total = 1 + int(need.sum()) + 3
    kv = torch.randn(total * page, 2, heads, depth, generator=g)
    kv[:page] = 0
    ids = torch.randperm(total - 1, generator=g) + 1
    table = torch.zeros(B, int(need.max()) + spare_blocks, dtype=torch.int64)
    poison = torch.ones(total * page, dtype=torch.bool)
    poison[:page] = False
    used = 0
    for b in range(B):
        for block in range(int(need[b])):
            page_id = int(ids[used])
            used += 1
            table[b, block] = page_id
            valid = min(page, int(ends[b]) - block * page)
            poison[page_id * page:page_id * page + valid] = False
    kv[poison] = float("nan")
    # Production layout: q is a strided [B, H, W, D] view of the fused projection.
    qkv = torch.randn(B, W, 3, heads, depth, generator=g) * scale
    q = qkv[:, :, 0].transpose(1, 2)
    starts = ends - counts
    return q, kv, table.to(index_dtype), starts.to(index_dtype), ends.to(index_dtype)


def dense_loop(q, kv, table, starts, ends, page=PAGE):
    """Float64 attention written from the definition, one stream and row at a time."""
    B, H, W, D = q.shape
    out = torch.zeros(B, H, W, D, dtype=torch.float64)
    for b in range(B):
        end, start = int(ends[b]), int(starts[b])
        keys = torch.arange(end)
        rows = table[b].long()[keys // page] * page + keys % page
        k = kv[rows, 0].double().transpose(0, 1)       # [H, end, D]
        v = kv[rows, 1].double().transpose(0, 1)
        for i in range(W):
            n = min(end, start + i + 1)
            logits = q[b, :, i].double()[:, None] @ k[:, :n].transpose(1, 2) / math.sqrt(D)
            out[b, :, i] = (torch.softmax(logits, -1) @ v[:, :n])[:, 0]
    return out


def assert_close(actual, expected, tolerance=1e-5):
    assert actual.shape == expected.shape
    assert bool(torch.isfinite(actual).all()), "non-finite output"
    error = float((actual.double() - expected.double()).abs().max())
    bound = tolerance * max(1.0, float(expected.abs().max()))
    assert error <= bound, f"max |error| {error:.3e} > {bound:.3e}"
    return error



@pytest.fixture(params=["interpret", "cuda"])
def device(request, monkeypatch):
    if paged.triton is None:
        pytest.skip("Triton is not installed")
    if request.param == "cuda":
        if paged.INTERPRETED or not torch.cuda.is_available():
            pytest.skip("compiled CUDA kernel acceptance")
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
        monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
        return torch.device("cuda")
    if not paged.INTERPRETED:
        pytest.skip("interpreter cases run in test_interpreter_cases_in_subprocess")
    return torch.device("cpu")


def check_kernel(case, device, **launch):
    q, kv, table, starts, ends = (t.to(device) for t in case)
    reference = paged_append_attention_reference(q, kv, table, starts, ends, PAGE)
    out = paged_append_attention_triton(q, kv, table, starts, ends, PAGE, **launch)
    error = assert_close(out, reference)          # every row, query padding included
    assert_close(out.cpu(), dense_loop(*(t.cpu() for t in case)))
    return out, error


def test_interpreter_cases_in_subprocess():
    if paged.triton is None:
        pytest.skip("Triton is not installed")
    if paged.INTERPRETED:
        pytest.skip("this process runs the interpreter cases itself")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYTEST_XDIST", "PYTEST_CURRENT_TEST"))}
    env["TRITON_INTERPRET"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(Path(__file__)),
         "-k", "interpret and not subprocess"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    summary = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    assert result.returncode == 0, result.stdout[-6000:] + result.stderr[-3000:]
    assert " passed" in summary and "skipped" not in summary, summary


# ---- reference and dispatch (no Triton needed) ----------------------------------

@pytest.mark.parametrize("ends,counts", [
    ([1, 5, 32], [1, 5, 3]),                     # span 32 is shorter than one page
    ([1, 2, 64, 65, 300], [1, 2, 3, 1, 5]),
    ([128, 129, 200], [128, 1, 64]),
])
def test_reference_is_masked_attention_over_the_page_table(ends, counts):
    case = make_case(ends, counts, spare_blocks=1)
    q, kv, table, starts, ends_t = case
    expected = dense_loop(*case)
    assert_close(paged_append_attention_reference(q, kv, table, starts, ends_t, PAGE), expected)
    # _append's power-of-two span, with the table _page_table builds for it.
    span = max(32, 1 << (int(ends_t.max()) - 1).bit_length())
    blocks = -(-span // PAGE)
    table = torch.nn.functional.pad(table, (0, max(blocks - table.shape[1], 0)))[:, :blocks]
    narrow = paged_append_attention_reference(q, kv, table, starts, ends_t, PAGE, span=span)
    assert_close(narrow, expected)


def test_compiled_mode_cpu_dispatch_is_the_reference():
    if paged.INTERPRETED:
        pytest.skip("CPU tensors run the interpreted kernel in this process")
    q, kv, table, starts, ends = make_case([5, 70, 130], [5, 2, 3])
    assert not kernel_eligible(q, kv, table, starts, ends, PAGE)
    actual = paged_append_attention(q, kv, table, starts, ends, PAGE, span=192)
    expected = paged_append_attention_reference(q, kv, table, starts, ends, PAGE, span=192)
    assert torch.equal(actual, expected)


def test_eligibility_limits(monkeypatch):
    monkeypatch.setattr(paged, "KERNEL_AVAILABLE", True)
    monkeypatch.setattr(paged, "INTERPRETED", True)
    q, kv, table, starts, ends = make_case([5, 70], [5, 2])
    assert kernel_eligible(q, kv, table, starts, ends, PAGE)
    assert kernel_eligible(q, kv, table.int(), starts.int(), ends.int(), PAGE)
    assert not kernel_eligible(q.double(), kv.double(), table, starts, ends, PAGE)
    assert not kernel_eligible(q, kv, table, starts.float(), ends, PAGE)
    assert not kernel_eligible(q, kv, table, starts, ends, 48)            # not a power of two
    assert not kernel_eligible(q[..., :8], kv[..., :8], table, starts, ends, PAGE)  # D < 16
    monkeypatch.setattr(paged, "INTERPRETED", False)
    assert not kernel_eligible(q, kv, table, starts, ends, PAGE)          # CPU, compiled mode
    monkeypatch.setattr(paged, "KERNEL_AVAILABLE", False)
    monkeypatch.setattr(paged, "INTERPRETED", True)
    assert not kernel_eligible(q, kv, table, starts, ends, PAGE)          # mixed modes


def test_inference_only():
    q, kv, table, starts, ends = make_case([5, 70], [5, 2])
    q = q.detach().requires_grad_()
    with pytest.raises(RuntimeError, match="inference-only"):
        paged_append_attention(q, kv, table, starts, ends, PAGE)
    with torch.no_grad():
        paged_append_attention(q, kv, table, starts, ends, PAGE)


def test_shape_validation():
    q, kv, table, starts, ends = make_case([5, 70], [5, 2])
    with pytest.raises(ValueError):
        paged_append_attention_reference(q, kv, table[:1], starts, ends, PAGE)
    with pytest.raises(ValueError):
        paged_append_attention_reference(q, kv[:, :, :1], table, starts, ends, PAGE)
    with pytest.raises(ValueError):
        paged_append_attention_reference(q, kv, table, starts[:1], ends, PAGE)


def test_launch_config():
    # Production-sized batches fill the GPU without splitting keys.
    assert launch_config(128, 8, 8, 47, sm_count=142) == LaunchConfig(16, 1, 47, 4)
    assert launch_config(64, 8, 128, 47, sm_count=142).block_q == 64
    assert launch_config(64, 8, 20, 47, sm_count=142).block_q == 32
    # Few streams: split keys, at least MIN_PAGES_PER_SPLIT pages each, no empty tail.
    small = launch_config(2, 8, 1, 47, sm_count=142)
    assert small.splits > 1 and small.pages_per_split >= paged.MIN_PAGES_PER_SPLIT
    assert (small.splits - 1) * small.pages_per_split < 47 <= small.splits * small.pages_per_split
    assert launch_config(2, 8, 1, 3, sm_count=142).splits == 1
    # The interpreter (no SM count) never splits unless asked; forced splits are trimmed.
    assert launch_config(1, 1, 1, 47).splits == 1
    forced = launch_config(1, 1, 1, 10, splits=4)
    assert (forced.splits, forced.pages_per_split) == (4, 3)
    assert launch_config(1, 1, 1, 10, splits=8) == LaunchConfig(16, 5, 2, 4)
    assert launch_config(1, 1, 1, 2, splits=8).splits == 2
    with pytest.raises(ValueError):
        launch_config(1, 1, 1, 2, block_q=24)


# ---- kernel against the reference -----------------------------------------------

@pytest.mark.parametrize("splits", [None, 3])
def test_decode_rows(device, splits):
    ends = [1, 2, 63, 64, 65, 128, 129, 2999]   # BOS-only streams, page boundaries, ~3k keys
    check_kernel(make_case(ends, [1] * len(ends), heads=3, seed=1), device, splits=splits)


@pytest.mark.parametrize("splits", [1, 2, 5])
def test_steady_state_append(device, splits):
    g = torch.Generator().manual_seed(2)
    counts = torch.randint(1, 9, (6,), generator=g).tolist()
    counts[0], counts[1] = 8, max(counts[1], 3)
    ends = [max(c, int(e)) for c, e in zip(counts, torch.randint(1, 3001, (6,), generator=g))]
    ends[1] = 63 + counts[1]                     # the append crosses a page boundary
    check_kernel(make_case(ends, counts, seed=3, spare_blocks=2), device, splits=splits)


@pytest.mark.parametrize("splits", [1, 2])
def test_prefill_chunk(device, splits):
    # W = 128: a stream starting at BOS, chunk-aligned and unaligned starts.
    check_kernel(make_case([128, 256, 1037, 3000], [128, 128, 37, 1], seed=4), device,
                 splits=splits)


@pytest.mark.parametrize("block_q", [16, 32, 64])
def test_query_tiles(device, block_q):
    check_kernel(make_case([40, 100, 700], [40, 7, 33], seed=5), device, block_q=block_q, splits=2)


def test_one_page_per_split_and_empty_splits(device):
    # 20 table columns (4 spare): short streams leave most splits empty.
    case = make_case([1, 10, 64, 1000], [1, 3, 2, 4], seed=6, spare_blocks=4)
    check_kernel(case, device, splits=20)        # one page per split
    check_kernel(case, device, splits=7)         # three pages per split


def test_large_logits(device):
    check_kernel(make_case([3, 90, 500], [3, 6, 2], scale=8.0, seed=7), device, splits=2)


def test_allow_tf32_spelling(device, monkeypatch):
    # Triton < 3.0 has no input_precision; the kernel then passes allow_tf32=False.
    monkeypatch.setattr(paged, "_ieee_dot", lambda: False)
    check_kernel(make_case([3, 90, 130], [3, 2, 5], seed=11), device, splits=2)


def test_wider_heads_and_int32_indices(device):
    case = make_case([33, 140], [4, 2], heads=1, depth=32, seed=8, index_dtype=torch.int32)
    check_kernel(case, device)


def test_strided_start_and_end_views(device):
    """starts/ends given as columns of one [B, 2] tensor (stride 2) still match
    the reference: the kernel entry makes them contiguous."""
    q, kv, table, starts, ends = make_case([5, 70, 130, 300], [5, 2, 3, 4], seed=9)
    both = torch.stack([starts, ends], 1)
    case = (q, kv, table, both[:, 0], both[:, 1])
    assert case[3].stride(0) == 2
    check_kernel(case, device)


def test_output_layout_and_launch_hints(device):
    q, kv, table, starts, ends = (t.to(device) for t in make_case([70, 2000], [3, 5], seed=9,
                                                                    spare_blocks=20))
    out = paged_append_attention_triton(q, kv, table, starts, ends, PAGE)
    assert out.shape == q.shape and out.dtype == torch.float32
    assert out.transpose(1, 2).is_contiguous()   # finish()'s reshape is a view
    hinted = paged_append_attention_triton(q, kv, table, starts, ends, PAGE, max_end=2000,
                                           splits=4)
    assert_close(hinted, out)


def test_dispatch_runs_the_kernel(device, monkeypatch):
    q, kv, table, starts, ends = (t.to(device) for t in make_case([5, 300], [5, 2], seed=10,
                                                                    spare_blocks=3))
    assert kernel_eligible(q, kv, table, starts, ends, PAGE)
    calls = []
    kernel = paged.paged_append_attention_triton
    monkeypatch.setattr(paged, "paged_append_attention_triton",
                        lambda *a, **k: calls.append(k) or kernel(*a, **k))
    out = paged_append_attention(q, kv, table, starts, ends, PAGE, span=512)
    assert len(calls) == 1
    assert_close(out, paged_append_attention_reference(q, kv, table, starts, ends, PAGE, span=512))
