"""CUDA-only paged-attention component gate and alternating benchmark.

This is gather-plus-attention, NOT complete PPO throughput. Dense row indices
and masks are prepared outside timing (once per append in the existing cache),
so the baseline does not pay their cost again for each layer. No production
checkpoint, job, environment, or global precision setting is changed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import time
from typing import Callable, Iterator

import numpy as np
import torch
from torch.nn import functional as F

from train.history_paged_attention import PagedAttentionLayout, dense_inputs, paged_attention, require_triton


@contextmanager
def strict_fp32() -> Iterator[None]:
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn


def source_facts() -> dict:
    root = Path(__file__).resolve().parents[1]
    paths = ("train/history_paged_attention.py", "train/history_paged_attention_kernel.py",
             "train/history_paged_cache.py", "bench/paged_attention.py", "bench/paged_attention_ppo.py")
    digests = {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
               for p in paths if (root / p).is_file()}
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return dict(revision=revision, dirty=dirty, source_sha256=digests)


def environment(device: torch.device) -> dict:
    properties = torch.cuda.get_device_properties(device)
    # No usernames, hostnames, paths, credentials, or device UUIDs in the report.
    return dict(python=platform.python_version(), torch=torch.__version__,
                cuda=torch.version.cuda, triton=importlib.metadata.version("triton"),
                device=str(device), gpu=properties.name,
                compute_capability=[properties.major, properties.minor],
                total_memory_bytes=properties.total_memory,
                visible_devices=torch.cuda.device_count(), tf32=False,
                shared_gpu_load="not controlled by this benchmark")


def make_fixture(batch: int, heads: int, depth: int, width: int, context: int, *,
                 device: str | torch.device, seed: int = 20261018):
    """Independent pages, ragged contexts, and the cache's strided QKV view."""
    if min(batch, heads, depth, width, context) < 1:
        raise ValueError("positive fixture dimensions required")
    rng = np.random.default_rng(seed)
    page = 64
    ends = rng.integers(max(1, context // 2), context + 1, size=batch, dtype=np.int64)
    ends[0] = context
    counts = np.minimum(ends, rng.integers(1, width + 1, size=batch, dtype=np.int64))
    counts[0] = min(width, context)
    starts = ends - counts
    span = (context + page - 1) // page * page
    page_counts = (ends + page - 1) // page
    pages = int(page_counts.sum()) + 1
    ids = rng.permutation(np.arange(1, pages, dtype=np.int64))
    table = np.zeros((batch, span // page), np.int64)
    offset = 0
    for i, count in enumerate(page_counts.tolist()):
        table[i, :count] = ids[offset:offset + count]
        offset += count
    generator = torch.Generator().manual_seed(seed)
    # Transfer backing QKV first so the CUDA query keeps the real strided layout.
    qkv = torch.randn(batch, width, 3, heads, depth, generator=generator).to(device)
    q = qkv[:, :, 0].transpose(1, 2)
    kv = torch.randn(pages * page, 2, heads, depth, generator=generator).to(device)
    kv[:page].zero_()
    td, sd, ed = (torch.from_numpy(a).to(device) for a in (table, starts, ends))
    layout = PagedAttentionLayout.from_host(
        table, starts, ends, td, sd, ed, page_tokens=page, span=span,
        query_width=width, pool_rows=kv.shape[0])
    return q, kv, layout


def gather_sdpa(q, kv, layout, prepared):
    """Exact original per-layer eager path, with no extra padding-zero kernel."""
    rows, mask = prepared
    b, h, _, d = q.shape
    packed = torch.index_select(kv, 0, rows).view(
        b, layout.span, 2, h, d).permute(2, 0, 3, 1, 4)
    return F.scaled_dot_product_attention(q, packed[0], packed[1],
                                          attn_mask=mask, dropout_p=0.0)


def fp64_sample(q, kv, layout, batch_samples: int = 3):
    """Independent CPU FP64 oracle on up to three streams; not a timed path."""
    query, pool = q.detach().cpu().double(), kv.detach().cpu().double()
    table, starts, ends = (t.cpu() for t in (layout.table, layout.starts, layout.ends))
    samples = {}
    for b in range(min(batch_samples, q.shape[0])):
        start, end = int(starts[b]), int(ends[b])
        count = end - start
        if not count:
            continue
        locations = [int(table[b, n // layout.page_tokens]) * layout.page_tokens
                     + n % layout.page_tokens for n in range(end)]
        values = pool[locations]
        keys = values[:, 0].permute(1, 2, 0)
        history = values[:, 1].permute(1, 0, 2)
        scores = query[b, :, :count] @ keys * q.shape[-1] ** -0.5
        mask = torch.arange(end)[None, :] > (start + torch.arange(count))[:, None]
        scores.masked_fill_(mask[None], -torch.inf)
        samples[b] = scores.softmax(-1) @ history
    return samples


def check_outputs(actual, baseline, q, kv, layout) -> dict:
    """Gate all valid outputs plus an independent FP64 sample before timing."""
    actual_cpu, baseline_cpu = actual.cpu(), baseline.cpu()
    counts = (layout.ends - layout.starts).cpu().tolist()
    maximum = 0.0
    for b, count in enumerate(counts):
        a, r = actual_cpu[b, :, :count], baseline_cpu[b, :, :count]
        torch.testing.assert_close(a, r, rtol=1e-4, atol=1e-5)
        if count:
            maximum = max(maximum, (a - r).abs().max().item())
        assert torch.isfinite(actual_cpu[b]).all(), "nonfinite prototype output"
        assert not torch.count_nonzero(actual_cpu[b, :, count:]), "nonzero unused query output"
    fp64_maximum = 0.0
    samples = fp64_sample(q, kv, layout)
    for b, reference in samples.items():
        a = actual_cpu[b, :, :reference.shape[1]].double()
        torch.testing.assert_close(a, reference, rtol=1e-4, atol=1e-5)
        fp64_maximum = max(fp64_maximum, (a - reference).abs().max().item())
    return dict(passed=True, max_absolute_difference=maximum,
                fp64_sample_max_absolute_difference=fp64_maximum,
                fp64_sample_streams=len(samples), output_rtol=1e-4, output_atol=1e-5,
                scope="synthetic attention outputs; not trained-policy logp/action acceptance")


def measure(call: Callable, device: torch.device, iterations: int) -> dict:
    torch.cuda.synchronize(device)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start.record()
    for _ in range(iterations):
        output = call()
        del output
    end.record()
    end.synchronize()
    wall_ms = 1000 * (time.perf_counter() - wall_start) / iterations
    return dict(cuda_elapsed_ms=start.elapsed_time(end) / iterations,
                wall_ms=wall_ms, iterations=iterations)


def extra_allocated(call: Callable, device: torch.device) -> int:
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = call()
    torch.cuda.synchronize(device)
    extra = torch.cuda.max_memory_allocated(device) - before
    del output
    return extra


def summarize(trials: list[dict]) -> dict:
    if not trials or {t["backend"] for t in trials} != {"torch", "triton"}:
        raise ValueError("both backends and nonempty trials required")
    medians = {b: statistics.median(t["wall_ms"] for t in trials if t["backend"] == b)
               for b in ("torch", "triton")}
    if any(not math.isfinite(v) or v <= 0 for v in medians.values()):
        raise ValueError("finite positive timing required")
    return dict(median_wall_ms=medians,
                ratio_of_median_rates=medians["torch"] / medians["triton"],
                interpretation="component-only; no inferred complete-PPO speedup")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch", type=int, default=61)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--contexts", type=int, nargs="+", default=[65, 1537, 4097])
    p.add_argument("--chunks", type=int, nargs="+", default=[1, 27, 128])
    p.add_argument("--blocks", type=int, default=2, help="number of complete ABBA blocks")
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261018)
    return p


def main() -> None:
    p = parser()
    args = p.parse_args()
    if min(args.batch, args.heads, args.depth, args.blocks, args.iterations, args.warmup,
           *args.contexts, *args.chunks) < 1:
        p.error("all benchmark dimensions/counts must be positive")
    if args.output.exists():
        p.error("choose a fresh output file; existing results are never overwritten")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        p.error("CUDA-only benchmark: request a GPU allocation, not a login node")
    require_triton(device, torch.float32, args.depth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve before the run, leaving failure evidence without overwriting a result.
    result = dict(schema=1, status="started", scope="synthetic gather-plus-attention only",
                  environment=environment(device), source=source_facts(), cases=[],
                  precision="FP32 IEEE dots; no reduced-precision probability conversion",
                  exclusions=["projections", "KV writes", "once-per-append metadata/mask creation",
                              "final memory gather", "PPO", "checkpointing"],
                  jit="each exact fixture warmed before timing; compilation excluded",
                  timer_note="CUDA elapsed events may include host submission gaps and shared-GPU contention")
    with args.output.open("x") as out:
        try:
            with torch.cuda.device(device), strict_fp32(), torch.no_grad():
                for context in args.contexts:
                    for width in args.chunks:
                        q, kv, layout = make_fixture(args.batch, args.heads, args.depth, width,
                                                     context, device=device, seed=args.seed)
                        prepared = dense_inputs(layout)
                        calls = {"torch": lambda: gather_sdpa(q, kv, layout, prepared),
                                 "triton": lambda: paged_attention(q, kv, layout)}
                        entry = dict(batch=args.batch, heads=args.heads, depth=args.depth,
                                     query_width=width, longest_context=context, span=layout.span,
                                     q_stride=list(q.stride()), trials=[])
                        result["cases"].append(entry)
                        entry["gate"] = check_outputs(calls["triton"](), calls["torch"](), q, kv, layout)
                        for _ in range(args.warmup):
                            for call in calls.values():
                                call()
                        for block in range(args.blocks):
                            for order, backend in enumerate(("torch", "triton", "triton", "torch")):
                                entry["trials"].append(dict(block=block, order=order, backend=backend,
                                                            **measure(calls[backend], device, args.iterations)))
                        entry["summary"] = summarize(entry["trials"])
                        entry["additional_peak_allocated_bytes"] = {
                            b: extra_allocated(call, device) for b, call in calls.items()}
                        print(json.dumps({k: entry[k] for k in ("longest_context", "query_width", "summary")}),
                              flush=True)
                        del calls, prepared, q, kv, layout
            result["status"] = "passed"
        except Exception as error:
            result["status"] = "failed"
            result["error_type"] = type(error).__name__
            raise
        finally:
            out.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
