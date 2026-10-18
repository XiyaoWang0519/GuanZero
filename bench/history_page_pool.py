"""Compare page reservation against the previous per-entry allocation loop.

This isolates allocator work from encoding and PPO. Cold batches, warm calls,
released-page reuse and restoration after a learner-pool reset are reported
separately. Timings are not complete training throughput measurements.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from train.history_model import HistoryActor, HistoryPolicyConfig, PublicStream
from train.history_paged_cache import KVPagePool, PagedHistoryCache


class MeasuredPool(KVPagePool):
    def __init__(self, *args, **kwargs):
        self.copied_bytes = 0
        super().__init__(*args, **kwargs)

    def _grow(self, pages):
        self.copied_bytes += self.reserved_bytes
        super()._grow(pages)


class PerEntryCache(PagedHistoryCache):
    """The pre-reservation allocation order, with the same tensor allocator."""

    def _allocate_pages(self, entries, ends):
        P = self.pool.page_tokens
        for entry, end in zip(entries, ends.tolist()):
            missing = -(-end // P) - len(entry.pages)
            if missing > 0:
                entry.pages.extend(self.pool.allocate(missing))


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def trial(actor, args, reserved):
    pool = MeasuredPool.for_actor(actor, page_tokens=args.page_tokens, pages=args.initial_pages)
    cache = (PagedHistoryCache if reserved else PerEntryCache)(actor, pool)
    ends = np.linspace(1, args.max_length, args.batch, dtype=np.int64)
    records = {}
    for phase in ("cold", "warm", "released", "restored"):
        if phase in ("released", "restored"):
            cache.clear()
            if phase == "restored":
                pool.reset()
        if phase != "warm":
            entries = [cache._entry((i, i), PublicStream(i)) for i in range(args.batch)]
        repeats = args.warm_calls if phase == "warm" else 1
        grows, copied = pool.grows, pool.copied_bytes
        synchronize(pool.device)
        started = time.perf_counter()
        for _ in range(repeats):
            cache._allocate_pages(entries, ends)
        synchronize(pool.device)
        records[phase] = dict(seconds_per_call=(time.perf_counter() - started) / repeats,
                              grows=pool.grows - grows, copied_bytes=pool.copied_bytes - copied,
                              reserved_bytes=pool.reserved_bytes, used_pages=pool.used_pages)
        pages = [page for entry in entries for page in entry.pages]
        assert len(pages) == len(set(pages)) == pool.used_pages
        assert 0 not in pages and not set(pages) & set(pool.free)
        assert [len(entry.pages) for entry in entries] == (
            -(-ends // args.page_tokens)).tolist()
    cache.clear()
    return records


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(1)
    actor = HistoryActor(HistoryPolicyConfig(width=args.width, layers=args.layers,
                                            heads=args.heads)).to(device).eval()
    rows = {"per_entry": [], "reserved": []}
    # Untimed warm-up of both code paths, then alternate which arm runs first.
    for reserved in (False, True):
        trial(actor, args, reserved)
    for index in range(args.repeats):
        for reserved in ((False, True) if index % 2 == 0 else (True, False)):
            rows["reserved" if reserved else "per_entry"].append(trial(actor, args, reserved))
    summary = {}
    for phase in rows["reserved"][0]:
        measured = {arm: statistics.median(row[phase]["seconds_per_call"] for row in samples)
                    for arm, samples in rows.items()}
        summary[phase] = dict(measured, speedup=measured["per_entry"] / measured["reserved"])
    return dict(scope="page allocation only; no encoder, PPO or playing-strength measurement",
                torch=torch.__version__, device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                settings={k: v for k, v in vars(args).items() if k != "output"},
                summary=summary, trials=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--page-tokens", type=int, default=64)
    parser.add_argument("--initial-pages", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warm-calls", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if min(args.batch, args.max_length, args.repeats, args.warm_calls, args.torch_threads) < 1:
        parser.error("batch, lengths, repeat counts and thread count must be positive")
    result = run(args)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"scope": result["scope"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
