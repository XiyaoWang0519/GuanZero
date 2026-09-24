"""Measure history collation avoided by belief controls (synthetic binary data).

    .venv/bin/python bench/belief_collate.py --batch-size 128 --repeats 12

This times CPU collation only, not model training or GPU transfers. It uses
the same sampled decisions for every arm and reports the tensor payload size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.belief_memory import attach_memory, memory_collate  # noqa: E402
from train.logs import TOKEN_DIM  # noqa: E402


def tensor_bytes(value) -> int:
    if isinstance(value, dict):
        return sum(tensor_bytes(item) for item in value.values())
    return value.numel() * value.element_size()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if min(vars(args).values()) <= 0:
        parser.error("batch size and repeats must be positive")
    torch.set_num_threads(1)
    rng = np.random.default_rng(31)
    records = []
    for match in range(16):
        for index in range(9):
            tokens = rng.integers(0, 2, (160, TOKEN_DIM), dtype=np.uint8)
            tokens[:, :4] = np.eye(4, dtype=np.uint8)[np.arange(160) % 4]
            records.append(dict(group=str(match), round_index=index, tokens=tokens,
                                obs=rng.integers(0, 2, (8, 1849), dtype=np.uint8),
                                hidden=rng.integers(0, 3, (8, 3, 54), dtype=np.uint8),
                                seat=np.arange(8) % 4, prefix=np.arange(8) * 20))
    attach_memory(records, 8)
    items = [(records[int(r)], int(i)) for r, i in zip(
        rng.integers(len(records), size=args.batch_size),
        rng.integers(8, size=args.batch_size))]
    variants = {"previous_all_inputs": {},
                "memory": {"include_history": False},
                "memory_masked": {"include_history": False, "include_memory": False}}
    timings = {name: [] for name in variants}
    payload = {}
    names = list(variants)
    for repeat in range(args.repeats + 2):
        for name in names if repeat % 2 else names[::-1]:
            start = time.perf_counter()
            batch = memory_collate(items, "cpu", 8, **variants[name])
            elapsed = time.perf_counter() - start
            payload[name] = tensor_bytes(batch)
            if repeat >= 2:
                timings[name].append(elapsed)
            del batch
    medians = {name: float(np.median(values) * 1000) for name, values in timings.items()}
    print(json.dumps({"config": vars(args), "scope": "CPU collation only; synthetic binary data",
                      "median_ms": medians, "tensor_bytes": payload,
                      "speedup": {name: medians[names[0]] / medians[name]
                                  for name in names[1:]}}, indent=2))


if __name__ == "__main__":
    main()
