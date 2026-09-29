"""Host-side microbenchmarks for the history trainer's per-row Python costs.

CPU only; run the same script on two source revisions and compare. Measures:

* ``from_streams``: ``StreamBatch.from_streams`` on 16 streams of 600 tokens
  (one learner minibatch of 16 matches);
* ``grad_norm``: ``train.history_ppo.grad_norm`` over the stream-encoder
  parameters of a width-128, 4-layer, 8-head actor (float-per-parameter host
  accumulation before, one device reduction after; on CPU there is no device
  synchronization to save, so this is Python overhead only);
* ``metadata``: ``TritonHistoryCache._metadata`` bookkeeping on 155 of 256 live
  entries with a different subset each call (the learner's pending set). CPU has
  no Triton path, so the eligibility gate is patched open, pinned memory and
  ``record_stream`` are skipped; no kernel is launched.

Usage: ``PYTHONPATH=python:oracle:. python bench/history_learn_host.py``
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from train.history_inference import Entry
from train.history_model import HistoryPolicyConfig, PublicStream, StreamBatch, fresh_player
from train.logs import TOKEN_DIM


def token(i: int) -> np.ndarray:
    value = np.zeros(TOKEN_DIM, np.uint8)
    value[i % 4] = 1
    value[4 + 7 + i % 90] = 1
    value[158 + 27 - i % 28] = 1
    return value


def best(fn, number: int, repeat: int = 7) -> float:
    """Best mean milliseconds per call over ``repeat`` runs of ``number`` calls."""
    fn()
    times = []
    for _ in range(repeat):
        begin = time.perf_counter()
        for _ in range(number):
            fn()
        times.append((time.perf_counter() - begin) / number)
    return min(times) * 1e3


def bench_from_streams() -> float:
    streams = []
    for s in range(16):
        stream = PublicStream(s)
        for i in range(600):
            stream.append_token(token(i + s), i // 40, i % 4)
        streams.append(stream)
    return best(lambda: StreamBatch.from_streams(streams, "cpu"), 50)


def bench_grad_norm() -> dict[str, float]:
    from train.history_ppo import grad_norm
    torch.manual_seed(0)
    actor, _ = fresh_player(HistoryPolicyConfig(width=128, layers=4, heads=8), 0)
    params = list(actor.stream.parameters())
    for p in params:
        p.grad = torch.randn_like(p)
    return dict(grad_norm_ms=best(lambda: grad_norm(params), 200),
                grad_norm_to_float_ms=best(lambda: float(grad_norm(params)), 200),
                tensors=len(params))


def bench_metadata() -> dict[str, float]:
    import train.history_triton_cache as module
    from train.history_model import HistoryActor
    actor = HistoryActor(HistoryPolicyConfig(width=128, layers=4, heads=8)).float()
    cache = module.TritonHistoryCache(actor, enabled=False)
    cache._supported_entries = lambda entries, counts: True
    original_empty, original_record = torch.empty, torch.Tensor.record_stream

    def empty(*args, pin_memory=False, **kwargs):
        return original_empty(*args, **kwargs)
    shape = (8, 64, 16)
    entries = [Entry(PublicStream(i), 0, 5, 64, [torch.zeros(shape) for _ in range(4)],
                     [torch.zeros(shape) for _ in range(4)], torch.zeros(64, 128))
               for i in range(256)]
    for i, entry in enumerate(entries):
        cache.entries[(i, i)] = entry
    rng = np.random.default_rng(0)
    subsets = [sorted(rng.choice(256, 155, replace=False).tolist()) for _ in range(64)]
    state = {"i": 0}

    def call():
        picked = [entries[j] for j in subsets[state["i"] % len(subsets)]]
        state["i"] += 1
        cache._metadata(picked, [1] * len(picked))
    module.torch.empty = empty
    torch.Tensor.record_stream = lambda self, stream: None
    try:
        ms = best(call, 64)
    finally:
        module.torch.empty = original_empty
        torch.Tensor.record_stream = original_record
    return dict(metadata_ms=ms, entries=155, live=256,
                copy_stats={k: v for k, v in sorted(cache.copy_stats.items())
                            if "upload" in k or "reuse" in k or "rebuild" in k})


def main() -> None:
    torch.set_num_threads(1)
    result = dict(from_streams_ms=bench_from_streams(), **bench_grad_norm(), **bench_metadata())
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
