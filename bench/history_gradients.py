"""Alternating benchmark of actor clipping plus encoder gradient diagnostics.

Uses production-shaped FP32 parameters with fixed gradients, checks exact
norm/clipped-gradient parity before timing, and measures both operations in
the same process. This is a component measurement, not PPO throughput.

Run: ``PYTHONPATH=python:. python -m bench.history_gradients --device cpu``.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import statistics
import time

import torch

from train.history_gradients import ActorGradientClipper
from train.history_model import HistoryPolicyConfig, fresh_player


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run(device: torch.device, width: int, layers: int, heads: int,
        iterations: int, repeats: int) -> dict:
    config = HistoryPolicyConfig(width=width, layers=layers, heads=heads,
                                 aux_heads="next,belief,outcome")
    actor, _ = fresh_player(config, 217)
    actor = actor.to(device)
    optimized = copy.deepcopy(actor)
    generator = torch.Generator(device=device).manual_seed(893)
    for original, candidate in zip(actor.parameters(), optimized.parameters()):
        original.grad = torch.randn(original.shape, device=device, generator=generator) * 0.001
        candidate.grad = original.grad.clone()
    parameters = tuple(actor.parameters())
    encoder_parameters = tuple(actor.stream.parameters())
    clip_optimized = ActorGradientClipper(optimized.parameters(), optimized.stream.parameters())
    # Keep gradients identical throughout the timing: the original unconditional
    # multiply still runs, with its coefficient clamped to one in both paths.
    max_norm = 1e10

    @torch.no_grad()
    def baseline():
        gradients = [p.grad.detach().float() for p in encoder_parameters if p.grad is not None]
        encoder = (torch.stack(torch._foreach_norm(gradients)) ** 2).unbind() if gradients else ()
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm), encoder

    def candidate():
        return clip_optimized(max_norm)

    original_norm, original_encoder = baseline()
    candidate_norm, candidate_encoder = candidate()
    torch.testing.assert_close(candidate_norm, original_norm, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(candidate_encoder), torch.stack(original_encoder),
                               rtol=0, atol=0)
    for original, proposed in zip(actor.parameters(), optimized.parameters()):
        torch.testing.assert_close(proposed.grad, original.grad, rtol=0, atol=0)
    methods = {"baseline": baseline, "reuse": candidate}
    for _ in range(20):
        baseline()
        candidate()
    blocks = []
    for repeat in range(repeats):
        block = {"index": repeat}
        order = ("baseline", "reuse") if repeat % 2 == 0 else ("reuse", "baseline")
        for name in order:
            call = methods[name]
            synchronize(device)
            started = time.perf_counter()
            for _ in range(iterations):
                call()
            synchronize(device)
            block[name + "_ms"] = (time.perf_counter() - started) * 1000 / iterations
        block["speedup"] = block["baseline_ms"] / block["reuse_ms"]
        blocks.append(block)
    base = statistics.median(block["baseline_ms"] for block in blocks)
    improved = statistics.median(block["reuse_ms"] for block in blocks)
    return dict(scope="actor gradient clipping and encoder metrics only; not full PPO",
                parity="exact norms, encoder squared norms and clipped gradients",
                device=str(device), torch=torch.__version__, python=platform.python_version(),
                threads=torch.get_num_threads(), config=dict(width=width, layers=layers, heads=heads),
                actor_parameters=sum(p.numel() for p in parameters),
                encoder_parameters=sum(p.numel() for p in encoder_parameters),
                iterations=iterations, repeats=repeats, warmup=20,
                baseline_ms=base, reuse_ms=improved, speedup=base / improved,
                median_paired_speedup=statistics.median(block["speedup"] for block in blocks),
                blocks=blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()
    if min(args.threads, args.iterations, args.repeats) < 1:
        parser.error("threads, iterations and repeats must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; GPU measurements must not fall back to CPU")
    torch.set_num_threads(args.threads)
    result = run(torch.device(args.device), args.width, args.layers, args.heads,
                 args.iterations, args.repeats)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
