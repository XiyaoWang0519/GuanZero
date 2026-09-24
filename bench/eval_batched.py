"""Time scalar versus batched evaluation and verify every deterministic result.

PYTHONPATH=python:. .venv/bin/python -m bench.eval_batched --agent path/to/model.pt
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from eval.arena import play_matches
from eval.batched import EvalConfig, play_duplicate_batch, play_matches_batch
from eval.duplicate import generate_deals, play_duplicate
from eval.policies import load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', required=True)
    parser.add_argument('--opponent', default='greedy')
    parser.add_argument('--deals', type=int, default=256)
    parser.add_argument('--matches', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(1)
    agent, opponent = load_policy(args.agent), load_policy(args.opponent)
    deals = generate_deals(args.deals, args.seed)
    config = EvalConfig(batch_size=args.batch_size)
    # Warm inference and engine setup before either timed run.
    play_duplicate(deals[0], agent, opponent, args.seed)
    started = time.perf_counter()
    scalar = [play_duplicate(d, agent, opponent, args.seed + i) for i, d in enumerate(deals)]
    scalar_s = time.perf_counter() - started
    started = time.perf_counter()
    batched = play_duplicate_batch(deals, (agent, agent), (opponent, opponent), args.seed, config)
    batched_s = time.perf_counter() - started
    report = {'agent': agent.name, 'opponent': opponent.name, 'deals': args.deals,
              'batch_size': args.batch_size, 'scalar_seconds': scalar_s, 'batched_seconds': batched_s,
              'speedup': scalar_s / batched_s,
              'identical_deals': sum(a == b for a, b in zip(scalar, batched))}
    if args.matches:
        started = time.perf_counter()
        scalar = play_matches(agent, opponent, range(args.matches), args.seed)
        scalar_s = time.perf_counter() - started
        started = time.perf_counter()
        batched = play_matches_batch(agent, opponent, range(args.matches), args.seed, config=config)
        batched_s = time.perf_counter() - started
        report['matches'] = {'count': args.matches, 'identical': scalar == batched,
                             'scalar_seconds': scalar_s, 'batched_seconds': batched_s,
                             'speedup': scalar_s / batched_s}
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
