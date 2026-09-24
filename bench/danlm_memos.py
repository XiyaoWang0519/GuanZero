"""Measure DanLM's exact observation/play memos against the uncached path.

Run with DanLM's Python environment and extension (see docs/STAGE_C_TODO.md)::

    PYTHONPATH=python:. .work/external/danlm-venv/bin/python -m bench.danlm_memos \\
        --checkpoint .work/runpod-b11/payload/artifacts/league.pt --deals 80

Both modes play the same complete duplicate deals and compare every round
record, including divergences. Timings exclude imports, loading and warmup.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np

from eval.danlm import arena, fast_obs
from eval.danlm.bridge import PlayIndex


@contextmanager
def uncached():
    """Restore pre-memo work while using exactly the same arena and weights."""
    class EagerPlayIndex(PlayIndex):
        def __init__(self, legal_plays, level, **kwargs):
            super().__init__(legal_plays, level)
            self._build()

    original_index, original_hand = arena.PlayIndex, fast_obs._hand_plays
    arena.PlayIndex = EagerPlayIndex
    fast_obs._hand_plays = lambda rnd, actions, hand, player, level: np.asarray(
        actions.hand_calculator_v2(hand, level))
    try:
        yield
    finally:
        arena.PlayIndex, fast_obs._hand_plays = original_index, original_hand


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--danlm", default=arena.DEFAULT_DANLM_CHECKPOINT)
    parser.add_argument("--deals", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--diff", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    arena._init_worker(args.checkpoint, args.danlm, 1)
    items = []
    for i, deal in enumerate(arena.generate_deals(args.deals, args.seed, 0.5)):
        payload = arena.deal_to_json(deal)
        for leg, seats in (("a", ["ours", "danlm", "ours", "danlm"]),
                           ("b", ["danlm", "ours", "danlm", "ours"])):
            items.append({"id": f"deal-{i}/{leg}", "deal": payload,
                          "seed": args.seed + i, "seats": seats})
    times = {"uncached": [], "cached": []}
    reference = None

    def measure(name: str) -> None:
        nonlocal reference
        arena._run_rounds({"rounds": items[:2], "diff": args.diff})
        started = time.perf_counter()
        records = arena._run_rounds({"rounds": items, "diff": args.diff})
        times[name].append(time.perf_counter() - started)
        if reference is None:
            reference = records
        elif records != reference:
            raise AssertionError(f"{name} changed a complete round record")

    for repeat in range(args.repeats):
        # Reverse the order every second pair to reduce thermal/drift bias.
        order = ("uncached", "cached") if repeat % 2 == 0 else ("cached", "uncached")
        for name in order:
            if name == "uncached":
                with uncached():
                    measure(name)
            else:
                measure(name)
    report = {"deals": args.deals, "seed": args.seed, "diff": args.diff,
              "repeats": args.repeats, "seconds": times,
              "speedup": statistics.median(times["uncached"]) / statistics.median(times["cached"]),
              "identical_round_records": len(reference),
              "decisions": sum(r["decisions"] for r in reference),
              "records_sha256": hashlib.sha256(
                  json.dumps(reference, sort_keys=True).encode()).hexdigest()}
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
