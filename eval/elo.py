"""Deterministic Elo updates from whole duplicate-pair results.

A pair is one win, draw or loss according to its signed level difference.
Level margins are reported separately; they are not interpreted as win odds.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Mapping


def expected_score(rating_a: float, rating_b: float) -> float:
    delta = max(-308.0, min(308.0, (rating_b - rating_a) / 400.0))
    return 1.0 / (1.0 + 10.0 ** delta)


def update_elo(rating_a: float, rating_b: float, score_a: float,
               k: float = 16.0) -> tuple[float, float]:
    if not all(math.isfinite(x) for x in (rating_a, rating_b, score_a, k)):
        raise ValueError("Elo inputs must be finite")
    if not 0 <= score_a <= 1 or k <= 0:
        raise ValueError("score must be in [0, 1] and K must be positive")
    delta = k * (score_a - expected_score(rating_a, rating_b))
    return rating_a + delta, rating_b - delta


def refresh_elo(reports: Iterable[Mapping], initial: float = 1500.0,
                k: float = 16.0) -> dict[str, float]:
    """Rebuild all ratings in the supplied chronological report/pair order.

    Supply the full historical report list on each refresh. Online Elo is
    order-dependent; the report order is deliberately not sorted or shuffled.
    """
    ratings: dict[str, float] = {}
    for report in reports:
        agent, opponent = str(report["agent"]), str(report["opponent"])
        if agent == opponent:
            continue
        for margin in report["duplicate"]["pair_scores"]:
            if not math.isfinite(margin):
                raise ValueError("duplicate margins must be finite")
            actual = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            ratings[agent], ratings[opponent] = update_elo(
                ratings.get(agent, initial), ratings.get(opponent, initial), actual, k)
    return dict(sorted(ratings.items(), key=lambda item: (-item[1], item[0])))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+", help="arena JSON files in chronological order")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--k", type=float, default=16.0)
    args = parser.parse_args(argv)
    reports = [json.loads(path.read_text()) for path in args.reports]
    output = json.dumps({"ratings": refresh_elo(reports, k=args.k), "k": args.k,
                         "reports": [str(path) for path in args.reports]}, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
