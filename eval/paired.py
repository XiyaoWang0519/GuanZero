"""Paired difference between two checkpoints scored on the same deal set.

    python -m eval.paired A.json B.json [--output diff.json] [--seed 0 --samples 5000]

Both inputs are ``eval.danlm.arena duplicate`` outputs (with ``per_deal``), or
both are ``eval.lineage_eval`` outputs against the same baseline. The deal
set, opponent and seed must match. Every deal score is the mean of its two
legs, so the deal is the resampling unit: the difference A - B is taken deal
by deal on the deals both runs scored, and its interval resamples whole
deals. Deals excluded by either run (DanLM mirror failures) are reported
with their reasons, and a worst-case bound shows how far they could move the
difference had they been scored.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from eval.duplicate import bootstrap_interval

# A leg's net level gain is within [-3, 3], so a deal score is too and a
# difference of two deal scores is within [-6, 6].
DEAL_SCORE_BOUND = 3.0
NORMAL_95 = 1.959963984540054


@dataclass(frozen=True)
class DealScores:
    kind: str                        # "danlm_duplicate" or "lineage_eval"
    name: str
    identity: dict                   # must match between the two runs
    deals: int
    scores: dict[int, float]         # deal -> net levels per round, scored deals only
    round_wins: dict[int, int]       # deal -> legs whose first finisher was ours (0..2)
    excluded: dict[int, list[str]]   # deal -> failure kinds of its failed legs


def deal_scores(report: dict) -> DealScores:
    """Per-deal scores of one evaluation report."""
    if report.get("mode") == "duplicate":
        if "per_deal" not in report:
            raise ValueError("arena output has no per_deal records; rerun with the current "
                             "eval.danlm.arena")
        identity = {key: report.get(key) for key in
                    ("danlm", "seed", "deals", "tribute_fraction", "deal_set_sha256")}
        scores, wins, excluded = {}, {}, {}
        for row in report["per_deal"]:
            if row["scored"]:
                scores[row["deal"]] = float(row["score"])
                wins[row["deal"]] = int(row["round_wins"])
            else:
                excluded[row["deal"]] = [(leg.get("failure") or {}).get("kind", "unrecorded")
                                         for leg in row["legs"] if leg["status"] != "ok"]
        return DealScores("danlm_duplicate", report.get("checkpoint_name") or report["checkpoint"],
                          identity, int(report["deals"]), scores, wins, excluded)
    if "duplicates" in report and "baseline_sha256" in report:
        identity = {key: report.get(key) for key in
                    ("baseline_sha256", "seed", "deals", "deal_set_sha256")}
        pairs = report["duplicates"]["pair_scores"]
        results = report["duplicates"]["results"]
        if len(pairs) != report["deals"] or len(results) != report["deals"]:
            raise ValueError("lineage_eval output does not hold one score per deal")
        # Team A sits at seats 0/2 in the first leg and 1/3 in the swapped one.
        wins = {i: int(r["first"]["winning_team"] == 0) + int(r["swapped"]["winning_team"] == 1)
                for i, r in enumerate(results)}
        return DealScores("lineage_eval", report["candidate"], identity, int(report["deals"]),
                          {i: float(v) for i, v in enumerate(pairs)}, wins, {})
    raise ValueError("not an eval.danlm.arena duplicate or eval.lineage_eval report")


def _interval(values: np.ndarray, seed: int, samples: int) -> dict:
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else math.nan
    return {"mean": mean, "standard_error": se,
            "normal_95_ci": [mean - NORMAL_95 * se, mean + NORMAL_95 * se],
            "bootstrap_95_ci": list(bootstrap_interval(values, seed, samples))}


def compare(a: DealScores, b: DealScores, seed: int = 0, samples: int = 5000) -> dict:
    """A - B on the deals both scored, resampling whole deals."""
    if a.kind != b.kind:
        raise ValueError(f"cannot pair a {a.kind} report with a {b.kind} report")
    mismatched = {key: [a.identity.get(key), b.identity.get(key)] for key in a.identity
                  if a.identity.get(key) != b.identity.get(key)
                  and not (key == "deal_set_sha256" and None in (a.identity.get(key),
                                                                  b.identity.get(key)))}
    if mismatched:
        raise ValueError(f"the two runs are not on the same deals/opponent: {mismatched}")
    common = sorted(set(a.scores) & set(b.scores))
    if len(common) < 2:
        raise ValueError("fewer than two deals scored by both runs")
    score_a = np.array([a.scores[i] for i in common])
    score_b = np.array([b.scores[i] for i in common])
    diff = score_a - score_b
    win_diff = np.array([a.round_wins[i] - b.round_wins[i] for i in common]) / 2
    union = sorted(set(a.excluded) | set(b.excluded))
    # Had the excluded deals been scored, the all-deal difference would lie in this range.
    total = a.deals
    bound = [(diff.sum() - 2 * DEAL_SCORE_BOUND * len(union)) / total,
             (diff.sum() + 2 * DEAL_SCORE_BOUND * len(union)) / total]
    return {"a": a.name, "b": b.name, "kind": a.kind, "identity": a.identity,
            "deal_set_verified": None not in (a.identity.get("deal_set_sha256"),
                                              b.identity.get("deal_set_sha256")),
            "deals": total, "common_deals": len(common), "seed": seed, "bootstrap_samples": samples,
            "a_levels_per_round": float(score_a.mean()), "b_levels_per_round": float(score_b.mean()),
            "difference": _interval(diff, seed, samples),
            "round_win_rate_difference": _interval(win_diff, seed, samples),
            "deals_with_different_score": int(np.count_nonzero(diff)),
            "excluded": {"a_only": sorted(set(a.excluded) - set(b.excluded)),
                         "b_only": sorted(set(b.excluded) - set(a.excluded)),
                         "both": sorted(set(a.excluded) & set(b.excluded)),
                         "reasons": {str(i): {"a": a.excluded.get(i), "b": b.excluded.get(i)}
                                     for i in union}},
            "difference_bound_if_excluded_scored": [float(bound[0]), float(bound[1])],
            "reading": "difference is A - B in net levels per round on deals both runs scored; "
                       "the bootstrap resamples whole deals, the normal interval uses the "
                       "per-deal standard error of the paired differences"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=5000)
    args = parser.parse_args(argv)
    result = compare(deal_scores(json.loads(args.a.read_text())),
                     deal_scores(json.loads(args.b.read_text())), args.seed, args.samples)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    d = result["difference"]
    lo, hi = d["bootstrap_95_ci"]
    print(f"{result['a']} - {result['b']}: {d['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}] "
          f"(normal [{d['normal_95_ci'][0]:+.3f}, {d['normal_95_ci'][1]:+.3f}]) over "
          f"{result['common_deals']}/{result['deals']} common deals; excluded "
          f"{len(result['excluded']['reasons'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
