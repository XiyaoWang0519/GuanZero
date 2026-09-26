"""Development-only evaluation of a history checkpoint against frozen MLPs.

This process alone loads the evaluation assets. It has no final-test switch,
training output or feedback path. All duplicate legs and full matches are saved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gd
import torch

from infra.history_artifacts import sha256, engine_digest, source_identity
from eval.arena import play_matches
from eval.duplicate import (bootstrap_interval, evaluate_duplicates)
from eval.policies import load_policy


def load_deals(records: list[dict]) -> list:
    deals = []
    for record in records:
        deal = gd.DealSpec()
        for key, value in record.items():
            setattr(deal, key, value)
        deals.append(deal)
    return deals


def evaluate(freeze_path: Path, candidate_path: Path, output: Path) -> dict:
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("engine_digest") != engine_digest():
        raise ValueError("frozen evaluation engine digest mismatch")
    root = freeze_path.parent
    dev_path = root / freeze["development"]["file"]
    if sha256(dev_path) != freeze["development"]["sha256"]:
        raise ValueError("development deal file digest mismatch")
    dev = json.loads(dev_path.read_text())
    candidate = load_policy(str(candidate_path), "cpu")
    if getattr(candidate, "stage", None) != "history_ppo":
        raise ValueError("candidate must be a history_ppo checkpoint")
    reports = {}
    for spec in freeze["baselines"]:
        path = Path(spec["path"])
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"frozen baseline changed: {spec['name']}")
        opponent = load_policy(str(path), "cpu", margin=0.0)
        duplicate = evaluate_duplicates(candidate, opponent, load_deals(dev["deals"]),
                                        seed=dev["policy_seed"], bootstrap_samples=2000)
        pairs = []
        for seed in dev["match_seeds"]:
            # Same engine seed twice, opposite team assignments. Greedy actions
            # eliminate the different seat RNG streams of play_matches.
            first = play_matches(candidate, opponent, [0], seed=seed)
            swapped = play_matches(candidate, opponent, [1], seed=seed - 1)
            pairs.append(dict(first=first, swapped=swapped,
                              win_rate=(first["wins"] + swapped["wins"]) / 2))
        wins = [p["win_rate"] for p in pairs]
        reports[spec["name"]] = dict(baseline_sha256=spec["sha256"], duplicates=duplicate,
                                    full_matches=dict(pairs=pairs, win_rate=sum(wins) / len(wins),
                                                      bootstrap_95_ci=list(bootstrap_interval(wins))))
    report = dict(candidate_sha256=sha256(candidate_path), freeze_sha256=sha256(freeze_path),
                  evaluation_source_sha256=source_identity()["source_sha256"],
                  reports=reports, split="development", selection="predeclared endpoint",
                  claim="small pilot diagnostic; no playing-strength promotion")
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    evaluate(args.freeze, args.candidate, args.output)


if __name__ == "__main__":
    main()
