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
from eval.batched import EvalConfig, play_duplicate_batch, play_match_slots_batch
from eval.duplicate import (bootstrap_interval, evaluate_duplicates, summarize_duplicates)
from eval.policies import load_policy


def load_deals(records: list[dict]) -> list:
    deals = []
    for record in records:
        deal = gd.DealSpec()
        for key, value in record.items():
            setattr(deal, key, value)
        deals.append(deal)
    return deals


def evaluate(freeze_path: Path, candidate_path: Path, output: Path, backend: str = "scalar",
             device: str = "cpu", batch_size: int = 512) -> dict:
    """Scalar (the reference) or batched VecEnv evaluation of one candidate.

    The batched backend plays the same deals and match seeds with the same greedy
    policies; on another device or batch shape it agrees with the scalar backend up
    to floating point ties, so compare its tables with the measured agreement rather
    than mixing them with scalar results.
    """
    if backend not in ("scalar", "batched"):
        raise ValueError("backend must be scalar or batched")
    if backend == "scalar" and device != "cpu":
        raise ValueError("the scalar reference backend runs on cpu")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("engine_digest") != engine_digest():
        raise ValueError("frozen evaluation engine digest mismatch")
    root = freeze_path.parent
    dev_path = root / freeze["development"]["file"]
    if sha256(dev_path) != freeze["development"]["sha256"]:
        raise ValueError("development deal file digest mismatch")
    dev = json.loads(dev_path.read_text())
    candidate = load_policy(str(candidate_path), device)
    if getattr(candidate, "stage", None) != "history_ppo":
        raise ValueError("candidate must be a history_ppo checkpoint")
    reports = {}
    for spec in freeze["baselines"]:
        path = Path(spec["path"])
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"frozen baseline changed: {spec['name']}")
        opponent = load_policy(str(path), device, margin=0.0)
        if backend == "batched":
            config = EvalConfig(batch_size=batch_size, kv_cache=True)
            scores = play_duplicate_batch(load_deals(dev["deals"]), (candidate, candidate),
                                          (opponent, opponent), dev["policy_seed"], config)
            duplicate = summarize_duplicates(scores, dev["policy_seed"], 2000)
            # Same engine seed twice, opposite team assignments, all in one wave.
            slots = [(seed, team) for seed in dev["match_seeds"] for team in (0, 1)]
            played = play_match_slots_batch(candidate, opponent, slots, config=config)
            matches = [(played[i], played[i + 1]) for i in range(0, len(played), 2)]
        else:
            duplicate = evaluate_duplicates(candidate, opponent, load_deals(dev["deals"]),
                                            seed=dev["policy_seed"], bootstrap_samples=2000)
            # Same engine seed twice, opposite team assignments. Greedy actions
            # eliminate the different seat RNG streams of play_matches.
            matches = [(play_matches(candidate, opponent, [0], seed=seed),
                        play_matches(candidate, opponent, [1], seed=seed - 1))
                       for seed in dev["match_seeds"]]
        pairs = [dict(first=first, swapped=swapped, win_rate=(first["wins"] + swapped["wins"]) / 2)
                 for first, swapped in matches]
        wins = [p["win_rate"] for p in pairs]
        reports[spec["name"]] = dict(baseline_sha256=spec["sha256"], duplicates=duplicate,
                                    full_matches=dict(pairs=pairs, win_rate=sum(wins) / len(wins),
                                                      bootstrap_95_ci=list(bootstrap_interval(wins))))
    report = dict(candidate_sha256=sha256(candidate_path), freeze_sha256=sha256(freeze_path),
                  evaluation_source_sha256=source_identity()["source_sha256"],
                  reports=reports, evaluator=dict(backend=backend, device=device,
                                                  batch_size=batch_size if backend == "batched" else None),
                  split="development", selection="predeclared endpoint",
                  claim="small pilot diagnostic; no playing-strength promotion")
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("scalar", "batched"), default="scalar")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="batched backend: deals or match slots per VecEnv wave")
    parser.add_argument("--threads", type=int, default=2, help="torch CPU threads")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    evaluate(args.freeze, args.candidate, args.output, args.backend, args.device, args.batch_size)


if __name__ == "__main__":
    main()
