"""Evaluate two checkpoints on paired house-rule deals and full matches.

Run from any directory with the project's Python environment, for example:
  .venv/bin/python bench/overnight_eval.py --candidate candidate.pt \
      --control control.pt --output results/overnight-eval.json --device cuda

The report is saved after each completed cell. A partial run has status
"running"; only status "complete" contains every requested comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

import numpy as np
import torch

from eval.arena import summarize_matches
from eval.batched import EvalConfig, play_duplicate_batch, play_matches_batch
from eval.duplicate import bootstrap_interval, generate_deals, summarize_duplicates
from eval.policies import StyledPolicy, load_policy
from train.styles import FIXED_STYLES, StyleSpace


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def paired_delta(candidate: dict, control: dict, seed: int, samples: int) -> dict:
    """Subtract whole-deal levels in their original, shared deal order."""
    a = np.asarray(candidate["pair_scores"], dtype=np.float64)
    b = np.asarray(control["pair_scores"], dtype=np.float64)
    if a.shape != b.shape or not len(a):
        raise ValueError("paired reports must cover the same nonempty deal set")
    delta = a - b
    return {"deals": len(delta), "mean_levels_per_round": float(delta.mean()),
            "bootstrap_95_ci": list(bootstrap_interval(delta, seed, samples)),
            "pair_differences": delta.tolist(),
            "definition": "candidate pair score minus control pair score, paired by deal"}


def duplicate(policy, opponent, deals, seed: int, config: EvalConfig,
              bootstrap_samples: int) -> dict:
    started = time.monotonic()
    scores = play_duplicate_batch(deals, (policy, policy), (opponent, opponent), seed, config)
    result = summarize_duplicates(scores, seed, bootstrap_samples)
    result["seconds"] = time.monotonic() - started
    return result


def checkpoint_description(path: Path, policy) -> dict:
    return {"path": str(path.resolve()), "file_sha256": file_sha256(path),
            "model_digest": getattr(policy, "checkpoint_id", None),
            "name": policy.name, "stage": getattr(policy, "stage", None)}


def run(candidate_path: Path, control_path: Path, output: Path, *, device: str,
        deals_count: int, matches: int, seed: int, batch_size: int,
        engine_threads: int, heldout_styles: int,
        bootstrap_samples: int = 2000) -> dict:
    if min(deals_count, matches, batch_size, engine_threads, heldout_styles,
           bootstrap_samples) < 1:
        raise ValueError("deals, matches, batch size, engine threads, heldout styles and bootstrap samples must be positive")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(2 if device == "cpu" else min(2, torch.get_num_threads()))
    config = EvalConfig(batch_size=batch_size, engine_threads=engine_threads)
    candidate = load_policy(str(candidate_path), device)
    control = load_policy(str(control_path), device)
    deals = generate_deals(deals_count, seed)
    space = StyleSpace.default()
    rng = np.random.default_rng(seed + 2_000_003)
    heldout = [space.sample(rng, "heldout") for _ in range(heldout_styles)]
    assert all(space.in_heldout(vector) for vector in heldout)
    opponents = [("greedy", load_policy("greedy"), None)]
    opponents += [(f"styled:{name}", load_policy(f"styled:{name}"), None)
                  for name in FIXED_STYLES]
    opponents += [(f"heldout:{index}", StyledPolicy(vector, f"heldout:{index}"),
                   vector.tolist()) for index, vector in enumerate(heldout)]
    started = time.monotonic()
    report = {
        "schema_version": 1, "status": "running", "protocol": "internal-house",
        "seed": seed, "requested": {"deals": deals_count, "matches": matches,
                             "heldout_styles": heldout_styles, "bootstrap_samples": bootstrap_samples},
        "evaluation": {**config.metadata(), "device": device,
                       "torch_threads": torch.get_num_threads()},
        "candidate": checkpoint_description(candidate_path, candidate),
        "control": checkpoint_description(control_path, control),
        "heldout_style_space": space.describe(),
        "h2h": {}, "suite": {},
        "notes": ["All duplicate cells reuse the same generated deals and order.",
                  "Heldout vectors come from train.styles.StyleSpace region heldout and are fixed across both checkpoints.",
                  "A paired bootstrap resamples whole deals; results retain original pair order."]}
    save_report(output, report)
    report["h2h"]["duplicate"] = duplicate(candidate, control, deals, seed, config,
                                           bootstrap_samples)
    save_report(output, report)
    match_start = time.monotonic()
    match_totals = play_matches_batch(candidate, control, range(matches), seed, config=config)
    report["h2h"]["matches"] = summarize_matches(match_totals)
    report["h2h"]["matches"]["seconds"] = time.monotonic() - match_start
    save_report(output, report)
    for index, (name, opponent, vector) in enumerate(opponents):
        cell = {"opponent": name, "style_vector": vector}
        for label, policy in (("candidate", candidate), ("control", control)):
            cell[label] = duplicate(policy, opponent, deals, seed, config, bootstrap_samples)
            report["suite"][name] = cell
            save_report(output, report)
        cell["paired_delta"] = paired_delta(cell["candidate"], cell["control"],
                                            seed + 1000 + index, bootstrap_samples)
        save_report(output, report)
    report["seconds"] = time.monotonic() - started
    report["status"] = "complete"
    save_report(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--deals", type=int, default=4000)
    parser.add_argument("--matches", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--engine-threads", type=int, default=2)
    parser.add_argument("--heldout-styles", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    report = run(args.candidate, args.control, args.output, device=args.device,
                 deals_count=args.deals, matches=args.matches, seed=args.seed,
                 batch_size=args.batch_size, engine_threads=args.engine_threads,
                 heldout_styles=args.heldout_styles,
                 bootstrap_samples=args.bootstrap_samples)
    print(json.dumps({"status": report["status"], "output": str(args.output),
                      "h2h_levels_per_round": report["h2h"]["duplicate"]["mean_net_levels_per_round"],
                      "h2h_win_rate": report["h2h"]["matches"]["win_rate"],
                      "seconds": report["seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
