"""Head-to-head duplicate evaluation of one history checkpoint against another.

    python -m eval.lineage_eval --candidate NEW.pt --baseline OLD.pt --deals 2000 \\
        --output out.json [--device mps --batch-size 512 --seed 2026100301]

The internal yardstick of the nightly report (October 3, 2026): the new
endpoint against the lineage's previous endpoint on a reproducible deal set,
both sides greedy, every deal played twice with the teams swapped. The score
is the candidate's net levels per round; identical checkpoints score exactly 0.
Batched VecEnv backend (eval/batched.py); the external yardstick stays DanLM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from eval.batched import EvalConfig, play_duplicate_batch
from eval.duplicate import generate_deals, summarize_duplicates
from eval.policies import load_policy
from infra.history_artifacts import sha256, source_identity


def evaluate(candidate_path: Path, baseline_path: Path, deals: int, seed: int,
             device: str = "cpu", batch_size: int = 512) -> dict:
    candidate = load_policy(str(candidate_path), device)
    baseline = load_policy(str(baseline_path), device, margin=0.0)
    for name, policy in (("candidate", candidate), ("baseline", baseline)):
        if getattr(policy, "stage", None) != "history_ppo":
            raise ValueError(f"{name} must be a history_ppo checkpoint")
    config = EvalConfig(batch_size=batch_size, kv_cache=True)
    scores = play_duplicate_batch(generate_deals(deals, seed), (candidate, candidate),
                                  (baseline, baseline), seed, config)
    summary = summarize_duplicates(scores, seed, 2000)
    return dict(candidate=str(candidate_path), candidate_sha256=sha256(candidate_path),
                baseline=str(baseline_path), baseline_sha256=sha256(baseline_path),
                deals=deals, seed=seed, duplicates=summary,
                mean_net_levels_per_round=summary["mean_net_levels_per_round"],
                bootstrap_95_ci=summary["bootstrap_95_ci"],
                evaluator=dict(backend="batched", device=device, batch_size=batch_size,
                               kv_cache=True, torch_threads=torch.get_num_threads(),
                               source_sha256=source_identity()["source_sha256"]),
                claim="development yardstick against the previous lineage endpoint")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deals", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026100301)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    report = evaluate(args.candidate, args.baseline, args.deals, args.seed, args.device,
                      args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    ci = report["bootstrap_95_ci"]
    print(f"{args.candidate.name} vs {args.baseline.name}: {report['mean_net_levels_per_round']:+.3f} "
          f"[{ci[0]:+.3f}, {ci[1]:+.3f}] over {args.deals} deals")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
