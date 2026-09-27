"""Stratified policy entropy of a history checkpoint (diagnostic only).

The checkpoint plays all four seats of fixed-seed self-play and samples its
own actions, so the decisions are its on-policy distribution. Every play-phase
decision with at least two legal candidates is scored with the full-candidate
distribution. Strata: legal-candidate count and lead versus follow (a decision
is a follow exactly when pass is legal). Also reported: the fraction of
decisions whose most likely candidate has probability above 0.99.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gd
import numpy as np
import torch

from train.history_model import load_history_checkpoint
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer

PASS_FEATURE = 108            # kActType + Type::Pass in the candidate encoding
COUNT_BINS = ((2, 2), (3, 5), (6, 15), (16, 10**9))
NEAR_DETERMINISTIC = 0.99


def decision_entropies(actor, buffer: SequenceRolloutBuffer, store: MatchEventStore,
                       chunk: int = 256) -> dict[str, np.ndarray]:
    """Per stored decision: entropy, top probability, legal count and follow flag."""
    entropy, top, count, follow = [], [], [], []
    for start in range(0, len(buffer), chunk):
        rows = np.arange(start, min(start + chunk, len(buffer)))
        inputs, _ = buffer.decision_inputs(rows, store, "cpu")
        with torch.no_grad():
            log_probs = actor.candidate_log_probs(inputs).double()
        offsets = inputs.offsets.tolist()
        passes = inputs.cand[:, PASS_FEATURE] > 0
        for i in range(inputs.decisions):
            lp = log_probs[offsets[i]:offsets[i + 1]]
            entropy.append(float(-(lp.exp() * lp).sum()))
            top.append(float(lp.max().exp()))
            count.append(offsets[i + 1] - offsets[i])
            follow.append(bool(passes[offsets[i]:offsets[i + 1]].any()))
    return dict(entropy=np.asarray(entropy), top=np.asarray(top), count=np.asarray(count),
                follow=np.asarray(follow))


def summarize(values: dict[str, np.ndarray]) -> dict:
    keep = values["count"] >= 2
    def stats(mask):
        mask = mask & keep
        n = int(mask.sum())
        if not n:
            return dict(decisions=0)
        h = values["entropy"][mask]
        return dict(decisions=n, mean_entropy=float(h.mean()),
                    mean_normalized_entropy=float((h / np.log(values["count"][mask])).mean()),
                    near_deterministic_fraction=float((values["top"][mask] > NEAR_DETERMINISTIC).mean()))
    strata = {}
    for role, flag in (("lead", False), ("follow", True)):
        for low, high in COUNT_BINS:
            name = f"{role}_n{low}" + ("" if low == high else f"-{high}" if high < 10**9 else "+")
            strata[name] = stats((values["follow"] == flag) & (values["count"] >= low)
                                 & (values["count"] <= high))
    return dict(all=stats(np.ones_like(keep)), lead=stats(~values["follow"]),
                follow=stats(values["follow"]), strata=strata,
                single_candidate_excluded=int((~keep).sum()))


def measure(checkpoint: Path, seed: int = 2026092731, envs: int = 16, steps: int = 256) -> dict:
    actor, _, payload = load_history_checkpoint(checkpoint, "cpu")
    actor.eval()
    env = gd.VecEnv(num_envs=envs, num_threads=2, seed=seed, log_public_actions=True,
                    log_env_limit=envs)
    store, buffer = MatchEventStore(), SequenceRolloutBuffer()
    collector = HistoryCollector(env, actor, store, buffer, torch.Generator().manual_seed(seed))
    collector.collect(steps)
    summary = summarize(decision_entropies(actor, buffer, store))
    return dict(checkpoint=str(checkpoint), updates=payload.get("progress", {}).get("updates"),
                seed=seed, envs=envs, steps=steps, **summary,
                claim="on-policy self-play diagnostic; not a strength measure")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=256)
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    report = [measure(path, steps=args.steps) for path in args.checkpoints]
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
