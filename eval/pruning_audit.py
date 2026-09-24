"""Audit the frozen-reference top-k candidate pruning of a Stage B policy.

Stage C task C1(a) (``docs/STAGE_C_TODO.md``). A Stage B policy chooses
among the frozen Stage A reference's top ``top_k`` candidates plus pass
(``train/policy.py``). This script plays the audited checkpoint against
itself in every seat, exactly as the arena evaluates it (argmax over the
pruned set, heuristic tribute), and for every play decision records:

- the size of the full canonical candidate set, so how often pruning bites;
- which play types, bomb sizes and wild usages get pruned;
- the reference rank of the action the policy actually chose, so whether
  choices pile up at the boundary of the allowed set;
- whether the policy's argmax over the FULL set lies outside the pruned set
  (a divergence indicator), and at what reference rank it sits, which gives
  a coverage curve over alternative ``top_k`` values.

The full-set argmax is an indicator only: the policy never trained on
candidates outside the pruned set, so its logits there are uncalibrated.
Coverage at a larger ``top_k`` says how many of those preferred actions a
wider set would admit, not that admitting them would help. That question is
task C1(c), a training comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import gd
import numpy as np
import torch

from eval.collect_critic import STAGES, file_sha256, stage_bins, write_json
from eval.policies import model_digest
from train.ckpt import load_checkpoint
from train.model import GuandanModel, ModelConfig, select_actions
from train.policy import (PASS_FEATURE, PolicyConfig, StageBPolicy, policy_from_payload,
                          segment_rows)

SCHEMA_VERSION = 1
# Action encoding layout, cpp/include/gd/encoder.h.
TYPE_SLICE = slice(108, 121)
BOMB_SIZE_SLICE = slice(136, 143)
WILDS_SLICE = slice(143, 146)
TYPE_NAMES = ("Pass", "Single", "Pair", "Triple", "FullHouse", "Straight", "Tube", "Plate",
              "Bomb", "StraightFlush", "JokerBomb", "Tribute", "BackTribute")
BOMB_SIZES = tuple(range(4, 11))
# Observation bit that says the acting seat leads the trick (kObsTrickLeading).
LEADING_BIT = 967
PLAY = int(gd.Phase.Play)


def segment_ranks(scores: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Rank of every candidate within its segment by descending score, 0 best.

    Ties break toward the lower original index, the same order that
    ``train.policy.prune_candidates`` keeps, so ``rank < top_k`` is exactly
    the pruning rule before the pass exception.
    """
    rows = segment_rows(offsets, len(scores))
    order = torch.argsort(scores, descending=True, stable=True)
    order = order[torch.argsort(rows[order], stable=True)]
    rank = torch.empty_like(order)
    rank[order] = torch.arange(len(scores), device=order.device) - offsets[:-1][rows[order]]
    return rank


def load_audited_policy(path: Path, device: str, top_k: int | None) -> tuple[StageBPolicy, dict]:
    """A Stage B checkpoint as it plays; a DMC checkpoint prunes with itself."""
    payload = load_checkpoint(path, device=device)
    stage = payload.get("stage", "dmc")
    if stage == "ppo":
        policy = policy_from_payload(payload, device=device)
        if top_k is not None and top_k != policy.config.top_k:
            from dataclasses import replace
            policy.config = replace(policy.config, top_k=top_k)
    elif stage == "dmc":
        model = GuandanModel(ModelConfig(**payload["model_config"]))
        model.load_state_dict(payload["model"])
        policy = StageBPolicy(model, model, PolicyConfig(top_k=top_k or 32)).to(device)
    else:
        raise ValueError(f"cannot audit a {stage} checkpoint")
    if payload.get("tribute_policy", "heuristic") != "heuristic":
        raise ValueError("the audit plays heuristic tribute only")
    info = {"stage": stage, "checkpoint_id": model_digest(payload["model"]),
            "reference_checkpoint_id": policy.reference_checkpoint_id
            if stage == "ppo" else model_digest(payload["model"]),
            "checkpoint_updates": int(payload.get("progress", {}).get("updates", 0)),
            "training_seed": payload.get("config", {}).get("seed"),
            "policy_config": {"temperature": policy.config.temperature,
                              "top_k": policy.config.top_k}}
    policy.net.eval()
    return policy, info


class Accumulator:
    """Counts over play decisions, split by stage bin and lead/follow."""

    def __init__(self, top_k: int, coverage_ks: tuple[int, ...], max_candidates: int) -> None:
        self.top_k = top_k
        self.coverage_ks = coverage_ks
        self.max_candidates = max_candidates
        groups = (len(STAGES), 2)  # stage x (follow, lead)
        self.decisions = np.zeros(groups, np.int64)
        self.pruned_decisions = np.zeros(groups, np.int64)  # n > top_k
        self.candidates = np.zeros(groups, np.int64)
        self.pruned_candidates = np.zeros(groups, np.int64)
        self.size_hist = np.zeros((*groups, max_candidates + 1), np.int64)
        self.divergent = np.zeros(groups, np.int64)
        self.full_argmax_rank_hist = np.zeros((*groups, max_candidates + 1), np.int64)
        self.chosen_rank_hist = np.zeros((*groups, max_candidates + 1), np.int64)
        self.coverage = np.zeros((*groups, len(coverage_ks)), np.int64)
        self.type_total = np.zeros(len(TYPE_NAMES), np.int64)
        self.type_pruned = np.zeros(len(TYPE_NAMES), np.int64)
        self.type_chosen = np.zeros(len(TYPE_NAMES), np.int64)
        self.type_full_argmax_divergent = np.zeros(len(TYPE_NAMES), np.int64)
        self.type_pruned_argmax_divergent = np.zeros(len(TYPE_NAMES), np.int64)
        self.bomb_total = np.zeros(len(BOMB_SIZES), np.int64)
        self.bomb_pruned = np.zeros(len(BOMB_SIZES), np.int64)
        self.wilds_total = np.zeros(3, np.int64)
        self.wilds_pruned = np.zeros(3, np.int64)
        self.logit_gap_sum = 0.0  # net logit(full argmax) - net logit(chosen), divergent only
        self.logit_gap_count = 0

    def add(self, *, sizes: np.ndarray, stage: np.ndarray, leading: np.ndarray,
            rank: np.ndarray, kept: np.ndarray, offsets: np.ndarray,
            full_argmax_local: np.ndarray, chosen_local: np.ndarray,
            cand_type: np.ndarray, cand_bomb: np.ndarray, cand_wilds: np.ndarray,
            full_logits: np.ndarray) -> None:
        starts = offsets[:-1]
        full_flat = starts + full_argmax_local
        chosen_flat = starts + chosen_local
        divergent = ~kept[full_flat]
        full_rank = np.minimum(rank[full_flat], self.max_candidates)
        chosen_rank = np.minimum(rank[chosen_flat], self.max_candidates)
        sizes_c = np.minimum(sizes, self.max_candidates)
        pruned_flat = ~kept
        rows = np.repeat(np.arange(len(sizes)), sizes)
        for s in range(len(STAGES)):
            for lead in (0, 1):
                mask = (stage == s) & (leading == lead)
                if not mask.any():
                    continue
                g = (s, lead)
                self.decisions[g] += int(mask.sum())
                self.pruned_decisions[g] += int((sizes[mask] > self.top_k).sum())
                self.candidates[g] += int(sizes[mask].sum())
                self.pruned_candidates[g] += int(pruned_flat[mask[rows]].sum())
                self.size_hist[g] += np.bincount(sizes_c[mask], minlength=self.max_candidates + 1)
                self.divergent[g] += int(divergent[mask].sum())
                self.full_argmax_rank_hist[g] += np.bincount(
                    full_rank[mask], minlength=self.max_candidates + 1)
                self.chosen_rank_hist[g] += np.bincount(
                    chosen_rank[mask], minlength=self.max_candidates + 1)
                is_pass = cand_type[full_flat[mask]] == 0
                for j, k in enumerate(self.coverage_ks):
                    self.coverage[g][j] += int(((rank[full_flat[mask]] < k) | is_pass).sum())
        self.type_total += np.bincount(cand_type, minlength=len(TYPE_NAMES))
        self.type_pruned += np.bincount(cand_type[pruned_flat], minlength=len(TYPE_NAMES))
        self.type_chosen += np.bincount(cand_type[chosen_flat], minlength=len(TYPE_NAMES))
        self.type_full_argmax_divergent += np.bincount(
            cand_type[full_flat[divergent]], minlength=len(TYPE_NAMES))
        self.type_pruned_argmax_divergent += np.bincount(
            cand_type[chosen_flat[divergent]], minlength=len(TYPE_NAMES))
        bombs = cand_bomb >= 0
        self.bomb_total += np.bincount(cand_bomb[bombs], minlength=len(BOMB_SIZES))
        self.bomb_pruned += np.bincount(cand_bomb[bombs & pruned_flat], minlength=len(BOMB_SIZES))
        self.wilds_total += np.bincount(cand_wilds, minlength=3)
        self.wilds_pruned += np.bincount(cand_wilds[pruned_flat], minlength=3)
        if divergent.any():
            gap = full_logits[full_flat[divergent]] - full_logits[chosen_flat[divergent]]
            self.logit_gap_sum += float(gap.sum())
            self.logit_gap_count += int(divergent.sum())

    @property
    def total_decisions(self) -> int:
        return int(self.decisions.sum())

    def report(self) -> dict:
        def group_table(values: np.ndarray, denominator: np.ndarray) -> dict:
            out = {}
            for s, name in enumerate(STAGES):
                for lead, kind in enumerate(("follow", "lead")):
                    d = int(denominator[s, lead])
                    out[f"{name}/{kind}"] = {"count": int(values[s, lead]),
                                             "fraction": float(values[s, lead] / d) if d else None}
            total, den = int(values.sum()), int(denominator.sum())
            out["all"] = {"count": total, "fraction": float(total / den) if den else None}
            return out

        def hist_summary(hist: np.ndarray) -> dict:
            flat = hist.reshape(-1, hist.shape[-1]).sum(0)
            n = int(flat.sum())
            if not n:
                return {"count": 0}
            cumulative = np.cumsum(flat) / n
            values = np.arange(len(flat))
            mean = float((flat * values).sum() / n)
            percentile = lambda p: int(values[np.searchsorted(cumulative, p)])  # noqa: E731
            return {"count": n, "mean": mean, "p50": percentile(0.5), "p90": percentile(0.9),
                    "p99": percentile(0.99), "max": int(values[flat > 0].max()),
                    "histogram": {str(i): int(c) for i, c in enumerate(flat) if c}}

        chosen_hist = self.chosen_rank_hist.reshape(-1, self.max_candidates + 1).sum(0)
        chosen_n = int(chosen_hist.sum())
        boundary = self.top_k - 1
        near_boundary = int(chosen_hist[max(0, self.top_k - 4):self.top_k].sum())
        coverage = {}
        for j, k in enumerate(self.coverage_ks):
            coverage[str(k)] = group_table(self.coverage[..., j], self.decisions)

        def by_name(names, total, pruned, extra=None):
            rows = {}
            for i, name in enumerate(names):
                t = int(total[i])
                row = {"candidates": t, "pruned": int(pruned[i]),
                       "pruned_fraction": float(pruned[i] / t) if t else None}
                if extra is not None:
                    row.update({key: int(value[i]) for key, value in extra.items()})
                rows[str(name)] = row
            return rows

        return {
            "play_decisions": self.total_decisions,
            "decisions_with_pruning": group_table(self.pruned_decisions, self.decisions),
            "candidates_pruned": group_table(self.pruned_candidates, self.candidates),
            "candidate_set_size": hist_summary(self.size_hist),
            "candidate_set_size_lead": hist_summary(self.size_hist[:, 1]),
            "candidate_set_size_follow": hist_summary(self.size_hist[:, 0]),
            "full_argmax_outside_pruned_set": group_table(self.divergent, self.decisions),
            "full_argmax_reference_rank": hist_summary(self.full_argmax_rank_hist),
            "coverage_of_full_argmax_at_top_k": coverage,
            "chosen_reference_rank": {
                **hist_summary(self.chosen_rank_hist),
                "at_boundary_rank": boundary,
                "at_boundary_count": int(chosen_hist[boundary]) if boundary < len(chosen_hist) else 0,
                "at_boundary_fraction": (float(chosen_hist[boundary] / chosen_n)
                                         if chosen_n and boundary < len(chosen_hist) else None),
                "last_four_ranks_fraction": float(near_boundary / chosen_n) if chosen_n else None,
                "rank0_fraction": float(chosen_hist[0] / chosen_n) if chosen_n else None,
            },
            "mean_logit_gap_when_divergent": (self.logit_gap_sum / self.logit_gap_count
                                              if self.logit_gap_count else None),
            "by_type": by_name(TYPE_NAMES, self.type_total, self.type_pruned, {
                "chosen": self.type_chosen,
                "full_argmax_when_divergent": self.type_full_argmax_divergent,
                "chosen_when_divergent": self.type_pruned_argmax_divergent}),
            "by_bomb_size": by_name(BOMB_SIZES, self.bomb_total, self.bomb_pruned),
            "by_wilds_used": by_name((0, 1, 2), self.wilds_total, self.wilds_pruned),
        }


def audit(checkpoint: str | Path, *, decisions: int = 100_000, num_envs: int = 256,
          threads: int = 4, torch_threads: int | None = None, seed: int = 20260926,
          device: str = "cpu", top_k: int | None = None,
          coverage_ks: tuple[int, ...] = (16, 32, 48, 64, 96, 128),
          max_candidates: int = 512, max_seconds: float = 3600.0,
          policy: StageBPolicy | None = None) -> dict:
    """Self-play the audited policy until ``decisions`` play decisions are seen."""
    if decisions < 1 or num_envs < 1 or threads < 1 or max_candidates < 1:
        raise ValueError("decisions, environments, threads and max_candidates must be positive")
    if not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("maximum seconds must be positive and finite")
    started = time.monotonic()
    torch.set_num_threads(torch_threads or threads)
    checkpoint = Path(checkpoint)
    if policy is None:
        policy, info = load_audited_policy(checkpoint, device, top_k)
    else:
        info = {"stage": "in-memory", "policy_config": {"temperature": policy.config.temperature,
                                                        "top_k": policy.config.top_k}}
    if info.get("training_seed") is not None and seed == info["training_seed"]:
        raise ValueError("audit seed must differ from the training seed")
    k = policy.config.top_k
    acc = Accumulator(k, tuple(sorted(set(coverage_ks) | {k})), max_candidates)
    env = gd.VecEnv(num_envs, num_threads=threads, seed=seed,
                    rules=gd.RuleConfig.house(), actions=gd.ActionConfig())
    env.reset()
    rounds = steps = 0
    status = "complete"
    while acc.total_decisions < decisions:
        if time.monotonic() - started > max_seconds:
            status = "time_limited"
            break
        batch = env.pending()
        rounds += len(env.drain_finished_rounds())
        phase = np.asarray(batch.phase)
        offsets_np = np.asarray(batch.offsets, np.int64)
        choices = np.array(batch.greedy_choice, dtype=np.int32, copy=True)
        play = np.flatnonzero(phase == PLAY)
        if play.size:
            obs_np = np.asarray(batch.obs)
            cand_np = np.asarray(batch.cand)
            sizes = (offsets_np[1:] - offsets_np[:-1])[play]
            play_offsets = np.concatenate(([0], np.cumsum(sizes)))
            cand_rows = np.concatenate([np.arange(offsets_np[r], offsets_np[r + 1]) for r in play])
            obs = torch.as_tensor(np.array(obs_np[play], copy=True), device=device)
            cand = torch.as_tensor(np.array(cand_np[cand_rows], copy=True), device=device)
            offsets = torch.as_tensor(play_offsets, device=device)
            phase_t = torch.full((len(play),), PLAY, device=device, dtype=torch.long)
            with torch.inference_mode():
                ref_scores = policy.reference.score_candidates(
                    obs, cand, offsets, phase_t, chunk_size=policy.config.chunk_size,
                    phase_code=PLAY)
                if not bool(torch.isfinite(ref_scores).all()):
                    raise FloatingPointError("non-finite reference scores")
                step = policy.act(obs, cand, offsets, phase_t, greedy=True, phase_code=PLAY,
                                  ref_scores=ref_scores)
                full_logits = policy.logits(obs, cand, offsets, phase_t, phase_code=PLAY)
                if not bool(torch.isfinite(full_logits).all()):
                    raise FloatingPointError("non-finite policy logits")
                full_argmax = select_actions(full_logits, offsets)
                rank = segment_ranks(ref_scores, offsets)
            kept = np.zeros(len(cand), bool)
            kept[step.keep_index.cpu().numpy()] = True
            cand_f = cand.cpu().numpy()
            cand_type = cand_f[:, TYPE_SLICE].argmax(1)
            bomb_bits = cand_f[:, BOMB_SIZE_SLICE]
            cand_bomb = np.where(bomb_bits.max(1) > 0.5, bomb_bits.argmax(1), -1)
            cand_wilds = cand_f[:, WILDS_SLICE].argmax(1)
            obs_f = obs.cpu().numpy()
            acc.add(sizes=sizes, stage=stage_bins(obs_f), leading=(obs_f[:, LEADING_BIT] > 0.5).astype(np.int64),
                    rank=rank.cpu().numpy(), kept=kept, offsets=play_offsets,
                    full_argmax_local=full_argmax.cpu().numpy(),
                    chosen_local=step.choice.cpu().numpy(), cand_type=cand_type,
                    cand_bomb=cand_bomb, cand_wilds=cand_wilds,
                    full_logits=full_logits.cpu().numpy())
            choices[play] = step.choice.cpu().numpy().astype(np.int32)
        env.step(choices)
        steps += 1
    elapsed = time.monotonic() - started
    report = {
        "schema_version": SCHEMA_VERSION, "status": status,
        "checkpoint": str(checkpoint), **info,
        "checkpoint_file_sha256": file_sha256(checkpoint) if checkpoint.is_file() else None,
        "seed": seed, "num_envs": num_envs, "threads": threads,
        "torch_threads": torch.get_num_threads(), "device": device,
        "play_mode": "audited policy in all four seats, argmax over the pruned set, "
                     "heuristic tribute, house rules, canonical actions",
        "requested_decisions": decisions, "vec_steps": steps, "finished_rounds": rounds,
        "elapsed_seconds": elapsed,
        "decisions_per_second": acc.total_decisions / elapsed if elapsed else None,
        "top_k": k, "coverage_ks": list(acc.coverage_ks),
        "reading": {
            "decisions_with_pruning": "fraction of play decisions whose canonical set exceeds top_k",
            "full_argmax_outside_pruned_set": "policy argmax over the full set not in the pruned set; "
                                              "an indicator only, those logits were never trained",
            "coverage_of_full_argmax_at_top_k": "fraction of decisions whose full-set argmax would be "
                                                "admitted by top_k, pass always admitted",
            "chosen_reference_rank": "reference (frozen M1) rank of the action actually chosen; mass "
                                     "at rank top_k-1 means the cap is binding",
        },
        "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        **acc.report(),
    }
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="docs/reports/stage-c-pruning-audit.json")
    parser.add_argument("--decisions", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=None,
                        help="override the checkpoint's top_k (DMC checkpoints default to 32)")
    parser.add_argument("--coverage-ks", type=int, nargs="+", default=(16, 32, 48, 64, 96, 128))
    parser.add_argument("--max-seconds", type=float, default=3600.0)
    args = parser.parse_args(argv)
    report = audit(args.checkpoint, decisions=args.decisions, num_envs=args.num_envs,
                   threads=args.threads, torch_threads=args.torch_threads, seed=args.seed,
                   device=args.device, top_k=args.top_k, coverage_ks=tuple(args.coverage_ks),
                   max_seconds=args.max_seconds)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    d = report["decisions_with_pruning"]["all"]
    v = report["full_argmax_outside_pruned_set"]["all"]
    c = report["chosen_reference_rank"]
    print(f"{report['play_decisions']} play decisions in {report['elapsed_seconds']:.0f}s "
          f"({report['decisions_per_second']:.0f}/s), status {report['status']}")
    print(f"pruning active: {d['fraction']:.3f} of decisions; "
          f"full-set argmax outside pruned set: {v['fraction']:.4f}")
    print(f"chosen rank: rank0 {c['rank0_fraction']:.3f}, boundary {c['at_boundary_fraction']:.4f}, "
          f"last four ranks {c['last_four_ranks_fraction']:.4f}")
    for k, table in report["coverage_of_full_argmax_at_top_k"].items():
        print(f"coverage at top {k}: {table['all']['fraction']:.4f}")
    print(f"written {output}")


if __name__ == "__main__":
    main()
