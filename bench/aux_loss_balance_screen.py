"""CPU aux-loss balance screen on saved PPO minibatches; never steps weights.

On each fixture (embedded actor + recorded minibatch labels), compute public-
encoder gradient norms and pairwise cosines for the PPO surrogate and the
three auxiliary heads (next / belief / outcome). When next dominates and
conflicts with PPO, also report single-knob next-coefficient ablations that
leave belief and outcome at their configured ``aux_coef``.

Usage::

    PYTHONPATH=python:oracle:. .venv/bin/python bench/aux_loss_balance_screen.py \\
        --fixtures .work/aux-loss-balance-2026-10-20/fixtures/*.pt \\
        --out .work/aux-loss-balance-2026-10-20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from bench.history_training_signal import parameter_group, policy_surrogate
from bench.qknorm_match_ablation import sha
from bench.qknorm_mechanism import unpack
from train.history_aux import belief_loss, next_token_loss, outcome_loss
from train.history_model import HistoryActor, segment_log_softmax
from train.history_ppo import HistoryPPOConfig

LOSS_NAMES = ("ppo", "next", "belief", "outcome")
# Default includes lower / mid-anneal / near-off / off; belief and outcome stay at aux_coef.
DEFAULT_NEXT_COEFS = (0.1, 0.05, 0.03, 0.01, 0.0)


def encoder_mask(named: list[tuple[str, torch.nn.Parameter]]) -> list[bool]:
    return [parameter_group(name) == "encoder" for name, _ in named]


def flat_encoder(grads: list[torch.Tensor | None],
                 params: list[torch.nn.Parameter],
                 mask: list[bool]) -> torch.Tensor:
    parts = []
    for keep, grad, param in zip(mask, grads, params):
        if not keep:
            continue
        parts.append((grad if grad is not None else torch.zeros_like(param)).detach()
                     .flatten().double())
    if not parts:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat(parts)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    denom = (a.norm() * b.norm()).clamp_min(1e-300)
    return float(torch.dot(a, b) / denom)


def group_norms(grads: list[torch.Tensor | None],
                named: list[tuple[str, torch.nn.Parameter]]) -> dict[str, float]:
    totals = {key: 0.0 for key in
              ("encoder", "private", "reader", "policy_fusion", "aux_heads")}
    for (name, _), grad in zip(named, grads):
        if grad is None:
            continue
        totals[parameter_group(name)] += float(grad.double().square().sum())
    return {key: value ** 0.5 for key, value in totals.items()}


def next_token_parts(actor: HistoryActor, inputs: Any, encoded: list) -> tuple[torch.Tensor, ...]:
    parts = []
    for memory, group in encoded:
        ids = group.matches if group is not None else torch.arange(len(inputs.streams.lengths))
        count = memory.shape[1] - 1
        tokens = inputs.streams.tokens[ids, :count]
        local = group.local_match if group is not None else inputs.match_index
        rows = group.rows if group is not None else torch.arange(inputs.decisions)
        cited = torch.zeros(len(ids), dtype=torch.long).scatter_reduce(
            0, local, inputs.prefix[rows], reduce="amax")
        lengths = inputs.streams.lengths[ids].clamp(max=count)
        limit = torch.minimum(lengths, cited)
        valid = torch.arange(count)[None] < limit[:, None]
        logits = actor.next_logits(memory[:, :count])
        parts.append((logits.reshape(-1, logits.shape[-1]),
                      tokens.reshape(-1, tokens.shape[-1]),
                      valid.reshape(-1)))
    if len(parts) == 1:
        return parts[0]
    return tuple(torch.cat([part[i] for part in parts]) for i in range(3))


def screen_fixture(path: Path, next_coefs: tuple[float, ...] = DEFAULT_NEXT_COEFS) -> dict:
    """Backward-only screen; parameters and ``.grad`` stay unchanged."""
    torch.set_num_threads(1)
    fixture = torch.load(path, map_location="cpu", weights_only=False)
    cfg = HistoryPPOConfig.from_payload(fixture["config"])
    actor = HistoryActor(cfg.policy_config())
    actor.load_state_dict(fixture["actor"], strict=True)
    actor.requires_grad_(True).train()
    actor.causal_sdpa = True
    actor.batched_match_attention = True
    before = {key: value.detach().clone() for key, value in actor.state_dict().items()}
    inputs = unpack(fixture["batch"]["inputs"])
    fields = fixture["batch"]["fields"]
    named = list(actor.named_parameters())
    params = [param for _, param in named]
    mask = encoder_mask(named)

    with sdpa_kernel(SDPBackend.MATH):
        encoded: list = []
        state = actor.decision_states(None, inputs, encoded)
        logits = actor.candidate_logits(state, inputs.cand, inputs.offsets, rows=inputs.rows)
        all_logp = segment_log_softmax(logits, inputs.rows, inputs.decisions)
        logp = all_logp[fixture["batch"]["chosen"]]
        old = fields["logp"]
        behaviour = fields["behaviour_logp"]
        per_row = policy_surrogate(logp, old, behaviour, fields["advantage"], cfg,
                                   fixture["advantage_moments"])
        raw = {
            "ppo": per_row.mean(),
            "next": next_token_loss(*next_token_parts(actor, inputs, encoded))["next_loss"],
            "belief": belief_loss(actor.belief_logits(state), fields["hidden"])["belief_loss"],
            "outcome": outcome_loss(actor.outcome_value(state),
                                    fields["outcome_target"])["outcome_loss"],
        }
        vectors: dict[str, torch.Tensor] = {}
        norms: dict[str, dict[str, float]] = {}
        for name, loss in raw.items():
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            norms[name] = group_norms(grads, named)
            vectors[name] = flat_encoder(grads, params, mask)
            assert all(grad is None or torch.isfinite(grad).all() for grad in grads), name

        pairwise = {a: {b: cosine(vectors[a], vectors[b]) for b in LOSS_NAMES}
                    for a in LOSS_NAMES}
        conflicts = {
            f"{a}_vs_{b}": pairwise[a][b] < 0
            for i, a in enumerate(LOSS_NAMES) for b in LOSS_NAMES[i + 1:]
        }
        aux_coef = float(cfg.aux_coef)
        weighted_encoder = {
            "ppo": norms["ppo"]["encoder"],
            "next": aux_coef * norms["next"]["encoder"],
            "belief": aux_coef * norms["belief"]["encoder"],
            "outcome": aux_coef * norms["outcome"]["encoder"],
        }
        dominance = {
            "next_over_ppo": weighted_encoder["next"] / max(weighted_encoder["ppo"], 1e-300),
            "belief_over_ppo": weighted_encoder["belief"] / max(weighted_encoder["ppo"], 1e-300),
            "outcome_over_ppo": weighted_encoder["outcome"] / max(weighted_encoder["ppo"], 1e-300),
        }

        ablations = []
        for next_coef in next_coefs:
            # Single knob: scale next only; belief/outcome stay at aux_coef.
            total = (raw["ppo"]
                     + next_coef * raw["next"]
                     + aux_coef * raw["belief"]
                     + aux_coef * raw["outcome"])
            grads = torch.autograd.grad(total, params, retain_graph=True, allow_unused=True)
            total_vec = flat_encoder(grads, params, mask)
            enc_norm = float(total_vec.norm())
            ablations.append(dict(
                next_coef=next_coef,
                belief_coef=aux_coef,
                outcome_coef=aux_coef,
                encoder_grad_norm=enc_norm,
                cosine_with_ppo=cosine(total_vec, vectors["ppo"]),
                ppo_encoder_fraction=(
                    float(vectors["ppo"].norm()) / max(enc_norm, 1e-300)),
            ))

    assert all(torch.equal(value, before[key]) for key, value in actor.state_dict().items())
    assert all(param.grad is None for param in actor.parameters())
    clip_fraction = float(((logp - old).exp().sub(1).abs() > cfg.clip).float().mean())
    return dict(
        path=str(path),
        sha256=sha(path),
        decisions=int(inputs.decisions),
        streams=int(len(inputs.streams.lengths)),
        prefix_max=int(inputs.prefix.max()),
        aux_coef=aux_coef,
        aux_heads=cfg.aux_heads,
        clip_fraction=clip_fraction,
        raw_loss={name: float(loss.detach()) for name, loss in raw.items()},
        encoder_grad_norm={name: norms[name]["encoder"] for name in LOSS_NAMES},
        group_grad_norm=norms,
        weighted_encoder_grad_norm=weighted_encoder,
        dominance_vs_ppo=dominance,
        pairwise_encoder_cosine=pairwise,
        conflict=conflicts,
        next_coef_ablations=ablations,
        parameters_unchanged=True,
    )


def summarize(results: list[dict]) -> dict:
    """Aggregate dominance and conflict across fixtures."""
    def mean(key_path: tuple[str, ...]) -> float:
        values = []
        for row in results:
            cur: Any = row
            for key in key_path:
                cur = cur[key]
            values.append(float(cur))
        return float(np.mean(values))

    conflict_rates = {}
    for i, a in enumerate(LOSS_NAMES):
        for b in LOSS_NAMES[i + 1:]:
            key = f"{a}_vs_{b}"
            conflict_rates[key] = float(np.mean([1.0 if row["conflict"][key] else 0.0
                                                 for row in results]))
    ablations = []
    coefs = [item["next_coef"] for item in results[0]["next_coef_ablations"]]
    for index, coef in enumerate(coefs):
        ablations.append(dict(
            next_coef=coef,
            mean_cosine_with_ppo=float(np.mean(
                [row["next_coef_ablations"][index]["cosine_with_ppo"] for row in results])),
            mean_encoder_grad_norm=float(np.mean(
                [row["next_coef_ablations"][index]["encoder_grad_norm"] for row in results])),
            mean_ppo_encoder_fraction=float(np.mean(
                [row["next_coef_ablations"][index]["ppo_encoder_fraction"] for row in results])),
        ))
    next_conflicts = [row["pairwise_encoder_cosine"]["next"]["ppo"] < 0 for row in results]
    next_dominates = [row["dominance_vs_ppo"]["next_over_ppo"] > 1.0 for row in results]
    n = len(results)
    conflict_rate = (sum(next_conflicts) / n) if n else 0.0
    # Stable enough for a cheap train proposal: next always dominates the
    # encoder and conflicts with PPO on a majority of real minibatches.
    warrant = bool(n and all(next_dominates) and conflict_rate >= 0.5)
    return dict(
        fixtures=n,
        mean_weighted_encoder_grad_norm={
            name: mean(("weighted_encoder_grad_norm", name)) for name in LOSS_NAMES},
        mean_dominance_vs_ppo={
            key: mean(("dominance_vs_ppo", key))
            for key in ("next_over_ppo", "belief_over_ppo", "outcome_over_ppo")},
        mean_pairwise_encoder_cosine={
            a: {b: mean(("pairwise_encoder_cosine", a, b)) for b in LOSS_NAMES}
            for a in LOSS_NAMES},
        conflict_frequency=conflict_rates,
        next_dominates_and_conflicts_ppo=bool(all(next_dominates) and all(next_conflicts)),
        fixtures_next_dominates=int(sum(next_dominates)),
        fixtures_next_conflicts_ppo=int(sum(next_conflicts)),
        next_ppo_conflict_rate=conflict_rate,
        next_coef_ablations=ablations,
        recommendation=(
            "Single-knob next ablations warranted: lower / anneal / off next only."
            if warrant else
            "No majority next-dominates-and-conflicts-PPO pattern across fixtures."
        ),
        train_ablation_proposal=(
            dict(
                knob="next_coef",
                keep_fixed=("belief_coef=aux_coef", "outcome_coef=aux_coef",
                            "response_coef", "ppo hyperparameters"),
                arms=("next_coef=0.1 (control)", "next_coef=0.03",
                      "anneal 0.1→0 over ~1k updates", "next_coef=0 (off)"),
                note=("Requires a next-specific coefficient; shared aux_coef would "
                      "also scale belief/outcome and is not a single-knob next change."),
            ) if warrant else None),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--next-coefs", nargs="+", type=float, default=list(DEFAULT_NEXT_COEFS))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for path in args.fixtures:
        print(f"screening {path}", flush=True)
        row = screen_fixture(path, tuple(args.next_coefs))
        results.append(row)
        (args.out / f"{path.stem}.json").write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps({
            "fixture": path.name,
            "dominance": row["dominance_vs_ppo"],
            "cosine_next_ppo": row["pairwise_encoder_cosine"]["next"]["ppo"],
            "conflict_next_ppo": row["conflict"]["ppo_vs_next"],
        }, indent=2), flush=True)
    summary = summarize(results)
    payload = dict(results=results, summary=summary)
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
