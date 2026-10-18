"""Multi-batch audit of the history actor's private cross-attention (CPU only).

Extends the single-fixture diagnosis of
``docs/reports/private-attention-diagnosis-2026-10-15.md`` to many self-play
minibatches. One sample (whole matches from a CPU self-play collection, split
into learner-style minibatches of whole matches) is collected once and every
audited actor is measured on the same inputs:

* saturation: per head and decision, FP64 softmax entropy, top probability,
  top-1 minus top-2 score gap and that gap in FP32 ulps of the top score;
* landing: BOS, last visible token, current round or an earlier round, and how
  many distinct positions one head selects across one match's decisions;
* mechanism: pairwise cosine of each head's q across decisions, whether the
  argmax survives replacing every q by the minibatch mean, the q_proj input's
  mean/residual decomposition and q_proj/k singular values;
* history sensitivity: decisions in round >= 1 re-scored with the tokens of
  their earlier rounds shuffled or swapped for another match's tokens (round,
  phase and length kept), plus a whole-stream shuffle as positive control;
* learning signal: math-SDPA gradient norms (ratio-1 PPO policy loss and,
  with an outcome head, the outcome loss) per module and per q/k head, and the
  checkpoint's Adam second moments when present;
* dead ReLU units in the private block and the encoder feed-forward layers.

Every number is stratified by round bucket and by tribute round (the decision's
current round contains a public exchange token). The explicit recomputation is
checked against ``HistoryActor.decision_states`` on every minibatch.

Actors are named by checkpoint path or ``fresh:SEED[:NORM]`` (random init with
``--fresh-*`` sizes and ``private_qk_norm=NORM``). ``--stress`` turns fresh actors
into the trained regime's numerics (a large constant query component and
scaled q/k projections) to test normalization arms on identical weights.

    PYTHONPATH=python:oracle:. python -m eval.private_attention_audit \\
        --out .work/attn-audit CKPT_A CKPT_B --sampler CKPT_A
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import gd
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from train.history_aux import round_outcomes
from train.history_model import (PLAY_PHASE, DecisionInputs, HistoryActor, HistoryPolicyConfig,
                                 StreamBatch, fresh_player, load_history_checkpoint,
                                 segment_log_softmax)
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer

ROUND_BUCKETS = ((0, 0, "round 0"), (1, 3, "rounds 1-3"), (4, 7, "rounds 4-7"),
                 (8, 1 << 30, "rounds 8+"))
PERTURBATIONS = ("identity", "swap_earlier", "shuffle_earlier", "shuffle_all")
ENTROPY_COEF = 0.03
SUBNORMAL_FLOOR = 1e-20
BATCH_FIELDS = ("tokens", "rounds", "phases", "lengths", "match_index", "prefix", "obs", "seat",
                "cand", "offsets", "chosen", "advantage", "outcome", "round")


# ---- sample -----------------------------------------------------------------

def collect_sample(actor: HistoryActor, *, envs: int, steps: int, seed: int, matches: int,
                   minibatch_matches: int, rows_per_match: int) -> list[dict[str, Tensor]]:
    """Self-play on CPU with ``actor`` in all seats; whole-match minibatches.

    Rows of each match are its completed-round decisions, subsampled evenly in
    time to ``rows_per_match`` (0 keeps all) so every round stays represented.
    """
    env = gd.VecEnv(num_envs=envs, num_threads=1, seed=seed, log_public_actions=True,
                    log_env_limit=envs)
    store, buffer = MatchEventStore(), SequenceRolloutBuffer()
    collector = HistoryCollector(env, actor, store, buffer,
                                 torch.Generator().manual_seed(seed), kv_cache=True)
    with torch.no_grad():
        collector.collect(steps)
    buffer.finalize(np.zeros(len(buffer), np.float32))
    groups = buffer.match_groups()
    keys = sorted(groups)[:matches] if matches > 0 else sorted(groups)
    data = buffer.compact()
    batches = []
    for begin in range(0, len(keys), minibatch_matches):
        picked = []
        for key in keys[begin:begin + minibatch_matches]:
            rows = np.sort(groups[key])
            if rows_per_match and len(rows) > rows_per_match:
                rows = rows[np.unique(np.linspace(0, len(rows) - 1, rows_per_match).round()
                                      .astype(np.int64))]
            picked.append(rows)
        rows = np.concatenate(picked)
        extra = {"outcome": round_outcomes(buffer, rows).astype(np.float32),
                 "round": data["round"][rows].astype(np.int64)}
        batch = buffer.training_batch(rows, store, "cpu", fields=("advantage",), extra=extra)
        i = batch.inputs
        values = (i.streams.tokens, i.streams.rounds, i.streams.phases, i.streams.lengths,
                  i.match_index, i.prefix, i.obs, i.seat, i.cand, i.offsets, batch.chosen,
                  batch.fields["advantage"], batch.fields["outcome"], batch.fields["round"])
        batches.append({name: value.clone() for name, value in zip(BATCH_FIELDS, values)})
    return batches


def batch_inputs(batch: dict[str, Tensor]) -> DecisionInputs:
    return DecisionInputs(streams=StreamBatch(batch["tokens"], batch["rounds"], batch["phases"],
                                              batch["lengths"]),
                          match_index=batch["match_index"], prefix=batch["prefix"],
                          obs=batch["obs"], seat=batch["seat"], cand=batch["cand"],
                          offsets=batch["offsets"])


def tribute_round(batch: dict[str, Tensor]) -> np.ndarray:
    """Whether each decision's current round has a public exchange token."""
    rounds, phases = batch["rounds"].numpy(), batch["phases"].numpy()
    match, prefix, current = (batch["match_index"].numpy(), batch["prefix"].numpy(),
                              batch["round"].numpy())
    out = np.zeros(len(match), bool)
    for i, (m, p, r) in enumerate(zip(match, prefix, current)):
        seen = slice(0, int(p))
        out[i] = bool(((rounds[m, seen] == r) & (phases[m, seen] != PLAY_PHASE)).any())
    return out


# ---- actors -----------------------------------------------------------------

@dataclass(frozen=True)
class Stress:
    """Synthetic trained-regime numerics for a fresh actor (identical across norm arms)."""
    query_mean: float = 2700.0   # norm of a constant added to every private query
    q_gain: float = 1.0          # q_proj weight scale
    k_gain: float = 1.0          # key rows of kv_proj scale
    seed: int = 0

    @staticmethod
    def parse(spec: str) -> "Stress":
        mean, q_gain, k_gain = (float(x) for x in spec.split(":"))
        return Stress(mean, q_gain, k_gain)


def apply_stress(actor: HistoryActor, stress: Stress) -> None:
    width = actor.config.width
    direction = torch.randn(width, generator=torch.Generator().manual_seed(stress.seed))
    with torch.no_grad():
        actor.seat.weight.add_(direction / direction.norm() * stress.query_mean)
        actor.q_proj.weight.mul_(stress.q_gain)
        actor.kv_proj.weight[:width].mul_(stress.k_gain)


def load_actor(spec: str, fresh: HistoryPolicyConfig, stress: Stress | None
               ) -> tuple[HistoryActor, dict[str, Any] | None]:
    if spec.startswith("fresh:"):
        parts = spec.split(":")
        config = replace(fresh, private_qk_norm=parts[2] if len(parts) > 2 else "none")
        actor, _ = fresh_player(config, int(parts[1]))
        if stress is not None:
            apply_stress(actor, stress)
        return actor.eval(), None
    actor, _, payload = load_history_checkpoint(spec, "cpu")
    return actor.eval(), payload.get("optimizer", {}).get("actor")


# ---- explicit attention ---------------------------------------------------------

def capture(actor: HistoryActor, inputs: DecisionInputs, chunk: int = 32,
            hooks: dict[str, Any] | None = None) -> dict[str, Any]:
    """Recompute the private attention with FP64 scores; per (row, head) arrays."""
    config = actor.config
    if config.window or config.style_input:
        raise ValueError("the audit covers full-history actors without the style input")
    heads, width = config.heads, config.width
    depth = width // heads
    with torch.no_grad():
        query = actor.private(inputs.obs.float()) + actor.seat(inputs.seat)
        handles = []
        if hooks is not None:
            lengths = inputs.streams.lengths
            for index, layer in enumerate(actor.stream.layers):
                def record(module, args, output, index=index):
                    valid = (torch.arange(output.shape[1])[None] <= lengths[:, None])
                    hooks.setdefault(f"encoder.{index}", []).append((output[valid] > 0))
                handles.append(layer.linear1.register_forward_hook(record))
        try:
            encoded = actor.encode_batch(inputs.streams)
        finally:
            for handle in handles:
                handle.remove()
        keys, values = actor.project_keys_values(encoded)
        q = actor.project_query(query)
        q_mean = q.mean(0, keepdim=True)
        k = keys.view(*keys.shape[:2], heads, depth)
        v = values.view(*values.shape[:2], heads, depth)
        n, positions = len(query), encoded.shape[1]
        rows_out = defaultdict(list)
        head_out = torch.empty(n, heads, depth)
        for start in range(0, n, chunk):
            rows = slice(start, min(start + chunk, n))
            match, prefix = inputs.match_index[rows], inputs.prefix[rows]
            kk = k[match].double()
            allowed = torch.arange(positions)[None] <= prefix[:, None]
            scale = 1.0 / math.sqrt(depth)
            scores = torch.einsum("chd,cshd->chs", q[rows].double(), kk) * scale
            scores = scores.masked_fill(~allowed[:, None], float("-inf"))
            mean_scores = torch.einsum("hd,cshd->chs", q_mean[0].double(), kk) * scale
            mean_scores = mean_scores.masked_fill(~allowed[:, None], float("-inf"))
            probs = scores.softmax(-1)
            top = scores.topk(min(2, positions), -1)
            top1 = top.values[..., 0]
            gap = (top1 - top.values[..., 1]) if positions > 1 else torch.full_like(top1, math.inf)
            gap = torch.where(prefix[:, None] >= 1, gap, torch.full_like(gap, math.inf))
            exponent = torch.frexp(top1.float())[1].double()
            key_norm = kk.norm(dim=-1).permute(0, 2, 1)                 # [c, h, S]
            key_norm = key_norm.masked_fill(~allowed[:, None], float("nan"))
            argmax = top.indices[..., 0]
            rows_out["entropy"].append(-torch.special.xlogy(probs, probs).sum(-1))
            rows_out["max_prob"].append(probs.max(-1).values)
            rows_out["gap"].append(gap)
            rows_out["gap_ulp"].append(gap / torch.pow(2.0, exponent - 24))
            rows_out["top1_abs"].append(top1.abs())
            rows_out["argmax"].append(argmax)
            rows_out["mean_q_same"].append(mean_scores.argmax(-1) == argmax)
            rows_out["key_ratio"].append(key_norm.gather(-1, argmax[..., None])[..., 0]
                                         / key_norm.nanmedian(-1).values)
            probs32 = scores.float().softmax(-1)
            head_out[rows] = torch.einsum("chs,cshd->chd", probs32, v[match])
        attended = actor.out_proj(head_out.reshape(n, width))
        state = actor.attention_norm(query + attended)
        hidden = actor.feed_forward[1](actor.feed_forward[0](state))
        state = actor.output_norm(state + actor.feed_forward[2](hidden))
        if hooks is not None:
            hooks.setdefault("private_block", []).append(hidden > 0)
        reference = actor.decision_states(encoded, inputs)
        logits = actor.candidate_logits(state, inputs.cand, inputs.offsets, rows=inputs.rows)
        log_probs = segment_log_softmax(logits, inputs.rows, n)
        q_input = actor.query_norm(query) if actor.query_norm is not None else query
    out = {name: torch.cat(parts).numpy() for name, parts in rows_out.items()}
    q_unit = F.normalize(q, dim=-1)
    pick = torch.randperm(n, generator=torch.Generator().manual_seed(0))[:256]
    cos = torch.einsum("ahd,bhd->hab", q_unit[pick], q_unit[pick])
    upper = torch.triu(torch.ones(len(pick), len(pick), dtype=torch.bool), 1)
    out.update(
        head_out=head_out.numpy(), attended=attended.numpy(), log_probs=log_probs.numpy(),
        q_cos_median=cos[:, upper].median(-1).values.numpy() if len(pick) > 1
        else np.ones(heads),
        query_norm=query.norm(dim=-1).numpy(), query_mean_norm=float(query.mean(0).norm()),
        query_residual_norm=(query - query.mean(0)).norm(dim=-1).numpy(),
        q_input_norm=q_input.norm(dim=-1).numpy(),
        state_error=float((state - reference).abs().max()))
    return out


def landing(batch: dict[str, Tensor], captured: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per (row, head): BOS, last visible, current round, earlier round."""
    argmax = captured["argmax"]
    prefix = batch["prefix"].numpy()[:, None]
    match = batch["match_index"].numpy()
    rounds = batch["rounds"].numpy()
    current = batch["round"].numpy()[:, None]
    token = np.clip(argmax - 1, 0, None)
    token_round = rounds[match[:, None], token]
    bos = argmax == 0
    return {"bos": bos, "last": (argmax == prefix) & (prefix > 0),
            "current_round": ~bos & (token_round == current),
            "earlier_round": ~bos & (token_round < current)}


def distinct_per_match(batch: dict[str, Tensor], argmax: np.ndarray, minimum: int = 4
                       ) -> np.ndarray:
    """``[matches, heads]`` distinct landing positions over each match's decisions
    (absolute positions; NaN for matches with fewer than ``minimum`` decisions)."""
    match = batch["match_index"].numpy()
    out = np.full((int(match.max()) + 1, argmax.shape[1]), np.nan)
    for m in np.unique(match):
        rows = match == m
        if rows.sum() >= minimum:
            out[m] = [len(np.unique(argmax[rows, h])) for h in range(argmax.shape[1])]
    return out


# ---- history perturbations ----------------------------------------------------

def perturbed_inputs(batch: dict[str, Tensor], kind: str, seed: int) -> DecisionInputs:
    """One stream per (match, decision round): the round's decisions see their
    earlier rounds' token contents perturbed; round index, phase and length kept."""
    rng = np.random.default_rng(seed)
    tokens, rounds, phases = (batch[name].numpy() for name in ("tokens", "rounds", "phases"))
    lengths, match = batch["lengths"].numpy(), batch["match_index"].numpy()
    prefix, current = batch["prefix"].numpy(), batch["round"].numpy()
    keys = sorted(set(zip(match.tolist(), current.tolist())))
    index = {key: i for i, key in enumerate(keys)}
    streams = []
    for m, r in keys:
        n = int(prefix[(match == m) & (current == r)].max())
        t, rr, pp = tokens[m, :n].copy(), rounds[m, :n], phases[m, :n]
        earlier = np.flatnonzero(rr < r)
        if kind == "shuffle_earlier" and len(earlier):
            t[earlier] = t[rng.permutation(earlier)]
        elif kind == "swap_earlier" and len(earlier):
            donor = tokens[(m + 1) % len(lengths), :int(lengths[(m + 1) % len(lengths)])]
            t[earlier] = np.resize(donor, (len(earlier), donor.shape[1])) if len(donor) else 0
        elif kind == "shuffle_all" and r > 0:
            t = t[rng.permutation(n)]
        elif kind != "identity" and kind not in PERTURBATIONS:
            raise ValueError(f"unknown perturbation {kind}")
        streams.append((t, rr, pp))
    remap = torch.as_tensor([index[(int(m), int(r))] for m, r in zip(match, current)])
    return DecisionInputs(streams=StreamBatch.from_arrays(streams, "cpu"), match_index=remap,
                          prefix=batch["prefix"], obs=batch["obs"], seat=batch["seat"],
                          cand=batch["cand"], offsets=batch["offsets"])


def sensitivity(base: dict[str, Any], other: dict[str, Any], batch: dict[str, Tensor]
                ) -> dict[str, np.ndarray]:
    offsets = batch["offsets"].numpy()
    counts = np.diff(offsets)
    rows = np.repeat(np.arange(len(counts)), counts)
    p, q = np.exp(base["log_probs"]), np.exp(other["log_probs"])
    tv = 0.5 * np.bincount(rows, np.abs(p - q), minlength=len(counts))
    greedy = np.array([np.argmax(base["log_probs"][a:b]) != np.argmax(other["log_probs"][a:b])
                       for a, b in zip(offsets[:-1], offsets[1:])]) & (counts > 1)
    chosen = batch["chosen"].numpy()
    norm = lambda x: np.linalg.norm(x, axis=-1)
    return {"tv": tv, "greedy": greedy,
            "chosen_dlogp": np.abs(base["log_probs"][chosen] - other["log_probs"][chosen]),
            "moved": base["argmax"] != other["argmax"],
            "head_rel": norm(other["head_out"] - base["head_out"])
            / np.maximum(norm(base["head_out"]), 1e-30),
            "attended_rel": norm(other["attended"] - base["attended"])
            / np.maximum(norm(base["attended"]), 1e-30)}


# ---- learning signal -----------------------------------------------------------

def parameter_groups(actor: HistoryActor) -> dict[str, list[tuple[str, Any]]]:
    """Module group -> [(parameter name, row slice or None)]."""
    width, heads = actor.config.width, actor.config.heads
    depth = width // heads
    groups: dict[str, list] = defaultdict(list)
    for name, _ in actor.named_parameters():
        if name.startswith("q_proj."):
            groups["q_proj"].append((name, None))
            for h in range(heads):
                groups[f"q_proj.head{h}"].append((name, slice(h * depth, (h + 1) * depth)))
        elif name.startswith("kv_proj."):
            groups["k_proj"].append((name, slice(0, width)))
            groups["v_proj"].append((name, slice(width, 2 * width)))
            for h in range(heads):
                groups[f"k_proj.head{h}"].append((name, slice(h * depth, (h + 1) * depth)))
        elif name.startswith("out_proj."):
            groups["out_proj"].append((name, None))
        elif name.startswith(("query_norm.", "q_norm.", "k_norm.")):
            groups["qk_norm"].append((name, None))
        elif name.startswith("stream.layers."):
            groups[f"encoder.{name.split('.')[2]}"].append((name, None))
        elif name.startswith(("public.", "round_embedding.", "phase_embedding.", "bos")):
            groups["encoder.embedding"].append((name, None))
        elif name.startswith("stream_norm."):
            groups["encoder.norm"].append((name, None))
        elif name.startswith(("private.", "seat.")):
            groups["private_mlp"].append((name, None))
        elif name.startswith(("attention_norm.", "feed_forward.", "output_norm.")):
            groups["private_block"].append((name, None))
        elif name.startswith(("action_tower.", "fusion.")):
            groups["decision_head"].append((name, None))
        else:
            groups[name.split(".")[0]].append((name, None))
    return dict(groups)


def _pick(tensor: Tensor, rows: slice | None) -> Tensor:
    return tensor if rows is None else tensor[rows]


def gradients(actor: HistoryActor, batch: dict[str, Tensor]) -> dict[str, dict[str, float]]:
    """Gradient norms under math SDPA: ratio-1 PPO policy loss and outcome loss."""
    inputs = batch_inputs(batch)
    groups = parameter_groups(actor)
    params = dict(actor.named_parameters())
    out = {}
    losses = ["policy"] + (["outcome"] if "outcome" in actor.aux_head_names else [])
    for loss_name in losses:
        actor.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.MATH):
            state = actor.decision_states(None, inputs)
            if loss_name == "policy":
                rows = inputs.rows
                log_probs = segment_log_softmax(
                    actor.candidate_logits(state, inputs.cand, inputs.offsets, rows=rows),
                    rows, inputs.decisions)
                entropy = -torch.zeros(inputs.decisions).index_add(
                    0, rows, log_probs.exp() * log_probs)
                advantage = batch["advantage"]
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                loss = (-(advantage * log_probs[batch["chosen"]]).mean()
                        - ENTROPY_COEF * entropy.mean())
            else:
                loss = F.mse_loss(actor.outcome_value(state), batch["outcome"])
        loss.backward()
        norms = {"loss": float(loss.detach())}
        for group, members in groups.items():
            total = sum(float(_pick(params[name].grad, rows).double().square().sum())
                        for name, rows in members if params[name].grad is not None)
            norms[group] = math.sqrt(total)
        out[loss_name] = norms
    actor.zero_grad(set_to_none=True)
    return out


def adam_state(actor: HistoryActor, optimizer: dict[str, Any] | None
               ) -> dict[str, dict[str, float]] | None:
    """sqrt(v) median, |m|/sqrt(v) median and the share of sqrt(v) below the
    FP32 subnormal floor, per module group (parameter order = named_parameters)."""
    if not optimizer or not optimizer.get("state"):
        return None
    names = [name for name, _ in actor.named_parameters()]
    state = {names[int(i)]: s for i, s in optimizer["state"].items() if int(i) < len(names)}
    out = {}
    for group, members in parameter_groups(actor).items():
        root, ratio = [], []
        for name, rows in members:
            if name not in state or "exp_avg_sq" not in state[name]:
                continue
            v = _pick(state[name]["exp_avg_sq"], rows).double().sqrt().reshape(-1)
            m = _pick(state[name]["exp_avg"], rows).double().abs().reshape(-1)
            root.append(v)
            ratio.append(m[v > 0] / v[v > 0])
        if root:
            root_all, ratio_all = torch.cat(root), torch.cat(ratio)
            out[group] = {"sqrt_v_median": float(root_all.median()),
                          "ratio_median": float(ratio_all.median()) if len(ratio_all) else 0.0,
                          "floor_share": float((root_all < SUBNORMAL_FLOOR).double().mean())}
    return out


def singular_values(actor: HistoryActor) -> dict[str, list[float]]:
    width, heads = actor.config.width, actor.config.heads
    depth = width // heads
    q, k = actor.q_proj.weight.detach().double(), actor.kv_proj.weight.detach()[:width].double()
    top2 = lambda w: torch.linalg.svdvals(w)[:2].tolist()
    return {"q_proj": top2(q), "k_proj": top2(k),
            "q_head_sigma1": [top2(q[h * depth:(h + 1) * depth])[0] for h in range(heads)],
            "k_head_sigma1": [top2(k[h * depth:(h + 1) * depth])[0] for h in range(heads)]}


# ---- audit -------------------------------------------------------------------

def _quantile(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if len(x) else float("nan")


def head_summary(rows: dict[str, np.ndarray], mask: np.ndarray) -> list[dict[str, float]]:
    """Per head over the decisions in ``mask`` that see at least one token."""
    out = []
    for h in range(rows["entropy"].shape[1]):
        pick = lambda name: rows[name][mask, h]
        out.append({
            "decisions": int(mask.sum()),
            "entropy_q99": _quantile(pick("entropy"), 0.99),
            "max_prob_q01": _quantile(pick("max_prob"), 0.01),
            "one_hot_share": float(np.mean(pick("max_prob") > 1 - 1e-6)) if mask.any() else 0.0,
            "gap_min": _quantile(pick("gap"), 0.0),
            "gap_median": _quantile(pick("gap"), 0.5),
            "gap_below_ulp_share": float(np.mean(pick("gap_ulp") < 1)) if mask.any() else 0.0,
            "top1_abs_median": _quantile(pick("top1_abs"), 0.5),
            "bos_share": float(np.mean(pick("bos"))) if mask.any() else 0.0,
            "last_share": float(np.mean(pick("last"))) if mask.any() else 0.0,
            "current_round_share": float(np.mean(pick("current_round"))) if mask.any() else 0.0,
            "earlier_round_share": float(np.mean(pick("earlier_round"))) if mask.any() else 0.0,
            "mean_q_same_share": float(np.mean(pick("mean_q_same"))) if mask.any() else 0.0,
            "key_ratio_median": _quantile(pick("key_ratio"), 0.5),
        })
    return out


def audit_actor(actor: HistoryActor, batches: list[dict[str, Tensor]],
                optimizer: dict[str, Any] | None = None, seed: int = 20261018,
                perturbations: tuple[str, ...] = PERTURBATIONS, grads: bool = True
                ) -> dict[str, Any]:
    rows, mechanism, grad_rows, hooks = defaultdict(list), defaultdict(list), [], {}
    distinct, sens = [], {kind: defaultdict(list) for kind in perturbations}
    sens_mask = []
    for b, batch in enumerate(batches):
        inputs = batch_inputs(batch)
        captured = capture(actor, inputs, hooks=hooks)
        for name in ("entropy", "max_prob", "gap", "gap_ulp", "top1_abs", "argmax",
                     "mean_q_same", "key_ratio"):
            rows[name].append(captured[name])
        for name, value in landing(batch, captured).items():
            rows[name].append(value)
        heads = captured["entropy"].shape[1]
        rows["round"].append(batch["round"].numpy())
        rows["tribute"].append(tribute_round(batch))
        rows["prefix"].append(batch["prefix"].numpy())
        rows["batch"].append(np.full(len(batch["prefix"]), b))
        distinct.append(distinct_per_match(batch, captured["argmax"]))
        mechanism["state_error"].append(captured["state_error"])
        mechanism["q_cos_median"].append(captured["q_cos_median"])
        mechanism["query_norm_median"].append(float(np.median(captured["query_norm"])))
        mechanism["query_mean_norm"].append(captured["query_mean_norm"])
        mechanism["query_residual_median"].append(
            float(np.median(captured["query_residual_norm"])))
        mechanism["q_input_norm_median"].append(float(np.median(captured["q_input_norm"])))
        with_history = batch["round"].numpy() >= 1
        sens_mask.append(with_history)
        for kind in perturbations:
            other = capture(actor, perturbed_inputs(batch, kind, seed + b))
            for name, value in sensitivity(captured, other, batch).items():
                sens[kind][name].append(value)
        if grads:
            grad_rows.append(gradients(actor, batch))
    rows = {name: np.concatenate(parts) for name, parts in rows.items()}
    seen = rows["prefix"] >= 1
    strata = {"all": seen}
    for low, high, label in ROUND_BUCKETS:
        strata[label] = seen & (rows["round"] >= low) & (rows["round"] <= high)
    strata["tribute round"] = seen & rows["tribute"]
    strata["no tribute round"] = seen & ~rows["tribute"]
    distinct_all = np.concatenate(distinct)
    history = np.concatenate(sens_mask)
    sensitivity_summary = {}
    for kind, parts in sens.items():
        values = {name: np.concatenate(v) for name, v in parts.items()}
        not_bos = ~rows["bos"][history]
        moved = values["moved"][history]
        sensitivity_summary[kind] = {
            "decisions": int(history.sum()),
            "tv_median": _quantile(values["tv"][history], 0.5),
            "tv_q95": _quantile(values["tv"][history], 0.95),
            "tv_max": _quantile(values["tv"][history], 1.0),
            "greedy_changes": int(values["greedy"][history].sum()),
            "chosen_dlogp_median": _quantile(values["chosen_dlogp"][history], 0.5),
            "moved_share_non_bos": float(moved[not_bos].mean()) if not_bos.any() else 0.0,
            "head_rel_median": _quantile(values["head_rel"][history].reshape(-1), 0.5),
            "attended_rel_median": _quantile(values["attended_rel"][history], 0.5),
            "round0_tv_max": _quantile(values["tv"][~history], 1.0),
        }
    dead = {}
    for name, parts in hooks.items():
        active = torch.cat(parts).double().mean(0)
        dead[name] = {"units": len(active), "dead_share": float((active == 0).double().mean()),
                      "rare_share": float((active < 0.01).double().mean())}
    result = {
        "decisions": int(len(rows["prefix"])), "minibatches": len(batches),
        "matches": int(sum(np.isfinite(d[:, 0]).sum() for d in distinct)),
        "round_counts": np.bincount(rows["round"]).tolist(),
        "tribute_round_decisions": int(rows["tribute"].sum()),
        "strata": {name: head_summary(rows, mask) for name, mask in strata.items()},
        "distinct_per_match_median": np.nanmedian(distinct_all, 0).tolist(),
        "decisions_per_match_median": float(np.median(
            [np.sum(batch["match_index"].numpy() == m) for batch in batches
             for m in np.unique(batch["match_index"].numpy())])),
        "mechanism": {
            "state_error_max": float(np.max(mechanism["state_error"])),
            "q_cos_median_by_head": np.median(np.stack(mechanism["q_cos_median"]), 0).tolist(),
            "query_norm_median": float(np.median(mechanism["query_norm_median"])),
            "query_mean_norm": float(np.median(mechanism["query_mean_norm"])),
            "query_residual_median": float(np.median(mechanism["query_residual_median"])),
            "q_input_norm_median": float(np.median(mechanism["q_input_norm_median"])),
            "singular_values": singular_values(actor),
        },
        "sensitivity": sensitivity_summary,
        "dead_relu": dead,
        "adam": adam_state(actor, optimizer),
    }
    if grad_rows:
        summary = {}
        for loss in grad_rows[0]:
            groups = grad_rows[0][loss]
            summary[loss] = {group: {
                "median": float(np.median([g[loss][group] for g in grad_rows])),
                "max": float(np.max([g[loss][group] for g in grad_rows])),
                "zero_share": float(np.mean([g[loss][group] == 0 for g in grad_rows]))}
                for group in groups}
        result["gradients"] = summary
    return result


def adam_probe(actor: HistoryActor, batches: list[dict[str, Tensor]], steps: int, lr: float
               ) -> dict[str, Any]:
    """``steps`` Adam updates of the ratio-1 policy loss over the fixed sample
    (cycling minibatches). A numerics probe of whether q/k can move, not training."""
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    before = {name: p.detach().clone() for name, p in actor.named_parameters()}
    for step in range(steps):
        batch = batches[step % len(batches)]
        inputs = batch_inputs(batch)
        optimizer.zero_grad(set_to_none=True)
        with sdpa_kernel(SDPBackend.MATH):
            state = actor.decision_states(None, inputs)
            rows = inputs.rows
            log_probs = segment_log_softmax(
                actor.candidate_logits(state, inputs.cand, inputs.offsets, rows=rows),
                rows, inputs.decisions)
            advantage = batch["advantage"]
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            loss = -(advantage * log_probs[batch["chosen"]]).mean()
        loss.backward()
        optimizer.step()
    change = {}
    params = dict(actor.named_parameters())
    for group, members in parameter_groups(actor).items():
        delta = sum(float((_pick(params[n].detach(), r) - _pick(before[n], r)).double()
                          .square().sum()) for n, r in members)
        base = sum(float(_pick(before[n], r).double().square().sum()) for n, r in members)
        change[group] = math.sqrt(delta) / max(math.sqrt(base), 1e-30)
    return {"steps": steps, "lr": lr, "relative_change": change}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("actors", nargs="+", help="checkpoint paths or fresh:SEED[:NORM]")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sampler", default=None, help="actor that plays the sample "
                        "(default: the first audited actor)")
    parser.add_argument("--load-sample", default=None)
    parser.add_argument("--save-sample", default=None)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--matches", type=int, default=48)
    parser.add_argument("--minibatch-matches", type=int, default=8)
    parser.add_argument("--rows-per-match", type=int, default=48)
    parser.add_argument("--fresh-width", type=int, default=128)
    parser.add_argument("--fresh-layers", type=int, default=4)
    parser.add_argument("--fresh-heads", type=int, default=8)
    parser.add_argument("--fresh-aux-heads", default="next,belief,outcome")
    parser.add_argument("--stress", default=None, metavar="MEAN:QGAIN:KGAIN",
                        help="trained-regime numerics for fresh actors (see Stress)")
    parser.add_argument("--adam-steps", type=int, default=0)
    parser.add_argument("--adam-lr", type=float, default=3e-4)
    parser.add_argument("--no-perturb", action="store_true")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    fresh = HistoryPolicyConfig(width=args.fresh_width, layers=args.fresh_layers,
                                heads=args.fresh_heads, aux_heads=args.fresh_aux_heads)
    stress = Stress.parse(args.stress) if args.stress else None
    if args.load_sample:
        batches = torch.load(args.load_sample)
    else:
        sampler, _ = load_actor(args.sampler or args.actors[0], fresh, None)
        batches = collect_sample(sampler, envs=args.envs, steps=args.steps, seed=args.seed,
                                 matches=args.matches, minibatch_matches=args.minibatch_matches,
                                 rows_per_match=args.rows_per_match)
    if args.save_sample:
        torch.save(batches, args.save_sample)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {"args": vars(args), "torch": torch.__version__, "actors": {}}
    for spec in args.actors:
        actor, optimizer = load_actor(spec, fresh, stress)
        result = audit_actor(actor, batches, optimizer, args.seed,
                             () if args.no_perturb else PERTURBATIONS)
        if args.adam_steps:
            result["adam_probe"] = adam_probe(actor, batches, args.adam_steps, args.adam_lr)
            result["after_adam_probe"] = audit_actor(actor, batches, None, args.seed, (), True)
        results["actors"][spec] = result
        worst = result["strata"]["all"]
        print(json.dumps({"actor": spec, "decisions": result["decisions"],
                          "one_hot_share": [round(h["one_hot_share"], 4) for h in worst],
                          "bos_share": [round(h["bos_share"], 4) for h in worst],
                          "state_error": result["mechanism"]["state_error_max"]}))
    (out / "audit.json").write_text(json.dumps(results, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
