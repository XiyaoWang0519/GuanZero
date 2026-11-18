"""Migrate a legacy ``private_decoder_layers=0`` history checkpoint to the
multi-layer relative-position private cross-decoder architecture.

    PYTHONPATH=python:oracle:. python -m train.history_sequence_migrate SRC OUT \\
        --report REPORT.json [--decoder-layers 3] [--query-slots 4] \\
        [--obs-mode private_only] [--residual-scale 0.02] \\
        [--initial-mask-prob 0.0] [--sample-seed 20261018] \\
        [--heldout-seed 20261019] [--matches 24] [--envs 16] [--steps 1200] \\
        [--population migrate|reset] [--archive-filter keep|thin|strength_behavior|nearest_pair] \\
        [--archive-max-age-updates N]

Calibration collects a self-play sample from ``source`` and computes
``r_bar = mean(attended_legacy)`` over decisions that see at least one public
token (``prefix >= 1``). Matching weights (stream encoder, private MLP, seat
embedding, norms, private FFN, action tower, fusion, response head/bridge, and
auxiliary heads) are copied bitwise into the target ``HistoryActor`` while
``private_decoder`` is initialized around ``attention_norm(query + r_bar)``
with small ``residual_scale`` residuals.

Actor Adam optimizer state is remapped by parameter name so shared parameters
keep their exact ``step``, ``exp_avg``, and ``exp_avg_sq`` moments at their new
parameter indices, while newly added ``private_decoder.*`` parameters are omitted
from ``state`` for lazy Adam initialization on the first step. When
``--population migrate`` is active, resident snapshots are converted with the
same ``r_bar`` and initial decoder weights (excluding auxiliary heads), with
snapshot ``obs_mask_prob = 0.0`` and recomputed ``sha256`` digests.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import gd
import numpy as np
import torch
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from eval.private_attention_audit import (
    BATCH_FIELDS,
    ENTROPY_COEF,
    batch_inputs,
    collect_sample,
    encode,
)
from infra.history_artifacts import sha256
from train.history_aux import belief_loss, next_token_loss, outcome_loss, round_outcomes
from train.history_decoder import (
    PRIVATE_OBS_MODES,
    _compute_rel_buckets,
    _gather_attn_bias,
    _wide_project,
    extract_stream_pos,
    infer_decision_rounds,
)
from train.history_model import (
    DecisionInputs,
    HistoryActor,
    config_record,
    load_history_checkpoint,
    save_history_checkpoint,
    segment_log_softmax,
)
from train.history_population import ARCHIVE_FILTERS, HistoryPopulation, weights_digest
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer

SAMPLE_ENVS = 16
SAMPLE_STEPS = 1200
MINIBATCH_MATCHES = 8
ROWS_PER_MATCH = 48

LEGACY_ATTENTION_KEYS = frozenset({
    "q_proj.weight",
    "q_proj.bias",
    "kv_proj.weight",
    "kv_proj.bias",
    "out_proj.weight",
    "out_proj.bias",
})


def validate_source(actor: HistoryActor) -> None:
    """Reject sources that cannot be migrated to the sequence private decoder."""
    config = actor.config
    if config.window:
        raise ValueError("windowed sources are not supported")
    if config.style_input:
        raise ValueError("style-input sources are not supported")
    if getattr(config, "private_decoder_layers", 0) > 0:
        raise ValueError("sources with a private decoder are not supported")
    if config.private_fusion != "sum":
        raise ValueError("the source must be a 'sum'-fusion checkpoint")
    if config.private_qk_norm != "none":
        raise ValueError("the source must have private_qk_norm 'none'")


def _batch_inputs(batch: dict[str, Tensor]) -> DecisionInputs:
    inputs = batch_inputs(batch)
    if "round" in batch:
        inputs.round_index = batch["round"]
    return inputs


def _collect_heldout_sample(
    actor: HistoryActor,
    *,
    envs: int,
    steps: int,
    seed: int,
    matches: int,
    minibatch_matches: int,
    rows_per_match: int,
) -> list[dict[str, Tensor]]:
    """Collect held-out self-play minibatches including ``hidden`` for belief loss."""
    env = gd.VecEnv(
        num_envs=envs,
        num_threads=1,
        seed=seed,
        log_public_actions=True,
        log_env_limit=envs,
    )
    store, buffer = MatchEventStore(), SequenceRolloutBuffer()
    collector = HistoryCollector(
        env,
        actor,
        store,
        buffer,
        torch.Generator().manual_seed(seed),
        kv_cache=True,
    )
    with torch.no_grad():
        collector.collect(steps)
    buffer.finalize(np.zeros(len(buffer), np.float32))
    groups = buffer.match_groups()
    keys = sorted(groups)[:matches] if matches > 0 else sorted(groups)
    data = buffer.compact()
    batches: list[dict[str, Tensor]] = []
    for begin in range(0, len(keys), minibatch_matches):
        picked = []
        for key in keys[begin : begin + minibatch_matches]:
            rows = np.sort(groups[key])
            if rows_per_match and len(rows) > rows_per_match:
                idx = np.linspace(0, len(rows) - 1, rows_per_match).round().astype(np.int64)
                rows = rows[np.unique(idx)]
            picked.append(rows)
        rows = np.concatenate(picked)
        extra = {
            "outcome": round_outcomes(buffer, rows).astype(np.float32),
            "round": data["round"][rows].astype(np.int64),
        }
        batch = buffer.training_batch(
            rows,
            store,
            "cpu",
            fields=("advantage", "hidden"),
            extra=extra,
        )
        i = batch.inputs
        values = (
            i.streams.tokens,
            i.streams.rounds,
            i.streams.phases,
            i.streams.lengths,
            i.match_index,
            i.prefix,
            i.obs,
            i.seat,
            i.cand,
            i.offsets,
            batch.chosen,
            batch.fields["advantage"],
            batch.fields["outcome"],
            batch.fields["round"],
        )
        record = {name: value.clone() for name, value in zip(BATCH_FIELDS, values)}
        record["hidden"] = batch.fields["hidden"].clone()
        batches.append(record)
    return batches


def calibrate_reader_bias(
    source: HistoryActor,
    batches: list[dict[str, Tensor]],
) -> tuple[Tensor, dict[str, Any]]:
    """Compute ``r_bar = mean(attended_legacy)`` over rows with ``prefix >= 1``."""
    attended_rows: list[Tensor] = []
    for batch in batches:
        inputs = _batch_inputs(batch)
        seen = inputs.prefix >= 1
        if not bool(seen.any()):
            continue
        with torch.no_grad():
            query = source.private_query(inputs)
            encoded = encode(source, inputs.streams)
            attended = source._attend_encoded(
                query, encoded, inputs.match_index, inputs.prefix, False
            )
            attended_rows.append(attended[seen])

    if not attended_rows:
        r_bar = torch.zeros(source.config.width, dtype=torch.float32)
        return r_bar, {
            "rows": 0,
            "fallback_zeros": True,
            "r_bar_norm": 0.0,
            "attended_rms": 0.0,
        }

    stacked = torch.cat(attended_rows, dim=0).double()
    r_bar = stacked.mean(dim=0).float()
    return r_bar, {
        "rows": int(stacked.shape[0]),
        "fallback_zeros": False,
        "r_bar_norm": float(r_bar.double().norm()),
        "attended_rms": float(stacked.square().mean().sqrt()),
    }


def convert_actor(
    source: HistoryActor,
    r_bar: Tensor,
    *,
    decoder_layers: int = 3,
    query_slots: int = 4,
    obs_mode: str = "private_only",
    residual_scale: float = 0.02,
    initial_mask_prob: float = 0.0,
    seed: int = 0,
) -> HistoryActor:
    """Create the target ``HistoryActor`` with bitwise-copied shared weights."""
    validate_source(source)
    if decoder_layers < 1:
        raise ValueError("decoder_layers must be at least 1")
    if query_slots < 1:
        raise ValueError("query_slots must be at least 1")
    if obs_mode not in PRIVATE_OBS_MODES:
        raise ValueError(f"obs_mode must be one of {PRIVATE_OBS_MODES}")
    if not (math.isfinite(residual_scale) and residual_scale >= 0.0):
        raise ValueError("residual_scale must be finite and non-negative")
    if not (math.isfinite(initial_mask_prob) and 0.0 <= initial_mask_prob <= 1.0):
        raise ValueError("initial_mask_prob must be in [0, 1]")

    target_config = replace(
        source.config,
        private_decoder_layers=decoder_layers,
        private_query_slots=query_slots,
        private_obs_mode=obs_mode,
        private_qk_norm="none",
        private_fusion="sum",
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        converted = HistoryActor(target_config)
        src_state = source.state_dict()
        filtered = {k: v for k, v in src_state.items() if k not in LEGACY_ATTENTION_KEYS}
        missing, unexpected = converted.load_state_dict(filtered, strict=False)
        expected_missing = {
            k
            for k in converted.state_dict()
            if k.startswith("private_decoder.") or k == "obs_mask_prob"
        }
        if unexpected or set(missing) != expected_missing:
            raise ValueError(
                f"converted actor weights do not fit: missing {missing}, extra {unexpected}"
            )
        converted.private_decoder.reset_for_migration(
            r_bar, residual_scale=residual_scale
        )
    if converted.obs_mask_prob is not None:
        with torch.no_grad():
            converted.obs_mask_prob.fill_(float(initial_mask_prob))
    return converted


def migrate_optimizer(
    optimizer_payload: dict[str, Any],
    source: HistoryActor,
    converted: HistoryActor,
) -> dict[str, Any]:
    """Remap actor Adam moments by parameter name; preserve critic optimizer intact."""
    if not optimizer_payload:
        return {}
    out: dict[str, Any] = {}
    if "actor" in optimizer_payload:
        src_opt = copy.deepcopy(optimizer_payload["actor"])
        groups = src_opt.get("param_groups", [])
        if len(groups) != 1:
            raise ValueError(f"expected single actor param_group, got {len(groups)}")
        src_names = [name for name, _ in source.named_parameters()]
        old_param_ids = groups[0]["params"]
        if len(old_param_ids) != len(src_names):
            raise ValueError(
                f"actor optimizer param count {len(old_param_ids)} != source params {len(src_names)}"
            )
        name_to_old_id = dict(zip(src_names, old_param_ids))
        old_state = src_opt.get("state", {})
        new_state: dict[int, Any] = {}
        conv_params = list(converted.named_parameters())
        for new_idx, (name, param) in enumerate(conv_params):
            if name not in name_to_old_id:
                continue
            old_id = name_to_old_id[name]
            if old_id not in old_state:
                continue
            entry = copy.deepcopy(old_state[old_id])
            for moment_key in ("exp_avg", "exp_avg_sq"):
                if moment_key in entry and tuple(entry[moment_key].shape) != tuple(param.shape):
                    raise ValueError(
                        f"optimizer {moment_key} shape mismatch for {name}: "
                        f"{tuple(entry[moment_key].shape)} != {tuple(param.shape)}"
                    )
            new_state[new_idx] = entry
        groups[0]["params"] = list(range(len(conv_params)))
        src_opt["state"] = new_state
        out["actor"] = src_opt
    if "critic" in optimizer_payload:
        out["critic"] = copy.deepcopy(optimizer_payload["critic"])
    for key, value in optimizer_payload.items():
        if key not in out:
            out[key] = copy.deepcopy(value)
    return out


def migrate_population(
    population_state: dict[str, Any],
    converted: HistoryActor,
    *,
    archive_filter: str = "keep",
    archive_max_age_updates: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Migrate every resident snapshot to the decoder architecture with ``obs_mask_prob=0.0``."""
    if archive_filter not in ("keep", *ARCHIVE_FILTERS):
        raise ValueError(
            f"archive_filter must be one of {('keep', *ARCHIVE_FILTERS)}, got {archive_filter!r}"
        )
    if archive_max_age_updates is not None and int(archive_max_age_updates) < 0:
        raise ValueError("archive_max_age_updates must be non-negative")

    pop = copy.deepcopy(population_state)
    aux_names = converted.auxiliary_parameter_names()
    decoder_weights = {
        k: v.detach().cpu().clone()
        for k, v in converted.state_dict().items()
        if k.startswith("private_decoder.")
    }
    has_obs_mask = "obs_mask_prob" in converted.state_dict()
    for _identity, record in pop.get("snapshots", {}).items():
        snap_state = {
            k: v.detach().cpu().clone()
            for k, v in record["model"].items()
            if k not in LEGACY_ATTENTION_KEYS and k not in aux_names
        }
        for k, v in decoder_weights.items():
            snap_state[k] = v.clone()
        if has_obs_mask:
            snap_state["obs_mask_prob"] = torch.tensor(0.0, dtype=torch.float32)
        record["model"] = snap_state
        record["metadata"] = dict(record["metadata"], sha256=weights_digest(snap_state))

    # Clear pre-migration behavioral probe/fingerprints (they came from the pre-decoder actor).
    pop["behaviors"] = {}
    pop["behavior_decisions"] = 0
    pop["behavior_probe"] = None
    pop["behavior_probe_meta"] = None
    if archive_filter != "keep":
        pop["archive_filter"] = archive_filter
    if archive_max_age_updates is not None:
        pop["archive_max_age_updates"] = int(archive_max_age_updates)

    resolved_filter = pop.get("archive_filter", "thin")
    resolved_max_age = int(pop.get("archive_max_age_updates", 0))

    # Verify the migrated population loads cleanly via HistoryPopulation.load_state_dict.
    check_pool = HistoryPopulation(
        converted,
        pop["lineage"],
        0,
        recent=pop["recent"],
        snapshot_probability=pop["probability"],
        archive_every=pop.get("archive_every", 0),
        archive_size=pop.get("archive_size", 16),
        archive_share=pop.get("archive_share", 0.5),
        archive_filter=resolved_filter,
        archive_min_tv=float(pop.get("archive_min_tv", 0.05)),
        archive_max_age_updates=resolved_max_age,
    )
    check_pool.load_state_dict(pop)
    summary = {
        "migrated": True,
        "snapshots": len(pop.get("snapshots", {})),
        "archive": len(pop.get("archive", [])),
        "snapshot_obs_mask_prob": 0.0 if has_obs_mask else None,
        "archive_filter": resolved_filter,
        "archive_max_age_updates": resolved_max_age,
    }
    return pop, summary


def capture_decoder_attention(
    actor: HistoryActor,
    inputs: DecisionInputs,
    encoded: Tensor | None = None,
    *,
    chunk: int = 32,
) -> dict[str, Any]:
    """Explicit FP64 per-layer, per-head attention audit on ``PrivateCrossDecoder``."""
    dec = actor.private_decoder
    if dec is None:
        raise ValueError("actor does not have a private_decoder")
    with torch.no_grad():
        if encoded is None:
            encoded = encode(actor, inputs.streams)
        query = actor.private_query(inputs)
        reading = actor.reader_input(query)
        pos_seat, pos_round = extract_stream_pos(
            inputs.streams, encoded.shape[1], actor.config.max_rounds, encoded.device
        )
        round_idx = infer_decision_rounds(
            inputs.round_index, pos_round, inputs.match_index, inputs.prefix
        )
        n = len(query)
        my_round = round_idx.to(device=query.device, dtype=torch.long).clamp(
            0, dec.max_rounds - 1
        )
        base = actor.attention_norm(reading + dec.reader_bias) + dec.query_round(my_round)
        if dec.slot_embeddings is None:
            z0 = base[:, None, :]
        else:
            z0 = base[:, None, :] + dec.slot_embeddings[None, :, :]
        z = z0

        k_slots = dec.query_slots
        heads = dec.heads
        depth = dec.depth
        scale = 1.0 / math.sqrt(depth)
        seen = inputs.prefix >= 1

        layer_max_probs: list[Tensor] = []
        layer_entropies: list[Tensor] = []

        for layer in dec.layers:
            k_all, v_all = layer.project_kv(encoded)
            q, r_seat, r_round, r_step = layer.project_queries(z, _wide_project)
            attended = z.new_zeros(n, k_slots, dec.width)
            max_prob_parts: list[Tensor] = []
            entropy_parts: list[Tensor] = []

            for start in range(0, n, chunk):
                rows = slice(start, min(start + chunk, n))
                c = rows.stop - rows.start
                m = inputs.match_index[rows]
                p = inputs.prefix[rows]
                my_seat = inputs.seat[rows]
                my_r = my_round[rows]
                allowed, seat_b, round_b, step_b = _compute_rel_buckets(
                    pos_seat[m],
                    pos_round[m],
                    p[:, None],
                    my_seat[:, None],
                    my_r[:, None],
                    dec.max_rounds,
                )
                allowed_q = allowed[:, None, :].expand(c, k_slots, -1)
                seat_q = seat_b[:, None, :].expand(c, k_slots, -1)
                round_q = round_b[:, None, :].expand(c, k_slots, -1)
                step_q = step_b[:, None, :].expand(c, k_slots, -1)
                bias = _gather_attn_bias(
                    r_seat[rows].permute(0, 2, 1, 3),
                    r_round[rows].permute(0, 2, 1, 3),
                    r_step[rows].permute(0, 2, 1, 3),
                    allowed_q,
                    seat_q,
                    round_q,
                    step_q,
                )
                q_chunk = q[rows].permute(0, 2, 1, 3).double()
                k_chunk = k_all[m].double()
                scores = (
                    torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale
                    + bias.double()
                )
                probs64 = scores.softmax(dim=-1)
                max_prob_parts.append(probs64.max(dim=-1).values)
                entropy_parts.append(-torch.special.xlogy(probs64, probs64).sum(dim=-1))

                probs32 = scores.float().softmax(dim=-1)
                out_chunk = torch.matmul(probs32, v_all[m])
                attended[rows] = (
                    out_chunk.permute(0, 2, 1, 3).reshape(c, k_slots, dec.width)
                )

            max_probs = torch.cat(max_prob_parts, dim=0) if max_prob_parts else z.new_zeros((0, heads, k_slots), dtype=torch.float64)
            entropies = torch.cat(entropy_parts, dim=0) if entropy_parts else z.new_zeros((0, heads, k_slots), dtype=torch.float64)
            layer_max_probs.append(max_probs[seen])
            layer_entropies.append(entropies[seen])
            z = layer.finish_layer(z, attended, _wide_project)

        dec_out = (
            z[:, 0]
            if dec.slot_proj is None
            else base + dec.slot_proj((z - z0).reshape(n, -1))
        )
        state = actor.output_norm(dec_out + actor.feed_forward(dec_out))
        reference = actor.decision_states(encoded, inputs)
        state_error = float((state - reference).abs().max()) if n else 0.0

    return {
        "max_probs": layer_max_probs,
        "entropies": layer_entropies,
        "state_error": state_error,
        "decisions_seen": int(seen.sum()),
    }


def heldout_diagnostics(
    source: HistoryActor,
    converted: HistoryActor,
    batches: list[dict[str, Tensor]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Measure policy shift vs ``source`` and decoder attention diagnostics."""
    tv_parts: list[Tensor] = []
    greedy_changes: list[bool] = []
    seen_total = 0

    layers = len(converted.private_decoder.layers)
    heads = converted.config.heads
    layer_probs: list[list[Tensor]] = [[] for _ in range(layers)]
    layer_entropies: list[list[Tensor]] = [[] for _ in range(layers)]
    state_error_max = 0.0

    for batch in batches:
        inputs = _batch_inputs(batch)
        seen = inputs.prefix >= 1
        with torch.no_grad():
            encoded = encode(source, inputs.streams)
            logp_src = source.candidate_log_probs(inputs, encoded=encoded)
            logp_new = converted.candidate_log_probs(inputs, encoded=encoded)
            rows = inputs.rows
            tv = 0.5 * torch.zeros(inputs.decisions, dtype=torch.float32).index_add_(
                0, rows, (logp_src.exp() - logp_new.exp()).abs()
            )
            if bool(seen.any()):
                tv_parts.append(tv[seen])
            offsets = inputs.offsets.tolist()
            counts = inputs.counts.tolist()
            seen_list = seen.tolist()
            for i in range(inputs.decisions):
                if seen_list[i] and counts[i] > 1:
                    src_best = logp_src[offsets[i] : offsets[i + 1]].argmax()
                    new_best = logp_new[offsets[i] : offsets[i + 1]].argmax()
                    greedy_changes.append(bool(src_best != new_best))
            captured = capture_decoder_attention(converted, inputs, encoded=encoded)
            state_error_max = max(state_error_max, captured["state_error"])
            for layer_idx in range(layers):
                layer_probs[layer_idx].append(captured["max_probs"][layer_idx])
                layer_entropies[layer_idx].append(captured["entropies"][layer_idx])
            seen_total += int(seen.sum())

    if tv_parts:
        tv_all = torch.cat(tv_parts).double()
        heldout_summary = {
            "decisions_seen": seen_total,
            "tv_mean": float(tv_all.mean()),
            "tv_median": float(tv_all.median()),
            "tv_p95": float(tv_all.quantile(0.95)),
            "tv_max": float(tv_all.max()),
            "greedy_change_share": (
                float(np.mean(greedy_changes)) if greedy_changes else 0.0
            ),
            "greedy_change_decisions": len(greedy_changes),
        }
    else:
        heldout_summary = {
            "decisions_seen": 0,
            "tv_mean": 0.0,
            "tv_median": 0.0,
            "tv_p95": 0.0,
            "tv_max": 0.0,
            "greedy_change_share": 0.0,
            "greedy_change_decisions": 0,
        }

    one_hot_share: list[list[float]] = []
    entropy_median: list[list[float]] = []
    max_prob_median: list[list[float]] = []
    for layer_idx in range(layers):
        if layer_probs[layer_idx] and seen_total > 0:
            probs_cat = torch.cat(layer_probs[layer_idx], dim=0)  # [N_seen, heads, slots]
            ent_cat = torch.cat(layer_entropies[layer_idx], dim=0)
            oh_row: list[float] = []
            ent_row: list[float] = []
            mp_row: list[float] = []
            for h in range(heads):
                ph = probs_cat[:, h, :].reshape(-1)
                eh = ent_cat[:, h, :].reshape(-1)
                oh_row.append(float((ph > 1.0 - 1e-6).double().mean()))
                ent_row.append(float(eh.median()))
                mp_row.append(float(ph.median()))
            one_hot_share.append(oh_row)
            entropy_median.append(ent_row)
            max_prob_median.append(mp_row)
        else:
            one_hot_share.append([0.0] * heads)
            entropy_median.append([0.0] * heads)
            max_prob_median.append([0.0] * heads)

    one_hot_max = max(
        (val for row in one_hot_share for val in row),
        default=0.0,
    )
    attention_summary = {
        "layers": layers,
        "heads": heads,
        "query_slots": converted.private_decoder.query_slots,
        "decisions_seen": seen_total,
        "one_hot_share": one_hot_share,
        "one_hot_share_max": float(one_hot_max),
        "entropy_median": entropy_median,
        "max_prob_median": max_prob_median,
        "state_error_max": state_error_max,
    }
    return heldout_summary, attention_summary


def gradient_audit(actor: HistoryActor, batch: dict[str, Tensor]) -> dict[str, Any]:
    """Run one backward pass on ``policy_loss + aux_loss`` and audit grad norms."""
    if actor.private_decoder is None:
        raise ValueError("actor does not have a private_decoder")
    was_training = actor.training
    actor.train()
    actor.zero_grad(set_to_none=True)
    inputs = _batch_inputs(batch)
    try:
        encoded = encode(actor, inputs.streams)
        with sdpa_kernel(SDPBackend.MATH):
            state = actor.decision_states(encoded, inputs)
            rows = inputs.rows
            logits = actor.candidate_logits(
                state, inputs.cand, inputs.offsets, rows=rows
            )
            log_probs = segment_log_softmax(logits, rows, inputs.decisions)
            entropy = -torch.zeros(inputs.decisions, device=log_probs.device).index_add(
                0, rows, log_probs.exp() * log_probs
            )
            advantage = batch["advantage"].to(device=log_probs.device)
            if len(advantage) > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            policy_loss = (
                -(advantage * log_probs[batch["chosen"]]).mean()
                - ENTROPY_COEF * entropy.mean()
            )
            aux_loss = state.new_zeros(())
            heads = actor.aux_head_names
            if "next" in heads:
                s = inputs.streams
                cited = torch.zeros_like(s.lengths).scatter_reduce(
                    0, inputs.match_index, inputs.prefix, reduce="amax"
                )
                count = s.tokens.shape[1]
                limit = torch.minimum(s.lengths, cited)
                valid = (
                    torch.arange(count, device=s.tokens.device)[None] < limit[:, None]
                )
                next_stats = next_token_loss(
                    actor.next_logits(encoded[:, :count]), s.tokens, valid
                )
                aux_loss = aux_loss + next_stats["next_loss"]
            if "belief" in heads and "hidden" in batch:
                belief_stats = belief_loss(
                    actor.belief_logits(state), batch["hidden"]
                )
                aux_loss = aux_loss + belief_stats["belief_loss"]
            if "outcome" in heads and "outcome" in batch:
                outcome_stats = outcome_loss(
                    actor.outcome_value(state), batch["outcome"]
                )
                aux_loss = aux_loss + outcome_stats["outcome_loss"]
            total_loss = policy_loss + aux_loss
        total_loss.backward()

        def summarize(module: torch.nn.Module) -> dict[str, Any]:
            norms: dict[str, float] = {}
            zero_params: list[str] = []
            nonfinite_params: list[str] = []
            for name, param in module.named_parameters():
                if param.grad is None:
                    norm = 0.0
                else:
                    norm = float(param.grad.detach().double().square().sum().sqrt())
                norms[name] = norm
                if not math.isfinite(norm):
                    nonfinite_params.append(name)
                elif norm <= 0.0:
                    zero_params.append(name)
            vals = list(norms.values())
            finite_count = len(vals) - len(nonfinite_params)
            nonzero_count = sum(1 for v in vals if math.isfinite(v) and v > 0.0)
            return {
                "count": len(vals),
                "nonzero_count": nonzero_count,
                "finite_count": finite_count,
                "all_finite_positive": (
                    len(vals) > 0 and not zero_params and not nonfinite_params
                ),
                "min_norm": min(vals) if vals else 0.0,
                "max_norm": max(vals) if vals else 0.0,
                "zero_params": zero_params,
                "nonfinite_params": nonfinite_params,
                "norms": norms,
            }

        result = {
            "loss": float(total_loss.detach()),
            "policy_loss": float(policy_loss.detach()),
            "aux_loss": float(aux_loss.detach()),
            "private_decoder": summarize(actor.private_decoder),
            "stream": summarize(actor.stream),
        }
    finally:
        actor.zero_grad(set_to_none=True)
        actor.train(was_training)
    return result


def migrate(
    source_path: str | Path,
    out_path: str | Path,
    *,
    decoder_layers: int = 3,
    query_slots: int = 4,
    obs_mode: str = "private_only",
    residual_scale: float = 0.02,
    initial_mask_prob: float = 0.0,
    sample_seed: int = 20261018,
    heldout_seed: int = 20261019,
    matches: int = 24,
    envs: int = SAMPLE_ENVS,
    steps: int = SAMPLE_STEPS,
    minibatch_matches: int = MINIBATCH_MATCHES,
    rows_per_match: int = ROWS_PER_MATCH,
    population: str = "migrate",
    archive_filter: str = "keep",
    archive_max_age_updates: int | None = None,
    verify_audit: bool = True,
) -> dict[str, Any]:
    """Migrate a ``private_decoder_layers=0`` checkpoint and run held-out audits."""
    if population not in ("migrate", "reset"):
        raise ValueError(f"population must be 'migrate' or 'reset', got {population!r}")
    if archive_filter not in ("keep", *ARCHIVE_FILTERS):
        raise ValueError(
            f"archive_filter must be one of {('keep', *ARCHIVE_FILTERS)}, got {archive_filter!r}"
        )
    if archive_max_age_updates is not None and int(archive_max_age_updates) < 0:
        raise ValueError("archive_max_age_updates must be non-negative")
    source, _, payload = load_history_checkpoint(source_path, "cpu")
    validate_source(source)
    source = source.eval()
    source.causal_sdpa = True

    cal_batches = collect_sample(
        source,
        envs=envs,
        steps=steps,
        seed=sample_seed,
        matches=matches,
        minibatch_matches=minibatch_matches,
        rows_per_match=rows_per_match,
    )
    r_bar, calibration = calibrate_reader_bias(source, cal_batches)
    calibration["seed"] = sample_seed

    converted = convert_actor(
        source,
        r_bar,
        decoder_layers=decoder_layers,
        query_slots=query_slots,
        obs_mode=obs_mode,
        residual_scale=residual_scale,
        initial_mask_prob=initial_mask_prob,
        seed=sample_seed,
    )
    converted = converted.eval()
    converted.causal_sdpa = True

    heldout_batches = _collect_heldout_sample(
        source,
        envs=envs,
        steps=steps,
        seed=heldout_seed,
        matches=matches,
        minibatch_matches=minibatch_matches,
        rows_per_match=rows_per_match,
    )
    if not heldout_batches:
        raise RuntimeError("heldout self-play sample produced no completed matches")
    heldout_summary, attention_summary = heldout_diagnostics(
        source, converted, heldout_batches
    )
    heldout_summary["seed"] = heldout_seed

    grad_summary = gradient_audit(converted, heldout_batches[0])

    if verify_audit:
        if attention_summary["one_hot_share_max"] > 0.0:
            raise RuntimeError(
                f"migrated decoder has one-hot attention saturation: "
                f"{attention_summary['one_hot_share']}"
            )
        if not grad_summary["private_decoder"]["all_finite_positive"]:
            raise RuntimeError(
                f"private_decoder gradient audit failed: "
                f"zero={grad_summary['private_decoder']['zero_params']}, "
                f"nonfinite={grad_summary['private_decoder']['nonfinite_params']}"
            )
        if not grad_summary["stream"]["all_finite_positive"]:
            raise RuntimeError(
                f"stream gradient audit failed: "
                f"zero={grad_summary['stream']['zero_params']}, "
                f"nonfinite={grad_summary['stream']['nonfinite_params']}"
            )

    out_payload = copy.deepcopy(payload)
    out_payload["model"] = {
        k: v.detach().cpu().clone() for k, v in converted.state_dict().items()
    }
    out_payload["model_config"] = config_record(converted.config)
    if payload.get("optimizer"):
        out_payload["optimizer"] = migrate_optimizer(
            payload["optimizer"], source, converted
        )

    if population == "reset":
        out_payload.pop("population", None)
        pop_summary: dict[str, Any] = {
            "migrated": False,
            "mode": "reset",
            "snapshots": 0,
        }
    elif payload.get("population") is not None:
        migrated_pop, pop_summary = migrate_population(
            payload["population"],
            converted,
            archive_filter=archive_filter,
            archive_max_age_updates=archive_max_age_updates,
        )
        out_payload["population"] = migrated_pop
        pop_summary["mode"] = "migrate"
    else:
        pop_summary = {"migrated": False, "mode": "migrate", "snapshots": 0}

    if isinstance(out_payload.get("config"), dict):
        cfg = dict(out_payload["config"])
        cfg["private_decoder_layers"] = decoder_layers
        cfg["private_query_slots"] = query_slots
        cfg["private_obs_mode"] = obs_mode
        cfg["private_qk_norm"] = "none"
        cfg["private_fusion"] = "sum"
        if archive_filter != "keep":
            cfg["population_archive_filter"] = archive_filter
        if archive_max_age_updates is not None:
            cfg["population_archive_max_age_updates"] = int(archive_max_age_updates)
        out_payload["config"] = cfg

    migration = {
        "source_path": str(source_path),
        "source_sha256": sha256(Path(source_path)),
        "target": {
            "private_decoder_layers": decoder_layers,
            "private_query_slots": query_slots,
            "private_obs_mode": obs_mode,
            "private_qk_norm": "none",
            "private_fusion": "sum",
        },
        "residual_scale": float(residual_scale),
        "initial_mask_prob": float(initial_mask_prob),
        "population": pop_summary,
        "calibration": calibration,
        "heldout": heldout_summary,
        "attention": attention_summary,
        "gradients": grad_summary,
        "out": str(out_path),
    }
    out_payload["sequence_migration"] = migration
    save_history_checkpoint(out_path, out_payload)
    reloaded, _, _ = load_history_checkpoint(out_path, "cpu")
    if reloaded.config != converted.config:
        raise RuntimeError("reloaded migrated checkpoint config mismatch")
    return migration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("out")
    parser.add_argument("--report", required=True)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--query-slots", type=int, default=4)
    parser.add_argument("--obs-mode", choices=PRIVATE_OBS_MODES, default="private_only")
    parser.add_argument("--residual-scale", type=float, default=0.02)
    parser.add_argument("--initial-mask-prob", type=float, default=0.0)
    parser.add_argument("--sample-seed", type=int, default=20261018)
    parser.add_argument("--heldout-seed", type=int, default=20261019)
    parser.add_argument("--matches", type=int, default=24)
    parser.add_argument("--envs", type=int, default=SAMPLE_ENVS)
    parser.add_argument("--steps", type=int, default=SAMPLE_STEPS)
    parser.add_argument("--minibatch-matches", type=int, default=MINIBATCH_MATCHES)
    parser.add_argument("--rows-per-match", type=int, default=ROWS_PER_MATCH)
    parser.add_argument(
        "--population", choices=("migrate", "reset"), default="migrate"
    )
    parser.add_argument(
        "--archive-filter", choices=("keep", *ARCHIVE_FILTERS), default="keep"
    )
    parser.add_argument("--archive-max-age-updates", type=int, default=None)
    args = parser.parse_args(argv)

    migration = migrate(
        args.source,
        args.out,
        decoder_layers=args.decoder_layers,
        query_slots=args.query_slots,
        obs_mode=args.obs_mode,
        residual_scale=args.residual_scale,
        initial_mask_prob=args.initial_mask_prob,
        sample_seed=args.sample_seed,
        heldout_seed=args.heldout_seed,
        matches=args.matches,
        envs=args.envs,
        steps=args.steps,
        minibatch_matches=args.minibatch_matches,
        rows_per_match=args.rows_per_match,
        population=args.population,
        archive_filter=args.archive_filter,
        archive_max_age_updates=args.archive_max_age_updates,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(migration, indent=1, default=float) + "\n")
    print(json.dumps({
        "out": args.out,
        "report": args.report,
        "tv_mean": migration["heldout"]["tv_mean"],
        "one_hot_share_max": migration["attention"]["one_hot_share_max"],
        "decoder_grads_ok": migration["gradients"]["private_decoder"]["all_finite_positive"],
        "stream_grads_ok": migration["gradients"]["stream"]["all_finite_positive"],
    }, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
