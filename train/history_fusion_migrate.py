"""Convert a "sum"-fusion history checkpoint into the "split" private-fusion arm.

    PYTHONPATH=python:oracle:. python -m train.history_fusion_migrate SRC OUT \\
        --report REPORT.json [--reader-gain calibrated|FLOAT]

The converted checkpoint loads through ``load_history_checkpoint`` and is
accepted by ``train.history_ppo.habit_init_player`` with
``private_fusion="split", private_qk_norm="qk"``: every source parameter is
copied bitwise, ``q_norm``/``k_norm`` start as identities (LayerNorm defaults)
and ``reader_norm`` starts as ``c * attention_norm.weight`` with zero bias, so
at init the reader enters the fusion with the same relative magnitude it had
inside ``attention_norm(query + attended)`` — now under its own gain and
centering. The new parameters are trailing, so ``load_extended_optimizers``
appends fresh Adam slots for them and keeps the source moments.

Calibration. With ``--reader-gain calibrated`` a self-play sample of the
source actor is collected (``eval.private_attention_audit.collect_sample``)
and per decision ``c_row = rms(r - mean(r)) / sqrt(var(q + r) + eps)`` with
``q`` the private query, ``r`` the reader output and ``eps`` the
``attention_norm`` epsilon; ``c`` is the median over rows that see at least
one public token. An explicit float skips the calibration sample entirely.

Held-out. A second, independent sample (``--heldout-seed``) measures the
starting-policy shift of the converted actor versus the source on identical
inputs: per-decision total-variation stats, the greedy-argmax change share
and, as a scale reference, the same stats for the source with its reader
output replaced by zeros. The converted actor's per-head attention saturation
(one-hot share, median top probability) is reported alongside.

Sources must be plain ``history_ppo`` checkpoints: ``private_fusion="sum"``,
``private_qk_norm="none"``, no style input and no auxiliary heads (their
parameters would not be trailing for the optimizer remap).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from infra.history_artifacts import sha256
from eval.private_attention_audit import batch_inputs, capture, collect_sample, encode
from train.history_model import (HistoryActor, HistoryPolicyConfig, config_record,
                                 load_history_checkpoint, save_history_checkpoint,
                                 segment_log_softmax)

# Self-play sizing, after the audit's defaults (collect_sample).
SAMPLE_ENVS = 16
SAMPLE_STEPS = 1500
MINIBATCH_MATCHES = 8
ROWS_PER_MATCH = 48
NEW_PARAMETERS = {"q_norm.weight", "q_norm.bias", "k_norm.weight", "k_norm.bias",
                  "reader_norm.weight", "reader_norm.bias"}


def validate_source(actor: HistoryActor) -> None:
    """Only plain "sum"/"none" actors convert: split reuses ``attention_norm``
    on the query and every other arm combination is undefined here."""
    config = actor.config
    if config.private_hidden > 0:
        raise ValueError("fusion migration does not support private_hidden > 0")
    if config.private_fusion != "sum":
        raise ValueError("the source must be a 'sum'-fusion checkpoint")
    if config.private_qk_norm != "none":
        raise ValueError("the source must have private_qk_norm 'none'")
    if config.style_input:
        raise ValueError("style-input sources are not supported")
    if config.aux_head_names:
        raise ValueError("auxiliary-head sources are not supported")
    if config.window:
        raise ValueError("windowed sources are not supported")


def convert(actor: HistoryActor, gain: float) -> HistoryActor:
    """The ``split``+``qk`` twin of ``actor``: identical weights, identity
    q/k norms and ``reader_norm.weight = gain * attention_norm.weight``."""
    if actor.config.private_hidden > 0:
        raise ValueError("fusion migration does not support private_hidden > 0")
    config = replace(actor.config, private_fusion="split", private_qk_norm="qk")
    converted = HistoryActor(config)
    missing, unexpected = converted.load_state_dict(actor.state_dict(), strict=False)
    if unexpected or set(missing) != NEW_PARAMETERS:
        raise ValueError(f"converted actor weights do not fit: missing {missing}, "
                         f"extra {unexpected}")
    with torch.no_grad():
        converted.reader_norm.weight.copy_(gain * converted.attention_norm.weight)
        converted.reader_norm.bias.zero_()
    return converted


def calibrate(actor: HistoryActor, batches: list[dict[str, Tensor]]
              ) -> tuple[float, dict[str, Any]]:
    """Median ``c_row`` over decisions that see at least one public token."""
    eps = actor.attention_norm.eps
    values = []
    for batch in batches:
        inputs = batch_inputs(batch)
        with torch.no_grad():
            query = actor.private_query(inputs)
            encoded = encode(actor, inputs.streams)
            attended = actor._attend_encoded(query, encoded, inputs.match_index,
                                             inputs.prefix, False)
            c_row = ((attended - attended.mean(-1, keepdim=True)).square().mean(-1).sqrt()
                     / ((query + attended).var(-1, unbiased=False) + eps).sqrt())
            values.append(c_row[inputs.prefix >= 1])
    c_rows = torch.cat(values).double()
    stats = {"rows": int(len(c_rows)), "eps": float(eps),
             "quantiles": {f"q{q:g}": float(c_rows.quantile(q))
                           for q in (0.05, 0.25, 0.5, 0.75, 0.95)}}
    return float(c_rows.median()), stats


def heldout_metrics(source: HistoryActor, converted: HistoryActor,
                    batches: list[dict[str, Tensor]]) -> dict[str, Any]:
    """Held-out policy shift and converted-actor attention diagnostics."""
    tvs = {"converted": [], "reader_zeroed": []}
    greedy = {"converted": [], "reader_zeroed": []}
    heads = converted.config.heads
    one_hot = torch.zeros(heads, dtype=torch.float64)
    top_probs: list[Tensor] = []
    seen_total = 0
    for batch in batches:
        inputs = batch_inputs(batch)
        mask = (inputs.prefix >= 1).numpy()
        with torch.no_grad():
            encoded = encode(source, inputs.streams)
            logp_src = source.candidate_log_probs(inputs, encoded=encoded)
            logp_new = converted.candidate_log_probs(inputs, encoded=encoded)
            query = source.private_query(inputs)
            state0 = source.fuse(query, query.new_zeros(query.shape))
            logits0 = source.candidate_logits(state0, inputs.cand, inputs.offsets,
                                              rows=inputs.rows)
            logp_0 = segment_log_softmax(logits0, inputs.rows, inputs.decisions)
            rows = inputs.rows
            offsets = inputs.offsets.tolist()
            counts = inputs.counts.tolist()
            for name, other in (("converted", logp_new), ("reader_zeroed", logp_0)):
                tv = 0.5 * torch.zeros(inputs.decisions).index_add_(
                    0, rows, (logp_src.exp() - other.exp()).abs())
                tvs[name].append(tv[torch.as_tensor(mask)])
                for i in range(inputs.decisions):
                    if mask[i] and counts[i] > 1:
                        greedy[name].append(bool(
                            logp_src[offsets[i]:offsets[i + 1]].argmax()
                            != other[offsets[i]:offsets[i + 1]].argmax()))
            captured = capture(converted, inputs)
        probs = torch.as_tensor(captured["max_prob"]).double()[torch.as_tensor(mask)]
        top_probs.append(probs)
        one_hot += (probs > 1 - 1e-6).sum(0)
        seen_total += int(mask.sum())
    out = {"decisions_seen": seen_total}
    for name in ("converted", "reader_zeroed"):
        tv = torch.cat(tvs[name]).double()
        picks = np.asarray(greedy[name], dtype=bool)
        out[name] = {"tv_mean": float(tv.mean()), "tv_median": float(tv.median()),
                     "tv_p95": float(tv.quantile(0.95)), "tv_max": float(tv.max()),
                     "greedy_change_share": float(picks.mean()) if len(picks) else 0.0,
                     "greedy_change_decisions": int(len(picks))}
    probs = torch.cat(top_probs)
    out["attention"] = {"heads": heads,
                        "one_hot_share": (one_hot / max(seen_total, 1)).tolist(),
                        "top_prob_median": probs.median(0).values.tolist()}
    return out


def converted_payload(source: HistoryActor, payload: dict[str, Any], gain: float
                      ) -> dict[str, Any]:
    """The source payload with the split+qk actor's weights and record."""
    converted = convert(source, gain)
    out = dict(payload)
    out["model"] = converted.state_dict()
    out["model_config"] = config_record(converted.config)
    return out


def migrate(source_path: str | Path, out_path: str | Path, *, sample_seed: int,
            heldout_seed: int, matches: int, reader_gain: str,
            envs: int = SAMPLE_ENVS, steps: int = SAMPLE_STEPS,
            minibatch_matches: int = MINIBATCH_MATCHES,
            rows_per_match: int = ROWS_PER_MATCH) -> dict[str, Any]:
    source, _, payload = load_history_checkpoint(source_path, "cpu")
    validate_source(source)
    source = source.eval()
    source.causal_sdpa = True            # chunked encode; weights untouched

    if reader_gain == "calibrated":
        batches = collect_sample(source, envs=envs, steps=steps, seed=sample_seed,
                                 matches=matches, minibatch_matches=minibatch_matches,
                                 rows_per_match=rows_per_match)
        gain, calibration = calibrate(source, batches)
    else:
        gain = float(reader_gain)
        calibration = None

    converted = convert(source, gain)
    converted.eval().causal_sdpa = True

    heldout = collect_sample(source, envs=envs, steps=steps, seed=heldout_seed,
                             matches=matches, minibatch_matches=minibatch_matches,
                             rows_per_match=rows_per_match)
    shift = heldout_metrics(source, converted, heldout)
    shift["seed"] = heldout_seed

    migration = {"source_path": str(source_path),
                 "source_sha256": sha256(Path(source_path)),
                 "target": {"private_fusion": "split", "private_qk_norm": "qk"},
                 "reader_gain": gain,
                 "calibration": calibration,
                 "calibration_seed": sample_seed if calibration is not None else None,
                 "heldout": shift}
    out = converted_payload(source, payload, gain)
    out["fusion_migration"] = migration
    save_history_checkpoint(out_path, out)
    return migration


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source")
    parser.add_argument("out")
    parser.add_argument("--report", required=True)
    parser.add_argument("--sample-seed", type=int, default=20261018)
    parser.add_argument("--heldout-seed", type=int, default=20261019)
    parser.add_argument("--matches", type=int, default=48)
    parser.add_argument("--reader-gain", default="calibrated",
                        metavar="calibrated|FLOAT")
    parser.add_argument("--envs", type=int, default=SAMPLE_ENVS)
    parser.add_argument("--steps", type=int, default=SAMPLE_STEPS)
    parser.add_argument("--minibatch-matches", type=int, default=MINIBATCH_MATCHES)
    parser.add_argument("--rows-per-match", type=int, default=ROWS_PER_MATCH)
    args = parser.parse_args(argv)
    migration = migrate(args.source, args.out, sample_seed=args.sample_seed,
                        heldout_seed=args.heldout_seed, matches=args.matches,
                        reader_gain=args.reader_gain, envs=args.envs, steps=args.steps,
                        minibatch_matches=args.minibatch_matches,
                        rows_per_match=args.rows_per_match)
    report = {"fusion_migration": migration, "out": args.out}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(report, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
