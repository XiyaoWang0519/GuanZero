"""Frozen mechanism readouts for the split-fusion habit A/B (no training).

    PYTHONPATH=python:oracle:. python -m bench.fusion_habit_mechanism CKPT --pack U2623 \
        --out OUT.json [--envs 16 --per-env 4 --seed 910001 --threads 8]

Plays ``per_env`` paired match slots per env against the styled pack exactly as
``eval.history_habit_eval`` does (axis pass, strength 4, slot-fixed team and z;
learner in its native view, full or oracle), keeps every learner Play row of
those matches and reads, on rows with at least two candidates:

* reader share: rms of the reader's term in the pre-FFN state over rms of that
  state. Split: ``reader_norm(r)``. Sum: ``LN(q + r) - LN(q + mean r)``, the mean
  taken over the scored rows.
* z readout: closed-form ridge (lambda = number of training rows) on the
  standardized decision state and on the raw reader output, rows of round
  index >= 1, two folds by match ((env + match) % 2), held-out AUC with a
  10,000-draw match bootstrap interval and a 200-permutation match-level null.
* style-specific history sensitivity: rows of round index >= 2. All tokens of
  earlier rounds are replaced, equal length and keeping the original round and
  phase metadata, by the last tokens of the earlier rounds of a donor match
  with the opposite z or with the same z (another match). Metric per match:
  mean TV(opposite) - mean TV(same) over its rows; reported as the mean over
  matches with a match bootstrap interval. Donors are drawn with a fixed
  generator among matches that have enough earlier-round tokens.

Information and sensitivity only: none of this says an action is better.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import gd
import numpy as np
import torch

from eval.history_habit_eval import slot_assignment
from train.history_habit import STYLE_IDENTITY, HabitPack
from train.history_model import (PLAY_PHASE, StreamBatch, load_history_checkpoint,
                                 segment_log_softmax)
from train.history_rollout import HistoryCollector, MatchEventStore, SequenceRolloutBuffer

BOOTSTRAP = 10_000
BOOT_SEED = 20261018
DONOR_SEED = 20261020
PERMUTATIONS = 200


def collect(actor, pack: str, envs: int, per_env: int, seed: int):
    habit = HabitPack(actor, "mechanism", seed + 17, pack, "pass", 4.0, "cpu")
    oracle = bool(actor.config.style_input)

    def seats(env: int, match: int) -> list[int]:
        team, z = slot_assignment(seed, env, match)
        habit.styles[(int(env), int(match))] = z
        assignment = [STYLE_IDENTITY[z]] * 4
        assignment[team] = assignment[team + 2] = 0
        return assignment

    styles: dict[tuple[int, int], int] = {}

    def style(env: int, match: int) -> int:
        styles[(int(env), int(match))] = habit.style(env, match)
        return styles[(int(env), int(match))]

    env = gd.VecEnv(num_envs=envs, num_threads=1, seed=seed, log_public_actions=True,
                    log_env_limit=envs)
    store = MatchEventStore()
    buffer = SequenceRolloutBuffer()
    collector = HistoryCollector(env, actor, store, buffer, torch.Generator().manual_seed(seed),
                                 "cpu", seat_policy=seats, resolve_policy=habit.resolve,
                                 kv_cache=True, row_style=style if oracle else None)
    wanted = {(e, m) for e in range(envs) for m in range(per_env)}
    finished: set[tuple[int, int]] = set()
    z_of: dict[tuple[int, int], int] = {}
    with torch.inference_mode():
        while not wanted <= finished:
            collector.collect(256)
            for r in collector.results:
                if r.match_winner >= 0:
                    finished.add((int(r.env_id), int(r.match_id)))
            z_of.update({k: v for k, v in habit.styles.items() if k in wanted})
    return buffer, store, wanted, z_of


def rms(x: torch.Tensor) -> torch.Tensor:
    return x.square().mean(-1).sqrt()


def readout_parts(actor, buffer, store, rows, z_rows, streams=None):
    """(state, reader output, reader term, pre-FFN state, log-probs, inputs)."""
    inputs, _ = buffer.decision_inputs(rows, store, "cpu")
    if streams is not None:
        inputs.streams = streams
    if actor.config.style_input:
        inputs.style = torch.as_tensor(z_rows, dtype=torch.long)
    with torch.no_grad():
        encoded = actor.encode_batch(inputs.streams)
        query = actor.private_query(inputs)
        reading = actor.reader_input(query)
        attended = actor._attend_encoded(reading, encoded, inputs.match_index, inputs.prefix,
                                         False)
        state = actor.fuse(reading, attended)
        logits = actor.candidate_logits(state, inputs.cand, inputs.offsets, rows=inputs.rows)
        logp = segment_log_softmax(logits, inputs.rows, inputs.decisions)
        reference = actor.decision_states(encoded, inputs)
    if not torch.allclose(state, reference, atol=1e-5, rtol=1e-5):
        raise AssertionError("explicit state differs from decision_states")
    return state, attended, reading, inputs, logp


def reader_share(actor, attended, reading) -> np.ndarray:
    with torch.no_grad():
        if actor.reader_norm is not None:
            term = actor.reader_norm(attended)
            pre = reading + term
        else:
            pre = actor.attention_norm(reading + attended)
            term = pre - actor.attention_norm(reading + attended.mean(0, keepdim=True))
    return (rms(term) / rms(pre)).numpy()


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos, neg = score[label > 0], score[label < 0]
    if not len(pos) or not len(neg):
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable"), kind="stable") + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cross_fit(x: np.ndarray, z: np.ndarray, fold: np.ndarray) -> np.ndarray:
    score = np.zeros(len(z))
    for f in (0, 1):
        train, test = fold != f, fold == f
        mu, sd = x[train].mean(0), x[train].std(0) + 1e-6
        a = (x[train] - mu) / sd
        w = np.linalg.solve(a.T @ a + len(a) * np.eye(a.shape[1]), a.T @ (z[train] - z[train].mean()))
        score[test] = ((x[test] - mu) / sd) @ w
    return score


def ridge_readout(x: np.ndarray, z: np.ndarray, groups: np.ndarray, fold: np.ndarray) -> dict:
    """Held-out AUC, its match bootstrap, and a null from permuting z across
    matches (z is constant within a match, so grouped cross-fitting is biased
    below 0.5 with few matches; compare against the null, not against 0.5)."""
    score = cross_fit(x, z, fold)
    point = auc(score, z)
    keys = np.unique(groups)
    index = {k: np.flatnonzero(groups == k) for k in keys}
    rng = np.random.default_rng(BOOT_SEED)
    draws = []
    for _ in range(BOOTSTRAP):
        picked = np.concatenate([index[k] for k in rng.choice(keys, len(keys))])
        draws.append(auc(score[picked], z[picked]))
    match_z = {k: z[index[k][0]] for k in keys}
    null = []
    for _ in range(PERMUTATIONS):
        shuffled = dict(zip(keys, rng.permutation([match_z[k] for k in keys])))
        zp = np.asarray([shuffled[g] for g in groups])
        null.append(auc(cross_fit(x, zp, fold), zp))
    null = np.asarray(null)
    return dict(auc=point, ci95=np.nanquantile(draws, [0.025, 0.975]).tolist(),
                null_mean=float(np.nanmean(null)), null_q95=float(np.nanquantile(null, 0.95)),
                p_value=float((1 + np.sum(null >= point)) / (1 + len(null))),
                rows=int(len(z)), matches=int(len(keys)))


def segment_tv(p: torch.Tensor, q: torch.Tensor, rows: torch.Tensor, n: int) -> np.ndarray:
    return (0.5 * torch.zeros(n).index_add_(0, rows, (p.exp() - q.exp()).abs())).numpy()


def match_bootstrap(values: dict) -> dict:
    keys = sorted(values)
    arr = np.asarray([values[k] for k in keys])
    rng = np.random.default_rng(BOOT_SEED)
    draws = arr[rng.integers(0, len(arr), (BOOTSTRAP, len(arr)))].mean(1)
    return dict(mean=float(arr.mean()), ci95=np.quantile(draws, [0.025, 0.975]).tolist(),
                matches=len(arr))


def history_swap(actor, buffer, store, rows, data, z_of, keys_all) -> dict:
    """Opposite-z minus same-z donor TV, grouped by current round index."""
    rng = np.random.default_rng(DONOR_SEED)
    stream_of = {k: store.stream(*k).arrays() for k in keys_all}
    per_match: dict[tuple, list] = {}
    raw = {"opposite": [], "same": []}
    flips = {"opposite": 0, "same": 0}
    missing = 0
    row_key = np.asarray([(int(e), int(m)) for e, m in zip(data["env"], data["match"])])
    for current in sorted(set(data["round"][rows].tolist())):
        picked = rows[data["round"][rows] == current]
        donors = {}
        for k in sorted({tuple(row_key[i]) for i in picked}):
            need = int(np.count_nonzero(stream_of[k][1] < current))
            choices = {}
            for kind, want in (("opposite", -z_of[k]), ("same", z_of[k])):
                pool = [d for d in keys_all if d != k and z_of[d] == want
                        and np.count_nonzero(stream_of[d][1] < current) >= need]
                choices[kind] = pool[int(rng.integers(len(pool)))] if pool else None
            if need > 0 and all(choices.values()):
                donors[k] = choices
            else:
                missing += 1
        picked = np.asarray([i for i in picked if tuple(row_key[i]) in donors], np.int64)
        if not len(picked):
            continue
        unique, _ = buffer._group_rows(picked)
        old = {k: int(np.count_nonzero(stream_of[k][1] < current)) for k in unique}
        cited = {k: int(max(data["prefix"][i] for i in picked if tuple(row_key[i]) == k))
                 for k in unique}
        z_rows = np.asarray([z_of[tuple(row_key[i])] for i in picked])
        _, _, _, inputs, base = readout_parts(actor, buffer, store, picked, z_rows)
        for kind in ("opposite", "same"):
            arrays = []
            for k in unique:
                t, r, p = (a[:cited[k]].copy() for a in stream_of[k])
                n = old[k]
                d = donors[k][kind]
                dt = stream_of[d][0][:np.count_nonzero(stream_of[d][1] < current)]
                t[:n] = dt[-n:]
                arrays.append((t, r, p))
            swapped = StreamBatch.from_arrays(arrays, "cpu")
            _, _, _, _, logp = readout_parts(actor, buffer, store, picked, z_rows, swapped)
            tv = segment_tv(base, logp, inputs.rows, inputs.decisions)
            raw[kind].append(tv)
            for i in range(inputs.decisions):
                a, b = int(inputs.offsets[i]), int(inputs.offsets[i + 1])
                flips[kind] += int(base[a:b].argmax() != logp[a:b].argmax())
            for i, row in enumerate(picked):
                per_match.setdefault(tuple(row_key[row]), {"opposite": [], "same": []})[
                    kind].append(float(tv[i]))
    diff = {k: np.mean(v["opposite"]) - np.mean(v["same"]) for k, v in per_match.items()}
    return dict(style_specific_tv=match_bootstrap(diff),
                tv_opposite_mean=float(np.concatenate(raw["opposite"]).mean()),
                tv_same_mean=float(np.concatenate(raw["same"]).mean()),
                argmax_changes=flips, rows=int(sum(len(x) for x in raw["same"])),
                skipped_match_rounds=missing)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("checkpoint")
    parser.add_argument("--pack", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--per-env", type=int, default=4)
    parser.add_argument("--seed", type=int, default=910001)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    torch.set_num_threads(args.threads)
    started = time.time()
    actor, _, payload = load_history_checkpoint(args.checkpoint, "cpu")
    actor.eval()
    buffer, store, wanted, z_of = collect(actor, args.pack, args.envs, args.per_env, args.seed)
    data = buffer.compact()
    keys = list(zip(data["env"].tolist(), data["match"].tolist()))
    rows = np.asarray([i for i, k in enumerate(keys) if k in wanted
                       and data["phase"][i] == PLAY_PHASE and data["cand_count"][i] > 1], np.int64)
    z_rows = np.asarray([z_of[keys[i]] for i in rows])
    state, attended, reading, _, _ = readout_parts(actor, buffer, store, rows, z_rows)
    share = reader_share(actor, attended, reading)
    groups = np.asarray([keys[i][0] * 1000 + keys[i][1] for i in rows])
    later = data["round"][rows] >= 1
    fold = (np.asarray([keys[i][0] + keys[i][1] for i in rows]) % 2)[later]
    result = dict(
        checkpoint=args.checkpoint, update=payload.get("progress", {}).get("updates"),
        architecture=dict(private_fusion=actor.config.private_fusion,
                          private_qk_norm=actor.config.private_qk_norm,
                          style_input=actor.config.style_input),
        seed=args.seed, envs=args.envs, per_env=args.per_env, matches=len(wanted),
        rows=int(len(rows)),
        reader_share=dict(median=float(np.median(share)), mean=float(share.mean()),
                          p90=float(np.quantile(share, 0.9))),
        z_readout=dict(state=ridge_readout(state.numpy()[later], z_rows[later], groups[later], fold),
                       reader=ridge_readout(attended.numpy()[later], z_rows[later], groups[later],
                                            fold)),
        history_swap=history_swap(actor, buffer, store, rows[data["round"][rows] >= 2], data, z_of,
                                  sorted(wanted)),
        seconds=time.time() - started)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("rows", "reader_share", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
