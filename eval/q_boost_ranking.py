"""Does the Q critic rank a state's candidate moves, and better than the policy does?

Follow-up to ``eval.q_boost_offline`` (docs/reports/vrpo-design-2026-10-04.md,
sections 5 and 7). At random self-play decisions it forces each of up to
``--max-candidates`` legal moves (the sampled one always included) and plays
``--rollouts`` sampled continuations to the end of the round, giving a noisy
true ``Q(s, c)`` per candidate. ``evaluate`` then scores rankers on the same
truth:

* Q critic (``train.q_critic``), * the policy's own log-probability, * random.

Metrics: the pooled within-state correlation of a ranker's scores with the
truth, and the top-1 gain, the true value of the ranker's best candidate minus
the state's average candidate. The truth's sampling noise is independent of
every ranker, so the top-1 gain is unbiased; correlations are attenuated by
that noise equally for all rankers.

    python -m eval.q_boost_ranking collect --checkpoint CKPT --rounds 250 --output P.pkl
    python -m eval.q_boost_ranking evaluate --checkpoint CKPT --qcritic Q.pt --points P.pkl --output R.json
"""
from __future__ import annotations

import argparse
import copy
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import random
import time

import gd
import numpy as np

from eval.critic_noise import PLAY, ROUND_END, _Listener, hidden_counts, make_deal
from eval.history_policy import apply_and_observe, explicit_passes, resolve_forced_passes
from eval.q_boost_offline import ACT_DIM, OBS_DIM, OfflineWorker, sample_rows, unpack


class RankWorker(OfflineWorker):
    def candidates_truth(self, state, stream, count: int, subset: list[int], k: int,
                         rng: np.random.Generator) -> dict:
        origin = int(state.to_move)
        branches, streams, group = [], [], []
        for slot, candidate in enumerate(subset):
            for _ in range(k):
                branch = gd.MatchState.deserialize(state.serialize())
                branch_stream = copy.deepcopy(stream)
                action = self.engine.legal_actions(branch)[candidate]
                apply_and_observe(self.engine, branch, action, [_Listener(branch_stream)])
                branches.append(branch)
                streams.append(branch_stream)
                group.append(slot)
        returns = np.asarray(self.to_end(branches, streams, origin, rng), np.float64)
        grid = returns.reshape(len(subset), k)
        return {"mean": grid.mean(1), "var": grid.var(1, ddof=1), "k": k}

    def play_points(self, job: dict) -> list[dict]:
        from train.history_model import PublicStream

        seed = job["seed"]
        rng = random.Random(seed)
        nprng = np.random.default_rng(seed)
        state = gd.MatchState()
        self.engine.set_deal(state, make_deal(rng, 0.5))
        stream = PublicStream()
        listener = _Listener(stream)
        points = []
        with explicit_passes(self.engine, [listener]):
            while True:
                resolve_forced_passes(self.engine, state, [listener])
                phase = int(state.phase)
                if phase == ROUND_END:
                    break
                actions = self.engine.legal_actions(state)
                if phase != PLAY:
                    apply_and_observe(self.engine, state, actions[self.engine.greedy(state)], [listener])
                    continue
                seat = int(state.to_move)
                row = self.table([state], [stream])[0][:len(actions)]
                choice = int(sample_rows(row[None], nprng)[0])
                if len(actions) > 2 and nprng.random() < job["point_rate"]:
                    others = [i for i in range(len(actions)) if i != choice]
                    nprng.shuffle(others)
                    subset = sorted([choice] + others[:job["max_candidates"] - 1])
                    truth = self.candidates_truth(state, stream, len(actions), subset,
                                                  job["rollouts"], nprng)
                    cand = np.stack([np.asarray(self.engine.encode_action(a, state, seat), np.uint8)
                                     for a in actions])
                    points.append({"seat": seat, "obs": np.packbits(np.asarray(state.observation(seat), np.uint8)[None], axis=1),
                                   "hidden": hidden_counts(state, seat).astype(np.uint8).reshape(1, -1),
                                   "cand": np.packbits(cand, axis=1), "probs": row.astype(np.float32),
                                   "subset": np.asarray(subset), "chosen": choice,
                                   "cards_left": len(state.hand(seat)), **truth})
                apply_and_observe(self.engine, state, actions[choice], [listener])
        return points


_WORKER = None


def _init(checkpoint: str, device: str) -> None:
    global _WORKER
    _WORKER = RankWorker(checkpoint, 1, device)


def _run(job: dict) -> list[dict]:
    return _WORKER.play_points(job)


def collect(args: argparse.Namespace) -> None:
    jobs = [{"seed": 4_000_000 + args.seed_offset + i, "point_rate": args.point_rate,
             "max_candidates": args.max_candidates, "rollouts": args.rollouts}
            for i in range(args.rounds)]
    started = time.perf_counter()
    points: list[dict] = []
    with get_context("spawn").Pool(args.workers, _init, (args.checkpoint, args.device)) as pool:
        for i, part in enumerate(pool.imap(_run, jobs, chunksize=1), 1):
            points.extend(part)
            if i % 10 == 0 or i == len(jobs):
                print(json.dumps({"rounds": i, "of": len(jobs), "points": len(points),
                                  "elapsed": round(time.perf_counter() - started, 1)}), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as handle:
        pickle.dump({"args": vars(args), "points": points}, handle, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate(args: argparse.Namespace) -> None:
    import torch
    from train.history_model import load_history_checkpoint
    from train.q_critic import QCritic

    torch.set_num_threads(args.threads)
    _, critic, _ = load_history_checkpoint(args.checkpoint, "cpu")
    model = QCritic(critic, ACT_DIM)
    model.load_state_dict(torch.load(args.qcritic, map_location="cpu")["model"])
    model.eval()
    points = []
    for path in args.points:
        with open(path, "rb") as handle:
            points.extend(pickle.load(handle)["points"])
    scores = {"q_critic": [], "policy": [], "random": []}
    truth, cards = [], []
    rng = np.random.default_rng(args.seed)
    for p in points:
        n = len(p["probs"])
        with torch.no_grad():
            q = model.forward_all(torch.as_tensor(unpack(p["obs"], OBS_DIM)),
                                  torch.as_tensor(p["hidden"]),
                                  torch.as_tensor(unpack(p["cand"], ACT_DIM)),
                                  torch.as_tensor([n])).numpy()
        subset = p["subset"]
        scores["q_critic"].append(q[subset])
        scores["policy"].append(np.log(np.maximum(p["probs"][subset], 1e-12)))
        scores["random"].append(rng.random(len(subset)))
        truth.append(p["mean"])
        cards.append(p["cards_left"])
    cards = np.asarray(cards)

    def per_state_corr(score: np.ndarray, true: np.ndarray) -> float:
        if np.std(score) < 1e-12:
            return 0.0
        return float(np.corrcoef(score, true)[0, 1])

    def metrics(idx: np.ndarray) -> dict:
        out = {"states": int(len(idx))}
        spread = np.mean([np.var(truth[i], ddof=1) - np.mean(points[i]["var"]) / points[i]["k"]
                          for i in idx])
        out["true_q_spread_var"] = float(spread)
        for name, values in scores.items():
            out[f"{name}_corr"] = float(np.mean([per_state_corr(values[i], truth[i]) for i in idx]))
            out[f"{name}_top1_gain"] = float(np.mean([truth[i][int(np.argmax(values[i]))] - truth[i].mean()
                                                      for i in idx]))
        return out

    everyone = np.arange(len(points))
    point = metrics(everyone)
    boots = [metrics(rng.integers(0, len(points), len(points))) for _ in range(500)]
    keys = [k for k in point if k not in ("states",)]
    ci = {k: [float(np.percentile([b[k] for b in boots], 2.5)),
              float(np.percentile([b[k] for b in boots], 97.5))] for k in keys}
    paired = {}
    for key in ("corr", "top1_gain"):
        diffs = [b[f"q_critic_{key}"] - b[f"policy_{key}"] for b in boots]
        paired[key] = {"point": point[f"q_critic_{key}"] - point[f"policy_{key}"],
                       "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}
    strata = {}
    for label, low, high in (("19-27", 19, 27), ("11-18", 11, 18), ("1-10", 1, 10)):
        idx = np.flatnonzero((cards >= low) & (cards <= high))
        if len(idx) > 20:
            strata[label] = metrics(idx)
    report = {"args": vars(args), "point": point, "ci95": ci, "q_minus_policy": paired,
              "by_cards_left": strata,
              "candidates_per_state": float(np.mean([len(p["subset"]) for p in points]))}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"point": point, "q_minus_policy": paired}, indent=1))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--checkpoint", required=True)
    c.add_argument("--rounds", type=int, required=True)
    c.add_argument("--point-rate", type=float, default=0.05)
    c.add_argument("--max-candidates", type=int, default=12)
    c.add_argument("--rollouts", type=int, default=32)
    c.add_argument("--seed-offset", type=int, default=0)
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--device", default="cpu")
    c.add_argument("--output", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--qcritic", required=True)
    e.add_argument("--points", nargs="+", required=True)
    e.add_argument("--threads", type=int, default=8)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    {"collect": collect, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
