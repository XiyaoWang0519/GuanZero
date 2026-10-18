"""How much of the critic's residual is irreducible action-sampling noise?

The history trainer's critic V(obs, hidden hands) sees every hand, so after the
deal the only randomness left in a round is the four players' sampled actions.
For a set of play-phase states s drawn from sampled self-play, this script runs
K independent sampled rollouts of the current policy to the end of the round
and records each return G (the acting team's round return, the trainer's
target with gamma 1). Per state:

* within-state variance of G: the action-sampling noise at s; no critic,
  however good, can predict it, and it is what Q-boosting averages out;
* (mean G - V(s))^2 minus its sampling share: the critic's own squared error.

Over all states the critic's mean squared error against a single return
splits as noise + critic error, and ``1 - mse / var(G)`` is the Monte Carlo
explained variance (the trainer logs it against lambda-returns instead).

    python -m eval.critic_noise --checkpoint CKPT --deals 60 --states-per-deal 4 \\
        --rollouts 32 --workers 4 --output OUT.json
"""
from __future__ import annotations

import argparse
import copy
import json
from multiprocessing import get_context
from pathlib import Path
import random
import time

import gd
import numpy as np

from eval.history_policy import apply_and_observe, explicit_passes, resolve_forced_passes

PLAY = int(gd.Phase.Play)
ROUND_END = int(gd.Phase.RoundEnd)


class _Listener:
    def __init__(self, stream) -> None:
        self.stream = stream

    def observe(self, event) -> None:
        self.stream.append(event)


def hidden_counts(state: gd.MatchState, seat: int) -> np.ndarray:
    """The critic's privileged input: per-card counts of seats seat+1..seat+3."""
    out = np.zeros((3, 54), np.float32)
    for rel in range(1, 4):
        for card in state.hand((seat + rel) % 4):
            out[rel - 1, card] += 1
    return out


def make_deal(rng: random.Random, tribute_fraction: float) -> gd.DealSpec:
    deck = [card for card in range(54) for _ in range(2)]
    rng.shuffle(deck)
    deal = gd.DealSpec()
    deal.hands = [sorted(deck[seat * 27:(seat + 1) * 27]) for seat in range(4)]
    deal.level = rng.randrange(13)
    deal.team_levels = [deal.level, deal.level]
    deal.owner = -1
    if rng.random() < tribute_fraction:
        order = list(range(4))
        rng.shuffle(order)
        deal.prev_order = order
    else:
        deal.leader = rng.randrange(4)
    return deal


class Worker:
    def __init__(self, checkpoint: str, threads: int) -> None:
        import torch
        from train.history_model import load_history_checkpoint
        from eval.history_policy import HistoryPolicy

        torch.set_num_threads(threads)
        actor, critic, _ = load_history_checkpoint(checkpoint, "cpu")
        self.torch = torch
        self.critic = critic.eval()
        self.policy = HistoryPolicy(actor, sample=True)
        self.engine = gd.Engine(gd.RuleConfig.house(), gd.ActionConfig())
        self.engine.auto_pass = False

    def value(self, state: gd.MatchState) -> float:
        seat = int(state.to_move)
        obs = np.asarray(state.observation(seat), np.float32)[None]
        hidden = hidden_counts(state, seat)[None]
        with self.torch.no_grad():
            return float(self.critic(self.torch.as_tensor(obs), self.torch.as_tensor(hidden))[0])

    def act(self, branches: list, streams: list, generator) -> list[int]:
        """One sampled choice per branch, all in one actor call."""
        engine = self.engine
        legal, obs, cand, offsets, seats = [], [], [], [0], []
        for state in branches:
            actions = engine.legal_actions(state)
            seat = int(state.to_move)
            legal.append(actions)
            seats.append(seat)
            obs.append(np.asarray(state.observation(seat), np.float32))
            cand.extend(np.asarray(engine.encode_action(a, state, seat), np.float32) for a in actions)
            offsets.append(offsets[-1] + len(actions))
        inputs = self.policy.batch_inputs(streams, np.arange(len(branches)), np.asarray(seats),
                                          np.stack(obs), np.stack(cand), np.asarray(offsets))
        choice = self.policy.act(inputs, generator)
        return [int(c) for c in choice]

    def rollouts(self, state: gd.MatchState, stream, k: int, generator) -> list[float]:
        """K sampled continuations to the round end; the acting team's returns."""
        seat = int(state.to_move)
        branches = [gd.MatchState.deserialize(state.serialize()) for _ in range(k)]
        streams = [copy.deepcopy(stream) for _ in range(k)]
        listeners = [_Listener(s) for s in streams]
        returns = [None] * k
        active = list(range(k))
        guard = 0
        while active:
            guard += 1
            if guard > 2000:
                raise RuntimeError("rollout does not terminate")
            for i in list(active):
                resolve_forced_passes(self.engine, branches[i], [listeners[i]])
                if int(branches[i].phase) == ROUND_END:
                    returns[i] = float(self.engine.end_round(branches[i]).seat_return[seat])
                    active.remove(i)
            if not active:
                break
            choices = self.act([branches[i] for i in active], [streams[i] for i in active],
                               generator)
            for i, c in zip(list(active), choices):
                actions = self.engine.legal_actions(branches[i])
                apply_and_observe(self.engine, branches[i], actions[c], [listeners[i]])
        return returns

    def deal(self, job: dict) -> list[dict]:
        from train.history_model import PublicStream

        rng = random.Random(job["seed"])
        generator = self.torch.Generator()
        generator.manual_seed(job["seed"])
        state = gd.MatchState()
        self.engine.set_deal(state, make_deal(rng, job["tribute_fraction"]))
        stream = PublicStream()
        listener = _Listener(stream)
        records = []
        decisions = 0
        targets = None
        with explicit_passes(self.engine, [listener]):
            while True:
                resolve_forced_passes(self.engine, state, [listener])
                phase = int(state.phase)
                if phase == ROUND_END:
                    break
                actions = self.engine.legal_actions(state)
                if phase != PLAY:
                    apply_and_observe(self.engine, state, actions[self.engine.greedy(state)],
                                      [listener])
                    continue
                if targets is None:
                    # Pick measurement points among the first 60 multi-choice decisions.
                    targets = set(rng.sample(range(60), job["states_per_deal"]))
                if decisions in targets and len(actions) > 1:
                    started = time.perf_counter()
                    value = self.value(state)
                    returns = self.rollouts(state, stream, job["rollouts"], generator)
                    records.append({"deal": job["index"], "decision": decisions,
                                    "seat": int(state.to_move), "cards_left": len(state.hand(int(state.to_move))),
                                    "value": value, "returns": returns,
                                    "seconds": round(time.perf_counter() - started, 2)})
                if len(actions) > 1:
                    decisions += 1
                choice = self.act([state], [stream], generator)[0]
                apply_and_observe(self.engine, state, actions[choice], [listener])
        return records


_WORKER = None


def _init(checkpoint: str, threads: int) -> None:
    global _WORKER
    _WORKER = Worker(checkpoint, threads)


def _run(job: dict) -> list[dict]:
    return _WORKER.deal(job)


def summarize(records: list[dict], rng: np.random.Generator, boots: int = 1000) -> dict:
    v = np.array([r["value"] for r in records])
    g = np.array([r["returns"] for r in records], dtype=np.float64)     # [n, k]
    n, k = g.shape

    def stats(idx: np.ndarray) -> dict:
        gi, vi = g[idx], v[idx]
        noise = gi.var(axis=1, ddof=1).mean()                 # E[Var(G | s)]
        mean = gi.mean(axis=1)
        critic_err = ((mean - vi) ** 2).mean() - noise / k    # E[(mu(s) - V(s))^2]
        total_var = gi.var()                                  # Var(G) over states and draws
        mse = noise + critic_err                              # E[(G - V)^2] for a single draw
        return {"noise": noise, "critic_error": critic_err, "mse": mse, "var_return": total_var,
                "noise_share_of_mse": noise / mse, "ev_mc": 1 - mse / total_var,
                "ev_ceiling": 1 - noise / total_var}

    point = stats(np.arange(n))
    draws = [stats(rng.integers(0, n, n)) for _ in range(boots)]
    ci = {key: [float(np.percentile([d[key] for d in draws], 2.5)),
                float(np.percentile([d[key] for d in draws], 97.5))] for key in point}
    return {"states": int(n), "rollouts": int(k),
            "point": {key: float(val) for key, val in point.items()}, "ci95": ci}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--deals", type=int, default=60)
    parser.add_argument("--states-per-deal", type=int, default=4)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--tribute-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    jobs = [{"index": i, "seed": args.seed * 1000 + i, "states_per_deal": args.states_per_deal,
             "rollouts": args.rollouts, "tribute_fraction": args.tribute_fraction}
            for i in range(args.deals)]
    started = time.perf_counter()
    records: list[dict] = []
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with get_context("spawn").Pool(args.workers, _init, (args.checkpoint, args.threads)) as pool:
        for i, part in enumerate(pool.imap_unordered(_run, jobs), 1):
            records.extend(part)
            print(json.dumps({"deals": i, "of": len(jobs), "states": len(records),
                              "elapsed": round(time.perf_counter() - started, 1)}), flush=True)
    report = {"checkpoint": args.checkpoint, "args": vars(args),
              "elapsed_seconds": round(time.perf_counter() - started, 1),
              "summary": summarize(records, np.random.default_rng(args.seed)),
              "records": records}
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
