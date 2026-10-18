"""Whose sampling is the action-sampling noise: our team's or the opponents'?

``eval/critic_noise.py`` showed most of the critic's residual is the variance
of the round return given the state, caused by the four players' sampled
actions. This script splits that variance by team. For each measured state it
plays an ``I x J`` grid of continuations to the round end: our team's
(acting seat and partner) sampling uses uniform stream ``a``, the opponents'
uses stream ``b``, and rollout ``(a, b)`` pairs them. The policy is
unchanged (every draw is an ordinary sample from the actor's softmax); only
the randomness is arranged so that one side can be held fixed.

Two-way ANOVA with one replicate per cell gives per-state variance components

    own team (main effect of a), opponents (main effect of b), interaction,

and their sum is the within-state variance of the return. A large interaction
share means the two sides' randomness cannot be separated additively (one
changed move reroutes the whole rest of the round), so the useful numbers are
the total effects: the share of variance left when one side's sampling is
removed (``left_if_own_fixed``, ``left_if_opponent_fixed``). See
docs/reports/vrpo-design-2026-10-04.md, section 4.

    python -m eval.critic_noise_split --checkpoint CKPT --deals 300 --grid 8 8 \\
        --workers 4 --output OUT.json
"""
from __future__ import annotations

import argparse
import copy
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np

from eval.critic_noise import PLAY, ROUND_END, Worker, _Listener
from eval.history_policy import apply_and_observe, resolve_forced_passes

ROUND_END_PHASE = ROUND_END


def anova_components(grid: np.ndarray) -> dict[str, float]:
    """Variance components of a ``[I, J]`` return grid (own x opponent stream)."""
    rows, cols = grid.shape
    row_mean = grid.mean(1)
    col_mean = grid.mean(0)
    grand = grid.mean()
    ms_own = cols * row_mean.var(ddof=1)
    ms_opp = rows * col_mean.var(ddof=1)
    resid = grid - row_mean[:, None] - col_mean[None, :] + grand
    ms_inter = (resid ** 2).sum() / ((rows - 1) * (cols - 1))
    return {"own": max((ms_own - ms_inter) / cols, 0.0),
            "opponent": max((ms_opp - ms_inter) / rows, 0.0),
            "interaction": float(ms_inter)}


class SplitWorker(Worker):
    """Worker whose rollouts form the I x J grid with side-specific randomness."""

    grid = (8, 8)
    job_seed = 0
    calls = 0

    def deal(self, job: dict) -> list[dict]:
        self.job_seed = int(job["seed"])
        self.calls = 0
        return super().deal(job)

    def table(self, branches: list, streams: list) -> np.ndarray:
        """Candidate probabilities, one row per branch (``-inf`` log-probs -> 0)."""
        import torch

        engine = self.engine
        obs, cand, offsets, seats = [], [], [0], []
        for state in branches:
            actions = engine.legal_actions(state)
            seat = int(state.to_move)
            seats.append(seat)
            obs.append(np.asarray(state.observation(seat), np.float32))
            cand.extend(np.asarray(engine.encode_action(a, state, seat), np.float32) for a in actions)
            offsets.append(offsets[-1] + len(actions))
        inputs = self.policy.batch_inputs(streams, np.arange(len(branches)), np.asarray(seats),
                                          np.stack(obs), np.stack(cand), np.asarray(offsets))
        with torch.inference_mode():
            log_table = self.policy.actor._log_prob_table(inputs, None, None)
        # float64 where the device has it (CPU, as before); MPS has no float64
        probs = torch.exp(log_table.double() if log_table.device.type == "cpu" else log_table.float())
        return probs.cpu().numpy()

    def rollouts(self, state, stream, k: int, generator) -> list[float]:
        rows, cols = self.grid
        total = rows * cols
        self.calls += 1
        origin = int(state.to_move)
        branches = [type(state).deserialize(state.serialize()) for _ in range(total)]
        streams = [copy.deepcopy(stream) for _ in range(total)]
        listeners = [_Listener(s) for s in streams]
        # Independent uniform streams per side and per grid line; every rollout
        # owns its own copy, so what a draw does cannot depend on batch makeup.
        own_rng = [np.random.default_rng([self.job_seed, self.calls, 0, i // cols])
                   for i in range(total)]
        opp_rng = [np.random.default_rng([self.job_seed, self.calls, 1, i % cols])
                   for i in range(total)]
        returns = [None] * total
        active = list(range(total))
        guard = 0
        while active:
            guard += 1
            if guard > 2000:
                raise RuntimeError("rollout does not terminate")
            for i in list(active):
                resolve_forced_passes(self.engine, branches[i], [listeners[i]])
                if int(branches[i].phase) == ROUND_END_PHASE:
                    returns[i] = float(self.engine.end_round(branches[i]).seat_return[origin])
                    active.remove(i)
            if not active:
                break
            probs = self.table([branches[i] for i in active], [streams[i] for i in active])
            for row, i in enumerate(active):
                mover = int(branches[i].to_move)
                rng = own_rng[i] if (mover - origin) % 2 == 0 else opp_rng[i]
                cumulative = probs[row].cumsum()
                choice = int(min(np.searchsorted(cumulative, rng.random() * cumulative[-1],
                                                 side="right"), len(cumulative) - 1))
                actions = self.engine.legal_actions(branches[i])
                apply_and_observe(self.engine, branches[i], actions[choice], [listeners[i]])
        return returns


_WORKER = None


def _init(checkpoint: str, threads: int, grid: tuple[int, int]) -> None:
    global _WORKER
    _WORKER = SplitWorker(checkpoint, threads)
    _WORKER.grid = tuple(grid)


def _run(job: dict) -> list[dict]:
    return _WORKER.deal(job)


def summarize(records: list[dict], grid: tuple[int, int], rng: np.random.Generator,
              boots: int = 1000) -> dict:
    comps = [anova_components(np.asarray(r["returns"], np.float64).reshape(grid)) for r in records]
    own = np.array([c["own"] for c in comps])
    opp = np.array([c["opponent"] for c in comps])
    inter = np.array([c["interaction"] for c in comps])
    cards = np.array([r["cards_left"] for r in records])

    def shares(idx: np.ndarray) -> dict:
        total = own[idx].sum() + opp[idx].sum() + inter[idx].sum()
        return {"total_variance": float(total / len(idx)), "own": float(own[idx].sum() / total),
                "opponent": float(opp[idx].sum() / total),
                "interaction": float(inter[idx].sum() / total),
                # variance left when ONE side's randomness is removed (total effects):
                # without our team's sampling only opponent + interaction remain, and vice versa
                "left_if_own_fixed": float((opp[idx].sum() + inter[idx].sum()) / total),
                "left_if_opponent_fixed": float((own[idx].sum() + inter[idx].sum()) / total)}

    everyone = np.arange(len(records))
    point = shares(everyone)
    draws = [shares(rng.integers(0, len(records), len(records))) for _ in range(boots)]
    ci = {key: [float(np.percentile([d[key] for d in draws], 2.5)),
                float(np.percentile([d[key] for d in draws], 97.5))] for key in point}
    by_cards = {}
    for name, low, high in (("19-27", 19, 27), ("11-18", 11, 18), ("1-10", 1, 10)):
        idx = np.nonzero((cards >= low) & (cards <= high))[0]
        if len(idx) > 10:
            by_cards[name] = {"states": int(len(idx)), **shares(idx)}
    return {"states": len(records), "grid": list(grid), "point": point, "ci95": ci,
            "by_cards_left": by_cards}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--deals", type=int, default=300)
    parser.add_argument("--states-per-deal", type=int, default=4)
    parser.add_argument("--grid", type=int, nargs=2, default=[8, 8], metavar=("OWN", "OPP"))
    parser.add_argument("--tribute-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    grid = tuple(args.grid)
    if min(grid) < 2:
        raise SystemExit("--grid needs at least 2 streams per side")
    jobs = [{"index": i, "seed": args.seed * 1000 + i, "states_per_deal": args.states_per_deal,
             "rollouts": grid[0] * grid[1], "tribute_fraction": args.tribute_fraction}
            for i in range(args.deals)]
    started = time.perf_counter()
    records: list[dict] = []
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with get_context("spawn").Pool(args.workers, _init, (args.checkpoint, args.threads, grid)) as pool:
        for i, part in enumerate(pool.imap_unordered(_run, jobs), 1):
            records.extend(part)
            print(json.dumps({"deals": i, "of": len(jobs), "states": len(records),
                              "elapsed": round(time.perf_counter() - started, 1)}), flush=True)
    report = {"checkpoint": args.checkpoint, "args": vars(args),
              "elapsed_seconds": round(time.perf_counter() - started, 1),
              "summary": summarize(records, grid, np.random.default_rng(args.seed)),
              "records": records}
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
