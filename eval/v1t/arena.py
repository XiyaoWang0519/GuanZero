"""Duplicate evaluation of a history checkpoint against DanZero-V1T, our engine as referee.

    PYTHONPATH=python:oracle:. python -m eval.v1t.arena --checkpoint X.pt \\
        --deals 1000 --seed 20260929 --workers 8 --output out.json

Needs only ``gd``, numpy, torch and onnxruntime (no DanLM binaries), so it runs
on Linux. The V1T model file comes from ``--model``, ``V1T_ONNX`` or the DanLM
checkout. Deals follow ``eval.danlm.arena`` exactly (same generator and seed:
random level, both teams at the round level, a ``--tribute-fraction`` of deals
open with a tribute phase from a random previous order); every deal is played
twice with the teams swapped, house rules, our engine the only referee.

Our seats play exactly as in ``eval.danlm.arena``: canonical candidates,
greedy, engine heuristic tribute for a ``history_ppo`` checkpoint. V1T seats
use ``eval.v1t.player`` on the full-mode legal list. The score is our team's
mean net levels per round over both legs, with a bootstrap interval over deals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Any

import gd
import numpy as np

from eval.danlm.arena import deal_from_json, deal_to_json, generate_deals, has_tribute
from eval.duplicate import bootstrap_interval
from eval.history_policy import apply_and_observe, needs_history
from eval.v1t.player import V1TModel, V1TPlayer, default_model_path

_WORKER: dict[str, Any] = {}


def tribute_kind(deal: gd.DealSpec, state: gd.MatchState) -> str:
    """none, single, double or anti, read right after ``set_deal``."""
    if not has_tribute(deal):
        return "none"
    if state.phase == gd.Phase.Play:
        return "anti"
    p3, p4 = int(deal.prev_order[2]), int(deal.prev_order[3])
    return "double" if (p3 - p4) % 4 == 2 else "single"


def play_leg(deal: gd.DealSpec, seats: tuple[str, str, str, str], policy, v1t: V1TPlayer,
             rng: random.Random, max_decisions: int = 2000) -> dict:
    """One round; ``seats[s]`` is ``"ours"`` or ``"v1t"``. Returns a JSON-ready record."""
    rules = gd.RuleConfig.house()
    engine_full = gd.Engine(rules, gd.ActionConfig.full())
    engine_canon = gd.Engine(rules, gd.ActionConfig())
    engine_full.auto_pass = engine_canon.auto_pass = False
    state = gd.MatchState()
    engine_full.set_deal(state, deal)
    kind = tribute_kind(deal, state)
    v1t.begin_round(deal, state)
    v1t.start_match()
    listeners = [v1t]
    if needs_history(policy):
        policy.start_match()
        listeners.append(policy)
    elif hasattr(policy, "start_match"):
        policy.start_match()
    decisions = declared = 0
    while state.phase != gd.Phase.RoundEnd:
        if decisions >= max_decisions:
            raise RuntimeError(f"round exceeded {max_decisions} decisions")
        seat = int(state.to_move)
        if seats[seat] == "ours":
            actions = engine_canon.legal_actions(state)
            if len(actions) == 1:
                action = actions[0]
            else:
                action = actions[policy.select(engine_canon, state, actions, rng)]
        else:
            actions = engine_full.legal_actions(state)
            if len(actions) == 1:
                action = actions[0]
            else:
                decision = v1t.decide(state)
                action, declared = decision.action, declared + int(decision.declared)
        apply_and_observe(engine_full, state, action, listeners)
        decisions += 1
    if needs_history(policy) and policy.events_seen != v1t.events_seen:
        raise RuntimeError("the two players saw different event counts")
    result = engine_full.end_round(state)
    return {"order": [int(s) for s in result.order], "returns": [int(r) for r in result.seat_return],
            "decisions": decisions, "tribute": kind, "v1t_declared_wild": declared}


def _init_worker(checkpoint: str, model: str | None, torch_threads: int) -> None:
    import torch

    from eval.policies import load_policy

    torch.set_num_threads(torch_threads)
    _WORKER["policy"] = load_policy(checkpoint)
    _WORKER["v1t"] = V1TPlayer(V1TModel(model))


def _run_deals(job: list[dict]) -> list[dict]:
    policy, v1t = _WORKER["policy"], _WORKER["v1t"]
    out = []
    for item in job:
        deal = deal_from_json(item["deal"])
        legs = []
        for seats in (("ours", "v1t", "ours", "v1t"), ("v1t", "ours", "v1t", "ours")):
            legs.append(play_leg(deal, seats, policy, v1t, random.Random(item["seed"])))
        out.append({"id": item["id"], "legs": legs})
    return out


def run(checkpoint: str, deals: int, seed: int, workers: int, tribute_fraction: float = 0.5,
        model: str | None = None, torch_threads: int = 1, chunk: int = 8,
        bootstrap_samples: int = 2000) -> dict:
    deal_specs = generate_deals(deals, seed, tribute_fraction)
    items = [{"id": i, "deal": deal_to_json(deal), "seed": seed + i} for i, deal in enumerate(deal_specs)]
    jobs = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    started = time.monotonic()
    if workers <= 1:
        _init_worker(checkpoint, model, torch_threads)
        results = [r for job in jobs for r in _run_deals(job)]
    else:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing

        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context, initializer=_init_worker,
                                 initargs=(checkpoint, model, torch_threads)) as pool:
            results = [r for group in pool.map(_run_deals, jobs) for r in group]
    elapsed = time.monotonic() - started
    results.sort(key=lambda r: r["id"])
    per_deal, wins, by_tribute = [], 0, {}
    declared, decisions = 0, 0
    for record in results:
        a, b = record["legs"]
        score = (a["returns"][0] + b["returns"][1]) / 2
        per_deal.append(score)
        by_tribute.setdefault(a["tribute"], []).append(score)
        wins += int(a["order"][0] % 2 == 0) + int(b["order"][0] % 2 == 1)
        declared += a["v1t_declared_wild"] + b["v1t_declared_wild"]
        decisions += a["decisions"] + b["decisions"]
    from eval.policies import load_policy
    from infra.history_artifacts import sha256

    policy = load_policy(checkpoint)
    model_path = Path(model) if model else default_model_path()
    return {
        "mode": "duplicate", "referee": "gd (house rules)", "opponent": "danzero-v1t (eval.v1t)",
        "checkpoint": checkpoint, "checkpoint_name": policy.name,
        "checkpoint_id": getattr(policy, "checkpoint_id", None),
        "checkpoint_sha256": sha256(Path(checkpoint)),
        "v1t_model": str(model_path), "v1t_model_sha256": sha256(model_path),
        "deals": deals, "seed": seed, "tribute_fraction": tribute_fraction,
        "workers": workers, "elapsed_seconds": elapsed, "decisions": decisions,
        "scored_deals": len(per_deal),
        "levels_per_round": float(np.mean(per_deal)),
        "levels_per_round_ci95": list(bootstrap_interval(per_deal, seed, bootstrap_samples))
        if len(per_deal) > 1 else None,
        "round_win_rate": wins / (2 * len(per_deal)), "round_legs": 2 * len(per_deal),
        "by_tribute": {k: {"deals": len(v), "levels_per_round": float(np.mean(v))}
                       for k, v in sorted(by_tribute.items())},
        "v1t_declared_wild_choices": declared,
        "pair_scores": per_deal,
        "reading": "levels_per_round is our team's mean net level gain per round over both legs "
                   "of each deal (gd's seat_return); round_win_rate is the fraction of legs whose "
                   "first finisher is on our team. v1t_declared_wild_choices counts V1T picks of a "
                   "DanLM-only wild declaration, played at the level rank here.",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="our load_policy spec (history_ppo .pt)")
    parser.add_argument("--deals", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tribute-fraction", type=float, default=0.5)
    parser.add_argument("--model", default=None, help="V1T int8 ONNX file (default: V1T_ONNX or "
                                                      "the DanLM checkout)")
    parser.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.checkpoint, args.deals, args.seed, args.workers, args.tribute_fraction,
                 args.model, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    ci = report["levels_per_round_ci95"] or [float("nan")] * 2
    print(f"{Path(args.checkpoint).name} vs V1T: {report['levels_per_round']:+.3f} "
          f"[{ci[0]:+.3f}, {ci[1]:+.3f}] over {report['scored_deals']} deals, "
          f"win rate {report['round_win_rate']:.3f}, {report['elapsed_seconds']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
