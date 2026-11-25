"""Whole-match ladder: our policy against each competition bot on OpenGuanDan.

Each match runs from level 2 until a team passes A, refereed by the
OpenGuanDan table, the protocol the 1st National Guandan AI competition was
played on. Every seed is played twice with the teams swapped (our policy on
seats 1/3, then on 0/2); the table's own RNG is reseeded per match, so both
legs open with the same deal. Reported per opponent:

* match win rate (the metric of DanZero's Table I and DanLM's whole-game
  table), with a Wilson interval;
* per-round first-finisher rate for our team (DanLM's single-round metric,
  here on the levels a real match visits rather than random ones) and net
  levels per round (+3/+2/+1 for the first finisher's team);
* mirror health and bot failures (crashes, timeouts, out-of-range answers),
  where every failed bot decision was replaced by a random legal one.

Usage::

    BOTS_ROOT=.../DanLM/baselines OGD_ROOT=.../OpenGuanDan \
    python -m eval.ogd_ladder.ladder --policy <spec> --bots all --seeds 50 \
        --workers 3 --out .work/ogd-ladder/run1
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import math
from multiprocessing import get_context
import os
from pathlib import Path
import random
import time
from typing import Any

from eval.ogd_ladder.bots import BOT_DIRS, BotProcess, RandomAgent

GAIN = {1: 3, 2: 2, 3: 1}   # partner's finishing place -> levels for the first finisher's team


def round_score(order: list[int]) -> tuple[int, int]:
    """(team of the first finisher, levels it gains) from a finishing order."""
    first = order[0]
    partner_place = order.index((first + 2) % 4)
    return first % 2, GAIN[partner_place]


_POLICY: dict[str, Any] = {}


def _policy(spec: str) -> Any:
    if spec not in _POLICY:
        import torch

        torch.set_num_threads(1)
        from eval.policies import load_policy

        _POLICY[spec] = load_policy(spec)
    return _POLICY[spec]


def make_agent(kind: str, rng: random.Random):
    if kind == "random":
        return RandomAgent(rng)
    return BotProcess(kind, rng)


def play_match(task: dict) -> dict:
    """One full match. ``task``: opponent, policy, seed, our_team (0 or 1)."""
    from eval.ogd_adapter import bridge
    from eval.ogd_ladder.ours import OurSeats

    logging.getLogger().setLevel(logging.CRITICAL)
    seed, our_team = int(task["seed"]), int(task["our_team"])
    rng = random.Random(seed * 2 + our_team)
    our_seats = (our_team, our_team + 2)
    ours = OurSeats(_policy(task["policy"]), our_seats, rng)
    agents = {s: make_agent(task["opponent"], random.Random(rng.getrandbits(32)))
              for s in range(4) if s not in our_seats}
    started = time.perf_counter()
    random.seed(seed)
    env = bridge.make_env()
    rounds: list[dict] = []
    fn, idx, steps, status = env.start, None, 0, "ok"
    try:
        while True:
            msgs = fn({"actIndex": idx}) if idx is not None else fn()
            idx = None
            for seat, body in msgs:
                ours.notify(seat, body)
                if body.get("type") == "notify" and body.get("stage") == "episodeOver" and seat == 0:
                    order = [int(s) for s in body["order"]]
                    team, gain = round_score(order)
                    rounds.append({"level": str(body["curRank"]), "order": order,
                                   "first_team": team, "gain": gain})
                if body.get("type") == "act":
                    if seat in our_seats:
                        idx = ours.act(seat, body, None)
                    else:
                        idx = agents[seat].handle(body)
                        ours.act(seat, body, idx)
                elif seat not in our_seats:
                    agents[seat].handle(body)
            if env.results is not None:
                break
            steps += 1
            if steps > 200_000:
                status = "step_limit"
                break
            fn = env.loop
    finally:
        for agent in agents.values():
            agent.close()
    victories = env.results or [0, 0, 0, 0]
    winner = 0 if victories[0] > victories[1] else 1 if victories[1] > victories[0] else -1
    bot_stats: Counter = Counter()
    bot_errors: list[str] = []
    for agent in agents.values():
        bot_stats.update(agent.stats)
        bot_errors.extend(getattr(agent, "errors", [])[:2])
    return {"opponent": task["opponent"], "seed": seed, "our_team": our_team, "status": status,
            "winner_team": winner, "rounds": rounds, "seconds": time.perf_counter() - started,
            "mirror": dict(ours.stats), "mirror_examples": ours.examples,
            "bot": dict(bot_stats), "bot_errors": bot_errors[:4]}


# -- summary ------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def summarize(records: list[dict]) -> dict:
    out: dict[str, Any] = {}
    for opponent in sorted({r["opponent"] for r in records}):
        rs = [r for r in records if r["opponent"] == opponent]
        decided = [r for r in rs if r["winner_team"] in (0, 1)]
        wins = sum(r["winner_team"] == r["our_team"] for r in decided)
        rounds = [(rd, r["our_team"]) for r in rs for rd in r["rounds"]]
        first = sum(rd["first_team"] == team for rd, team in rounds)
        net = [rd["gain"] if rd["first_team"] == team else -rd["gain"] for rd, team in rounds]
        mirror, bot = Counter(), Counter()
        for r in rs:
            mirror.update(r["mirror"])
            bot.update(r["bot"])
        n_net = len(net)
        mean = sum(net) / n_net if n_net else math.nan
        sd = math.sqrt(sum((x - mean) ** 2 for x in net) / (n_net - 1)) if n_net > 1 else math.nan
        out[opponent] = {
            "matches": len(rs), "decided": len(decided), "match_wins": wins,
            "match_win_rate": wins / len(decided) if decided else math.nan,
            "match_win_ci": wilson(wins, len(decided)),
            "rounds": len(rounds), "round_first_rate": first / len(rounds) if rounds else math.nan,
            "round_first_ci": wilson(first, len(rounds)),
            "net_levels_per_round": mean,
            # Rounds within a match are dependent; this naive interval is a lower bound on width.
            "net_levels_se_naive": sd / math.sqrt(n_net) if n_net > 1 else math.nan,
            "rounds_mirror_failed": mirror.get("rounds_mirror_failed", 0),
            "our_decisions": mirror.get("our_decisions", 0),
            "our_fallback_decisions": mirror.get("our_fallback_decisions", 0),
            "bot_decisions": bot.get("decisions", 0),
            "bot_failures": {k: v for k, v in bot.items() if k != "decisions"},
            "mean_match_seconds": sum(r["seconds"] for r in rs) / len(rs),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="eval.policies.load_policy spec (random, greedy, a checkpoint)")
    ap.add_argument("--bots", default="all", help="comma list of bot names, 'all' or 'random'")
    ap.add_argument("--seeds", type=int, default=20, help="seeds per opponent; each is played with both seatings")
    ap.add_argument("--seed0", type=int, default=20261003)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    bots = sorted(BOT_DIRS) if args.bots == "all" else args.bots.split(",")
    args.out.mkdir(parents=True, exist_ok=True)
    raw = args.out / "matches.jsonl"
    done = set()
    if raw.exists():
        for line in raw.read_text().splitlines():
            r = json.loads(line)
            done.add((r["opponent"], r["seed"], r["our_team"]))
    tasks = [{"opponent": b, "policy": args.policy, "seed": args.seed0 + i, "our_team": t}
             for b in bots for i in range(args.seeds) for t in (1, 0)
             if (b, args.seed0 + i, t) not in done]
    print(f"{len(tasks)} matches to play ({len(done)} already in {raw})", flush=True)
    started = time.perf_counter()
    with raw.open("a") as sink, get_context("spawn").Pool(args.workers, maxtasksperchild=20) as pool:
        for n, record in enumerate(pool.imap_unordered(play_match, tasks), 1):
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            if n % 10 == 0 or n == len(tasks):
                print(f"[{n}/{len(tasks)}] {time.perf_counter() - started:.0f}s "
                      f"last={record['opponent']} winner={record['winner_team']} "
                      f"ours={record['our_team']} rounds={len(record['rounds'])}", flush=True)
    records = [json.loads(line) for line in raw.read_text().splitlines()]
    summary = {"policy": args.policy, "seeds": args.seeds, "per_opponent": summarize(records)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["per_opponent"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
