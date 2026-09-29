"""Record the review games (viewer matches) or statistics-only duplicate rounds.

  record.py matches <id,...> <out_prefix>     full matches from MATCHES below
  record.py dup <opponent:b11|u0844> <deal_seed> <first> <count> <out_prefix>
"""
import json
import sys
import time

import torch

from eval.duplicate import generate_deals
from eval.policies import load_policy
from eval.record_games import validate_round
import review_lib as R

torch.set_num_threads(2)
ROOT = "/Users/xiyaowang/Developer/Projects/GuanZero/"
CKPT = {
    "u2623": ROOT + ".work/longrun-batched-eval-2026-09-29/ckpt/u2623.pt",
    "u0844": ROOT + ".work/longrun-batched-eval-2026-09-29/ckpt/u0844.pt",
    "b11": ROOT + ".work/runpod-longrun-2026-09-25/payload/artifacts/b11-main.pt",
}
T = "Transformer u2623"
B = "B11 主线（MLP 基线）"
# id: (title, engine seed, team names, seat keys for seats 0..3)
MATCHES = {
    1: ("第1场 · Transformer u2623（南北）对 B11（东西）", 92901,
        ("Transformer u2623（1.72亿步）", B), ("u2623", "b11", "u2623", "b11")),
    2: ("第2场 · B11（南北）对 Transformer u2623（东西）", 92902,
        (B, "Transformer u2623（1.72亿步）"), ("b11", "u2623", "b11", "u2623")),
    3: ("第3场 · Transformer u2623（南北）对 B11（东西）", 92903,
        ("Transformer u2623（1.72亿步）", B), ("u2623", "b11", "u2623", "b11")),
    4: ("第4场 · 新 Transformer u2623（南北）对 旧 Transformer u0844（东西）", 92904,
        ("新 Transformer u2623（1.72亿步）", "旧 Transformer u0844（5530万步）"),
        ("u2623", "u0844", "u2623", "u0844")),
    5: ("第5场 · Transformer u2623 自对弈（四个座位都是 u2623）", 92905,
        ("Transformer u2623 · 南北", "Transformer u2623 · 东西"),
        ("u2623", "u2623", "u2623", "u2623")),
}
_cache = {}


def policy(key):
    if key not in _cache:
        p = load_policy(CKPT[key], "cpu", margin=0.0) if key == "b11" else load_policy(CKPT[key], "cpu")
        if key != "b11":
            R.instrument_history(p)
        _cache[key] = p
    return _cache[key]


def run_matches(ids, prefix):
    games, decisions = [], []
    for mid in ids:
        title, seed, names, keys = MATCHES[mid]
        seats = [policy(k) for k in keys]
        t = time.time()
        game = R.record_match(title, seed, names, seats, list(keys), decisions,
                              max_rounds=60, match_tag=f"M{mid}")
        for rnd in game["rounds"]:
            validate_round(rnd)
            del rnd["_raw_hands"]
        game["match_id"] = mid
        steps = sum(len(r["steps"]) for r in game["rounds"])
        print(f"M{mid} seed={seed} rounds={len(game['rounds'])} winner={game['winner_team']} "
              f"steps={steps} sec={time.time() - t:.0f}", flush=True)
        games.append(game)
    json.dump(games, open(prefix + ".games.json", "w"), ensure_ascii=False)
    with open(prefix + ".decisions.jsonl", "w") as f:
        for d in decisions:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def run_dup(opp, deal_seed, first, count, prefix):
    deals = generate_deals(first + count, seed=deal_seed)[first:]
    cand, other = policy("u2623"), policy(opp)
    decisions, rounds = [], []
    t = time.time()
    for i, deal in enumerate(deals, start=first):
        for leg, keys in enumerate((("u2623", opp, "u2623", opp), (opp, "u2623", opp, "u2623"))):
            seats = [cand if k == "u2623" else other for k in keys]
            rnd = R.play_deal_recorded(deal, seats, list(keys), decisions, seed=deal_seed + i,
                                       match_tag=f"D{i}L{leg}")
            validate_round(rnd)
            rounds.append({"deal": i, "leg": leg, "keys": keys, "level": deal.level,
                           "finish_order": rnd["finish_order"], "winning_team": rnd["winning_team"],
                           "gain": rnd["gain"], "seat_return": rnd["seat_return"],
                           "steps": len(rnd["steps"])})
        print(f"deal {i} done, {time.time() - t:.0f}s", flush=True)
    if True:
        json.dump(rounds, open(prefix + ".rounds.json", "w"))
        with open(prefix + ".decisions.jsonl", "w") as f:
            for d in decisions:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "matches":
        run_matches([int(x) for x in sys.argv[2].split(",")], sys.argv[3])
    else:
        run_dup(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
