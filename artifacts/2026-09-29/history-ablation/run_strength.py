"""Strength runs.

  run_strength.py dup   <full_b11|none_b11|full_none> <first> <count> <out.json>
  run_strength.py match <full_b11|round_b11|none_b11|full_round> <first_pair> <npairs> <out.json>

Single rounds: deals = eval.duplicate.generate_deals(1000, seed=DEAL_SEED), each deal
played with eval.duplicate.play_duplicate_teams (A at seats 0/2 in leg 1, at 1/3 in
leg 2), policy seed = deal index (greedy play, the seed is unused).
Matches: pair p uses engine seed MATCH_SEED0 + p twice, A on team 0 then team 1.
A is the first-named condition, B the second.
"""
import json
import sys
import time

import numpy as np
import torch

torch.set_num_threads(1)

from eval.duplicate import generate_deals, play_duplicate_teams
import ablate_lib as L

DEAL_SEED = 20260929
MATCH_SEED0 = 29092600


def make(cond, base, b11):
    a_name, b_name = cond.split("_")
    def one(name):
        return b11 if name == "b11" else L.AblatedPolicy(base, name)
    return one(a_name), one(b_name)


def stats(p):
    if not isinstance(p, L.AblatedPolicy):
        return None
    pre = [x[1] for x in p.prefix_log]
    return {"mode": p.mode, "decisions": len(pre), "withheld_events": p.withheld,
            "mean_prefix": float(np.mean(pre)) if pre else 0.0, "max_prefix": max(pre, default=0)}


def main():
    kind, cond, first, count, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    base, b11 = L.load_base(), L.load_b11()
    A, B = make(cond, base, b11)
    t0 = time.time()
    res = {"kind": kind, "cond": cond, "first": first, "count": count}
    if kind == "dup":
        deals = generate_deals(1000, seed=DEAL_SEED)[first:first + count]
        rows = []
        for i, deal in enumerate(deals, start=first):
            s = play_duplicate_teams(deal, (A, A), (B, B), seed=i)
            rows.append({"deal": i, "legs": [
                {"order": list(r.order), "winning_team": r.winning_team, "gain": r.gain,
                 "seat_return": list(r.seat_return), "decisions": r.decisions}
                for r in (s.first, s.swapped)], "score": s.levels_per_round})
        res["rows"] = rows
    else:
        rows, donors = [], []
        for p in range(first, first + count):
            seed = MATCH_SEED0 + p
            for a_team in (0, 1):
                m = L.play_match(A, B, a_team, seed)
                m["pair"] = p
                rows.append(m)
                if cond == "full_b11":
                    tok, rnd, ph = A.stream.arrays()
                    donors.append((f"P{p}T{a_team}", a_team, tok, rnd, ph))
            with open(out.replace(".json", ".pairs.jsonl"), "a") as f:
                for m in rows[-2:]:
                    f.write(json.dumps(m) + "\n")
            print(f"pair {p} done {time.time() - t0:.0f}s", flush=True)
        res["rows"] = rows
        if donors:
            np.savez_compressed(out.replace(".json", ".donors.npz"),
                                tags=np.array([d[0] for d in donors]),
                                teams=np.array([d[1] for d in donors]),
                                lengths=np.array([len(d[2]) for d in donors]),
                                tokens=np.concatenate([d[2] for d in donors]),
                                rounds=np.concatenate([d[3] for d in donors]),
                                phases=np.concatenate([d[4] for d in donors]))
    res["policy_stats"] = {"A": stats(A), "B": stats(B)}
    res["seconds"] = time.time() - t0
    json.dump(res, open(out, "w"))
    print("done", cond, res["seconds"], flush=True)


if __name__ == "__main__":
    main()
