"""Teacher-forced probe along FULL trajectories (FULL Transformer team vs B11).

  run_probe.py match <first_pair> <npairs> <out_prefix>   same seeds as the strength runs
  run_probe.py dup   <first> <count> <out_prefix>         same deals as the strength runs

Writes <out_prefix>.records.jsonl.gz (one record per non-forced play decision of the
Transformer team) and <out_prefix>.meta.json (per match: result, round token counts,
public event log in text).
"""
import glob
import gzip
import json
import sys
import time

import gd
import numpy as np
import torch

torch.set_num_threads(1)

from eval.duplicate import generate_deals, play_round
from eval.history_policy import history_listeners
import ablate_lib as L
from review_lib import action_desc, cards_str
from run_strength import DEAL_SEED, MATCH_SEED0


def describe(a, level):
    try:
        return action_desc(a, level)
    except Exception:
        return f"{a.type} {cards_str(list(a.cards), level)}"


def load_donors():
    donors = []
    for f in sorted(glob.glob("out/donor_full_b11_*.donors.npz")):
        z = np.load(f)
        start = 0
        for tag, team, n in zip(z["tags"], z["teams"], z["lengths"]):
            donors.append({"tag": str(tag), "team": int(team),
                           "tokens": z["tokens"][start:start + n],
                           "rounds": z["rounds"][start:start + n],
                           "phases": z["phases"][start:start + n]})
            start += n
    return donors


def main():
    kind, first, count, prefix = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    base, b11 = L.load_base(), L.load_b11()
    donors = load_donors() if kind == "match" else []
    probe = L.ProbePolicy(base, donors=donors, describe=describe)
    metas = []
    t0 = time.time()
    engine = gd.Engine()
    if kind == "match":
        for p in range(first, first + count):
            seed = MATCH_SEED0 + p
            for team in (0, 1):
                probe.match_tag, probe.team = f"P{p}T{team}", team
                policies = (probe, b11) if team == 0 else (b11, probe)
                state = gd.MatchState()
                engine.new_match(state, seed)
                for listener in history_listeners(policies):
                    listener.start_match()
                rounds = []
                while state.winner < 0:
                    probe.cur_level = int(state.level)
                    s = play_round(engine, state, policies, seed + team * 1000003 + len(rounds))
                    rounds.append({"net": s.net_gain(team), "order": list(s.order)})
                    if state.winner < 0:
                        engine.begin_round(state)
                tok, rnd, ph = probe.stream.arrays()
                metas.append({"tag": probe.match_tag, "seed": seed, "team": team,
                              "won": int(state.winner) == team, "rounds": rounds,
                              "round_tokens": np.bincount(rnd).tolist(),
                              "events": probe.event_desc})
            print(f"pair {p} {time.time() - t0:.0f}s", flush=True)
    else:
        deals = generate_deals(1000, seed=DEAL_SEED)[first:first + count]
        for i, deal in enumerate(deals, start=first):
            for leg, seats in enumerate(((probe, b11, probe, b11), (b11, probe, b11, probe))):
                probe.match_tag, probe.team = f"D{i}L{leg}", leg
                state = gd.MatchState()
                engine.set_deal(state, deal)
                probe.start_match()
                probe.cur_level = int(state.level)
                from eval.duplicate import play_round_seats
                s = play_round_seats(engine, state, seats, i)
                tok, rnd, ph = probe.stream.arrays()
                metas.append({"tag": probe.match_tag, "rounds": [{"net": s.net_gain(leg)}],
                              "round_tokens": [len(tok)], "events": probe.event_desc})
    with gzip.open(prefix + ".records.jsonl.gz", "wt") as f:
        for r in probe.records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(metas, open(prefix + ".meta.json", "w"), ensure_ascii=False)
    print("done", len(probe.records), time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
