"""Check that the review recorder feeds history exactly like the evaluator.

1. Negative control: the export's eval/record_games.py cannot drive a history policy.
2. Duplicate legs: eval.duplicate.play_duplicate_teams vs review_lib.play_deal_recorded.
3. Full match: eval.arena.play_matches (used by eval.history_frozen) vs review_lib.record_match.
Comparison: every non-forced play-phase decision (round, seat, chosen cards) and the
history policy's stream prefix length at that decision.
"""
import json
import sys
import time

import gd
import torch

from eval import record_games
from eval.arena import play_matches
from eval.duplicate import generate_deals, play_duplicate_teams
from eval.history_policy import needs_history
from eval.policies import load_policy
import review_lib as R

torch.set_num_threads(2)
ROOT = "/Users/xiyaowang/Developer/Projects/GuanZero/"
U2623 = ROOT + ".work/longrun-batched-eval-2026-09-29/ckpt/u2623.pt"
B11 = ROOT + ".work/runpod-longrun-2026-09-25/payload/artifacts/b11-main.pt"


def traced(policy, trace):
    orig = policy.select

    def select(engine, state, actions, rng):
        prefix = policy.events_seen if needs_history(policy) else None
        i = orig(engine, state, actions, rng)
        if int(state.phase) == int(gd.Phase.Play):
            a = actions[i]
            trace.append((int(state.round_index), int(state.to_move), bool(a.is_pass),
                          tuple(sorted(a.cards)), prefix))
        return i
    policy.select = select
    return orig


def compare(name, a, b):
    same = a == b
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    print(f"{name}: evaluator {len(a)} decisions, recorder {len(b)}, identical={same}"
          + ("" if same else f", first difference at {first}"), flush=True)
    return {"check": name, "evaluator_decisions": len(a), "recorder_decisions": len(b),
            "identical": same, "first_difference": first,
            "history_decisions": sum(1 for t in a if t[4] is not None),
            "max_prefix": max((t[4] for t in a if t[4] is not None), default=0)}


def main():
    out = {}
    cand = load_policy(U2623, "cpu")
    R.instrument_history(cand)
    b11 = load_policy(B11, "cpu", margin=0.0)
    print("loaded", cand.name, b11.name, flush=True)

    # 1. negative control
    try:
        record_games.record_match("x", 5, ("a", "b"), (cand, b11), max_rounds=1)
        out["original_recorder"] = "ran (unexpected)"
    except RuntimeError as e:
        out["original_recorder"] = f"RuntimeError: {e}"
    print("original recorder:", out["original_recorder"], flush=True)

    results = []
    # 2. duplicate legs on 3 deals
    deals = generate_deals(3, seed=777)
    for d_i, deal in enumerate(deals):
        ev, rec = [], []
        o1, o2 = traced(cand, ev), traced(b11, ev)
        t = time.time()
        play_duplicate_teams(deal, (cand, cand), (b11, b11), seed=d_i)
        cand.select, b11.select = o1, o2
        # recorder: leg 1 team0 = cand; leg 2 team1 = cand
        for seats in ((cand, b11, cand, b11), (b11, cand, b11, cand)):
            labels = ["x"] * 4
            R.play_deal_recorded(deal, seats, labels, [], seed=d_i, trace=rec)
        results.append(compare(f"duplicate deal {d_i} (2 legs)", ev, rec))
        print("  seconds", round(time.time() - t, 1), flush=True)

    # 3. one full match, evaluator seat assignment match index 0 -> candidate team 0
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 424242
    ev, rec = [], []
    o1, o2 = traced(cand, ev), traced(b11, ev)
    t = time.time()
    res = play_matches(cand, b11, [0], seed=seed)
    cand.select, b11.select = o1, o2
    print("evaluator match", res["records"], round(time.time() - t, 1), flush=True)
    game = R.record_match("v", seed, ("a", "b"), (cand, b11, cand, b11), ["x"] * 4, [],
                          max_rounds=1000, trace=rec)
    for rnd in game["rounds"]:
        record_games.validate_round(rnd)
    r = compare(f"full match seed {seed}", ev, rec)
    r["evaluator_rounds"] = res["records"][0]["rounds"]
    r["recorder_rounds"] = len(game["rounds"])
    r["evaluator_winner"] = res["records"][0]["winner"]
    r["recorder_winner"] = game["winner_team"]
    results.append(r)
    out["checks"] = results
    json.dump(out, open("verify.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
