"""Verification of the ablation wrappers and the teacher-forced probe.

1. FULL wrapper == export evaluator, decision for decision (duplicate legs and a full match).
2. NONE/ROUND really change the stream the model reads (prefix lengths, round content).
3. Probe variants == what the ablated policy chooses in the same state: the NONE (ROUND)
   game follows the FULL game until the first decision where the probe says the NONE
   (ROUND) argmax differs, and there it plays exactly the probe's NONE (ROUND) argmax.
"""
import json
import time

import gd
import torch

torch.set_num_threads(1)

from eval.arena import play_matches
from eval.duplicate import generate_deals, play_duplicate_teams
from eval.history_policy import needs_history
import ablate_lib as L


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


def untrace(policy, orig):
    policy.select = orig


def first_diff(a, b, n=4):
    return next((k for k, (x, y) in enumerate(zip(a, b)) if x[:n] != y[:n]),
                None if len(a) == len(b) else min(len(a), len(b)))


def main():
    out = {}
    base, b11 = L.load_base(), L.load_b11()
    # 1a duplicate legs
    deals = generate_deals(20, seed=777)
    same = 0
    total = 0
    for i, deal in enumerate(deals):
        ev, ab = [], []
        o1, o2 = traced(base, ev), traced(b11, ev)
        play_duplicate_teams(deal, (base, base), (b11, b11), seed=i)
        untrace(base, o1), untrace(b11, o2)
        full = L.AblatedPolicy(base, "full")
        o1, o2 = traced(full, ab), traced(b11, ab)
        play_duplicate_teams(deal, (full, full), (b11, b11), seed=i)
        untrace(full, o1), untrace(b11, o2)
        same += ev == ab
        total += len(ev)
    out["dup_full_equals_evaluator"] = {"deals": len(deals), "identical_deals": same,
                                        "decisions": total}
    print(out, flush=True)
    # 1b full match, same seed as the game-review verification (1239 decisions, 17 rounds)
    seed = 424242
    ev, ab = [], []
    o1, o2 = traced(base, ev), traced(b11, ev)
    res = play_matches(base, b11, [0], seed=seed)
    untrace(base, o1), untrace(b11, o2)
    full = L.AblatedPolicy(base, "full")
    o1, o2 = traced(full, ab), traced(b11, ab)
    m = L.play_match(full, b11, 0, seed)
    untrace(full, o1), untrace(b11, o2)
    out["match_full_equals_evaluator"] = {
        "seed": seed, "evaluator_decisions": len(ev), "wrapper_decisions": len(ab),
        "identical": ev == ab, "evaluator_rounds": res["records"][0]["rounds"],
        "wrapper_rounds": len(m["rounds"]), "evaluator_winner": res["records"][0]["winner"],
        "wrapper_winner": m["winner"], "max_prefix": max(t[4] for t in ev if t[4] is not None)}
    print(out["match_full_equals_evaluator"], flush=True)

    # 2 stream content under ROUND / NONE in one match each
    for mode in ("round", "none"):
        pol = L.AblatedPolicy(base, mode)
        bad = 0
        orig = pol.select

        def check(engine, state, actions, rng, pol=pol, orig=orig):
            nonlocal bad
            if pol.mode == "round":
                # after the wrapper's own reset the stream may only hold this round
                i = orig(engine, state, actions, rng)
                if any(r != int(state.round_index) for r in pol.stream.rounds):
                    bad += 1
                return i
            i = orig(engine, state, actions, rng)
            bad += pol.stream.prefix != 0
            return i
        pol.select = check
        mm = L.play_match(pol, b11, 0, seed)
        pre = [p for _, p in pol.prefix_log]
        out[f"stream_{mode}"] = {"decisions": len(pre), "violations": bad,
                                 "mean_prefix": sum(pre) / len(pre), "max_prefix": max(pre),
                                 "withheld_events": pol.withheld, "rounds": len(mm["rounds"])}
        print(mode, out[f"stream_{mode}"], flush=True)

    # 3 probe consistency on duplicate legs (NONE) and full matches (ROUND)
    def describe(a, level):
        return ""
    checks = {"none": [], "round": []}
    for i, deal in enumerate(generate_deals(40, seed=4242)):
        probe = L.ProbePolicy(base)
        tr_p = []
        o = traced(probe, tr_p)
        play_duplicate_teams(deal, (probe, probe), (b11, b11), seed=i)
        untrace(probe, o)
        none = L.AblatedPolicy(base, "none")
        tr_n = []
        o = traced(none, tr_n)
        play_duplicate_teams(deal, (none, none), (b11, b11), seed=i)
        untrace(none, o)
        recs = probe.records
        # map: probe records are in the same order as tr_p entries of the probe's own seats
        flip = next((k for k, r in enumerate(recs) if r["arg"]["none"] != r["arg"]["full"]), None)
        d = first_diff(tr_p, tr_n)
        checks["none"].append({"deal": i, "probe_first_flip": flip, "traj_first_diff": d,
                               "probe_decisions": len(recs)})
    for s in range(3):
        probe = L.ProbePolicy(base)
        probe.match_tag = f"V{s}"
        tr_p = []
        o = traced(probe, tr_p)
        L.play_match(probe, b11, 0, 555000 + s)
        untrace(probe, o)
        rp = L.AblatedPolicy(base, "round")
        tr_r = []
        o = traced(rp, tr_r)
        L.play_match(rp, b11, 0, 555000 + s)
        untrace(rp, o)
        recs = probe.records
        flip = next((k for k, r in enumerate(recs) if r["arg"]["round"] != r["arg"]["full"]), None)
        d = first_diff(tr_p, tr_r)
        # the trajectory entry at index d must be the probe's decision number `flip`
        ok = flip == d
        checks["round"].append({"seed": 555000 + s, "probe_first_flip": flip,
                                "traj_first_diff": d, "flip_at_same_decision": ok,
                                "probe_decisions": len(recs)})
        print(checks["round"][-1], flush=True)
    out["probe_consistency"] = checks
    json.dump(out, open("out/verify.json", "w"), indent=1)


if __name__ == "__main__":
    main()
