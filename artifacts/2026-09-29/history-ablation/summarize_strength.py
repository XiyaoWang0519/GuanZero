"""Strength tables from out/dup_*.json and out/match_*.json (bootstrap seed 0, 4000 samples)."""
import glob
import json

import numpy as np

B = 4000


def boot(fn, n, seed=0):
    rng = np.random.default_rng(seed)
    vals = np.array([fn(rng.integers(n, size=n)) for _ in range(B)])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def load(pattern):
    rows = []
    files = set(glob.glob(pattern)) - set(glob.glob(pattern.replace(".json", ".donors.json")))
    for f in sorted(files):
        rows += json.load(open(f))["rows"]
    # unfinished chunks: per-pair lines written as each pair completed
    for f in sorted(glob.glob(pattern.replace(".json", ".pairs.jsonl"))):
        if f.replace(".pairs.jsonl", ".json") not in files:
            rows += [json.loads(l) for l in open(f)]
    return rows


out = {}
print("## single rounds (1000 duplicate deals, generate_deals(1000, seed=20260929))")
dup = {}
for cond in ("full_b11", "none_b11", "full_none"):
    rows = sorted(load(f"out/dup_{cond}_*.json"), key=lambda r: r["deal"])
    s = np.array([r["score"] for r in rows])
    legs = [(leg, t) for r in rows for t, leg in enumerate(r["legs"])]
    first = np.mean([l["winning_team"] == t for l, t in legs])
    dbl = np.mean([l["order"][0] % 2 == t and l["order"][1] % 2 == t for l, t in legs])
    odbl = np.mean([l["order"][0] % 2 != t and l["order"][1] % 2 != t for l, t in legs])
    lo, hi = boot(lambda i: s[i].mean(), len(s))
    dup[cond] = s
    out[f"dup_{cond}"] = dict(deals=len(s), rounds=len(legs), mean=float(s.mean()), ci=[lo, hi],
                              first_place=float(first), double=float(dbl), opp_double=float(odbl))
    print(f"{cond:10s} deals={len(s)} net/round={s.mean():+.3f} [{lo:+.3f},{hi:+.3f}] "
          f"first={first:.3f} double={dbl:.3f} opp_double={odbl:.3f}")
d = dup["full_b11"] - dup["none_b11"]
lo, hi = boot(lambda i: d[i].mean(), len(d))
out["dup_full_minus_none_vs_b11"] = dict(mean=float(d.mean()), ci=[lo, hi])
print(f"paired (FULL vs B11) - (NONE vs B11) per deal: {d.mean():+.3f} [{lo:+.3f},{hi:+.3f}]")

print("\n## full matches (pair p: seed 29092600+p, A on team 0 and team 1)")
per = {}
for cond in ("full_b11", "round_b11", "none_b11", "full_round"):
    rows = load(f"out/match_{cond}_*.json")
    if not rows:
        continue
    pairs = {}
    for m in rows:
        pr = pairs.setdefault(m["pair"], {"wins": 0, "net": 0, "rounds": 0, "first": 0, "double": 0,
                                          "opp_double": 0, "n": 0})
        pr["wins"] += m["a_won"]
        pr["n"] += 1
        pr["rounds"] += len(m["rounds"])
        for r in m["rounds"]:
            pr["net"] += r["net"]
            pr["first"] += r["first"]
            pr["double"] += r["double"]
            pr["opp_double"] += r["opp_double"]
    keys = sorted(k for k, v in pairs.items() if v["n"] == 2)
    arr = {f: np.array([pairs[k][f] for k in keys], dtype=float) for f in pairs[keys[0]]}
    per[cond] = (keys, arr)
    n = len(keys)
    wr = arr["wins"].sum() / (2 * n)
    npr = arr["net"].sum() / arr["rounds"].sum()
    wlo, whi = boot(lambda i: arr["wins"][i].sum() / (2 * n), n)
    nlo, nhi = boot(lambda i: arr["net"][i].sum() / arr["rounds"][i].sum(), n)
    fr = arr["first"].sum() / arr["rounds"].sum()
    db = arr["double"].sum() / arr["rounds"].sum()
    out[f"match_{cond}"] = dict(pairs=n, matches=2 * n, rounds=int(arr["rounds"].sum()),
                                win_rate=float(wr), win_ci=[wlo, whi], net_per_round=float(npr),
                                net_ci=[nlo, nhi], first_place=float(fr), double=float(db),
                                mean_rounds=float(arr["rounds"].sum() / (2 * n)))
    print(f"{cond:10s} pairs={n} matches={2*n} rounds={int(arr['rounds'].sum())} "
          f"win={wr:.3f} [{wlo:.3f},{whi:.3f}] net/round={npr:+.3f} [{nlo:+.3f},{nhi:+.3f}] "
          f"first={fr:.3f} double={db:.3f}")
for a, b in (("full_b11", "round_b11"), ("full_b11", "none_b11"), ("round_b11", "none_b11")):
    if a not in per or b not in per:
        continue
    ka, aa = per[a]
    kb, bb = per[b]
    common = sorted(set(ka) & set(kb))
    ia = [ka.index(k) for k in common]
    ib = [kb.index(k) for k in common]
    A = {f: v[ia] for f, v in aa.items()}
    Bb = {f: v[ib] for f, v in bb.items()}
    n = len(common)

    def dnet(i):
        return A["net"][i].sum() / A["rounds"][i].sum() - Bb["net"][i].sum() / Bb["rounds"][i].sum()

    def dwin(i):
        return (A["wins"][i].sum() - Bb["wins"][i].sum()) / (2 * n)
    allidx = np.arange(n)
    lo, hi = boot(dnet, n)
    wlo, whi = boot(dwin, n)
    out[f"paired_{a}_minus_{b}"] = dict(pairs=n, net=float(dnet(allidx)), net_ci=[lo, hi],
                                        win=float(dwin(allidx)), win_ci=[wlo, whi])
    print(f"paired same seeds {a} - {b}: net/round {dnet(allidx):+.3f} [{lo:+.3f},{hi:+.3f}], "
          f"win rate {dwin(allidx):+.3f} [{wlo:+.3f},{whi:+.3f}]")
json.dump(out, open("out/strength_summary.json", "w"), indent=1)
