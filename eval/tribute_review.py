"""Extract every tribute / back-tribute event from the Oct 6 DanLM duplicate records.

  python3 eval/tribute_review.py <records_dir> <out_dir>   (stdlib only)

<records_dir> holds the per-checkpoint *.jsonl files written by the Oct 6 game
recorder (.work/danlm-games-2026-10-06/out, see
docs/reports/danlm-game-review-2026-10-06.md).

Writes tribute_events.jsonl (one row per exchange, all checkpoints),
paired_choices.jsonl (same hand, our heuristic vs DanLM, deduplicated) and summary.json.
Card ids: 0..51 = rank*4+suit (rank 0=2 .. 12=A; suit S,H,C,D), 52 small joker, 53 big joker.
Structure checks ignore wild cards except where noted; they are diagnostics, not rules.
"""
import collections, glob, json, os, random, sys

SRC, HERE = sys.argv[1], sys.argv[2]
os.makedirs(HERE, exist_ok=True)
RANKS = "23456789TJQKA"
SUITS = "SHCD"
WINDOWS = [[12, 0, 1, 2, 3]] + [[s + i for i in range(5)] for s in range(9)]  # A-low .. T-A


def rank(c): return c // 4 if c < 52 else c - 39
def suit(c): return c % 4 if c < 52 else -1
def name(c): return "BJ" if c == 52 else "RJ" if c == 53 else SUITS[c % 4] + RANKS[c // 4]
def power(r, level):
    if r >= 13: return r
    if r == level: return 12
    return r if r < level else r - 1


def sf_missing(hand, s, level):
    """Fewest natural cards missing from a straight flush of suit s (wilds not counted)."""
    have = {rank(c) for c in hand if suit(c) == s and not (rank(c) == level and s == 1)}
    return min(sum(r not in have for r in w) for w in WINDOWS)


def structure(hand, card, level):
    """What adding `card` to `hand` does to that hand (receiver view)."""
    r, s = rank(card), suit(card)
    nat = sum(1 for c in hand if rank(c) == r)
    wild = sum(1 for c in hand if c == level * 4 + 1)
    out = {"copies_before": nat, "wilds": wild,
           "makes_pair": nat == 1, "makes_trips": nat == 2, "makes_bomb": nat >= 3}
    if s >= 0 and not (r == level and s == 1):
        before = sf_missing(hand, s, level)
        after = sf_missing(hand + [card], s, level)
        out["sf_missing_before"], out["sf_missing_after"] = before, after
        out["completes_natural_sf"] = before > 0 and after == 0
        out["sf_within_wilds_after"] = after <= wild < before
    return out


def giver_cost(hand, card, level):
    """What removing `card` from `hand` costs the giver."""
    r, s = rank(card), suit(card)
    nat = sum(1 for c in hand if rank(c) == r)
    out = {"copies_held": nat, "breaks_pair": nat == 2, "breaks_trips": nat == 3, "breaks_bomb": nat >= 4}
    if s >= 0:
        rest = list(hand); rest.remove(card)
        out["breaks_natural_sf"] = sf_missing(hand, s, level) == 0 and sf_missing(rest, s, level) > 0
    return out


events, outcomes = [], []
for path in sorted(glob.glob(os.path.join(SRC, "*.jsonl"))):
    for line in open(path):
        g = json.loads(line)
        if g.get("status") != "ok":
            continue
        ours = [s for s in range(4) if g["seats"][s] == "ours"]
        ex = [e for e in g["log"] if e["phase"] != "play"]
        paid = {}  # tribute payer -> card
        for e in ex:
            if e["phase"] == "tribute":
                paid[e["seat"]] = e["cards"][0]
        recv_of = {}  # back-tribute giver -> tribute payer it pays back
        trib = [e for e in ex if e["phase"] == "tribute"]
        # receiver of each tribute: match by tribute card in giver's hand at back time
        for e in ex:
            if e["phase"] != "back":
                continue
            for p, c in paid.items():
                if c in e["hand"] and p not in recv_of.values() and (p - e["seat"]) % 2:
                    recv_of[e["seat"]] = p
                    break
            else:
                for p, c in paid.items():
                    if c in e["hand"] and p not in recv_of.values():
                        recv_of[e["seat"]] = p
                        break
        for e in ex:
            seat, card = e["seat"], e["cards"][0]
            if e["phase"] == "tribute":
                target = None  # filled below when known
                for b, p in recv_of.items():
                    if p == seat:
                        target = b
            else:
                target = recv_of.get(seat)
            row = {"ckpt": g["key"], "deal": g["deal"], "leg": g["leg"], "level": g["level"],
                   "level_name": RANKS[g["level"]], "phase": e["phase"], "seat": seat,
                   "side": e["side"], "to_seat": target,
                   "to_partner": None if target is None else (target - seat) % 4 == 2,
                   "double_tribute": len(trib) == 2, "hand": e["hand"],
                   "card": card, "card_name": name(card),
                   "card_power": power(rank(card), g["level"]),
                   "giver_cost": giver_cost(e["hand"], card, g["level"]),
                   "round_reward_giver_team": g["rewards"][seat], "finish_order": g["finish_order"]}
            if e["phase"] == "back" and target is not None:
                recv_hand = list(g["start_hands"][target]); recv_hand.remove(paid[target])
                row["receiver_effect"] = structure(recv_hand, card, g["level"])
            events.append(row)
        outcomes.append({"ckpt": g["key"], "deal": g["deal"], "leg": g["leg"],
                         "tribute_deal": bool(g["tribute_deal"]), "anti": bool(g["tribute_deal"]) and not trib,
                         "ours_pay": any(e["side"] == "ours" for e in trib),
                         "ours_reward": g["rewards"][ours[0]]})

with open(os.path.join(HERE, "tribute_events.jsonl"), "w") as f:
    for r in events:
        f.write(json.dumps(r) + "\n")

# Paired choices: same (deal, phase, seat) in leg a and b -> one ours, one DanLM, same hand.
by = collections.defaultdict(dict)
for r in events:
    k = (r["deal"], r["phase"], r["seat"])
    by[k].setdefault(r["side"], []).append(r)
pairs = []
for (deal, phase, seat), d in sorted(by.items()):
    if "ours" not in d or "danlm" not in d:
        continue
    o = d["ours"][0]
    dl = collections.Counter(x["card_name"] for x in d["danlm"])
    pairs.append({"deal": deal, "phase": phase, "seat": seat, "level_name": o["level_name"],
                  "to_partner": o["to_partner"], "double_tribute": o["double_tribute"],
                  "hand": [name(c) for c in sorted(o["hand"])],
                  "heuristic": o["card_name"], "danlm": dict(dl),
                  "same_hand": all(sorted(x["hand"]) == sorted(o["hand"]) for x in d["danlm"]),
                  "agree": dl.most_common(1)[0][0] == o["card_name"],
                  "heuristic_receiver_effect": o.get("receiver_effect"),
                  "danlm_receiver_effect": d["danlm"][0].get("receiver_effect"),
                  "heuristic_giver_cost": o["giver_cost"], "danlm_giver_cost": d["danlm"][0]["giver_cost"]})
with open(os.path.join(HERE, "paired_choices.jsonl"), "w") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")


def rate(rows, f):
    rows = [r for r in rows if r is not None]
    return {"n": len(rows), "rate": round(sum(1 for r in rows if f(r)) / len(rows), 3) if rows else None}


summary = {"source": "danlm-games-2026-10-06 (seed 20260929 DanLM set, deals 0-299, 4 checkpoints, both legs)"}
for phase in ("tribute", "back"):
    for rel in (True, False):
        ps = [p for p in pairs if p["phase"] == phase and p["to_partner"] is rel and p["same_hand"]]
        key = f"{phase}_to_{'partner' if rel else 'opponent'}"
        s = {"pairs": len(ps), "agree": rate(ps, lambda p: p["agree"])}
        if phase == "back":
            for who in ("heuristic", "danlm"):
                eff = [p[f"{who}_receiver_effect"] for p in ps]
                cost = [p[f"{who}_giver_cost"] for p in ps]
                s[who] = {
                    "receiver_gets_pair_or_better": rate(eff, lambda e: e["copies_before"] >= 1),
                    "receiver_gets_new_bomb": rate(eff, lambda e: e["makes_bomb"]),
                    "receiver_natural_sf_completed": rate(eff, lambda e: e.get("completes_natural_sf", False)),
                    "receiver_sf_within_wilds": rate(eff, lambda e: e.get("sf_within_wilds_after", False)),
                    "giver_breaks_pair_or_more": rate(cost, lambda c: c["copies_held"] >= 2),
                    "giver_breaks_bomb": rate(cost, lambda c: c["breaks_bomb"]),
                    "giver_breaks_natural_sf": rate(cost, lambda c: c.get("breaks_natural_sf", False)),
                }
        summary[key] = s

rng = random.Random(0)
def boot(vals, n=2000):
    if not vals: return None
    m = sum(vals) / len(vals)
    bs = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return {"n": len(vals), "mean": round(m, 3), "ci95": [round(bs[int(.025 * n)], 3), round(bs[int(.975 * n)], 3)]}

out = {}
for ck in sorted({o["ckpt"] for o in outcomes}):
    rows = [o for o in outcomes if o["ckpt"] == ck]
    deal = collections.defaultdict(list)
    for o in rows:
        deal[(o["deal"], o["tribute_deal"])].append(o["ours_reward"])
    out[ck] = {
        "all_rounds": boot([o["ours_reward"] for o in rows]),
        "no_tribute_deals_paired": boot([sum(v) / 2 for (d, t), v in deal.items() if not t and len(v) == 2]),
        "tribute_deals_paired": boot([sum(v) / 2 for (d, t), v in deal.items() if t and len(v) == 2]),
        "rounds_we_pay_tribute": boot([o["ours_reward"] for o in rows if o["tribute_deal"] and not o["anti"] and o["ours_pay"]]),
        "rounds_we_receive_tribute": boot([o["ours_reward"] for o in rows if o["tribute_deal"] and not o["anti"] and not o["ours_pay"]]),
        "anti_tribute_rounds": boot([o["ours_reward"] for o in rows if o["anti"]]),
    }
summary["levels_per_round_vs_danlm"] = out
summary["notes"] = [
    "In every tribute deal the previous finishing order is set by the eval generator (tribute fraction 0.5), not by a played previous round.",
    "Our tribute/back-tribute choices come from the fixed C++ tribute_bot heuristic for every checkpoint; only play differs between checkpoints.",
    "Paired rows: identical hand and incoming tribute card in the two duplicate legs; one leg ours (heuristic), one DanLM.",
    "Receiver/giver structure flags use the full hands from the record, which the giver cannot see; they describe consequences, not information available at decision time.",
    "Straight-flush flags count natural cards only; sf_within_wilds_after allows the receiver's own heart level cards.",
]
json.dump(summary, open(os.path.join(HERE, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
