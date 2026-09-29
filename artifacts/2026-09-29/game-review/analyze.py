"""Play-style statistics from decisions.jsonl + rounds.json (duplicate sample).

Usage: analyze.py <prefix> <opponent label> [--matches <prefix>]
Clusters for the bootstrap: deal (both legs of one deal share the hands).
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

import gd

BOMBS = ("Bomb", "StraightFlush", "JokerBomb")
TYPE_ZH = {"Single": "单张", "Pair": "对子", "Triple": "三张", "FullHouse": "三带二",
           "Straight": "顺子", "Tube": "三连对", "Plate": "钢板", "Bomb": "炸弹",
           "StraightFlush": "同花顺", "JokerBomb": "天王炸"}


def rank_of(c):
    return 13 if c == 52 else 14 if c == 53 else c // 4


def is_wild(c, level):
    return c < 52 and c // 4 == level and c % 4 == 1


def power(c, level):
    return gd.power(rank_of(c), level)


def natural_counts(hand, level):
    """Rank counts over non-wild cards; 13/14 are jokers."""
    cnt = Counter(rank_of(c) for c in hand if not is_wild(c, level))
    return cnt


def seq_windows(width):
    # sequence positions: A-low = 12 mapped to -1 conceptually; ranks 0..12 = 2..A
    wins = []
    for start in range(-1, 13 - width + 1):
        wins.append([12 if r == -1 else r for r in range(start, start + width)])
    return wins


STRAIGHT_W = seq_windows(5)   # A2345 .. TJQKA
TUBE_W = seq_windows(3)       # AA2233 .. QQKKAA
PLATE_W = seq_windows(2)      # AAA222 .. KKKAAA


def combos_with(cnt, r):
    """Which natural combination categories containing rank r the counts allow."""
    out = set()
    if r <= 12 and cnt[r] >= 4:
        out.add("bomb")
    if r >= 13 and cnt[13] == 2 and cnt[14] == 2:
        out.add("jokerbomb")
    if r <= 12:
        if any(r in w and all(cnt[x] >= 1 for x in w) for w in STRAIGHT_W):
            out.add("straight")
        if any(r in w and all(cnt[x] >= 2 for x in w) for w in TUBE_W):
            out.add("tube")
        if any(r in w and all(cnt[x] >= 3 for x in w) for w in PLATE_W):
            out.add("plate")
    return out


def broken(d):
    """Categories destroyed by a single/pair play (natural cards only)."""
    level = d["level"]
    played = [c for c in d["cards"] if not is_wild(c, level)]
    if not played:
        return set()
    r = rank_of(played[0])
    before = natural_counts(d["hand"], level)
    after = before.copy()
    for c in played:
        after[rank_of(c)] -= 1
    b, a = combos_with(before, r), combos_with(after, r)
    return b - a, b


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    s = 1 + z * z / n
    c = (p + z * z / (2 * n)) / s
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / s
    return c - h, c + h


def cluster_boot_diff(groups, num_a, den_a, num_b, den_b, reps=2000, seed=0):
    """Bootstrap CI for rate_a - rate_b resampling clusters. Inputs: per-cluster arrays."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    na, da, nb, db = map(np.asarray, (num_a, den_a, num_b, den_b))
    diffs = []
    for _ in range(reps):
        idx = rng.integers(n, size=n)
        A, B = da[idx].sum(), db[idx].sum()
        if A == 0 or B == 0:
            continue
        diffs.append(na[idx].sum() / A - nb[idx].sum() / B)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


class Rate:
    """Numerator/denominator per (label, cluster)."""

    def __init__(self):
        self.num = defaultdict(Counter)
        self.den = defaultdict(Counter)

    def add(self, label, cluster, hit, weight=1):
        self.den[label][cluster] += weight
        self.num[label][cluster] += weight if hit else 0

    def summary(self, label):
        k, n = sum(self.num[label].values()), sum(self.den[label].values())
        return k, n

    def fmt(self, label):
        k, n = self.summary(label)
        if n == 0:
            return "n=0"
        lo, hi = wilson(k, n)
        return f"{k}/{n} = {k / n:.1%} (Wilson 95% {lo:.1%}–{hi:.1%})"

    def diff(self, a, b, clusters):
        return cluster_boot_diff(clusters, [self.num[a][c] for c in clusters], [self.den[a][c] for c in clusters],
                                 [self.num[b][c] for c in clusters], [self.den[b][c] for c in clusters])


def cluster_of(tag):
    return tag.split("L")[0] if tag.startswith("D") else tag


def main():
    prefix, opp = sys.argv[1], sys.argv[2]
    labels = ["u2623", opp]
    decs = [json.loads(l) for l in open(prefix + ".decisions.jsonl")]
    rounds = json.load(open(prefix + ".rounds.json"))
    clusters = sorted({cluster_of(d["match"]) for d in decs})
    R = defaultdict(Rate)
    lead_types = defaultdict(Counter)
    lead_power = defaultdict(lambda: defaultdict(list))
    bomb_rows = defaultdict(list)
    wild = defaultdict(Counter)
    break_cat = defaultdict(Counter)
    decisions_n = Counter()
    for d in decs:
        lab = d["label"]
        cl = cluster_of(d["match"])
        if d["forced"]:
            continue
        decisions_n[lab] += 1
        seat = d["seat"]
        left = d["left"]
        opps = [left[s] for s in range(4) if s % 2 != seat % 2 and left[s] > 0]
        min_opp = min(opps) if opps else 99
        partner_left = left[(seat + 2) % 4]
        if not d["lead"]:
            if d["has_nonpass"]:
                R[f"pass_{d['top_rel']}"].add(lab, cl, d["pass"])
            if d["has_nonbomb"]:
                R[f"pass_nonbomb_{d['top_rel']}"].add(lab, cl, d["pass"])
            if d["top_rel"] == "partner" and d["has_nonpass"] and not d["pass"]:
                R["overtake_is_bomb"].add(lab, cl, d["type"] in BOMBS)
            if d["top_rel"] == "opponent" and d["has_nonpass"] and not d["has_nonbomb"]:
                # only bombs beat the opponent's play
                R["bomb_when_only_bombs_vs_opp"].add(lab, cl, not d["pass"])
                R["bomb_when_only_bombs_vs_opp_opp_le5"].add(lab, cl, not d["pass"]) if min_opp <= 5 else None
                R["bomb_when_only_bombs_vs_opp_opp_gt5"].add(lab, cl, not d["pass"]) if min_opp > 5 else None
        if d["type"] in BOMBS:
            bomb_rows[lab].append(dict(own=len(d["hand"]), min_opp=min_opp, lead=d["lead"],
                                       top_rel=d["top_rel"], type=d["type"], size=d["bomb_size"], cl=cl))
            R["bomb_opp_le5"].add(lab, cl, min_opp <= 5)
            R["bomb_on_partner"].add(lab, cl, d["top_rel"] == "partner")
            R["bomb_as_lead"].add(lab, cl, d["lead"])
        if d["lead"] and 1 in opps:
            lvl = d["level"]
            nat = natural_counts(d["hand"], lvl)
            wilds_n = sum(is_wild(c, lvl) for c in d["hand"])
            nonjoker = sum(v for r_, v in nat.items() if r_ <= 12)
            can_multi = (any(v >= 2 for v in nat.values()) or (wilds_n >= 1 and nonjoker >= 1)
                         or wilds_n >= 2)
            if can_multi:
                R["lead_single_vs_1card_opp_when_multi_possible"].add(lab, cl, d["type"] == "Single")
                if d["type"] == "Single":
                    maxp = max(power(c, lvl) for c in d["hand"])
                    R["  of_which_single_is_highest_card"].add(lab, cl, power(d["cards"][0], lvl) == maxp)
        if d["lead"]:
            lead_types[lab][d["type"]] += 1
            R["lead_single"].add(lab, cl, d["type"] == "Single")
            R["lead_pair"].add(lab, cl, d["type"] == "Pair")
            R["lead_bomb"].add(lab, cl, d["type"] in BOMBS)
            if len(d["hand"]) > len(d["cards"]):
                minp = min(power(c, d["level"]) for c in d["hand"])
                R["lead_has_min"].add(lab, cl, any(power(c, d["level"]) == minp for c in d["cards"]))
                if d["type"] in ("Single", "Pair"):
                    lead_power[lab][d["type"]].append(power(d["cards"][0], d["level"]) if not is_wild(d["cards"][0], d["level"]) else 12)
                if d["type"] == "Single":
                    R["lead_single_is_min"].add(lab, cl, power(d["cards"][0], d["level"]) == minp)
                    if power(d["cards"][0], d["level"]) != minp:
                        lvl = d["level"]
                        mins = [c for c in d["hand"] if power(c, lvl) == minp and not is_wild(c, lvl)]
                        cnt = natural_counts(d["hand"], lvl)
                        in_combo = any(combos_with(cnt, rank_of(c)) or cnt[rank_of(c)] >= 2 for c in mins)
                        R["lead_single_notmin__min_card_in_pair_or_combo"].add(lab, cl, in_combo)
        if d["type"] in ("Single", "Pair"):
            res = broken(d)
            if res:
                br, before = res
                R["break_any"].add(lab, cl, bool(br))
                for cat in ("bomb", "straight", "tube", "plate", "jokerbomb"):
                    if cat in br:
                        break_cat[lab][cat] += 1
                if d["lead"]:
                    R["break_any_lead"].add(lab, cl, bool(br))
                else:
                    R["break_any_follow"].add(lab, cl, bool(br))
        wilds = [c for c in d["cards"] if is_wild(c, d["level"])]
        if wilds:
            others = [c for c in d["cards"] if not is_wild(c, d["level"])]
            if d["type"] == "Single":
                wild[lab]["single"] += 1
            elif all(rank_of(c) == d["level"] for c in others):
                wild[lab]["level-rank group (pair/triple/bomb of level)"] += 1
            else:
                wild[lab][f"with other ranks: {d['type']}"] += 1
        # hand-level wild holding at round start
    # wild held but never played: from first decision of each seat-round
    first_seen = {}
    last_left = {}
    for d in decs:
        key = (d["match"], d["seat"])
        if key not in first_seen:
            first_seen[key] = d
    wild_held = Counter()
    wild_rounds = Counter()
    for (m, seat), d in first_seen.items():
        n = sum(is_wild(c, d["level"]) for c in d["hand"])
        wild_held[d["label"]] += n
        wild_rounds[d["label"]] += 1

    out = {}
    print(f"sample: {len(rounds)} rounds, {len(clusters)} deals; non-forced decisions {dict(decisions_n)}")
    for key in sorted(R):
        print(f"\n[{key}]")
        for lab in labels:
            print(f"  {lab}: {R[key].fmt(lab)}")
        try:
            lo, hi = R[key].diff(labels[0], labels[1], clusters)
            print(f"  diff {labels[0]}-{labels[1]} cluster-bootstrap 95%: {lo:+.1%} .. {hi:+.1%}")
        except Exception as e:  # noqa
            print("  diff n/a", e)
        out[key] = {lab: R[key].summary(lab) for lab in labels}
    print("\n[lead type distribution]")
    for lab in labels:
        tot = sum(lead_types[lab].values())
        print(f"  {lab} (n={tot}): " + ", ".join(f"{TYPE_ZH.get(t, t)} {n} ({n / tot:.0%})" for t, n in lead_types[lab].most_common()))
    print("\n[lead power of singles/pairs] (power 0=2 .. 11=A, 12=level card, 13/14 jokers)")
    for lab in labels:
        for t in ("Single", "Pair"):
            v = lead_power[lab][t]
            if v:
                print(f"  {lab} {t}: n={len(v)} mean={np.mean(v):.2f} median={np.median(v):.1f} share power<=5 (2..7-ish): {np.mean(np.array(v) <= 5):.0%}")
    print("\n[bombs]")
    player_rounds = Counter()
    for r in rounds:
        for s in range(4):
            player_rounds[r["keys"][s]] += 1
    for lab in labels:
        rows = bomb_rows[lab]
        pr = player_rounds[lab]
        print(f"  {lab}: {len(rows)} bomb plays over {pr} player-rounds = {len(rows) / pr:.2f} per player-round")
        if rows:
            own = np.array([r["own"] for r in rows]); mo = np.array([r["min_opp"] for r in rows])
            print(f"    own cards before bomb: mean {own.mean():.1f}, median {np.median(own):.0f}; "
                  f"min active opponent cards: median {np.median(mo):.0f}")
            print("    types:", dict(Counter((r["type"], r["size"]) for r in rows)))
    print("\n[break categories among single/pair plays]")
    for lab in labels:
        print(f"  {lab}: {dict(break_cat[lab])}")
    print("\n[wild card plays]")
    for lab in labels:
        print(f"  {lab}: {dict(wild[lab])}; wild cards held at start of play: {wild_held[lab]} over {wild_rounds[lab]} player-rounds")

    # finishing order per team label
    print("\n[finishing order]")
    fin = defaultdict(Counter)
    net = defaultdict(list)
    per_deal = defaultdict(float)
    for r in rounds:
        for team in (0, 1):
            lab = r["keys"][team]
            order = r["finish_order"]
            first = order[0] % 2 == team
            double = order[0] % 2 == team and order[1] % 2 == team
            ddown = order[0] % 2 != team and order[1] % 2 != team
            won = r["winning_team"] == team
            last = (not double and not ddown) and order[3] % 2 == team
            fin[lab]["rounds"] += 1
            fin[lab]["first"] += first
            fin[lab]["double"] += double
            fin[lab]["won"] += won
            fin[lab]["gain_when_won"] += r["gain"] if won else 0
            fin[lab]["partner_last_after_first"] += first and not double and order[3] % 2 == team
            fin[lab]["last_nondouble"] += last
            net[lab].append(r["seat_return"][team])
            if lab == "u2623":
                per_deal[r["deal"]] += r["seat_return"][team] / 2
    for lab in labels:
        f = fin[lab]; n = f["rounds"]
        print(f"  {lab}: rounds {n}; first place {f['first']} ({f['first'] / n:.1%}); double win {f['double']} ({f['double'] / n:.1%}); "
              f"team won round {f['won']}; own team has 1st but partner last (1st+4th) {f['partner_last_after_first']}; "
              f"team holds last place in a non-double round {f['last_nondouble']}; mean seat_return {np.mean(net[lab]):+.3f}")
    vals = np.array(list(per_deal.values()))
    rng = np.random.default_rng(1)
    boots = [rng.choice(vals, len(vals)).mean() for _ in range(2000)]
    print(f"  u2623 net levels/round vs {opp} (duplicate): {vals.mean():+.3f}, bootstrap 95% {np.quantile(boots, .025):+.3f}..{np.quantile(boots, .975):+.3f}, deals {len(vals)}")


if __name__ == "__main__":
    main()
