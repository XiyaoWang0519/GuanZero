"""Decision-change statistics from the teacher-forced probe (out/probe_*)."""
import glob
import gzip
import json
from collections import defaultdict

import numpy as np

SEATS = ["S0", "S1", "S2", "S3"]


def load(kind):
    recs, metas = [], {}
    for f in sorted(glob.glob(f"out/probe_{kind}_*.records.jsonl.gz")):
        recs += [json.loads(l) for l in gzip.open(f, "rt")]
        for m in json.load(open(f.replace(".records.jsonl.gz", ".meta.json"))):
            metas[m["tag"]] = m
    return recs, metas


def own_bin(n):
    return "01-05" if n <= 5 else "06-10" if n <= 10 else "11-18" if n <= 18 else "19-27"


def opp_bin(n):
    return "1-2" if n <= 2 else "3-5" if n <= 5 else "6-10" if n <= 10 else "11+"


def round_bin(r):
    return str(r) if r <= 3 else "4-5" if r <= 5 else "6-8" if r <= 8 else "9+"


def features(r, metas):
    total = metas[r["tag"]]["round_tokens"][r["round"]]
    frac = r["round_events"] / max(total, 1)
    return {
        "all": "all",
        "lead": "lead" if r["lead"] else "follow",
        "third": "early" if frac < 1 / 3 else "mid" if frac < 2 / 3 else "late",
        "own_left": own_bin(r["own_left"]),
        "opp_min_left": opp_bin(min(r["opp_left"])),
        "partner_left": "out" if r["partner_left"] == 0 else own_bin(r["partner_left"]),
        "chosen_type": "Pass" if r["pass"] else r["type"],
        "pass_play": "pass" if r["pass"] else "play",
        "round_index": round_bin(r["round"]),
        "n_legal": "2" if r["n_legal"] == 2 else "3-5" if r["n_legal"] <= 5 else "6-20" if r["n_legal"] <= 20 else "21+",
    }


def table(recs, metas, variant, keys, min_n=30):
    lines = []
    for key in keys:
        groups = defaultdict(list)
        for r in recs:
            if r["tv"][variant] is None:
                continue
            groups[features(r, metas)[key]].append(r)
        lines.append(f"  by {key}:")
        for g in sorted(groups):
            rs = groups[g]
            if len(rs) < min_n:
                continue
            flips = sum(r["arg"][variant] != r["arg"]["full"] for r in rs)
            tvs = np.array([r["tv"][variant] for r in rs])
            lines.append(f"    {g:8s} n={len(rs):6d} flips={flips:5d} ({flips / len(rs):6.2%}) "
                         f"meanTV={tvs.mean():.4f} p90TV={np.quantile(tvs, .9):.4f} maxTV={tvs.max():.3f}")
    return lines


def overall(recs, variant, label):
    rs = [r for r in recs if r["tv"][variant] is not None]
    flips = sum(r["arg"][variant] != r["arg"]["full"] for r in rs)
    tvs = np.array([r["tv"][variant] for r in rs])
    return (f"{label}: n={len(rs)} flips={flips} ({flips / max(len(rs), 1):.2%}) meanTV={tvs.mean():.4f} "
            f"medianTV={np.median(tvs):.4f} share TV>0.1={np.mean(tvs > 0.1):.3%} TV>0.3={np.mean(tvs > 0.3):.3%}")


def example(r, metas, variant):
    m = metas[r["tag"]]
    ev = m["events"][:r["history_len"]]
    cur = [e for e in ev if e[0] == r["round"]]
    earlier = [e for e in ev if e[0] < r["round"]]
    seat = r["seat"]
    rel = {seat: "自己", (seat + 1) % 4: "下家", (seat + 2) % 4: "对家", (seat + 3) % 4: "上家"}
    lines = [f"[{r['tag']} 第{r['round']}局 座位{seat} {'领出' if r['lead'] else '跟牌'} "
             f"本局已发生事件 {r['round_events']} / 全场前缀 {r['prefix']}; "
             f"剩牌 自己{r['own_left']} 对家{r['partner_left']} 下家{r['opp_left'][0]} 上家{r['opp_left'][1]}; "
             f"TV({variant})={r['tv'][variant]:.3f}]",
             f"  手牌: {r['hand']}",
             "  本局公开事件: " + " | ".join(f"{rel[s]}:{t}" for _, s, t in cur)]
    if variant in ("round", "swap"):
        lines.append(f"  之前各局: {len(earlier)} 个事件; 各局结果 "
                     + str([x.get('order') for x in m['rounds'][:r['round']]]))
    for c in sorted(r["cands"], key=lambda c: -c["full"]):
        mark = []
        if c["i"] == r["arg"]["full"]:
            mark.append("FULL选")
        if c["i"] == r["arg"][variant]:
            mark.append(f"{variant.upper()}选")
        sw = "" if c["swap"] is None else f" swap={c['swap']:.3f}"
        lines.append(f"    {c['a']:40s} full={c['full']:.3f} none={c['none']:.3f} round={c['round']:.3f}{sw} {' '.join(mark)}")
    return "\n".join(lines)


def main():
    out = []
    keys = ["lead", "third", "own_left", "opp_min_left", "partner_left", "chosen_type", "pass_play", "n_legal"]
    dup, dmeta = load("dup")
    match, mmeta = load("match")
    for recs, metas, name in ((dup, dmeta, "single rounds"), (match, mmeta, "full matches")):
        if not recs:
            continue
        multi = [r for r in recs if r["n_legal"] >= 2]
        out.append(f"\n==== {name}: {len(recs)} probe decisions, {len(multi)} with >=2 legal ====")
        out.append(overall(multi, "none", "FULL vs NONE"))
        if name == "full matches":
            later = [r for r in multi if r["round"] > 0]
            out.append(overall(later, "round", "FULL vs ROUND (rounds>0)"))
            out.append(overall(later, "swap", "FULL vs SWAP (rounds>0 with donor)"))
            sw = [r for r in later if r["tv"]["swap"] is not None]
            a = np.array([r["tv"]["round"] for r in sw])
            b = np.array([r["tv"]["swap"] for r in sw])
            c = np.array([r["tv"]["swap_vs_round"] for r in sw])
            out.append(f"same decisions (n={len(sw)}): meanTV FULL-ROUND={a.mean():.4f} FULL-SWAP={b.mean():.4f} "
                       f"SWAP-ROUND={c.mean():.4f}; flips FULL-ROUND={sum(r['arg']['round'] != r['arg']['full'] for r in sw)} "
                       f"FULL-SWAP={sum(r['arg']['swap'] != r['arg']['full'] for r in sw)} "
                       f"SWAP-ROUND={sum(r['arg']['swap'] != r['arg']['round'] for r in sw)}")
            rng = np.random.default_rng(0)
            # match-clustered bootstrap for mean TV(FULL,ROUND) - TV(FULL,SWAP)
            tags = sorted({r["tag"][:-2] for r in sw})
            by = defaultdict(list)
            for r in sw:
                by[r["tag"][:-2]].append(r["tv"]["round"] - r["tv"]["swap"])
            vals = []
            for _ in range(2000):
                pick = rng.integers(len(tags), size=len(tags))
                allv = np.concatenate([by[tags[i]] for i in pick])
                vals.append(allv.mean())
            out.append(f"  TV(FULL,ROUND)-TV(FULL,SWAP) = {(a - b).mean():+.4f} "
                       f"[{np.quantile(vals, .025):+.4f},{np.quantile(vals, .975):+.4f}] (pair-clustered bootstrap)")
            out.append(" FULL vs ROUND by round index:")
            out += table(later, metas, "round", ["round_index"], min_n=20)
            out.append(" FULL vs SWAP by round index:")
            out += table(later, metas, "swap", ["round_index"], min_n=20)
            out.append(" FULL vs NONE by round index:")
            out += table(multi, metas, "none", ["round_index"], min_n=20)
            out.append(" FULL vs ROUND splits (rounds>0):")
            out += table(later, metas, "round", keys)
        out.append(f" FULL vs NONE splits ({name}):")
        out += table(multi, metas, "none", keys)
    print("\n".join(out))
    # examples: in each category, the flip decision with the largest TV
    ex = []
    cats = [("single rounds NONE, lead", dup, dmeta, "none", True),
            ("single rounds NONE, follow", dup, dmeta, "none", False),
            ("matches NONE, lead", match, mmeta, "none", True),
            ("matches NONE, follow", match, mmeta, "none", False),
            ("matches ROUND, lead", match, mmeta, "round", True),
            ("matches ROUND, follow", match, mmeta, "round", False),
            ("matches SWAP, any", match, mmeta, "swap", None)]
    for label, recs, metas, v, lead in cats:
        cand = [r for r in recs if r["tv"][v] is not None and r["arg"][v] != r["arg"]["full"]
                and (lead is None or r["lead"] == lead)]
        if not cand:
            continue
        best = max(cand, key=lambda r: r["tv"][v])
        ex.append(f"### {label} (rule: largest TV among argmax flips in this category)\n" + example(best, metas, v))
    open("out/examples.txt", "w").write("\n\n".join(ex))
    open("out/probe_analysis.txt", "w").write("\n".join(out))


if __name__ == "__main__":
    main()
