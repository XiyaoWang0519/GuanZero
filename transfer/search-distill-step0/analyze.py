"""Summaries of partA/raw.jsonl and partB/raw.jsonl into results.json (numpy only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MARGIN = 0.5


def decide(means):
    w = int(np.argmax(means))
    return w if w and means[w] >= means[0] + MARGIN else 0


def ranks(x):
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(a, b):
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def bootstrap(values, seed=0, samples=4000):
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(len(data), size=(samples, len(data)))].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def compare(positions, name, ref="REF"):
    rows = []
    for p in positions:
        a = np.asarray(p["configs"][ref]["values"])
        b = np.asarray(p["configs"][name]["values"])
        ma, mb = a.mean(0), b.mean(0)
        best_a = set(np.flatnonzero(ma == ma.max()))
        rows.append({
            "argmax": int(np.argmax(ma) == np.argmax(mb)),
            "argmax_tie_tolerant": int(int(np.argmax(mb)) in best_a),
            "decision": int(decide(ma) == decide(mb)),
            "override_ref": int(decide(ma) != 0), "override_x": int(decide(mb) != 0),
            "spearman": spearman(ma, mb),
            "abs_dq": float(np.abs(ma - mb).mean()),
            "bias": float((mb - ma).mean()),
            "own_cards": p["own_cards"]})
    out = {"n": len(rows)}
    for key in ("argmax", "argmax_tie_tolerant", "decision", "override_ref", "override_x", "abs_dq", "bias"):
        out[key] = float(np.mean([r[key] for r in rows]))
    sp = [r["spearman"] for r in rows if r["spearman"] is not None]
    out["spearman_mean"] = float(np.mean(sp)) if sp else None
    out["spearman_n"] = len(sp)
    # By stage of the round (cards in the deciding seat's hand).
    out["by_stage"] = {}
    for label, lo, hi in (("own>20", 21, 27), ("own11-20", 11, 20), ("own<=10", 0, 10)):
        sub = [r for r in rows if lo <= r["own_cards"] <= hi]
        if sub:
            sps = [r["spearman"] for r in sub if r["spearman"] is not None]
            out["by_stage"][label] = {"n": len(sub),
                                      "argmax": float(np.mean([r["argmax"] for r in sub])),
                                      "decision": float(np.mean([r["decision"] for r in sub])),
                                      "spearman": float(np.mean(sps)) if sps else None,
                                      "abs_dq": float(np.mean([r["abs_dq"] for r in sub]))}
    return out


def noise(positions, name):
    se, se_diff, secs = [], [], []
    for p in positions:
        v = np.asarray(p["configs"][name]["values"])
        n = len(v)
        se.append(float((v.std(0, ddof=1) / np.sqrt(n)).mean()))
        d = v[:, 1:] - v[:, :1]
        se_diff.append(float((d.std(0, ddof=1) / np.sqrt(n)).mean()))
        if p["configs"][name]["seconds"] is not None:
            secs.append(p["configs"][name]["seconds"])
    return {"q_se_mean": float(np.mean(se)), "q_minus_baseline_se_mean": float(np.mean(se_diff)),
            "seconds_mean": float(np.mean(secs)) if secs else None,
            "seconds_p50_p95": [float(np.quantile(secs, q)) for q in (.5, .95)] if secs else None}


def judged_gain(positions, names, judge="REF2"):
    """Value, under an independent estimate (REF2), of each config's decision over the baseline."""
    out = {}
    for name in names:
        g = []
        for p in positions:
            j = np.asarray(p["configs"][judge]["values"]).mean(0)
            m = np.asarray(p["configs"][name]["values"]).mean(0)
            g.append(float(j[decide(m)] - j[0]))
        out[name] = {"mean": float(np.mean(g)), "ci95": bootstrap(g, 7)}
    return out


def calibration(positions):
    crit, ret = [], []
    for p in positions:
        c = np.asarray(p["calibration"]["critic"]).ravel()
        r = np.asarray(p["configs"]["REF"]["values"]).ravel()
        keep = ~np.isnan(c)
        crit.append(c[keep])
        ret.append(r[keep])
    c, r = np.concatenate(crit), np.concatenate(ret)
    edges = np.quantile(c, np.linspace(0, 1, 11))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c >= lo) & (c <= hi)
        bins.append({"critic_mean": float(c[m].mean()), "return_mean": float(r[m].mean()), "n": int(m.sum())})
    return {"pairs": int(len(c)), "corr": float(np.corrcoef(c, r)[0, 1]),
            "bias_return_minus_critic": float((r - c).mean()),
            "mse": float(((r - c) ** 2).mean()), "return_var": float(r.var()),
            "r2": float(1 - ((r - c) ** 2).mean() / r.var()), "deciles": bins,
            "ended_in_first_trick": int(sum(p["calibration"]["ended_first_trick"] for p in positions))}


def part_a():
    raw = HERE / "partA" / "raw.jsonl"
    if not raw.exists():
        return None
    deals = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
    positions = [p for d in deals for p in d["positions"]]
    sub = [p for p in positions if "REF2" in p["configs"]]
    names = ["C1", "C2", "C3", "REF_T"]
    out = {"deals": len(deals), "positions": len(positions), "ref2_positions": len(sub),
           "unsure_per_round": float(np.mean([d["unsure"] for d in deals])),
           "decisions_per_round": float(np.mean([d["decisions"] for d in deals])),
           "candidates_mean": float(np.mean([p["candidates"] for p in positions])),
           "own_cards_mean": float(np.mean([p["own_cards"] for p in positions])),
           "vs_REF_all": {n: compare(positions, n) for n in names},
           "vs_REF_subset": {n: compare(sub, n) for n in names + ["REF2"]} if sub else None,
           "noise": {n: noise(positions, n) for n in ["REF", "C1", "C2", "C3"]},
           "calibration": calibration(positions)}
    if sub:
        out["noise"]["REF2"] = noise(sub, "REF2")
        out["ref2_judged_gain_over_baseline"] = judged_gain(sub, ["REF", "C1", "C2", "C3", "REF_T"])
    return out


def part_b():
    raw = HERE / "partB" / "raw.jsonl"
    if not raw.exists():
        return None
    merged: dict[int, dict] = {}
    for line in raw.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            merged.setdefault(row["deal"], {}).update(row["arms"])
    configs = sorted({c for arms in merged.values() for c in arms})
    out = {}
    for c in configs:
        rows = [arms[c] for arms in merged.values() if c in arms]
        net = [r["net"] for r in rows]
        secs = [s for r in rows for s in r["search_seconds"]]
        trig = sum(r["triggered"] for r in rows)
        out[c] = {"deals": len(rows), "levels_per_round": float(np.mean(net)),
                  "ci95": bootstrap(net, 11), "deals_changed": int(sum(n != 0 for n in net)),
                  "searches": trig, "override_rate": sum(r["overrides"] for r in rows) / max(trig, 1),
                  "searches_per_leg": trig / (2 * len(rows)),
                  "seconds_per_search": float(np.mean(secs)) if secs else None,
                  "seconds_p50_p95": [float(np.quantile(secs, q)) for q in (.5, .95)] if secs else None,
                  "seconds_per_deal": float(np.mean([r["wall_seconds"] for r in rows]))}
    both = [d for d, arms in merged.items() if len(arms) >= 2]
    if len(configs) >= 2 and both:
        a, b = configs[0], configs[1]
        diff = [merged[d][a]["net"] - merged[d][b]["net"] for d in both]
        out[f"{a}_minus_{b}"] = {"deals": len(both), "mean": float(np.mean(diff)), "ci95": bootstrap(diff, 13)}
    return out


def main():
    results = {"partA": part_a(), "partB": part_b()}
    (HERE / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
