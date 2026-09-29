"""Collect finished checkpoint evaluations into results.json and results.md.

Per-checkpoint numbers come unchanged from eval.history_frozen (run_point.py).
Paired differences resample the 256 per-deal scores together
(eval.duplicate.bootstrap_interval, 4,000 samples, seed 0).
"""
import json
import statistics
from pathlib import Path

import numpy as np

from eval.duplicate import bootstrap_interval

ROOT = Path("/Users/xiyaowang/Developer/Projects/GuanZero")
W = ROOT / ".work/longrun-batched-eval-2026-09-29"
METRICS = {
    "w4": ROOT / ".work/actor-ranks-2026-09-28/download/results/segments/main-w4/metrics.jsonl",
    "overnight": ROOT / ".work/longrun-batched-2026-09-28/download/results/segments/main-overnight/metrics.jsonl",
}
U675 = ROOT / ".work/longrun-large-2026-09-28/final/vs-b11-256.json"
ORDER = ["u0844", "u1333", "u1612", "u1829", "u2015", "u2046", "u2077", "u2108", "u2139", "u2170"]


def metric_rows():
    rows = {}
    for path in METRICS.values():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["update"]] = r
    return rows


def pair_scores(path):
    return np.array(json.loads(Path(path).read_text())["reports"]["b11-main"]["duplicates"]["pair_scores"])


def paired(a, b):
    d = b - a
    lo, hi = bootstrap_interval(d, 0, 4000)
    return dict(mean=float(d.mean()), ci95=[lo, hi])


def main():
    rows = metric_rows()
    done = []
    for out in sorted((W / "out").glob("u*.json")):
        name = out.stem
        rep = json.loads(out.read_text())
        b11 = rep["reports"]["b11-main"]
        update = int(name[1:])
        window = [rows[u]["entropy"] for u in range(update - 30, update + 1) if u in rows]
        timef = W / "out" / f"{name}.time"
        done.append(dict(
            name=name, update=update,
            global_decisions=rows[update].get("global_decisions", rows[update]["decisions"]),
            entropy_at_update=rows[update]["entropy"],
            entropy_mean_31=statistics.mean(window), entropy_min_31=min(window), entropy_max_31=max(window),
            candidate_sha256=rep["candidate_sha256"], evaluation_source_sha256=rep["evaluation_source_sha256"],
            freeze_sha256=rep["freeze_sha256"],
            vs_b11=b11["duplicates"]["mean_net_levels_per_round"], vs_b11_ci95=b11["duplicates"]["bootstrap_95_ci"],
            match_win_rate=b11["full_matches"]["win_rate"], match_win_ci95=b11["full_matches"]["bootstrap_95_ci"],
            timing=" ".join(p.read_text().strip() for p in (timef, W / "out" / f"{name}.log")
                            if p.exists() and "wall_seconds" in p.read_text()) or None,
        ))
    scores = {d["name"]: pair_scores(W / "out" / f"{d['name']}.json") for d in done}
    ref = scores.get("u0844")
    u675 = pair_scores(U675)
    by_update = sorted(done, key=lambda d: d["update"])
    for i, d in enumerate(by_update):
        s = scores[d["name"]]
        d["paired_vs_u0675"] = paired(u675, s)
        if ref is not None and d["name"] != "u0844":
            d["paired_vs_u0844"] = paired(ref, s)
        if i > 0:
            prev = by_update[i - 1]
            d["paired_vs_previous"] = dict(previous=prev["name"], **paired(scores[prev["name"]], s))
    # Before vs after the entropy step, and latest vs step.
    (W / "results.json").write_text(json.dumps(dict(
        method=dict(
            evaluator="eval.history_frozen.evaluate via run_point.py (copied unchanged from actor-ranks-2026-09-28/eval-256), torch 4 threads",
            freeze=".work/history-budget-2026-09-27/evaluation/freeze.json (sha256 f77ea34d...), 256 development deals, 64 match seeds",
            opponent="b11-main (frozen baseline, sha checked by evaluator)",
            per_checkpoint_ci="evaluator's own bootstrap (2,000 samples) over 256 paired deals",
            paired_ci="bootstrap_interval over per-deal score differences, 4,000 samples, seed 0",
        ),
        checkpoints=by_update), indent=2) + "\n")

    f = lambda x: f"{x:+.2f}"
    ci = lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]"
    lines = ["# Overnight main-lineage checkpoints vs B11 (256 duplicate deals)", "",
             "| ckpt | update | decisions | entropy (31-upd mean) | vs B11 | 95% CI | match win | paired vs u844 | paired vs previous | wall |",
             "|---|---:|---:|---:|---:|---|---:|---|---|---|"]
    for d in by_update:
        pv = d.get("paired_vs_u0844")
        pp = d.get("paired_vs_previous")
        wall = ""
        if d["timing"] and "wall_seconds" in d["timing"]:
            wall = f"{int(d['timing'].split('wall_seconds')[1].split()[0]) / 60:.0f} min"
        lines.append(
            f"| {d['name']} | {d['update']} | {d['global_decisions'] / 1e6:.1f}M | {d['entropy_mean_31']:.3f} | "
            f"{f(d['vs_b11'])} | {ci(d['vs_b11_ci95'])} | {d['match_win_rate']:.0%} | "
            f"{(f(pv['mean']) + ' ' + ci(pv['ci95'])) if pv else '-'} | "
            f"{(pp['previous'] + ': ' + f(pp['mean']) + ' ' + ci(pp['ci95'])) if pp else '-'} | {wall} |")
    lines += ["", "Paired vs u675 (44.2M, overnight-large final): " + "; ".join(
        f"{d['name']} {f(d['paired_vs_u0675']['mean'])} {ci(d['paired_vs_u0675']['ci95'])}" for d in by_update), "",
        "Method: see results.json `method`."]
    (W / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
