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


LATE_FROM = 2108  # predeclared in the task: all evaluated checkpoints from u2108 onward


def boot_stat(M, stat, seed=0, samples=4000):
    """Percentile CI of stat(M[:, idx]) resampling whole deals (columns), shared across checkpoints."""
    rng = np.random.default_rng(seed)
    n = M.shape[1]
    vals = np.array([stat(M[:, rng.integers(n, size=n)]) for _ in range(samples)])
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def late_analysis(by_update, scores):
    late = [d for d in by_update if d["update"] >= LATE_FROM]
    if len(late) < 2:
        return None
    names = [d["name"] for d in late]
    ups = np.array([d["update"] for d in late], dtype=float)
    M = np.stack([scores[n] for n in names])
    pooled = M.mean(axis=0)
    out = dict(checkpoints=names, n_deals=int(M.shape[1]))
    out["pooled_vs_b11"] = dict(mean=float(pooled.mean()), ci95=list(bootstrap_interval(pooled, 0, 4000)))
    if "u0844" in scores:
        out["pooled_minus_u0844"] = paired(scores["u0844"], pooled)
    slope = lambda X: float(np.polyfit(ups, X.mean(axis=1), 1)[0] * 100)
    out["ols_slope_per_100_updates"] = dict(mean=slope(M), ci95=list(boot_stat(M, slope)),
                                            span_updates=float(ups.max() - ups.min()))
    h = len(names) // 2
    first, second = M[:h].mean(axis=0), M[-h:].mean(axis=0)
    out["second_half_minus_first_half"] = dict(first=names[:h], second=names[-h:], **paired(first, second))
    end = names[-1]
    others = M[:-1].mean(axis=0)
    out["endpoint_minus_other_late_mean"] = dict(endpoint=end, **paired(others, M[-1]))
    if "u2201" in names and end != "u2201":
        out["endpoint_minus_u2201"] = paired(scores["u2201"], scores[end])
    means = M.mean(axis=1)
    out["late_point_range"] = dict(min=float(means.min()), max=float(means.max()),
                                   argmin=names[int(means.argmin())], argmax=names[int(means.argmax())])
    return out


def late_lines(late, f, ci):
    if not late:
        return []
    fp = lambda p: f"{f(p['mean'])} {ci(p['ci95'])}"
    L = ["", f"## Late segment (u{LATE_FROM} onward; {', '.join(late['checkpoints'])})", "",
         "All intervals resample the same 256 deals jointly across checkpoints (4,000 samples, seed 0). "
         "The checkpoints share deals and are consecutive in one lineage, so the pooled number is a smoothed "
         "estimate of the late plateau, not independent replication.", "",
         f"- Pooled late mean vs B11 (per-deal scores averaged over the {len(late['checkpoints'])} checkpoints, then bootstrap over deals): "
         f"{fp(late['pooled_vs_b11'])}"]
    if "pooled_minus_u0844" in late:
        L.append(f"- Pooled late mean minus u844: {fp(late['pooled_minus_u0844'])}")
    s = late["ols_slope_per_100_updates"]
    L.append(f"- OLS slope of checkpoint means on update: {fp(s)} levels/round per 100 updates "
             f"(span {s['span_updates']:.0f} updates, so {f(s['mean'] * s['span_updates'] / 100)} over the span)")
    h = late["second_half_minus_first_half"]
    L.append(f"- Later half ({', '.join(h['second'])}) minus earlier half ({', '.join(h['first'])}): {fp(h)}")
    e = late["endpoint_minus_other_late_mean"]
    L.append(f"- Endpoint {e['endpoint']} minus mean of the other late checkpoints: {fp(e)}")
    if "endpoint_minus_u2201" in late:
        L.append(f"- Endpoint minus u2201 (best in-run point): {fp(late['endpoint_minus_u2201'])}")
    r = late["late_point_range"]
    L.append(f"- Range of late point estimates: {f(r['min'])} ({r['argmin']}) to {f(r['max'])} ({r['argmax']})")
    return L


NOTES_V1 = """
## Notes from the in-run evaluation (added after the first eight points; post hoc, exploratory)

- u844 reproduction: bit-identical to `.work/actor-ranks-2026-09-28/eval-256/main-w4.json` (same candidate sha, same source sha 14e72581..., all 256 per-deal scores and all 64 match pairs equal).
- Entropy step starts at about update 1838-1840; u1829 is the last synced checkpoint before it.
- Pooled, post hoc: mean of u2015/u2108/u2201/u2356 minus u1829 = +0.09 [-0.08, +0.27]; minus mean of u1612/u1829 = +0.15 [+0.02, +0.29]; u2356 - u1829 = +0.12 [-0.10, +0.35].
- Match win-rate 95% CIs (64 seed pairs): u1829 [0.43, 0.59], u2201 [0.40, 0.55], u2356 [0.41, 0.58].
- Wall times are with 2-4 evaluations running concurrently (4 torch threads each); single-run time will be lower.
"""


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
    late = late_analysis(by_update, scores)
    (W / "results.json").write_text(json.dumps(dict(
        method=dict(
            evaluator="eval.history_frozen.evaluate via run_point.py (copied unchanged from actor-ranks-2026-09-28/eval-256), torch 4 threads",
            freeze=".work/history-budget-2026-09-27/evaluation/freeze.json (sha256 f77ea34d...), 256 development deals, 64 match seeds",
            opponent="b11-main (frozen baseline, sha checked by evaluator)",
            per_checkpoint_ci="evaluator's own bootstrap (2,000 samples) over 256 paired deals",
            paired_ci="bootstrap_interval over per-deal score differences, 4,000 samples, seed 0",
            late_segment=("all evaluated checkpoints with update >= 2108; pooled = per-deal scores averaged over them; "
                          "slope/halves/endpoint use a joint deal bootstrap (4,000 samples, seed 0); checkpoints share "
                          "deals and are consecutive in one lineage, so this is smoothing, not replication"),
        ),
        checkpoints=by_update, late_segment=late), indent=2) + "\n")

    f = lambda x: f"{x:+.2f}"
    ci = lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]"
    lines = ["# Overnight main-lineage checkpoints vs B11 (256 duplicate deals)", "",
             "| ckpt | update | decisions | entropy (31-upd mean) | vs B11 | 95% CI | match win [95% CI] | paired vs u844 | paired vs previous | wall |",
             "|---|---:|---:|---:|---:|---|---|---|---|---|"]
    for d in by_update:
        pv = d.get("paired_vs_u0844")
        pp = d.get("paired_vs_previous")
        wall = ""
        if d["timing"] and "wall_seconds" in d["timing"]:
            wall = f"{int(d['timing'].split('wall_seconds')[1].split()[0]) / 60:.0f} min"
        lines.append(
            f"| {d['name']} | {d['update']} | {d['global_decisions'] / 1e6:.1f}M | {d['entropy_mean_31']:.3f} | "
            f"{f(d['vs_b11'])} | {ci(d['vs_b11_ci95'])} | "
            f"{d['match_win_rate']:.0%} [{d['match_win_ci95'][0]:.0%}, {d['match_win_ci95'][1]:.0%}] | "
            f"{(f(pv['mean']) + ' ' + ci(pv['ci95'])) if pv else '-'} | "
            f"{(pp['previous'] + ': ' + f(pp['mean']) + ' ' + ci(pp['ci95'])) if pp else '-'} | {wall} |")
    lines += ["", "Paired vs u675 (44.2M, overnight-large final): " + "; ".join(
        f"{d['name']} {f(d['paired_vs_u0675']['mean'])} {ci(d['paired_vs_u0675']['ci95'])}" for d in by_update), "",
        "Method: see results.json `method`."]
    lines += late_lines(late, f, ci)
    lines += ["", NOTES_V1.strip()]
    extra = W / "results-notes.md"
    if extra.exists():
        lines += ["", extra.read_text().strip()]
    (W / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
