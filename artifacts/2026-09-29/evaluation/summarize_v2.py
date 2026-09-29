"""Collect old (two-baseline) and new (B11-only) checkpoint evaluations into results.v2.{md,json}.

Old points: out/uNNNN.json (freeze f77ea34d..., two baselines). New points: out-b11/uNNNN.json
(derived freeze b11only/freeze.b11-only.json, b11-main only), used only when b11only/VERDICT is PASS;
otherwise out-full/ (original freeze) is used. Both from the b8c0c54 export (source 14e72581...).
Paired intervals: eval.duplicate.bootstrap_interval over per-deal differences, 4,000 samples, seed 0.
Run with PYTHONPATH=src-b8c0c54/python:src-b8c0c54/oracle:src-b8c0c54.
"""
import json
import statistics
from pathlib import Path

import numpy as np

from eval.duplicate import bootstrap_interval

ROOT = Path("/Users/xiyaowang/Developer/Projects/GuanZero")
W = ROOT / ".work/longrun-batched-eval-2026-09-29"
METRICS = [ROOT / ".work/actor-ranks-2026-09-28/download/results/segments/main-w4/metrics.jsonl",  # u844 lives here
           ROOT / ".work/longrun-batched-2026-09-28/download/results/segments/main-overnight/metrics.jsonl"]
LATE_FROM = 2108
ENDPOINT = "u2623"
SOURCE = "14e72581d628409032acf88b646f7eb7a1ded7e954fa06d2e367bcce67889270"


def verdict():
    p = W / "b11only/VERDICT"
    return p.read_text().strip() if p.exists() else None


def point_files():
    files = {p.stem: (p, "two-baseline (original freeze)") for p in sorted((W / "out").glob("u*.json"))}
    v = verdict()
    if v == "PASS":
        for p in sorted((W / "out-b11").glob("u*.json")):
            if p.stem not in files:
                files[p.stem] = (p, "B11-only (derived freeze)")
    for p in sorted((W / "out-full").glob("u*.json")) if (W / "out-full").exists() else []:
        if p.stem not in files:
            files[p.stem] = (p, "two-baseline (original freeze), fallback")
    return files


def metric_rows():
    rows = {}
    for path in METRICS:
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["update"]] = r
    return rows


def paired(a, b):
    d = b - a
    lo, hi = bootstrap_interval(d, 0, 4000)
    return dict(mean=float(d.mean()), ci95=[lo, hi])


def boot_stat(M, stat, seed=0, samples=4000):
    rng = np.random.default_rng(seed)
    n = M.shape[1]
    vals = np.array([stat(M[:, rng.integers(n, size=n)]) for _ in range(samples)])
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def timing(path):
    t = path.with_suffix(".time")
    if not t.exists():
        return None, None
    txt = t.read_text() + " " + (path.with_suffix(".log").read_text() if path.with_suffix(".log").exists() else "")
    wall = int(txt.split("wall_seconds")[1].split()[0]) if "wall_seconds" in txt else None
    conc = "2-4"
    if "concurrent=" in txt:  # new runs: measured from start/end stamps of all runs in out-b11 and out-full
        conc = measured_concurrency(t)
    return wall, conc


def _interval(t):
    from datetime import datetime
    txt = t.read_text()
    st = datetime.fromisoformat(txt.split("start ")[1].split()[0].replace("Z", "+00:00")).timestamp()
    en = (datetime.fromisoformat(txt.split(" end ")[1].split()[0].replace("Z", "+00:00")).timestamp()
          if " end " in txt else float("inf"))
    return st, en


def measured_concurrency(t):
    """Min-max number of evaluations running at once (incl. this one) over this run's interval."""
    runs = [_interval(p) for d in ("out-b11", "out-full") if (W / d).exists() for p in (W / d).glob("u*.time")]
    st, en = _interval(t)
    edges = sorted({st} | {x for r in runs for x in r if st <= x < en})
    counts = [sum(1 for a, b in runs if a <= e < b) for e in edges]
    lo, hi = min(counts), max(counts)
    return str(lo) if lo == hi else f"{lo}-{hi}"


def main():
    rows = metric_rows()
    done = []
    for name, (path, kind) in point_files().items():
        rep = json.loads(path.read_text())
        if rep["evaluation_source_sha256"] != SOURCE:
            raise SystemExit(f"{path}: unexpected evaluator source {rep['evaluation_source_sha256']}")
        b11 = rep["reports"]["b11-main"]
        u = int(name[1:])
        window = [rows[k]["entropy"] for k in range(u - 30, u + 1) if k in rows]
        wall, conc = timing(path)
        done.append(dict(
            name=name, update=u, run=kind, file=str(path.relative_to(W)),
            global_decisions=rows[u].get("global_decisions", rows[u]["decisions"]),
            entropy_mean_31=statistics.mean(window), n_entropy_updates=len(window),
            candidate_sha256=rep["candidate_sha256"], evaluation_source_sha256=rep["evaluation_source_sha256"],
            freeze_sha256=rep["freeze_sha256"],
            vs_b11=b11["duplicates"]["mean_net_levels_per_round"], vs_b11_ci95=b11["duplicates"]["bootstrap_95_ci"],
            match_win_rate=b11["full_matches"]["win_rate"], match_win_ci95=b11["full_matches"]["bootstrap_95_ci"],
            wall_seconds=wall, concurrent=conc))
    done.sort(key=lambda d: d["update"])
    scores = {d["name"]: np.array(json.loads((W / d["file"]).read_text())["reports"]["b11-main"]["duplicates"]["pair_scores"])
              for d in done}
    for i, d in enumerate(done):
        if d["name"] != "u0844":
            d["paired_vs_u0844"] = paired(scores["u0844"], scores[d["name"]])
        if i:
            prev = done[i - 1]["name"]
            d["paired_vs_previous"] = dict(previous=prev, **paired(scores[prev], scores[d["name"]]))

    late = [d["name"] for d in done if d["update"] >= LATE_FROM]
    L = {}
    if len(late) >= 2:
        M = np.stack([scores[n] for n in late])
        ups = np.array([int(n[1:]) for n in late], dtype=float)
        pooled = M.mean(axis=0)
        L["checkpoints"] = late
        L["pooled_vs_b11"] = dict(mean=float(pooled.mean()), ci95=list(bootstrap_interval(pooled, 0, 4000)))
        L["pooled_minus_u0844"] = paired(scores["u0844"], pooled)
        slope = lambda X: float(np.polyfit(ups, X.mean(axis=1), 1)[0] * 100)
        L["ols_slope_per_100_updates"] = dict(mean=slope(M), ci95=list(boot_stat(M, slope)),
                                              span_updates=float(ups.max() - ups.min()))
        h = len(late) // 2
        L["later_half_minus_earlier_half"] = dict(earlier=late[:h], later=late[-h:],
                                                  **paired(M[:h].mean(axis=0), M[-h:].mean(axis=0)))
        if ENDPOINT in scores:
            L["other_minus_endpoint"] = {n: paired(scores[ENDPOINT], scores[n]) for n in late if n != ENDPOINT}
    v = verdict()
    verify = json.loads((W / "b11only/verify.json").read_text()) if (W / "b11only/verify.json").exists() else None

    method = dict(
        evaluator="eval.history_frozen.evaluate from the read-only export src-b8c0c54 (source identity 14e72581...), run_point.py, torch 4 threads, nice 10",
        freeze_original=".work/history-budget-2026-09-27/evaluation/freeze.json sha256 f77ea34dbd4bae50dfb088b441a1c46868d7c0b1b013205975655517de3f7760 (b11-main + longrun-segment2-raw-endpoint)",
        freeze_b11_only="b11only/freeze.b11-only.json sha256 5926f7516786f0c34ef431dbb1bd98270f0aace05899a74b51960a4173864350 (b11-main only; development file path made absolute, same file and sha256)",
        deals="256 development duplicate deals, policy seed 2026092852, 64 match seeds",
        per_checkpoint_ci="evaluator bootstrap (2,000 samples) over 256 paired deals; match win CI = evaluator bootstrap over 64 seed pairs",
        paired_ci="eval.duplicate.bootstrap_interval over per-deal differences, 4,000 samples, seed 0",
        late_segment=f"all evaluated checkpoints with update >= {LATE_FROM}; pooled = per-deal scores averaged over checkpoints first; slope/halves use a joint deal bootstrap (4,000 samples, seed 0)",
    )
    (W / "results.v2.json").write_text(json.dumps(dict(method=method, b11_only_verification=verify,
                                                       checkpoints=done, late_segment=L), indent=2) + "\n")

    f = lambda x: f"{x:+.2f}"
    ci = lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]"
    fp = lambda p: f"{f(p['mean'])} {ci(p['ci95'])}"
    out = ["# Overnight main lineage vs B11, 256 duplicate deals (v2: adds late checkpoints)", ""]
    if verify is None:
        out += ["**B11-only verification: pending** (u2201 B11-only still running). New points are listed only after it passes.", ""]
    else:
        out += [f"**B11-only verification (u2201 B11-only vs `out/u2201.json`): {v}.** "
                f"Per-deal scores differing: {verify['pair_scores_differing']} of {verify['n_pair_scores'][0]}; "
                f"match pairs differing: {verify['match_pairs_differing']} of {verify['n_match_pairs'][0]}; "
                f"whole b11-main report identical: {verify['whole_b11_report_identical']}; "
                f"same candidate sha {verify['candidate_sha_equal']}, same evaluator source {verify['source_sha_equal']}. "
                f"freeze_sha256 recorded: original {verify['freeze_sha256_original'][:16]}..., B11-only {verify['freeze_sha256_b11_only'][:16]}... "
                "(differs by design; evaluator does not check it against a list). "
                f"u2201 B11-only wall time: {timing(W / 'out-b11/u2201.json')[0] / 60:.0f} min "
                f"(concurrent {timing(W / 'out-b11/u2201.json')[1]}) vs 36 min for the two-baseline run.", ""]
    out += ["| ckpt | update | decisions | entropy (31-upd mean) | vs B11 [95% CI] | match win [95% CI] | paired vs u844 | paired vs previous | run | wall (concurrent) |",
            "|---|---:|---:|---:|---|---|---|---|---|---|"]
    for d in done:
        pv, pp = d.get("paired_vs_u0844"), d.get("paired_vs_previous")
        wall = f"{d['wall_seconds'] / 60:.0f} min ({d['concurrent']})" if d["wall_seconds"] else "-"
        out.append(f"| {d['name']} | {d['update']} | {d['global_decisions'] / 1e6:.1f}M | {d['entropy_mean_31']:.3f} | "
                   f"{f(d['vs_b11'])} {ci(d['vs_b11_ci95'])} | "
                   f"{d['match_win_rate']:.0%} [{d['match_win_ci95'][0]:.0%}, {d['match_win_ci95'][1]:.0%}] | "
                   f"{fp(pv) if pv else '-'} | {(pp['previous'] + ': ' + fp(pp)) if pp else '-'} | "
                   f"{'B11-only' if 'B11-only' in d['run'] else 'two-baseline'} | {wall} |")
    pending = [n for n in ["u2201", "u2623", "u2480", "u2542", "u2418", "u2263"] if n not in scores]
    if pending:
        out += ["", f"Still pending: {', '.join(pending)}."]
    if L:
        out += ["", f"## Late segment (u{LATE_FROM} onward: {', '.join(L['checkpoints'])})", "",
                "The checkpoints share the same 256 deals and are consecutive in one lineage, so the pooled number is a "
                "smoothed estimate of the late plateau, not independent replication. Intervals resample deals jointly.", "",
                f"- Pooled late mean vs B11: {fp(L['pooled_vs_b11'])}",
                f"- Pooled late mean minus u844: {fp(L['pooled_minus_u0844'])}"]
        s = L["ols_slope_per_100_updates"]
        trend = "cannot distinguish from zero" if s["ci95"][0] <= 0 <= s["ci95"][1] else "interval excludes zero"
        out.append(f"- Trend: OLS slope {fp(s)} levels/round per 100 updates over {s['span_updates']:.0f} updates ({trend})")
        h = L["later_half_minus_earlier_half"]
        trend = "cannot distinguish" if h["ci95"][0] <= 0 <= h["ci95"][1] else "interval excludes zero"
        out.append(f"- Later half ({', '.join(h['later'])}) minus earlier half ({', '.join(h['earlier'])}): {fp(h)} ({trend})")
        if "other_minus_endpoint" in L:
            oe = L["other_minus_endpoint"]
            out.append("- Each late checkpoint minus endpoint u2623: " + "; ".join(f"{n} {fp(p)}" for n, p in oe.items()))
            better = [n for n, p in oe.items() if p["ci95"][0] > 0]
            rec = (f"{', '.join(better)} beats u2623 with a paired interval above zero; consider it."
                   if better else "no late checkpoint beats u2623 with a paired interval entirely above zero; "
                   "default to the endpoint u2623.")
            out += ["", f"**Starting checkpoint recommendation:** {rec}"]
    out += ["", "Limits: one training lineage, one opponent (B11), 256 development deals, no final-test evaluation. "
            "Old points ran 2-4 evaluations concurrently with two baselines; new points ran B11 only with 2-4 concurrent (measured per point from start/end stamps, see column), always 4 torch threads each. "
            "Method details: results.v2.json `method`."]
    (W / "results.v2.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
