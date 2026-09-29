"""Summarize the pod evaluation (runs on the Mac, after or during the download).

(a) Pod table: every finished pod point, paired against the POD-run u0844 and the
    previous pod point. Always valid: one machine, one build, one environment.
(b) Merged table with the 8 local points (.work/longrun-batched-eval-2026-09-29/out):
    only if BOTH pod references (u0844, u2201) are bit-identical to their local results
    (whole `reports` object: both baselines, all per-deal scores, all match pairs).
    Otherwise the platform difference is reported and nothing is merged.

Per-checkpoint numbers come unchanged from eval.history_frozen (run_point.py).
Paired differences: eval.duplicate.bootstrap_interval over per-deal differences,
4,000 samples, seed 0 (imported from the packed b8c0c54 source in local-src/).
Usage: summarize.py --pod-results DIR --out DIR [--local-out DIR]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

KIT = Path(__file__).resolve().parent
sys.path[:0] = [str(KIT / "local-src" / "python"), str(KIT / "local-src")]
from eval.duplicate import bootstrap_interval  # noqa: E402

REPO = KIT.parents[1]
LOCAL_OUT = REPO / ".work/longrun-batched-eval-2026-09-29/out"
METRICS = (REPO / ".work/actor-ranks-2026-09-28/download/results/segments/main-w4/metrics.jsonl",
           REPO / ".work/longrun-batched-2026-09-28/download/results/segments/main-overnight/metrics.jsonl")
CKPT_PATHS = REPO / ".work/longrun-batched-2026-09-28/download/results/segments/main-overnight"
REFERENCES = ("u0844", "u2201")
ENDPOINT = "u2623"
LATE_FROM = 2108
BASE = "b11-main"


def metric_rows():
    rows = {}
    for path in METRICS:
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["update"]] = r
    return rows


def finished(out: Path, require_exit: bool = True) -> dict:
    """name -> report. Pod: .time records exit 0 and the json exists. Local (the 8
    finished points; some .time files lack an exit line): the json exists, which
    run_point.py writes only at the end."""
    done = {}
    for js in sorted(out.glob("u*.json")):
        timef = out / f"{js.stem}.time"
        text = timef.read_text() if timef.exists() else ""
        if "exit 0 " in text or not require_exit:
            done[js.stem] = dict(report=json.loads(js.read_text()), timing=text.strip())
    return done


def scores(report):
    return np.array(report["reports"][BASE]["duplicates"]["pair_scores"], dtype=float)


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


def excludes_zero(ci):
    return ci[0] > 0 or ci[1] < 0


def row(name, rep, rows, timing, origin):
    update = int(name[1:])
    b = rep["reports"][BASE]
    window = [rows[u]["entropy"] for u in range(update - 30, update + 1) if u in rows]
    wall = None
    if timing and "wall_seconds" in timing:
        wall = int(timing.split("wall_seconds")[1].split()[0])
    return dict(name=name, update=update, origin=origin,
                global_decisions=rows[update].get("global_decisions", rows[update]["decisions"]),
                entropy_mean_31=statistics.mean(window), entropy_updates_in_window=len(window),
                candidate_sha256=rep["candidate_sha256"], freeze_sha256=rep["freeze_sha256"],
                evaluation_source_sha256=rep["evaluation_source_sha256"],
                vs_b11=b["duplicates"]["mean_net_levels_per_round"],
                vs_b11_ci95=b["duplicates"]["bootstrap_95_ci"],
                match_win_rate=b["full_matches"]["win_rate"],
                match_win_ci95=b["full_matches"]["bootstrap_95_ci"], wall_seconds=wall)


def table(points: dict, rows) -> dict:
    """points: name -> (report, timing, origin). Rows sorted by update with paired columns."""
    S = {n: scores(v[0]) for n, v in points.items()}
    out = sorted((row(n, v[0], rows, v[1], v[2]) for n, v in points.items()), key=lambda d: d["update"])
    for i, d in enumerate(out):
        if "u0844" in S and d["name"] != "u0844":
            d["paired_vs_u0844"] = paired(S["u0844"], S[d["name"]])
        if i > 0:
            prev = out[i - 1]["name"]
            d["paired_vs_previous"] = dict(previous=prev, **paired(S[prev], S[d["name"]]))
    return dict(rows=out, late=late(out, S), recommendation=recommend(out, S))


def late(rows, S):
    names = [d["name"] for d in rows if d["update"] >= LATE_FROM]
    if len(names) < 2:
        return None
    ups = np.array([int(n[1:]) for n in names], dtype=float)
    M = np.stack([S[n] for n in names])
    pooled = M.mean(axis=0)
    res = dict(checkpoints=names, n_deals=int(M.shape[1]))
    res["pooled_vs_b11"] = dict(mean=float(pooled.mean()), ci95=list(bootstrap_interval(pooled, 0, 4000)))
    if "u0844" in S:
        res["pooled_minus_u0844"] = paired(S["u0844"], pooled)
    slope = lambda X: float(np.polyfit(ups, X.mean(axis=1), 1)[0] * 100)
    res["ols_slope_per_100_updates"] = dict(mean=slope(M), ci95=list(boot_stat(M, slope)),
                                            span_updates=float(ups.max() - ups.min()))
    h = len(names) // 2
    res["second_half_minus_first_half"] = dict(first=names[:h], second=names[-h:],
                                               **paired(M[:h].mean(axis=0), M[-h:].mean(axis=0)))
    res["endpoint_minus_other_late_mean"] = dict(endpoint=names[-1], **paired(M[:-1].mean(axis=0), M[-1]))
    means = M.mean(axis=1)
    res["late_point_range"] = dict(min=float(means.min()), max=float(means.max()),
                                   argmin=names[int(means.argmin())], argmax=names[int(means.argmax())])
    s, hh = res["ols_slope_per_100_updates"], res["second_half_minus_first_half"]
    if not excludes_zero(s["ci95"]) and not excludes_zero(hh["ci95"]):
        verdict = ("cannot distinguish the late segment from flat: neither the slope nor the "
                   "later-minus-earlier difference excludes zero")
    elif excludes_zero(s["ci95"]) and excludes_zero(hh["ci95"]) and np.sign(s["mean"]) == np.sign(hh["mean"]):
        verdict = ("still " + ("rising" if s["mean"] > 0 else "falling") + ": slope and later-minus-earlier "
                   "both exclude zero (same deals, one lineage; a diagnostic, not a replicated claim)")
    else:
        verdict = ("mixed: only one of slope / later-minus-earlier excludes zero; treat as cannot distinguish "
                   "a trend")
    res["trend_statement"] = verdict
    return res


def recommend(rows, S):
    names = [d["name"] for d in rows]
    if ENDPOINT not in names:
        return dict(choice=None, reason=f"{ENDPOINT} has no finished evaluation; no recommendation")
    worse_than = []
    for n in names:
        if n == ENDPOINT:
            continue
        p = paired(S[n], S[ENDPOINT])
        if p["ci95"][1] < 0:
            worse_than.append(dict(point=n, endpoint_minus_point=p))
    ckpt = str(CKPT_PATHS / "latest.pt")
    if not worse_than:
        return dict(choice=ENDPOINT, checkpoint=ckpt,
                    reason=(f"default: the endpoint {ENDPOINT} (latest.pt); no evaluated earlier "
                            "checkpoint is better beyond noise (every paired CI of endpoint minus "
                            "point reaches zero or above)"))
    best = max(worse_than, key=lambda w: float(S[w["point"]].mean()))
    return dict(choice=best["point"], checkpoint=str(CKPT_PATHS / f"update-{int(best['point'][1:]):06d}.pt"),
                worse_than=worse_than,
                reason=(f"the endpoint is worse than {', '.join(w['point'] for w in worse_than)} beyond "
                        f"noise (paired CI entirely below zero); {best['point']} has the highest mean of "
                        "those. Several comparisons on shared deals: confirm before relying on it"))


def comparability(pod: dict, local: dict) -> dict:
    res = {}
    for ref in REFERENCES:
        if ref not in pod:
            res[ref] = dict(available=False)
            continue
        a, b = local[ref]["report"], pod[ref]["report"]
        la, pb = scores(a), scores(b)
        diff = pb - la
        fa, fb = a["reports"][BASE]["full_matches"]["pairs"], b["reports"][BASE]["full_matches"]["pairs"]
        other = {n: a["reports"][n] == b["reports"].get(n) for n in a["reports"] if n != BASE}
        res[ref] = dict(
            available=True,
            identical=a["reports"] == b["reports"],
            same_candidate=a["candidate_sha256"] == b["candidate_sha256"],
            same_freeze=a["freeze_sha256"] == b["freeze_sha256"],
            same_source=a["evaluation_source_sha256"] == b["evaluation_source_sha256"],
            b11_deals_differing=int((diff != 0).sum()), b11_deals=int(len(diff)),
            b11_mean_local=float(la.mean()), b11_mean_pod=float(pb.mean()),
            b11_mean_pod_minus_local=paired(la, pb),
            b11_match_pairs_differing=int(sum(x != y for x, y in zip(fa, fb))),
            b11_match_win_rate_local=a["reports"][BASE]["full_matches"]["win_rate"],
            b11_match_win_rate_pod=b["reports"][BASE]["full_matches"]["win_rate"],
            other_baselines_identical=other)
    res["both_identical"] = all(res[r].get("identical") for r in REFERENCES)
    return res


def fmt_table(t, title):
    f = lambda x: f"{x:+.2f}"
    ci = lambda c: f"[{c[0]:+.2f}, {c[1]:+.2f}]"
    L = [f"## {title}", "",
         "| ckpt | ran on | update | decisions | entropy (31-upd mean) | vs B11 [95% CI] | match win [95% CI] | paired vs u844 | paired vs previous | wall |",
         "|---|---|---:|---:|---:|---|---|---|---|---|"]
    for d in t["rows"]:
        pv, pp = d.get("paired_vs_u0844"), d.get("paired_vs_previous")
        L.append(f"| {d['name']} | {d['origin']} | {d['update']} | {d['global_decisions'] / 1e6:.1f}M | "
                 f"{d['entropy_mean_31']:.3f} | {f(d['vs_b11'])} {ci(d['vs_b11_ci95'])} | "
                 f"{d['match_win_rate']:.0%} [{d['match_win_ci95'][0]:.0%}, {d['match_win_ci95'][1]:.0%}] | "
                 f"{(f(pv['mean']) + ' ' + ci(pv['ci95'])) if pv else '-'} | "
                 f"{(pp['previous'] + ': ' + f(pp['mean']) + ' ' + ci(pp['ci95'])) if pp else '-'} | "
                 f"{(str(round(d['wall_seconds'] / 60)) + ' min') if d['wall_seconds'] else ''} |")
    lt = t["late"]
    if lt:
        fp = lambda p: f"{f(p['mean'])} {ci(p['ci95'])}"
        s, h, e, r = (lt["ols_slope_per_100_updates"], lt["second_half_minus_first_half"],
                      lt["endpoint_minus_other_late_mean"], lt["late_point_range"])
        L += ["", f"Late segment (evaluated checkpoints from u{LATE_FROM} on: {', '.join(lt['checkpoints'])}). "
              "Joint bootstrap over the same 256 deals (4,000 samples, seed 0). The checkpoints share deals "
              "and are consecutive in one lineage: the pooled number is a smoothed estimate of the late "
              "plateau, not independent replication.", "",
              f"- Pooled late mean vs B11: {fp(lt['pooled_vs_b11'])}"]
        if "pooled_minus_u0844" in lt:
            L.append(f"- Pooled late mean minus u844: {fp(lt['pooled_minus_u0844'])}")
        L += [f"- OLS slope: {fp(s)} levels/round per 100 updates (span {s['span_updates']:.0f} updates)",
              f"- Later half ({', '.join(h['second'])}) minus earlier half ({', '.join(h['first'])}): {fp(h)}",
              f"- Endpoint {e['endpoint']} minus mean of the other late points: {fp(e)}",
              f"- Range of late point estimates: {f(r['min'])} ({r['argmin']}) to {f(r['max'])} ({r['argmax']})",
              f"- **Trend: {lt['trend_statement']}.**"]
    rec = t["recommendation"]
    L += ["", f"**Next run's starting checkpoint: {rec.get('choice')}** ({rec.get('checkpoint', '-')}). {rec['reason']}."]
    return L


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod-results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--local-out", type=Path, default=LOCAL_OUT)
    args = ap.parse_args(argv)
    rows = metric_rows()
    pod = finished(args.pod_results / "out")
    local = finished(args.local_out, require_exit=False)
    status = {}
    if (args.pod_results / "status.json").exists():
        status = json.loads((args.pod_results / "status.json").read_text())
    env = json.loads((args.pod_results / "env.json").read_text()) if (args.pod_results / "env.json").exists() else {}
    facts = json.loads((args.pod_results / "host-facts.json").read_text()) \
        if (args.pod_results / "host-facts.json").exists() else {}
    comp = comparability(pod, local)
    result = dict(method=dict(
        evaluator="eval.history_frozen.evaluate via run_point.py (unchanged reference), torch 4 threads, "
                  "source git b8c0c54 (identity 14e72581...)",
        freeze=".work/history-budget-2026-09-27/evaluation/freeze.json (sha256 f77ea34d...), 256 development "
               "deals, policy seed 2026092852, 64 match seeds",
        opponent="b11-main (frozen, sha checked by the evaluator)",
        per_checkpoint_ci="evaluator's bootstrap (2,000 samples) over 256 paired deals",
        paired_ci="bootstrap_interval over per-deal differences, 4,000 samples, seed 0",
        entropy="mean training entropy of the 31 updates ending at the checkpoint (metrics.jsonl)"),
        pod_environment={k: env.get(k) for k in ("python", "machine", "torch", "numpy",
                                                 "matched_local_versions", "source_sha256")},
        pod_cpu=dict(usable=facts.get("usable_cpus"), model=facts.get("cpu_model"), plan=facts.get("plan")),
        pod_status={n: s.get("state") for n, s in status.get("points", {}).items()},
        comparability=comp)
    if pod:
        result["pod_table"] = table({n: (v["report"], v["timing"], "pod") for n, v in pod.items()}, rows)
    if comp["both_identical"]:
        merged = {n: (v["report"], v["timing"], "local") for n, v in local.items()}
        for n, v in pod.items():
            if n not in merged:
                merged[n] = (v["report"], v["timing"], "pod")
        result["merged_table"] = table(merged, rows)
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    L = ["# Overnight main lineage: pod CPU evaluation vs B11 (256 duplicate deals)", "",
         f"Pod: {facts.get('cpu_model')} , usable CPUs {facts.get('usable_cpus')}, "
         f"parallel {((facts.get('plan') or {}).get('parallel'))}; python {str(env.get('python', '')).split()[0] if env else '?'}, "
         f"torch {env.get('torch')}, numpy {env.get('numpy')} (local versions matched: {env.get('matched_local_versions')}).",
         "Point states: " + (", ".join(f"{n} {s}" for n, s in result["pod_status"].items()) or "none yet"), "",
         "## Cross-platform comparability (pod Linux x86-64 vs local macOS arm64)", ""]
    for ref in REFERENCES:
        c = comp[ref]
        if not c.get("available"):
            L.append(f"- {ref}: no finished pod result")
            continue
        p = c["b11_mean_pod_minus_local"]
        L.append(f"- {ref}: {'bit-identical' if c['identical'] else 'NOT identical'}; B11 per-deal scores differing: "
                 f"{c['b11_deals_differing']}/{c['b11_deals']}; mean pod {c['b11_mean_pod']:+.3f} vs local "
                 f"{c['b11_mean_local']:+.3f}, pod minus local {p['mean']:+.3f} [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}]; "
                 f"match pairs differing {c['b11_match_pairs_differing']}/64; same candidate/freeze/source: "
                 f"{c['same_candidate']}/{c['same_freeze']}/{c['same_source']}")
    if comp["both_identical"]:
        L.append("- Both references are bit-identical: the pod and local results are the same computation; "
                 "the merged table below is valid.")
    else:
        L.append("- The references are not both bit-identical (or not both finished): the pod table stands on its "
                 "own and is NOT merged with the local points. The measured platform difference is the pod-minus-"
                 "local line above; compare new points only with pod-run points.")
    L.append("")
    if "pod_table" in result:
        L += fmt_table(result["pod_table"], "(a) Pod-run points, paired against pod-run references (always valid)")
    if "merged_table" in result:
        L += [""] + fmt_table(result["merged_table"], "(b) Merged with the 8 local points (references bit-identical)")
    L += ["", "Method: see results.json `method`."]
    (args.out / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
