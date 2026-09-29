"""Turn the profile-diag metrics of every rank into profile-summary.json and .md.

    python summarize.py RESULTS_DIR [--window 15] [--out DIR]

RESULTS_DIR is the pod's /workspace/results (locally: download/results). Only
the last WINDOW updates that were profiled (``collection_profile_synchronized``)
are summarized; the last unprofiled warmup updates are reported next to them
as the unsynchronized reference. Profile mode synchronizes CUDA at every phase
boundary, so its seconds attribute time to phases; they are not absolute speed.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SEGMENT = "profile-diag"
PHASES = ("env_pending", "events_and_rounds", "metadata_and_assignment", "input_indexing",
          "decision_upload", "public_cache_or_collation", "actor_and_sampling",
          "decision_download", "buffer_and_counters", "env_step")
SPLIT = ("public_cache_or_collation", "actor_and_sampling")
GROUPS = ("learner", "snapshot")
REFERENCE = 5   # unprofiled updates just before the window


def read_lines(path: Path) -> list[dict]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except OSError:
        return []


def rank_dirs(output: Path) -> list[Path]:
    ranks = [output]
    r = 1
    while (output / f"rank-{r}" / "metrics.jsonl").exists():
        ranks.append(output / f"rank-{r}")
        r += 1
    return ranks


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def quantile(histogram: dict[int, int], q: float) -> int | None:
    """Nearest-rank quantile of a {value: count} histogram."""
    total = sum(histogram.values())
    if not total:
        return None
    rank = max(1, math.ceil(q * total))
    seen = 0
    for value in sorted(histogram):
        seen += histogram[value]
        if seen >= rank:
            return value
    return max(histogram)


def call_shape(histogram: dict[int, int], steps: int) -> dict:
    calls = sum(histogram.values())
    rows = sum(size * count for size, count in histogram.items())
    return dict(calls=calls, rows=rows, calls_per_step=calls / steps if steps else None,
                mean_rows=rows / calls if calls else None,
                median_rows=quantile(histogram, 0.5), p10_rows=quantile(histogram, 0.1),
                p90_rows=quantile(histogram, 0.9),
                min_rows=min(histogram) if histogram else None,
                max_rows=max(histogram) if histogram else None)


def rank_summary(lines: list[dict]) -> dict:
    count = len(lines)
    steps = sum(l.get("collection_steps") or 0 for l in lines)
    phases = {p: sum(l["collection_phase_seconds"].get(p, 0.0) for l in lines) / count
              for p in PHASES}
    collect = mean(l["collect_seconds"] for l in lines)
    phase_total = sum(phases.values())
    split = {}
    for group in GROUPS:
        split[group] = {p: sum(l.get("collection_group_phase_seconds", {}).get(group, {}).get(p, 0.0)
                               for l in lines) / count for p in SPLIT}
    histograms = {}
    for group in GROUPS:
        merged: dict[int, int] = {}
        for l in lines:
            for size, calls in l.get("collection_policy_call_rows", {}).get(group, {}).items():
                merged[int(size)] = merged.get(int(size), 0) + int(calls)
        histograms[group] = merged
    shapes = {g: call_shape(histograms[g], steps) for g in GROUPS}
    for group in GROUPS:   # seconds per call, from the per-group totals
        calls = shapes[group]["calls"] / count if shapes[group]["calls"] else 0
        for p in SPLIT:
            shapes[group][f"{p}_ms_per_call"] = (1000 * split[group][p] / calls) if calls else None
    return dict(
        updates=[lines[0]["update"], lines[-1]["update"]], profiled=count,
        steps_per_update=steps / count,
        collect_seconds=collect, learn_seconds=mean(l["learn_seconds"] for l in lines),
        phase_seconds=phases, phase_share_of_collect={p: v / collect for p, v in phases.items()},
        unattributed_seconds=collect - phase_total,
        group_phase_seconds=split, call_shapes=shapes,
        call_rows_histogram={g: {str(k): v for k, v in sorted(h.items())}
                             for g, h in histograms.items()},
        policy_calls_per_update=mean(l["collection_policy_batches"] for l in lines),
        local_decisions_per_update=mean(l["step_decisions"] for l in lines),
        cuda_peak_reserved_gb=max(l.get("cuda_peak_reserved_bytes") or 0 for l in lines) / 1e9,
        cuda_peak_allocated_gb=max(l.get("cuda_peak_allocated_bytes") or 0 for l in lines) / 1e9,
        allocation_retries=max(l.get("cuda_allocation_retries") or 0 for l in lines))


def gpu_window(results: Path, first_update: int, last_update: int) -> dict:
    """GPU samples between the end of the update before the window and the window's end."""
    stamps = {r["update"]: r["epoch"] for r in read_lines(results / "update-epochs.jsonl")}
    begin, end = stamps.get(first_update - 1), stamps.get(last_update)
    if begin is None or end is None:
        return dict(gpu_samples=0, note="no update epochs for the window")
    samples = [s for s in read_lines(results / "gpu-samples.jsonl")
               if "util" in s and begin <= s["epoch"] <= end]
    if not samples:
        return dict(gpu_samples=0)
    return dict(gpu_samples=len(samples), gpu_util_mean=mean(s["util"] for s in samples),
                gpu_mem_util_mean=mean(s.get("mem_util") for s in samples),
                gpu_mem_used_mib_max=max(s["mem_used_mib"] for s in samples),
                gpu_power_w_mean=mean(s.get("power_w") for s in samples))


def global_block(base: list[dict], all_lines: list[dict]) -> dict:
    """Throughput of a run of consecutive rank-0 lines (global_* fields when DDP)."""
    if not base:
        return {}
    before = next((l for l in all_lines if l["update"] == base[0]["update"] - 1), None)
    wall = base[-1]["elapsed_seconds"] - before["elapsed_seconds"] if before else None
    decisions = sum(l.get("global_step_decisions", l["step_decisions"]) for l in base)
    return dict(updates=[base[0]["update"], base[-1]["update"]], count=len(base),
                decisions_per_sec_wall=decisions / wall if wall else None,
                decisions_per_sec_collect=mean(l.get("global_decisions_per_sec",
                                                     l["decisions_per_sec"]) for l in base),
                wall_per_update=wall / len(base) if wall else None,
                collect_seconds_max=mean(l.get("global_collect_seconds", l["collect_seconds"])
                                         for l in base),
                learn_seconds_max=mean(l.get("global_learn_seconds", l["learn_seconds"])
                                       for l in base),
                resident_snapshots=[l["population"]["resident_snapshots"] for l in base],
                mean_prefix=mean(l["mean_prefix"] for l in base))


def summarize(results: Path, window: int) -> dict:
    output = results / "segments" / SEGMENT
    ranks = [read_lines(d / "metrics.jsonl") for d in rank_dirs(output)]
    base = ranks[0]
    profiled = [l for l in base if l.get("collection_profile_synchronized")]
    if not profiled:
        return dict(error="no profiled updates in rank-0 metrics", updates=len(base))
    chosen = [l["update"] for l in profiled[-window:]]
    per_rank = []
    for index, lines in enumerate(ranks):
        by_update = {l["update"]: l for l in lines}
        rows = [by_update[u] for u in chosen if u in by_update
                and by_update[u].get("collection_profile_synchronized")]
        if len(rows) != len(chosen):
            per_rank.append(dict(rank=index, error=f"{len(rows)} of {len(chosen)} window updates"))
            continue
        per_rank.append(dict(rank=index, **rank_summary(rows)))
    good = [r for r in per_rank if "error" not in r]
    window_lines = [l for l in base if l["update"] in chosen]
    unprofiled = [l for l in base if not l.get("collection_profile_synchronized")
                  and l["update"] < chosen[0]][-REFERENCE:]
    summary = dict(
        segment=SEGMENT, ranks=len(ranks), window_updates=[chosen[0], chosen[-1]],
        window=len(chosen), requested_window=window, first_update=base[0]["update"],
        last_update=base[-1]["update"], lineage_decisions=base[-1]["decisions"],
        caveat="profile mode synchronizes CUDA at every phase boundary: seconds attribute "
               "collection time to phases, they are not absolute speed",
        window_global=dict(global_block(window_lines, base),
                           **gpu_window(results, chosen[0], chosen[-1])),
        unprofiled_reference=dict(global_block(unprofiled, base),
                                  **(gpu_window(results, unprofiled[0]["update"],
                                                unprofiled[-1]["update"]) if unprofiled else {})),
        per_rank=per_rank)
    if good:
        summary["rank_mean"] = dict(
            collect_seconds=mean(r["collect_seconds"] for r in good),
            learn_seconds=mean(r["learn_seconds"] for r in good),
            phase_seconds={p: mean(r["phase_seconds"][p] for r in good) for p in PHASES},
            unattributed_seconds=mean(r["unattributed_seconds"] for r in good),
            group_phase_seconds={g: {p: mean(r["group_phase_seconds"][g][p] for r in good)
                                     for p in SPLIT} for g in GROUPS},
            calls_per_step={g: mean(r["call_shapes"][g]["calls_per_step"] for r in good)
                            for g in GROUPS},
            mean_rows_per_call={g: mean(r["call_shapes"][g]["mean_rows"] for r in good)
                                for g in GROUPS})
    return summary


def fmt(value, digits=2):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def markdown(s: dict) -> str:
    if "error" in s:
        return f"# Collection profile\n\nNo summary: {s['error']}.\n"
    ranks = [r for r in s["per_rank"] if "error" not in r]
    head = "| | " + " | ".join(f"rank {r['rank']}" for r in ranks) + " | mean |"
    rule = "|---|" + "---:|" * (len(ranks) + 1)
    out = [f"# Collection profile, updates {s['window_updates'][0]}-{s['window_updates'][1]}",
           "", f"{s['window']} profiled updates, {s['ranks']} ranks. {s['caveat']}.", ""]
    bad = [r for r in s["per_rank"] if "error" in r]
    if bad:
        out += ["Ranks left out: " + "; ".join(f"rank {r['rank']}: {r['error']}" for r in bad), ""]
    out += ["## Phase seconds per update (synchronized)", "", head, rule]
    for p in PHASES:
        out.append(f"| {p} | " + " | ".join(fmt(r["phase_seconds"][p]) for r in ranks)
                   + f" | {fmt(s['rank_mean']['phase_seconds'][p])} |")
    out.append("| unattributed (collect - phases) | " + " | ".join(
        fmt(r["unattributed_seconds"]) for r in ranks) + f" | {fmt(s['rank_mean']['unattributed_seconds'])} |")
    out.append("| **collect seconds** | " + " | ".join(fmt(r["collect_seconds"]) for r in ranks)
               + f" | {fmt(s['rank_mean']['collect_seconds'])} |")
    out.append("| learn seconds | " + " | ".join(fmt(r["learn_seconds"]) for r in ranks)
               + f" | {fmt(s['rank_mean']['learn_seconds'])} |")
    out += ["", "## Learner vs snapshot calls (seconds per update)", "", head, rule]
    for g in GROUPS:
        for p in SPLIT:
            out.append(f"| {g}: {p} | " + " | ".join(fmt(r["group_phase_seconds"][g][p]) for r in ranks)
                       + f" | {fmt(s['rank_mean']['group_phase_seconds'][g][p])} |")
    out += ["", "## Policy-call batch sizes (rows per call, window total)", "",
            "| rank | group | calls | calls/step | mean | median | p10 | p90 | min | max | "
            "cache ms/call | actor ms/call |", "|---|---|" + "---:|" * 10]
    for r in ranks:
        for g in GROUPS:
            c = r["call_shapes"][g]
            out.append(f"| {r['rank']} | {g} | {fmt(c['calls'])} | {fmt(c['calls_per_step'])} | "
                       f"{fmt(c['mean_rows'], 1)} | {fmt(c['median_rows'])} | {fmt(c['p10_rows'])} | "
                       f"{fmt(c['p90_rows'])} | {fmt(c['min_rows'])} | {fmt(c['max_rows'])} | "
                       f"{fmt(c['public_cache_or_collation_ms_per_call'])} | "
                       f"{fmt(c['actor_and_sampling_ms_per_call'])} |")
    out += ["", "## Whole run (global fields, rank 0)", "",
            "| | profiled window | unprofiled reference |", "|---|---:|---:|"]
    w, u = s["window_global"], s["unprofiled_reference"]
    for key, label in (("updates", "updates"), ("decisions_per_sec_wall", "decisions/s (wall)"),
                       ("decisions_per_sec_collect", "decisions/s (collect)"),
                       ("wall_per_update", "wall s/update"),
                       ("collect_seconds_max", "collect s (slowest rank)"),
                       ("learn_seconds_max", "learn s (slowest rank)"),
                       ("resident_snapshots", "resident snapshots"), ("mean_prefix", "mean prefix"),
                       ("gpu_util_mean", "GPU util %"), ("gpu_mem_used_mib_max", "GPU mem max MiB"),
                       ("gpu_samples", "GPU samples")):
        a, b = w.get(key), u.get(key)
        if isinstance(a, list):
            a = f"{min(a)}-{max(a)}" if key == "resident_snapshots" else f"{a[0]}-{a[-1]}"
        if isinstance(b, list):
            b = (f"{min(b)}-{max(b)}" if key == "resident_snapshots" else f"{b[0]}-{b[-1]}") if b else None
        out.append(f"| {label} | {fmt(a)} | {fmt(b)} |")
    out.append("")
    out.append("CUDA peak reserved per rank (GB): "
               + ", ".join(fmt(r["cuda_peak_reserved_gb"]) for r in ranks)
               + f"; allocation retries max {max(r['allocation_retries'] for r in ranks)}.")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--out", type=Path, default=None, help="default: RESULTS_DIR")
    args = parser.parse_args()
    out = args.out or args.results
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(args.results, args.window)
    (out / "profile-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = markdown(summary)
    (out / "profile-summary.md").write_text(text)
    print(text)
    return 0 if "error" not in summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
