"""Allocator-trim check summary: trim-summary.json and trim-summary.md.

    python summarize.py RESULTS_DIR [--settle 2] [--out DIR]

RESULTS_DIR is the pod's /workspace/results (locally: download/results). Every
rank-0 update of the segment gets one row (arm, wall time, speed, collect/learn,
entropy, KL, retries, nvidia-smi max/mean over the update's samples, and the
sums over ranks of reserved/allocated bytes before/after each trim point, plus
trim seconds). Per arm, the first SETTLE updates are dropped from the steady
figures (the switch releases the snapshot graphs and admits the stacked heads).

Verdict. With the arm on and no trim, the A/B (same layout, machine 67872) saw
nvidia-smi climb as a staircase (+460 MiB over 8 updates, max 23,578 MiB vs an
off-arm max of 21,250 MiB) while allocated memory did not grow. Here the on-arm
is right after a resume (histories and resident snapshots still growing for
~25 updates), so a rise in live memory is expected; the staircase test is the
rise of nvidia-smi *beyond* the rise of the summed per-rank peak allocated:

    excess = (smi max, last 10 settled on-updates - first 10) -
             (sum peak allocated, same windows)

FLAT if excess <= 256 MiB and the smi least-squares slope minus the allocated
slope is <= 10 MiB/update; otherwise STAIRCASE. Standard library only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SEGMENT = "trim-check-diag"
POINTS = ("graphs_released", "after_collect", "after_learn")
FIELDS = ("reserved_before", "reserved_after", "allocated_before", "allocated_after")
MIB = 2 ** 20
# The A/B of merged snapshot inference (no trim), machine 67872, settled blocks.
AB_REFERENCE = dict(machine_id=67872, off_smi_max_mib=21250, on_smi_max_mib=23578,
                    off_wall_s_per_update=20.84, on_wall_s_per_update=15.39,
                    source=".work/snapshot-batching-2026-09-28/ab-summary.md")
EXCESS_LIMIT_MIB = 256
SLOPE_LIMIT_MIB = 10


def read_lines(path: Path) -> list[dict]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except OSError:
        return []


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def rank_dirs(output: Path) -> list[Path]:
    ranks, r = [output], 1
    while (output / f"rank-{r}" / "metrics.jsonl").exists():
        ranks.append(output / f"rank-{r}")
        r += 1
    return ranks


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def slope(values):
    """Least-squares slope per index step; None with fewer than 3 points."""
    points = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(points) < 3:
        return None
    mx = mean(i for i, _ in points)
    my = mean(v for _, v in points)
    den = sum((i - mx) ** 2 for i, _ in points)
    return sum((i - mx) * (v - my) for i, v in points) / den if den else None


def host(results: Path) -> dict:
    machine, facts = read_json(results / "machine.json"), read_json(results / "host-facts.json")
    return dict(machine_id=machine.get("machine_id"), offer_id=machine.get("offer_id"),
                cpu_model=facts.get("cpu_model"), usable_cpus=facts.get("usable_cpus"),
                gpu=facts.get("gpu"))


def update_rows(results: Path) -> list[dict]:
    output = results / "segments" / SEGMENT
    ranks = [{l["update"]: l for l in read_lines(d / "metrics.jsonl")} for d in rank_dirs(output)]
    stamps = {r["update"]: r["epoch"] for r in read_lines(results / "update-epochs.jsonl")}
    samples = [s for s in read_lines(results / "gpu-samples.jsonl") if "mem_used_mib" in s]
    rows, previous = [], None
    for update in sorted(ranks[0]):
        line = ranks[0][update]
        lines = [r[update] for r in ranks if update in r]
        wall = (line["elapsed_seconds"] - previous["elapsed_seconds"]) if previous else None
        decisions = line.get("global_step_decisions", line["step_decisions"])
        begin, end = stamps.get(update - 1), stamps.get(update)
        window = [s for s in samples if begin is not None and end is not None
                  and begin <= s["epoch"] <= end]
        row = dict(update=update, arm="on" if line.get("rollout_batch_snapshot_policies") else "off",
                   trim=bool(line.get("rollout_trim_cuda_cache")), ranks=len(lines),
                   wall_s=wall, decisions_per_sec_wall=decisions / wall if wall else None,
                   collect_s=line.get("global_collect_seconds", line["collect_seconds"]),
                   learn_s=line.get("global_learn_seconds", line["learn_seconds"]),
                   entropy=line.get("entropy"), approx_kl=line.get("approx_kl"),
                   retries_max=max((l.get("cuda_allocation_retries") or 0) for l in lines),
                   smi_samples=len(window),
                   smi_max_mib=max((s["mem_used_mib"] for s in window), default=None),
                   smi_mean_mib=mean(s["mem_used_mib"] for s in window),
                   sum_peak_allocated_mib=sum(l.get("cuda_peak_allocated_bytes") or 0
                                              for l in lines) / MIB,
                   sum_peak_reserved_mib=sum(l.get("cuda_peak_reserved_bytes") or 0
                                             for l in lines) / MIB,
                   sum_end_reserved_mib=sum(l.get("cuda_reserved_bytes") or 0 for l in lines) / MIB,
                   sum_end_allocated_mib=sum(l.get("cuda_allocated_bytes") or 0 for l in lines) / MIB,
                   trim_s_max_rank=max((l.get("cuda_trim_seconds") or 0.0) for l in lines),
                   trim_s_mean_rank=mean(l.get("cuda_trim_seconds") or 0.0 for l in lines),
                   resident_snapshots=line["population"]["resident_snapshots"],
                   mean_prefix=line.get("mean_prefix"))
        for point in POINTS:
            records = [l["cuda_trim"][point] for l in lines if point in (l.get("cuda_trim") or {})]
            if records:
                row[point] = dict(ranks=len(records),
                                  **{f"sum_{f}_mib": sum(r[f] for r in records) / MIB
                                     for f in FIELDS},
                                  seconds_max_rank=max(r["seconds"] for r in records))
        rows.append(row)
        previous = line
    return rows


def arm_stats(rows: list[dict]) -> dict:
    if not rows:
        return dict(updates=0)
    wall = [r["wall_s"] for r in rows if r["wall_s"]]
    decisions = sum(r["decisions_per_sec_wall"] * r["wall_s"] for r in rows if r["wall_s"])
    stats = dict(updates=len(rows), range=[rows[0]["update"], rows[-1]["update"]],
                 wall_s_per_update=mean(wall),
                 decisions_per_sec_wall=decisions / sum(wall) if wall else None,
                 collect_s=mean(r["collect_s"] for r in rows),
                 learn_s=mean(r["learn_s"] for r in rows),
                 entropy=mean(r["entropy"] for r in rows),
                 approx_kl_max=max((r["approx_kl"] or 0) for r in rows),
                 retries_max=max(r["retries_max"] for r in rows),
                 smi_max_mib=max((r["smi_max_mib"] or 0) for r in rows),
                 smi_mean_mib=mean(r["smi_mean_mib"] for r in rows),
                 sum_peak_allocated_mib_max=max(r["sum_peak_allocated_mib"] for r in rows),
                 sum_peak_reserved_mib_max=max(r["sum_peak_reserved_mib"] for r in rows),
                 trim_s_per_update_max_rank=mean(r["trim_s_max_rank"] for r in rows),
                 trim_s_per_update_mean_rank=mean(r["trim_s_mean_rank"] for r in rows))
    for point in POINTS:
        present = [r[point] for r in rows if point in r]
        if present:
            stats[point] = dict(updates=len(present),
                                **{f"mean_sum_{f}_mib": mean(p[f"sum_{f}_mib"] for p in present)
                                   for f in FIELDS},
                                mean_released_mib=mean(p["sum_reserved_before_mib"]
                                                       - p["sum_reserved_after_mib"]
                                                       for p in present),
                                seconds_max_rank_mean=mean(p["seconds_max_rank"] for p in present))
    return stats


def verdict(on: list[dict], off: list[dict]) -> dict:
    window = min(10, len(on) // 2)
    if window < 3:
        return dict(verdict="INCONCLUSIVE", reason=f"only {len(on)} settled on-updates")
    smi = [r["smi_max_mib"] for r in on]
    alloc = [r["sum_peak_allocated_mib"] for r in on]
    if sum(v is not None for v in smi) < 2 * window:
        return dict(verdict="INCONCLUSIVE", reason="missing nvidia-smi samples")
    first, last = on[:window], on[-window:]
    smi_rise = mean(r["smi_max_mib"] for r in last) - mean(r["smi_max_mib"] for r in first)
    alloc_rise = mean(r["sum_peak_allocated_mib"] for r in last) - mean(
        r["sum_peak_allocated_mib"] for r in first)
    smi_slope, alloc_slope = slope(smi), slope(alloc)
    excess = smi_rise - alloc_rise
    flat = excess <= EXCESS_LIMIT_MIB and (smi_slope or 0) - (alloc_slope or 0) <= SLOPE_LIMIT_MIB
    on_max = max(v for v in smi if v is not None)
    off_max = max((r["smi_max_mib"] or 0) for r in off) if off else None
    trim_s = mean(r["trim_s_max_rank"] for r in on)
    wall = mean(r["wall_s"] for r in on)
    return dict(verdict="FLAT" if flat else "STAIRCASE", window=window,
                smi_rise_mib=smi_rise, sum_peak_allocated_rise_mib=alloc_rise,
                excess_rise_mib=excess, smi_slope_mib_per_update=smi_slope,
                sum_peak_allocated_slope_mib_per_update=alloc_slope,
                on_smi_max_mib=on_max, off_smi_max_mib=off_max,
                on_minus_off_max_mib=on_max - off_max if off_max else None,
                trim_s_per_update=trim_s,
                trim_share_of_wall=trim_s / wall if trim_s is not None and wall else None,
                limits=dict(excess_rise_mib=EXCESS_LIMIT_MIB,
                            slope_excess_mib_per_update=SLOPE_LIMIT_MIB))


def summarize(results: Path, settle: int) -> dict:
    rows = update_rows(results)
    if not rows:
        return dict(error="no rank-0 metrics", host=host(results))
    arms = {name: [r for r in rows if r["arm"] == name] for name in ("off", "on")}
    settled = {name: rs[settle:] if len(rs) > settle else rs for name, rs in arms.items()}
    summary = dict(segment=SEGMENT, host=host(results), plan=read_json(results / "plan.json"),
                   settle=settle, first_update=rows[0]["update"], last_update=rows[-1]["update"],
                   ab_reference=AB_REFERENCE,
                   caveat="the off arm is the first 10 updates after a resume: histories and "
                          "resident snapshots are still growing, so its memory is below the "
                          "steady state; ab_reference gives the A/B's settled off/on maxima",
                   arm_off=arm_stats(settled["off"]), arm_on=arm_stats(settled["on"]),
                   arm_on_all=arm_stats(arms["on"]),
                   verdict=verdict(settled["on"], arms["off"]), updates=rows)
    return summary


def fmt(value, digits=1):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, list):
        return f"{value[0]}-{value[1]}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def verdict_line(v: dict, ab: dict) -> str:
    if v.get("verdict") == "INCONCLUSIVE":
        return f"**Verdict: INCONCLUSIVE** ({v['reason']})."
    share = v["trim_share_of_wall"]
    return (f"**Verdict: {v['verdict']}.** Over the settled on-updates (flag + trim), nvidia-smi "
            f"per-update max rose {fmt(v['smi_rise_mib'], 0)} MiB (last {v['window']} vs first "
            f"{v['window']}) while the summed peak allocated rose "
            f"{fmt(v['sum_peak_allocated_rise_mib'], 0)} MiB: excess {fmt(v['excess_rise_mib'], 0)} "
            f"MiB (limit {EXCESS_LIMIT_MIB}); slopes {fmt(v['smi_slope_mib_per_update'])} vs "
            f"{fmt(v['sum_peak_allocated_slope_mib_per_update'])} MiB/update. On-arm max "
            f"{fmt(v['on_smi_max_mib'], 0)} MiB vs this run's off-arm max "
            f"{fmt(v['off_smi_max_mib'], 0)} MiB ({fmt(v['on_minus_off_max_mib'], 0)} MiB; the off "
            f"arm is early after the resume) and vs the A/B's settled off max "
            f"{ab['off_smi_max_mib']:,} MiB / untrimmed on max {ab['on_smi_max_mib']:,} MiB "
            f"(machine {ab['machine_id']}). Trim cost {fmt(v['trim_s_per_update'], 3)} s per "
            f"update (slowest rank), {fmt(100 * share if share is not None else None, 2)}% of "
            "wall time.")


def markdown(s: dict) -> str:
    h = s.get("host") or {}
    head = (f"Machine {h.get('machine_id') or '-'} (offer {h.get('offer_id') or '-'}); CPU "
            f"{h.get('cpu_model') or '-'}, {h.get('usable_cpus') or '-'} usable CPUs; "
            f"GPU {h.get('gpu') or '-'}.")
    if "error" in s:
        return f"# Allocator trim check\n\n{head}\n\nNo summary: {s['error']}.\n"
    out = [f"# Allocator trim check, updates {s['first_update']}-{s['last_update']}", "", head, "",
           verdict_line(s["verdict"], s["ab_reference"]), "",
           f"Schedule `{s['plan'].get('schedule', '?')}`; first {s['settle']} updates of each arm "
           f"dropped from the per-arm figures. Caveat: {s['caveat']}.", "",
           "## Per arm (settled)", "", "| | off | on (flag + trim) |", "|---|---:|---:|"]
    for key, label, digits in (
            ("updates", "updates", 0), ("wall_s_per_update", "wall s/update", 2),
            ("decisions_per_sec_wall", "decisions/s (wall)", 0),
            ("collect_s", "collect s (slowest rank)", 2), ("learn_s", "learn s (slowest rank)", 2),
            ("smi_max_mib", "nvidia-smi max MiB", 0), ("smi_mean_mib", "nvidia-smi mean MiB", 0),
            ("sum_peak_allocated_mib_max", "sum peak allocated MiB (max)", 0),
            ("sum_peak_reserved_mib_max", "sum peak reserved MiB (max)", 0),
            ("trim_s_per_update_max_rank", "trim s/update (slowest rank)", 3),
            ("entropy", "entropy", 4), ("approx_kl_max", "approx KL max", 4),
            ("retries_max", "allocation retries (max)", 0)):
        out.append(f"| {label} | {fmt(s['arm_off'].get(key), digits)} | "
                   f"{fmt(s['arm_on'].get(key), digits)} |")
    out += ["", "## Trim points, on arm (sums over ranks, MiB, mean over settled updates)", "",
            "| point | updates | reserved before | reserved after | released | allocated before "
            "| allocated after | s (slowest rank) |", "|---|" + "---:|" * 7]
    for point in POINTS:
        p = s["arm_on_all" if point == "graphs_released" else "arm_on"].get(point)
        if p:
            out.append(f"| {point} | {p['updates']} | {fmt(p['mean_sum_reserved_before_mib'], 0)} | "
                       f"{fmt(p['mean_sum_reserved_after_mib'], 0)} | "
                       f"{fmt(p['mean_released_mib'], 0)} | "
                       f"{fmt(p['mean_sum_allocated_before_mib'], 0)} | "
                       f"{fmt(p['mean_sum_allocated_after_mib'], 0)} | "
                       f"{fmt(p['seconds_max_rank_mean'], 3)} |")
    out += ["", "## Per update", "",
            "| update | arm | wall s | dec/s | collect s | learn s | smi max | smi mean | "
            "sum peak alloc | sum peak res | res before trim (learn) | res after | trim s | "
            "snapshots | entropy | KL | retries |", "|---:|---|" + "---:|" * 15]
    for r in s["updates"]:
        learn = r.get("after_learn") or {}
        out.append(f"| {r['update']} | {r['arm']} | {fmt(r['wall_s'], 2)} | "
                   f"{fmt(r['decisions_per_sec_wall'], 0)} | {fmt(r['collect_s'], 2)} | "
                   f"{fmt(r['learn_s'], 2)} | {fmt(r['smi_max_mib'], 0)} | "
                   f"{fmt(r['smi_mean_mib'], 0)} | {fmt(r['sum_peak_allocated_mib'], 0)} | "
                   f"{fmt(r['sum_peak_reserved_mib'], 0)} | "
                   f"{fmt(learn.get('sum_reserved_before_mib'), 0)} | "
                   f"{fmt(learn.get('sum_reserved_after_mib'), 0)} | "
                   f"{fmt(r['trim_s_max_rank'], 3)} | {fmt(r['resident_snapshots'])} | "
                   f"{fmt(r['entropy'], 4)} | {fmt(r['approx_kl'], 4)} | {fmt(r['retries_max'])} |")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--settle", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None, help="default: RESULTS_DIR")
    args = parser.parse_args()
    out = args.out or args.results
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(args.results, args.settle)
    (out / "trim-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = markdown(summary)
    (out / "trim-summary.md").write_text(text)
    print(text)
    return 0 if "error" not in summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
