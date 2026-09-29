"""A/B summary of merged snapshot inference: ab-summary.json and ab-summary.md.

    python summarize.py RESULTS_DIR [--settle 2] [--out DIR]

RESULTS_DIR is the pod's /workspace/results (locally: download/results). Every
rank-0 update after the warmup belongs to a block: a maximal run of updates
with the same arm (``rollout_batch_snapshot_policies``) and profile flag. The
first SETTLE updates of each block are dropped from the steady-state figures
(the arm switch clears or re-admits snapshot private graphs and allocates or
frees the stacked heads). Unprofiled blocks give speed; the profiled blocks
give the call shapes and the learner/snapshot phase split (synchronized: not
absolute speed). Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SEGMENT = "snapshot-batching-diag"
SPLIT = ("public_cache_or_collation", "actor_and_sampling")
GROUPS = ("learner", "snapshot")


def read_lines(path: Path) -> list[dict]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except OSError:
        return []


def rank_dirs(output: Path) -> list[Path]:
    ranks, r = [output], 1
    while (output / f"rank-{r}" / "metrics.jsonl").exists():
        ranks.append(output / f"rank-{r}")
        r += 1
    return ranks


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def quantile(histogram: dict[int, int], q: float):
    total = sum(histogram.values())
    if not total:
        return None
    rank, seen = max(1, math.ceil(q * total)), 0
    for value in sorted(histogram):
        seen += histogram[value]
        if seen >= rank:
            return value
    return max(histogram)


def gpu(results: Path, first: int, last: int) -> dict:
    stamps = {r["update"]: r["epoch"] for r in read_lines(results / "update-epochs.jsonl")}
    begin, end = stamps.get(first - 1), stamps.get(last)
    if begin is None or end is None:
        return dict(gpu_samples=0)
    samples = [s for s in read_lines(results / "gpu-samples.jsonl")
               if "util" in s and begin <= s["epoch"] <= end]
    if not samples:
        return dict(gpu_samples=0)
    return dict(gpu_samples=len(samples), gpu_util_mean=mean(s["util"] for s in samples),
                gpu_mem_used_mib_max=max(s["mem_used_mib"] for s in samples))


def block_stats(results: Path, lines: list[dict], ranks: list[dict[int, dict]],
                previous: dict[int, dict]) -> dict:
    """Speed and health of consecutive rank-0 lines, plus every rank's counters."""
    first, last = lines[0]["update"], lines[-1]["update"]
    before = previous.get(first - 1)
    wall = lines[-1]["elapsed_seconds"] - before["elapsed_seconds"] if before else None
    decisions = sum(l.get("global_step_decisions", l["step_decisions"]) for l in lines)
    updates = [l["update"] for l in lines]
    per_rank = [[r[u] for u in updates if u in r] for r in ranks]
    calls = [l["collection_policy_batches"] / l["collection_steps"]
             for rows in per_rank for l in rows if l.get("collection_steps")]
    retries = [l.get("cuda_allocation_retries") or 0 for rows in per_rank for l in rows]
    return dict(
        updates=[first, last], count=len(lines),
        wall_s_per_update=wall / len(lines) if wall else None,
        decisions_per_sec_wall=decisions / wall if wall else None,
        collect_s_slowest_rank=mean(l.get("global_collect_seconds", l["collect_seconds"])
                                    for l in lines),
        learn_s_slowest_rank=mean(l.get("global_learn_seconds", l["learn_seconds"])
                                  for l in lines),
        policy_calls_per_step_rank_mean=mean(calls),
        entropy=mean(l.get("entropy") for l in lines),
        approx_kl_mean=mean(l.get("approx_kl") for l in lines),
        approx_kl_max=max((l.get("approx_kl") or 0) for l in lines),
        allocation_retries_max=max(retries) if retries else None,
        cuda_peak_reserved_gb_max=max((l.get("cuda_peak_reserved_bytes") or 0)
                                      for rows in per_rank for l in rows) / 1e9,
        snapshot_heads_mb_max=max(((l.get("snapshot_heads") or {}).get("bytes", 0))
                                  for rows in per_rank for l in rows) / 2**20,
        resident_snapshots=[min(l["population"]["resident_snapshots"] for l in lines),
                            max(l["population"]["resident_snapshots"] for l in lines)],
        mean_prefix=mean(l["mean_prefix"] for l in lines),
        **gpu(results, first, last))


def profile_stats(lines_per_rank: list[list[dict]]) -> dict:
    """Call shapes and learner/snapshot phase split of profiled updates, all ranks."""
    histograms = {g: {} for g in GROUPS}
    steps = 0
    split = {g: {p: [] for p in SPLIT} for g in GROUPS}
    collect = []
    for rows in lines_per_rank:
        for l in rows:
            steps += l.get("collection_steps") or 0
            collect.append(l["collect_seconds"])
            for g in GROUPS:
                for size, count in l.get("collection_policy_call_rows", {}).get(g, {}).items():
                    histograms[g][int(size)] = histograms[g].get(int(size), 0) + int(count)
                for p in SPLIT:
                    split[g][p].append(l.get("collection_group_phase_seconds", {}).get(g, {}).get(p, 0.0))
    shapes = {}
    for g in GROUPS:
        h = histograms[g]
        calls = sum(h.values())
        rows = sum(s * c for s, c in h.items())
        shapes[g] = dict(calls_per_step=calls / steps if steps else None,
                         mean_rows=rows / calls if calls else None,
                         p10_rows=quantile(h, 0.1), median_rows=quantile(h, 0.5),
                         p90_rows=quantile(h, 0.9), max_rows=max(h) if h else None,
                         **{f"{p}_s_per_update": mean(split[g][p]) for p in SPLIT})
    return dict(collect_s_rank_mean_synchronized=mean(collect), call_shapes=shapes,
                call_rows_histogram={g: {str(k): v for k, v in sorted(h.items())}
                                     for g, h in histograms.items()})


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def host(results: Path) -> dict:
    """Machine and host facts of the run (machine.json from remote_start.sh,
    host-facts.json from infra.cpu_budget). Absolute speed is only comparable
    within one machine; the on/off ratio is the in-run comparison."""
    machine, facts = read_json(results / "machine.json"), read_json(results / "host-facts.json")
    return dict(machine_id=machine.get("machine_id"), offer_id=machine.get("offer_id"),
                cpu_model=facts.get("cpu_model"), cpu_count=facts.get("cpu_count"),
                usable_cpus=facts.get("usable_cpus"), affinity_cpus=facts.get("affinity_cpus"),
                cgroup_quota_cpus=facts.get("cgroup_quota_cpus"), gpu=facts.get("gpu"))


def summarize(results: Path, settle: int) -> dict:
    plan = json.loads((results / "plan.json").read_text()) if (results / "plan.json").exists() else {}
    output = results / "segments" / SEGMENT
    ranks = [{l["update"]: l for l in read_lines(d / "metrics.jsonl")} for d in rank_dirs(output)]
    base = [ranks[0][u] for u in sorted(ranks[0])]
    if not base:
        return dict(error="no rank-0 metrics", host=host(results))
    start = plan.get("resume_update", base[0]["update"] - 1) + plan.get("warmup", 25)
    blocks, current = [], []
    for line in base:
        if line["update"] <= start:
            continue
        key = (bool(line.get("rollout_batch_snapshot_policies")),
               bool(line.get("collection_profile_synchronized")))
        if current and key != current[0][0]:
            blocks.append(current)
            current = []
        current.append((key, line))
    if current:
        blocks.append(current)
    previous = ranks[0]
    records, arms = [], {False: [], True: []}
    for block in blocks:
        (arm, profiled), lines = block[0][0], [l for _, l in block]
        settled = lines[settle:] if len(lines) > settle else lines
        record = dict(arm="on" if arm else "off", profiled=profiled,
                      all=block_stats(results, lines, ranks, previous),
                      settled=block_stats(results, settled, ranks, previous))
        if profiled:
            record["profile"] = profile_stats([[r[l["update"]] for l in settled if l["update"] in r]
                                               for r in ranks])
        else:
            arms[arm].append(record["settled"])
        records.append(record)
    summary = dict(segment=SEGMENT, host=host(results), ranks=len(ranks), plan=plan, settle=settle,
                   first_update=base[0]["update"], last_update=base[-1]["update"],
                   caveat="profiled blocks synchronize CUDA at every phase boundary; their "
                          "seconds attribute time, speed comes from unprofiled blocks",
                   blocks=records)
    for arm, name in ((False, "off"), (True, "on")):
        summary[f"arm_{name}"] = dict(
            blocks=len(arms[arm]),
            wall_s_per_update=mean(b["wall_s_per_update"] for b in arms[arm]),
            decisions_per_sec_wall=mean(b["decisions_per_sec_wall"] for b in arms[arm]),
            collect_s_slowest_rank=mean(b["collect_s_slowest_rank"] for b in arms[arm]),
            learn_s_slowest_rank=mean(b["learn_s_slowest_rank"] for b in arms[arm]),
            policy_calls_per_step=mean(b["policy_calls_per_step_rank_mean"] for b in arms[arm]),
            gpu_util_mean=mean(b.get("gpu_util_mean") for b in arms[arm]),
            gpu_mem_used_mib_max=max((b.get("gpu_mem_used_mib_max") or 0) for b in arms[arm])
            if arms[arm] else None,
            entropy=mean(b["entropy"] for b in arms[arm]),
            approx_kl_max=max((b["approx_kl_max"] for b in arms[arm]), default=None),
            allocation_retries_max=max((b["allocation_retries_max"] or 0 for b in arms[arm]),
                                       default=None))
    off, on = summary["arm_off"], summary["arm_on"]
    if off["decisions_per_sec_wall"] and on["decisions_per_sec_wall"]:
        summary["speedup_decisions_per_sec_wall"] = (on["decisions_per_sec_wall"]
                                                     / off["decisions_per_sec_wall"])
    return summary


def fmt(value, digits=2):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, list):
        return f"{value[0]}-{value[1]}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def host_line(h: dict) -> str:
    return (f"Machine {h.get('machine_id') or '-'} (offer {h.get('offer_id') or '-'}); CPU "
            f"{h.get('cpu_model') or '-'}, {fmt(h.get('usable_cpus'))} usable of "
            f"{fmt(h.get('cpu_count'))} logical CPUs (cgroup quota {fmt(h.get('cgroup_quota_cpus'), 1)}); "
            f"GPU {h.get('gpu') or '-'}. Absolute speed is comparable only on the same machine; "
            "the on/off ratio is the in-run comparison.")


def markdown(s: dict) -> str:
    if "error" in s:
        return (f"# Merged snapshot inference A/B\n\n{host_line(s.get('host') or {})}\n\n"
                f"No summary: {s['error']}.\n")
    out = [f"# Merged snapshot inference A/B, updates {s['first_update']}-{s['last_update']}", "",
           host_line(s["host"]), "",
           f"{s['ranks']} ranks; schedule `{s['plan'].get('schedule', '?')}`; first "
           f"{s['settle']} updates of each block dropped. {s['caveat']}.", "",
           "## Per arm (unprofiled blocks, settled)", "", "| | off | on |", "|---|---:|---:|"]
    for key, label in (("blocks", "blocks"), ("wall_s_per_update", "wall s/update"),
                       ("decisions_per_sec_wall", "decisions/s (wall)"),
                       ("collect_s_slowest_rank", "collect s (slowest rank)"),
                       ("learn_s_slowest_rank", "learn s (slowest rank)"),
                       ("policy_calls_per_step", "policy calls/step (rank mean)"),
                       ("gpu_util_mean", "GPU util %"), ("gpu_mem_used_mib_max", "GPU mem max MiB"),
                       ("entropy", "entropy"), ("approx_kl_max", "approx KL max"),
                       ("allocation_retries_max", "allocation retries (max)")):
        digits = 4 if key in ("entropy", "approx_kl_max") else 2
        out.append(f"| {label} | {fmt(s['arm_off'].get(key), digits)} | "
                   f"{fmt(s['arm_on'].get(key), digits)} |")
    if "speedup_decisions_per_sec_wall" in s:
        out += ["", f"on / off decisions per wall second: **{s['speedup_decisions_per_sec_wall']:.3f}**"]
    out += ["", "## Blocks (settled)", "",
            "| arm | profiled | updates | wall s/upd | dec/s wall | collect s | learn s | calls/step "
            "| GPU % | GPU MiB | peak GB | heads MB | retries | entropy | KL max |",
            "|---|---|---|" + "---:|" * 12]
    for b in s["blocks"]:
        t = b["settled"]
        out.append(f"| {b['arm']} | {b['profiled']} | {fmt(t['updates'])} | "
                   f"{fmt(t['wall_s_per_update'])} | {fmt(t['decisions_per_sec_wall'], 0)} | "
                   f"{fmt(t['collect_s_slowest_rank'])} | {fmt(t['learn_s_slowest_rank'])} | "
                   f"{fmt(t['policy_calls_per_step_rank_mean'])} | {fmt(t.get('gpu_util_mean'), 1)} | "
                   f"{fmt(t.get('gpu_mem_used_mib_max'), 0)} | {fmt(t['cuda_peak_reserved_gb_max'])} | "
                   f"{fmt(t['snapshot_heads_mb_max'], 1)} | {fmt(t['allocation_retries_max'])} | "
                   f"{fmt(t['entropy'], 4)} | {fmt(t['approx_kl_max'], 4)} |")
    profiled = [b for b in s["blocks"] if b.get("profile")]
    if profiled:
        out += ["", "## Profiled blocks (synchronized; all ranks)", "",
                "| arm | group | calls/step | mean rows | p10 | median | p90 | max | cache s/upd "
                "| actor s/upd |", "|---|---|" + "---:|" * 8]
        for b in profiled:
            for g in GROUPS:
                c = b["profile"]["call_shapes"][g]
                out.append(f"| {b['arm']} | {g} | {fmt(c['calls_per_step'])} | "
                           f"{fmt(c['mean_rows'], 1)} | {fmt(c['p10_rows'])} | "
                           f"{fmt(c['median_rows'])} | {fmt(c['p90_rows'])} | {fmt(c['max_rows'])} | "
                           f"{fmt(c['public_cache_or_collation_s_per_update'])} | "
                           f"{fmt(c['actor_and_sampling_s_per_update'])} |")
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
    (out / "ab-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = markdown(summary)
    (out / "ab-summary.md").write_text(text)
    print(text)
    return 0 if "error" not in summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
