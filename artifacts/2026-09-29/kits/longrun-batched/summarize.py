"""Overnight run summary: longrun-summary.json and longrun-summary.md.

    python summarize.py RESULTS_DIR [--window 31] [--out DIR]

RESULTS_DIR is the pod's /workspace/results (locally: download/results). A crash
repeats the updates after the last checkpoint; for every update number the last
occurrence (latest attempt) counts. Windows of WINDOW updates (one checkpoint
interval, ~2M decisions): decisions/s from wall time (global decisions over the
orchestrator's wall-clock stamps, restart downtime included), collect/learn
seconds (slowest rank), resident snapshots, mean prefix, entropy, approx KL,
nvidia-smi max memory, summed reserved before/after the after-learn trim, trim
seconds, and the restarts/fallbacks logged in the window. Totals: updates,
decisions added, lineage decisions at the end, cost (create -> now or teardown at
the rented offer's price, from deadlines.env). Standard library only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

SEGMENT = "main-overnight"
INIT_UPDATE, INIT_DECISIONS = 844, 55_312_384
MIB = 2 ** 20


def read_lines(path: Path) -> list[dict]:
    out = []
    try:
        for raw in path.read_text().splitlines():
            try:
                out.append(json.loads(raw))
            except ValueError:
                pass
    except OSError:
        pass
    return out


def read_env(path: Path) -> dict:
    values = {}
    try:
        for raw in path.read_text().splitlines():
            key, sep, value = raw.partition("=")
            if sep:
                values[key.strip()] = value.strip()
    except OSError:
        pass
    return values


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def last_by_update(lines: list[dict]) -> dict[int, dict]:
    return {line["update"]: line for line in lines if "update" in line}


def summarize(results: Path, window: int, env: dict) -> dict:
    output = results / "segments" / SEGMENT
    ranks = [last_by_update(read_lines(output / "metrics.jsonl"))]
    r = 1
    while (output / f"rank-{r}" / "metrics.jsonl").exists():
        ranks.append(last_by_update(read_lines(output / f"rank-{r}" / "metrics.jsonl")))
        r += 1
    stamps = {e["update"]: e["epoch"] for e in read_lines(results / "update-epochs.jsonl")}
    events = read_lines(results / "orchestrator.jsonl")
    samples = [s for s in read_lines(results / "gpu-samples.jsonl") if "mem_used_mib" in s]
    updates = sorted(ranks[0])
    if not updates:
        return dict(error="no rank-0 metrics", events=[e["event"] for e in events][-20:])
    starts = [e for e in events if e["event"] == "train_start"]
    windows = []
    for begin in range(0, len(updates), window):
        chunk = updates[begin:begin + window]
        lines = [ranks[0][u] for u in chunk]
        all_lines = [rk[u] for rk in ranks for u in chunk if u in rk]
        t0 = stamps.get(chunk[0] - 1) or (starts[0]["time"] if starts else None)
        t1 = stamps.get(chunk[-1])
        decisions = sum(l.get("global_step_decisions", l.get("step_decisions", 0)) for l in lines)
        wall = t1 - t0 if t0 and t1 and t1 > t0 else None
        smi = [s["mem_used_mib"] for s in samples if t0 and t1 and t0 <= s["epoch"] <= t1]
        trims = [l["cuda_trim"]["after_learn"] for l in all_lines
                 if "after_learn" in (l.get("cuda_trim") or {})]
        in_window = [e for e in events if t0 and t1 and t0 <= e["time"] <= t1]
        windows.append(dict(
            updates=[chunk[0], chunk[-1]], count=len(chunk),
            decisions_per_sec_wall=decisions / wall if wall else None, wall_seconds=wall,
            collect_s=mean(l.get("global_collect_seconds", l.get("collect_seconds")) for l in lines),
            learn_s=mean(l.get("global_learn_seconds", l.get("learn_seconds")) for l in lines),
            resident_snapshots=[min(l["population"]["resident_snapshots"] for l in lines),
                                max(l["population"]["resident_snapshots"] for l in lines)],
            mean_prefix=mean(l.get("mean_prefix") for l in lines),
            entropy=mean(l.get("entropy") for l in lines),
            approx_kl_max=max((l.get("approx_kl") or 0) for l in lines),
            smi_max_mib=max(smi) if smi else None,
            batched=sum(bool(l.get("rollout_batch_snapshot_policies")) for l in lines),
            sum_reserved_before_trim_mib=(sum(t["reserved_before"] for t in trims) / MIB
                                          / max(1, len(chunk))) if trims else None,
            sum_reserved_after_trim_mib=(sum(t["reserved_after"] for t in trims) / MIB
                                         / max(1, len(chunk))) if trims else None,
            trim_s_per_update=mean(max((rk[u].get("cuda_trim_seconds") or 0.0) for rk in ranks
                                       if u in rk) for u in chunk),
            retries_max=max((l.get("cuda_allocation_retries") or 0) for l in all_lines),
            restarts=sum(e["event"] == "restart" for e in in_window),
            fallbacks=sum(e["event"] == "fallback" for e in in_window)))
    last = ranks[0][updates[-1]]
    end_decisions = last.get("global_decisions", last.get("decisions"))
    main_w4 = updates[0] == INIT_UPDATE + 1     # else a smoke or other checkpoint
    created = float(env.get("CREATED_EPOCH", 0) or 0)
    dph = float(env.get("KIT_DPH_TOTAL", 0) or 0)
    ended = max([e["time"] for e in events] + [time.time() if not events else 0])
    totals = dict(
        mode=env.get("KIT_MODE"), machine_id=env.get("KIT_MACHINE_ID"),
        first_update=updates[0], last_update=updates[-1], updates=updates[-1] - updates[0] + 1,
        lineage_decisions_end=end_decisions,
        decisions_added=end_decisions - INIT_DECISIONS if end_decisions and main_w4 else None,
        restarts=sum(e["event"] == "restart" for e in events),
        fallbacks=[e for e in events if e["event"] == "fallback"],
        stop_reason=next((e.get("reason") for e in reversed(events) if e["event"] == "run_done"), None),
        train_hours=((stamps.get(updates[-1], 0) - starts[0]["time"]) / 3600) if starts else None,
        pod_hours_to_last_event=(ended - created) / 3600 if created else None,
        estimated_cost_usd=(ended - created) / 3600 * dph if created and dph else None,
        cost_scope="create -> last orchestrator event at the offer's hourly price; the local "
                   "teardown-summary.json has the billed-to-destroy estimate")
    return dict(segment=SEGMENT, window=window, totals=totals, windows=windows)


def fmt(value, digits=1):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, list):
        return f"{value[0]}-{value[1]}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def markdown(s: dict) -> str:
    if "error" in s:
        return f"# Overnight run\n\nNo summary: {s['error']}.\n"
    t = s["totals"]
    out = [f"# Overnight run, main lineage, updates {t['first_update']}-{t['last_update']}", "",
           f"Mode `{t['mode']}`, machine {t['machine_id']}. {t['updates']} updates, "
           f"{fmt(t['decisions_added'])} decisions added, lineage at {fmt(t['lineage_decisions_end'])} "
           f"decisions. Restarts {t['restarts']}; fallbacks {len(t['fallbacks'])}; stop reason: "
           f"{t['stop_reason']}. Training {fmt(t['train_hours'], 2)} h; estimated cost "
           f"${fmt(t['estimated_cost_usd'], 2)} ({t['cost_scope']}).", "",
           "| updates | dec/s wall | collect s | learn s | snapshots | prefix | entropy | KL max "
           "| smi max MiB | res before trim | res after trim | trim s | batched | retries | restarts "
           "| fallbacks |", "|---|" + "---:|" * 15]
    for w in s["windows"]:
        out.append(f"| {fmt(w['updates'])} | {fmt(w['decisions_per_sec_wall'], 0)} | "
                   f"{fmt(w['collect_s'], 2)} | {fmt(w['learn_s'], 2)} | "
                   f"{fmt(w['resident_snapshots'])} | {fmt(w['mean_prefix'], 0)} | "
                   f"{fmt(w['entropy'], 4)} | {fmt(w['approx_kl_max'], 4)} | "
                   f"{fmt(w['smi_max_mib'], 0)} | {fmt(w['sum_reserved_before_trim_mib'], 0)} | "
                   f"{fmt(w['sum_reserved_after_trim_mib'], 0)} | {fmt(w['trim_s_per_update'], 3)} | "
                   f"{w['batched']}/{w['count']} | {w['retries_max']} | {w['restarts']} | "
                   f"{w['fallbacks']} |")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path)
    parser.add_argument("--window", type=int, default=31)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--env", type=Path, default=None, help="deadlines.env (default: pod kit)")
    args = parser.parse_args()
    env_path = args.env or Path("/workspace/kit/deadlines.env")
    env = read_env(env_path) or read_env(args.results / "deadlines.env")
    out = args.out or args.results
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(args.results, args.window, env)
    (out / "longrun-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = markdown(summary)
    (out / "longrun-summary.md").write_text(text)
    print(text)
    return 0 if "error" not in summary else 1


if __name__ == "__main__":
    raise SystemExit(main())
