"""Memory analysis of the merged-snapshot A/B (updates 845-919), downloaded data only.

    python analyze_memory.py            # writes tables.md, per_update.csv, analysis.json, timeline.svg

Inputs: ../download/results/{gpu-samples,update-epochs}.jsonl, plan.json and the
four ranks' metrics.jsonl. Standard library + numpy. No GPU, no network.

Conventions
- MiB = 2**20 bytes. nvidia-smi memory.used is MiB.
- Block = maximal run of rank-0 updates after the 25-update warmup with the same
  (arm, profiled) key; "settled" drops the first 2 updates (same as summarize.py).
- Update u spans (T[u-1], T[u]] where T[u] = offset + elapsed_seconds[u] of rank 0;
  offset = min_u(stamp_u - elapsed_u) + 1.0 s (stamps are polled every 2 s, so
  the true offset lies in [min, min + 2.02]; +-1 s alignment error).
  Within an update: learn = (T[u] - learn_seconds, T[u]], collect before it.
- "captures in update" per rank = increase of the private graph capture counters
  summed over identities (a counter that appears or restarts counts from 0).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "download" / "results"
SEG = RESULTS / "segments" / "snapshot-batching-diag"
MIB = 2 ** 20
SETTLE = 2


def lines(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load():
    ranks = [lines(SEG / "metrics.jsonl")] + [lines(SEG / f"rank-{r}" / "metrics.jsonl") for r in (1, 2, 3)]
    ranks = [{l["update"]: l for l in r} for r in ranks]
    samples = [s for s in lines(RESULTS / "gpu-samples.jsonl") if "mem_used_mib" in s]
    stamps = {r["update"]: r["epoch"] for r in lines(RESULTS / "update-epochs.jsonl")}
    plan = json.loads((RESULTS / "plan.json").read_text())
    return ranks, samples, stamps, plan


def graph_totals(line: dict) -> dict:
    pg = line.get("private_graphs") or {}
    return dict(ids=len(pg),
                snap_ids=sum(1 for k in pg if k != "0"),
                bytes=sum(v.get("bytes", 0) for v in pg.values()),
                pool=sum(v.get("pool_bytes", 0) for v in pg.values()),
                entries=sum(v.get("entries", 0) for v in pg.values()),
                learner_pool=(pg.get("0") or {}).get("pool_bytes", 0),
                captures={k: v.get("captures", 0) for k, v in pg.items()})


def per_update_rows(ranks, stamps):
    base = ranks[0]
    updates = sorted(base)
    offsets = [stamps[u] - base[u]["elapsed_seconds"] for u in updates if u in stamps]
    offset = min(offsets) + 1.0
    rows = []
    prev_caps = [dict() for _ in ranks]
    for u in updates:
        row = dict(update=u, arm=int(bool(base[u]["rollout_batch_snapshot_policies"])),
                   profiled=int(bool(base[u]["collection_profile_synchronized"])),
                   t_end=offset + base[u]["elapsed_seconds"],
                   collect_s=base[u]["collect_seconds"], learn_s=base[u]["learn_seconds"],
                   gcollect_s=base[u].get("global_collect_seconds"),
                   glearn_s=base[u].get("global_learn_seconds"))
        for r, rank in enumerate(ranks):
            l = rank[u]
            g = graph_totals(l)
            caps = 0
            for k, c in g["captures"].items():
                p = prev_caps[r].get(k, 0)
                caps += c - p if c >= p else c
            prev_caps[r] = g["captures"]
            row.update({
                f"r{r}_peak_alloc": l["cuda_peak_allocated_bytes"] / MIB,
                f"r{r}_peak_res": l["cuda_peak_reserved_bytes"] / MIB,
                f"r{r}_alloc": l["cuda_allocated_bytes"] / MIB,
                f"r{r}_res": l["cuda_reserved_bytes"] / MIB,
                f"r{r}_inact_peak": l["cuda_inactive_split_peak_bytes"] / MIB,
                f"r{r}_retries": l.get("cuda_allocation_retries") or 0,
                f"r{r}_cache": l["cache_bytes"] / MIB,
                f"r{r}_cache_entries": l["cache"]["entries"],
                f"r{r}_ccache": l["collection_cache"]["bytes"] / MIB,
                f"r{r}_ccache_entries": l["collection_cache"]["entries"],
                f"r{r}_graph_bytes": g["bytes"] / MIB, f"r{r}_graph_pool": g["pool"] / MIB,
                f"r{r}_graph_ids": g["ids"], f"r{r}_graph_entries": g["entries"],
                f"r{r}_captures": caps,
                f"r{r}_heads": ((l.get("snapshot_heads") or {}).get("bytes", 0)) / MIB,
                f"r{r}_resident": l["population"]["resident_snapshots"],
                f"r{r}_mean_prefix": l["mean_prefix"], f"r{r}_max_prefix": l["max_prefix"],
                f"r{r}_store_tokens": l["store_tokens"],
            })
        for key in ("peak_alloc", "peak_res", "alloc", "res", "cache", "ccache", "graph_bytes",
                    "graph_pool", "heads", "captures", "resident", "store_tokens"):
            row[f"sum_{key}"] = sum(row[f"r{r}_{key}"] for r in range(len(ranks)))
        row["sum_mean_prefix"] = float(np.mean([row[f"r{r}_mean_prefix"] for r in range(len(ranks))]))
        row["sum_res_minus_alloc_end"] = row["sum_res"] - row["sum_alloc"]
        row["sum_peakres_minus_res_end"] = row["sum_peak_res"] - row["sum_res"]
        row["ranks_trimmed"] = sum(1 for r in range(len(ranks))
                                   if row[f"r{r}_res"] < row[f"r{r}_peak_res"] - 0.5)
        rows.append(row)
    for i, row in enumerate(rows):
        row["t_start"] = rows[i - 1]["t_end"] if i else row["t_end"] - row["collect_s"] - row["learn_s"]
    return rows, offset


def attach_smi(rows, samples):
    for row in rows:
        inside = [s for s in samples if row["t_start"] < s["epoch"] <= row["t_end"]]
        row["smi_n"] = len(inside)
        vals = [s["mem_used_mib"] for s in inside]
        row["smi_max"] = max(vals) if vals else None
        row["smi_min"] = min(vals) if vals else None
        row["smi_mean"] = float(np.mean(vals)) if vals else None
        # nearest sample to the update end (reserved at end is known exactly there)
        near = min(samples, key=lambda s: abs(s["epoch"] - row["t_end"]))
        row["smi_end"] = near["mem_used_mib"]
        row["smi_end_dt"] = near["epoch"] - row["t_end"]
        # Lower bound on memory outside the allocator during the update:
        # smi(t) = outside + sum_r reserved_r(t) <= outside + sum_r peak_reserved_r.
        row["gap_lb"] = row["smi_max"] - row["sum_peak_res"] if vals else None
        # End-of-update estimate: nearest sample minus sum of end reserved.
        row["gap_end"] = row["smi_end"] - row["sum_res"]


def blocks_of(rows, plan):
    start = plan["resume_update"] + plan["warmup"]
    out, cur = [], []
    for row in rows:
        if row["update"] <= start:
            continue
        key = (row["arm"], row["profiled"])
        if cur and key != (cur[0]["arm"], cur[0]["profiled"]):
            out.append(cur)
            cur = []
        cur.append(row)
    if cur:
        out.append(cur)
    return out


def phase_of(sample_t, rows):
    for row in rows:
        if row["t_start"] < sample_t <= row["t_end"]:
            return row, ("learn" if sample_t > row["t_end"] - row["learn_s"] else "collect")
    return None, None


def fmt(v, d=0):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.{d}f}"
    return f"{v:,}"


def main() -> int:
    ranks, samples, stamps, plan = load()
    rows, offset = per_update_rows(ranks, stamps)
    attach_smi(rows, samples)
    blocks = blocks_of(rows, plan)
    nr = len(ranks)
    out = []
    A = out.append
    analysis = dict(offset=offset, blocks=[])

    # ---- Q1: per block per rank
    A("## Q1. Per block, per rank (settled updates; MiB)\n")
    A("Values: max over the block's settled updates of per-update peak allocated / peak reserved; "
      "mean end-of-update allocated / reserved; max inactive-split peak; mean cache (post-learn) and "
      "collection-cache (post-collect) bytes; mean private-graph pool bytes (all identities); heads.\n")
    A("| block | arm | updates | rank | peak alloc max | peak res max | end alloc mean | end res mean | "
      "end res max | inact split peak max | cache mean | coll-cache mean | graph ids | graph pool mean | heads |")
    A("|---|---|---|---|" + "---:|" * 11)
    for bi, b in enumerate(blocks):
        s = b[SETTLE:] if len(b) > SETTLE else b
        rec = dict(index=bi, arm="on" if s[0]["arm"] else "off", profiled=bool(s[0]["profiled"]),
                   updates=[s[0]["update"], s[-1]["update"]], ranks=[])
        for r in range(nr):
            k = lambda n: [x[f"r{r}_{n}"] for x in s]
            rr = dict(peak_alloc_max=max(k("peak_alloc")), peak_res_max=max(k("peak_res")),
                      alloc_mean=float(np.mean(k("alloc"))), res_mean=float(np.mean(k("res"))),
                      res_max=max(k("res")), inact_max=max(k("inact_peak")),
                      cache_mean=float(np.mean(k("cache"))), ccache_mean=float(np.mean(k("ccache"))),
                      graph_ids=f"{min(k('graph_ids'))}-{max(k('graph_ids'))}",
                      graph_pool_mean=float(np.mean(k("graph_pool"))), heads=max(k("heads")))
            rec["ranks"].append(rr)
            A(f"| {bi} | {rec['arm']}{'*' if rec['profiled'] else ''} | {s[0]['update']}-{s[-1]['update']} | {r} | "
              f"{fmt(rr['peak_alloc_max'])} | {fmt(rr['peak_res_max'])} | {fmt(rr['alloc_mean'])} | "
              f"{fmt(rr['res_mean'])} | {fmt(rr['res_max'])} | {fmt(rr['inact_max'])} | {fmt(rr['cache_mean'])} | "
              f"{fmt(rr['ccache_mean'])} | {rr['graph_ids']} | {fmt(rr['graph_pool_mean'])} | {fmt(rr['heads'], 1)} |")
        analysis["blocks"].append(rec)
    A("\n`*` = profiled block. `peak GB` in ab-summary = max over all 4 ranks and settled updates of "
      "`cuda_peak_reserved_bytes` / 1e9 (decimal GB, reserved, reset every update).\n")

    # ---- Q1/Q2: sums vs nvidia-smi
    A("## Q1-Q2. Sums across ranks vs nvidia-smi (settled updates; MiB)\n")
    A("| block | arm | updates | sum peak res (max) | sum end res mean | sum end alloc mean | "
      "sum end res-alloc mean | sum peak res - end res mean | ranks trimmed/upd | captures/upd (4 ranks) | "
      "smi max | smi mean | smi min | smi max - sum peak res (max over upd) | smi_end - sum end res (mean) |")
    A("|---|---|---|" + "---:|" * 12)
    for bi, b in enumerate(blocks):
        s = b[SETTLE:] if len(b) > SETTLE else b
        rec = analysis["blocks"][bi]
        smi = [x for x in s if x["smi_n"]]
        agg = dict(
            sum_peak_res_max=max(x["sum_peak_res"] for x in s),
            sum_res_mean=float(np.mean([x["sum_res"] for x in s])),
            sum_alloc_mean=float(np.mean([x["sum_alloc"] for x in s])),
            sum_cached_free_end_mean=float(np.mean([x["sum_res_minus_alloc_end"] for x in s])),
            sum_trim_mean=float(np.mean([x["sum_peakres_minus_res_end"] for x in s])),
            ranks_trimmed_mean=float(np.mean([x["ranks_trimmed"] for x in s])),
            captures_mean=float(np.mean([x["sum_captures"] for x in s])),
            smi_max=max(x["smi_max"] for x in smi), smi_mean=float(np.mean([x["smi_mean"] for x in smi])),
            smi_min=min(x["smi_min"] for x in smi),
            gap_lb_max=max(x["gap_lb"] for x in smi),
            gap_end_mean=float(np.mean([x["gap_end"] for x in s])))
        rec.update(agg)
        A(f"| {bi} | {rec['arm']}{'*' if rec['profiled'] else ''} | {s[0]['update']}-{s[-1]['update']} | "
          f"{fmt(agg['sum_peak_res_max'])} | {fmt(agg['sum_res_mean'])} | {fmt(agg['sum_alloc_mean'])} | "
          f"{fmt(agg['sum_cached_free_end_mean'])} | {fmt(agg['sum_trim_mean'])} | {fmt(agg['ranks_trimmed_mean'], 2)} | "
          f"{fmt(agg['captures_mean'], 2)} | {fmt(agg['smi_max'])} | {fmt(agg['smi_mean'])} | {fmt(agg['smi_min'])} | "
          f"{fmt(agg['gap_lb_max'])} | {fmt(agg['gap_end_mean'])} |")

    # Per-arm pooled (settled, all six blocks; and unprofiled only)
    A("\n### Pooled by arm (settled updates)\n")
    A("| arm | set | n upd | sum end res mean | sum end alloc mean | sum end res-alloc mean | "
      "ranks trimmed/upd | captures/upd | smi mean (upd means) | smi_end - sum end res mean | sd |")
    A("|---|---|" + "---:|" * 9)
    pooled = {}
    for label, keep in (("unprofiled", lambda b: not b[0]["profiled"]), ("all", lambda b: True)):
        for arm in (0, 1):
            s = [x for b in blocks if b[0]["arm"] == arm and keep(b) for x in (b[SETTLE:] if len(b) > SETTLE else b)]
            ge = [x["gap_end"] for x in s]
            p = dict(n=len(s), sum_res=float(np.mean([x["sum_res"] for x in s])),
                     sum_alloc=float(np.mean([x["sum_alloc"] for x in s])),
                     cached=float(np.mean([x["sum_res_minus_alloc_end"] for x in s])),
                     trimmed=float(np.mean([x["ranks_trimmed"] for x in s])),
                     captures=float(np.mean([x["sum_captures"] for x in s])),
                     smi_mean=float(np.mean([x["smi_mean"] for x in s if x["smi_n"]])),
                     gap_end=float(np.mean(ge)), gap_end_sd=float(np.std(ge, ddof=1)))
            pooled[f"{label}_{'on' if arm else 'off'}"] = p
            A(f"| {'on' if arm else 'off'} | {label} | {p['n']} | {fmt(p['sum_res'])} | {fmt(p['sum_alloc'])} | "
              f"{fmt(p['cached'])} | {fmt(p['trimmed'], 2)} | {fmt(p['captures'], 2)} | {fmt(p['smi_mean'])} | "
              f"{fmt(p['gap_end'])} | {fmt(p['gap_end_sd'])} |")
    analysis["pooled"] = pooled

    # Outside-allocator estimate at startup (first update: tiny reserved)
    first = rows[0]
    A(f"\nFirst update {first['update']}: sum end reserved {fmt(first['sum_res'])} MiB, nearest smi sample "
      f"{fmt(first['smi_end'])} MiB (dt {first['smi_end_dt']:+.1f} s) -> outside-allocator estimate "
      f"{fmt(first['gap_end'])} MiB.\n")
    analysis["first_update_gap_end"] = first["gap_end"]

    # Sum of per-update peak allocated (the live-tensor requirement) by arm
    A("### Sum over ranks of per-update peak allocated (settled; MiB; ranks peak at different times, so an upper bound on simultaneous live bytes)\n")
    A("| arm | set | n upd | mean | max |")
    A("|---|---|---:|---:|---:|")
    for label, keep in (("unprofiled", lambda b: not b[0]["profiled"]), ("all", lambda b: True)):
        for arm in (0, 1):
            s = [x["sum_peak_alloc"] for b in blocks if b[0]["arm"] == arm and keep(b)
                 for x in (b[SETTLE:] if len(b) > SETTLE else b)]
            analysis.setdefault("sum_peak_alloc", {})[f"{label}_{arm}"] = [float(np.mean(s)), max(s)]
            A(f"| {'on' if arm else 'off'} | {label} | {len(s)} | {fmt(float(np.mean(s)))} | {fmt(max(s))} |")
    # Per-sample residual smi(t) - sum peak reserved of its update (<= outside-allocator memory)
    A("\n### Per-sample smi(t) minus the sum of the 4 ranks' peak reserved of the update containing t (MiB; post-warmup)\n")
    A("This is a lower bound on memory outside the PyTorch allocator at t (exact when every rank's reserved equals its peak at t).\n")
    A("| arm | samples | max | p90 | median | min | samples within 20 MiB of max |")
    A("|---|---:|---:|---:|---:|---:|---:|")
    for arm in (0, 1):
        v = []
        for s_ in samples:
            row, _ = phase_of(s_["epoch"], rows)
            if row and row["update"] > plan["resume_update"] + plan["warmup"] and row["arm"] == arm:
                v.append(s_["mem_used_mib"] - row["sum_peak_res"])
        v = np.array(v)
        analysis.setdefault("residual", {})[arm] = dict(max=float(v.max()), median=float(np.median(v)))
        A(f"| {'on' if arm else 'off'} | {len(v)} | {fmt(float(v.max()))} | {fmt(float(np.percentile(v, 90)))} | "
          f"{fmt(float(np.median(v)))} | {fmt(float(v.min()))} | {int((v >= v.max() - 20).sum())} |")
    A("")

    # Trimming vs captures (all updates after warmup, per rank)
    A("## Reserved trimming vs private-graph captures (per rank-update, updates after warmup)\n")
    post = [x for x in rows if x["update"] > plan["resume_update"] + plan["warmup"]]
    table = {}
    for x in post:
        for r in range(nr):
            cap = x[f"r{r}_captures"] > 0
            trim = x[f"r{r}_res"] < x[f"r{r}_peak_res"] - 0.5
            key = ("on" if x["arm"] else "off", cap)
            t = table.setdefault(key, [0, 0, []])
            t[0] += 1
            t[1] += trim
            if trim:
                t[2].append(x[f"r{r}_peak_res"] - x[f"r{r}_res"])
    A("| arm | >=1 capture in update | rank-updates | with end reserved < peak reserved | mean drop MiB (when dropped) |")
    A("|---|---|---:|---:|---:|")
    for key in sorted(table):
        n, t, drops = table[key]
        A(f"| {key[0]} | {key[1]} | {n} | {t} | {fmt(float(np.mean(drops))) if drops else '-'} |")
    analysis["trim_vs_capture"] = {f"{k[0]}_{k[1]}": v[:2] for k, v in table.items()}

    # ---- Q3: time structure
    A("\n## Q3. Time structure of nvidia-smi samples\n")
    start = plan["resume_update"] + plan["warmup"]
    off_rows = [x for b in blocks if not b[0]["arm"] for x in b]
    off_max = max(x["smi_max"] for x in off_rows if x["smi_n"])
    lab = []
    for s_ in samples:
        row, ph = phase_of(s_["epoch"], rows)
        if row is None:
            continue
        lab.append((s_, row, ph))
    post_lab = [t for t in lab if t[1]["update"] > start]
    above = [t for t in post_lab if t[0]["mem_used_mib"] > off_max]
    A(f"Off-arm maximum over all post-warmup off updates (settling included): {fmt(off_max)} MiB.\n")
    A(f"Samples above it: {len(above)} of {len(post_lab)} post-warmup samples.\n")
    A("| epoch-offset s | update | arm | block pos | phase | smi MiB | excess over off max |")
    A("|---:|---:|---|---:|---|---:|---:|")
    t0 = rows[0]["t_start"]
    pos = {}
    for b in blocks:
        for i, x in enumerate(b):
            pos[x["update"]] = i
    for s_, row, ph in above:
        A(f"| {s_['epoch'] - t0:.0f} | {row['update']} | {'on' if row['arm'] else 'off'} | {pos.get(row['update'])} | "
          f"{ph} | {fmt(s_['mem_used_mib'])} | {fmt(s_['mem_used_mib'] - off_max)} |")
    analysis["samples_above_off_max"] = [dict(update=r["update"], arm=r["arm"], phase=p, mib=s["mem_used_mib"])
                                         for s, r, p in above]
    # Per-block sample distribution and phase split
    A("\n### Per-block sample distribution (all updates of the block, settling included)\n")
    A("| block | arm | updates | samples | collect n / mean / max | learn n / mean / max | p50 | p90 | max | n at max |")
    A("|---|---|---|---:|---|---|---:|---:|---:|---:|")
    for bi, b in enumerate(blocks):
        us = {x["update"] for x in b}
        ss = [t for t in post_lab if t[1]["update"] in us]
        v = np.array([t[0]["mem_used_mib"] for t in ss])
        c = [t[0]["mem_used_mib"] for t in ss if t[2] == "collect"]
        l_ = [t[0]["mem_used_mib"] for t in ss if t[2] == "learn"]
        A(f"| {bi} | {'on' if b[0]['arm'] else 'off'}{'*' if b[0]['profiled'] else ''} | {b[0]['update']}-{b[-1]['update']} | "
          f"{len(v)} | {len(c)} / {fmt(float(np.mean(c)))} / {fmt(max(c))} | {len(l_)} / {fmt(float(np.mean(l_)))} / {fmt(max(l_))} | "
          f"{fmt(float(np.percentile(v, 50)))} | {fmt(float(np.percentile(v, 90)))} | {fmt(float(v.max()))} | "
          f"{int((v == v.max()).sum())} |")
    # Transitions: last 3 samples of a block vs first 3 of the next
    A("\n### Arm switches: nvidia-smi around each switch (MiB)\n")
    A("| switch at update | from -> to | last 3 samples before | first 3 samples after | sum end res before -> after 1st upd |")
    A("|---:|---|---|---|---|")
    allb = []
    cur = []
    for x in rows:
        if cur and x["arm"] != cur[-1]["arm"]:
            allb.append(cur)
            cur = []
        cur.append(x)
    allb.append(cur)
    for prev, nxt in zip(allb, allb[1:]):
        tsw = prev[-1]["t_end"]
        before = [s["mem_used_mib"] for s in samples if tsw - 15 < s["epoch"] <= tsw][-3:]
        after = [s["mem_used_mib"] for s in samples if tsw < s["epoch"] <= tsw + 15][:3]
        A(f"| {nxt[0]['update']} | {'on' if prev[0]['arm'] else 'off'} -> {'on' if nxt[0]['arm'] else 'off'} | "
          f"{', '.join(fmt(v) for v in before)} | {', '.join(fmt(v) for v in after)} | "
          f"{fmt(prev[-1]['sum_res'])} -> {fmt(nxt[0]['sum_res'])} |")

    # Drift: linear trend within arms over post-warmup settled updates
    A("\n## Q4. Confounders and regression (per update, settled, after warmup)\n")
    s_all = [x for b in blocks for x in (b[SETTLE:] if len(b) > SETTLE else b)]
    feats = ["arm", "sum_resident", "sum_mean_prefix", "sum_cache", "sum_ccache", "sum_store_tokens", "update"]
    A("| block | arm | resident (4 ranks) mean | mean prefix (rank mean) | max prefix max | cache MiB sum mean | "
      "coll-cache MiB sum mean | store tokens sum mean | graph pool MiB sum mean | heads MiB sum |")
    A("|---|---|" + "---:|" * 8)
    for bi, b in enumerate(blocks):
        s = b[SETTLE:] if len(b) > SETTLE else b
        A(f"| {bi} | {'on' if s[0]['arm'] else 'off'}{'*' if s[0]['profiled'] else ''} | "
          f"{fmt(float(np.mean([x['sum_resident'] for x in s])), 1)} | {fmt(float(np.mean([x['sum_mean_prefix'] for x in s])), 0)} | "
          f"{max(max(x[f'r{r}_max_prefix'] for r in range(nr)) for x in s)} | {fmt(float(np.mean([x['sum_cache'] for x in s])))} | "
          f"{fmt(float(np.mean([x['sum_ccache'] for x in s])))} | {fmt(float(np.mean([x['sum_store_tokens'] for x in s])))} | "
          f"{fmt(float(np.mean([x['sum_graph_pool'] for x in s])))} | {fmt(float(np.mean([x['sum_heads'] for x in s])), 1)} |")
    y_names = ["sum_res", "sum_alloc", "sum_res_minus_alloc_end", "smi_mean", "smi_max"]
    reg = {}
    A("\nOLS on settled post-warmup updates (n = %d). Coefficients per unit; t = coef / se "
      "(ordinary SE, updates are autocorrelated, so t is optimistic).\n" % len(s_all))
    A("| y | model | arm coef (t) | other coefs (t) | R2 |")
    A("|---|---|---|---|---:|")
    models = {"arm only": ["arm"],
              "arm + collection cache": ["arm", "sum_ccache"],
              "arm + prefix + resident + update": ["arm", "sum_mean_prefix", "sum_resident", "update"],
              "no arm: ccache + prefix + resident + update": ["sum_ccache", "sum_mean_prefix", "sum_resident", "update"]}
    for y in y_names:
        s = [x for x in s_all if x.get(y) is not None]
        Y = np.array([x[y] for x in s], float)
        for mname, cols in models.items():
            X = np.column_stack([np.ones(len(s))] + [np.array([x[c] for x in s], float) for c in cols])
            beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
            res = Y - X @ beta
            dof = len(Y) - X.shape[1]
            sigma2 = res @ res / dof
            cov = sigma2 * np.linalg.inv(X.T @ X)
            se = np.sqrt(np.diag(cov))
            r2 = 1 - (res @ res) / ((Y - Y.mean()) @ (Y - Y.mean()))
            terms = {c: (float(beta[i + 1]), float(beta[i + 1] / se[i + 1])) for i, c in enumerate(cols)}
            reg[f"{y}|{mname}"] = dict(terms=terms, r2=float(r2), n=len(Y))
            armtxt = f"{terms['arm'][0]:,.0f} ({terms['arm'][1]:.1f})" if "arm" in terms else "-"
            other = "; ".join(f"{c} {v[0]:,.3g} ({v[1]:.1f})" for c, v in terms.items() if c != "arm")
            A(f"| {y} | {mname} | {armtxt} | {other or '-'} | {r2:.2f} |")
    analysis["regression"] = reg

    # Paired adjacent blocks
    A("\n### Adjacent-block differences (on minus neighbouring off, settled block means)\n")
    A("| pair | d smi mean | d smi max | d sum end res | d sum end alloc | d sum end res-alloc | d coll-cache | d mean prefix | d resident |")
    A("|---|" + "---:|" * 8)
    bm = []
    for b in blocks:
        s = b[SETTLE:] if len(b) > SETTLE else b
        bm.append(dict(arm=s[0]["arm"], u=f"{s[0]['update']}-{s[-1]['update']}",
                       smi_mean=float(np.mean([x["smi_mean"] for x in s if x["smi_n"]])),
                       smi_max=max(x["smi_max"] for x in s if x["smi_n"]),
                       res=float(np.mean([x["sum_res"] for x in s])), alloc=float(np.mean([x["sum_alloc"] for x in s])),
                       cached=float(np.mean([x["sum_res_minus_alloc_end"] for x in s])),
                       cc=float(np.mean([x["sum_ccache"] for x in s])), pref=float(np.mean([x["sum_mean_prefix"] for x in s])),
                       res_n=float(np.mean([x["sum_resident"] for x in s]))))
    pairs = []
    for i in range(len(bm) - 1):
        a, b = bm[i], bm[i + 1]
        on, off = (b, a) if b["arm"] else (a, b)
        d = {k: on[k] - off[k] for k in ("smi_mean", "smi_max", "res", "alloc", "cached", "cc", "pref", "res_n")}
        pairs.append(dict(pair=f"on {on['u']} - off {off['u']}", **d))
        A(f"| on {on['u']} - off {off['u']} | {fmt(d['smi_mean'])} | {fmt(d['smi_max'])} | {fmt(d['res'])} | "
          f"{fmt(d['alloc'])} | {fmt(d['cached'])} | {fmt(d['cc'])} | {fmt(d['pref'], 1)} | {fmt(d['res_n'], 1)} |")
    analysis["adjacent_pairs"] = pairs

    (HERE / "tables.md").write_text("\n".join(out) + "\n")
    with (HERE / "per_update.csv").open("w", newline="") as f:
        keys = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    (HERE / "analysis.json").write_text(json.dumps(analysis, indent=2, default=float) + "\n")
    svg(rows, samples, blocks, HERE / "timeline.svg")
    print("\n".join(out))
    return 0


def svg(rows, samples, blocks, path):
    W, H, L, R, T, B = 1100, 460, 70, 20, 20, 50
    t0, t1 = rows[0]["t_start"], rows[-1]["t_end"]
    ymin, ymax = 0, 25000
    X = lambda t: L + (t - t0) / (t1 - t0) * (W - L - R)
    Y = lambda v: T + (1 - (v - ymin) / (ymax - ymin)) * (H - T - B)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="11">',
         f'<rect width="{W}" height="{H}" fill="white"/>']
    for x in rows:
        if x["arm"]:
            p.append(f'<rect x="{X(x["t_start"]):.1f}" y="{T}" width="{X(x["t_end"]) - X(x["t_start"]):.1f}" '
                     f'height="{H - T - B}" fill="#dbe9ff"/>')
    for v in range(0, 25001, 5000):
        p.append(f'<line x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#ddd"/>'
                 f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for x in rows:
        if x["update"] % 5 == 0:
            p.append(f'<text x="{X(x["t_end"]):.1f}" y="{H - B + 15}" text-anchor="middle">{x["update"]}</text>')
    pts = " ".join(f"{X(s['epoch']):.1f},{Y(s['mem_used_mib']):.1f}" for s in samples if t0 <= s["epoch"] <= t1)
    p.append(f'<polyline points="{pts}" fill="none" stroke="#222" stroke-width="1.3"/>')
    # sum of end reserved and sum of peak reserved as steps
    for key, color in (("sum_res", "#d9480f"), ("sum_peak_res", "#2b8a3e"), ("sum_alloc", "#7048e8")):
        seg = " ".join(f"{X(x['t_start']):.1f},{Y(x[key]):.1f} {X(x['t_end']):.1f},{Y(x[key]):.1f}" for x in rows)
        p.append(f'<polyline points="{seg}" fill="none" stroke="{color}" stroke-width="1.2"/>')
    legend = [("#222", "nvidia-smi memory.used"), ("#2b8a3e", "sum of 4 ranks' per-update peak reserved"),
              ("#d9480f", "sum of end-of-update reserved"), ("#7048e8", "sum of end-of-update allocated"),
              ("#dbe9ff", "blue band = flag on")]
    for i, (c, name) in enumerate(legend):
        p.append(f'<rect x="{L + 10}" y="{T + 8 + i * 15}" width="12" height="8" fill="{c}"/>'
                 f'<text x="{L + 28}" y="{T + 16 + i * 15}">{name}</text>')
    p.append(f'<text x="{W / 2}" y="{H - 8}" text-anchor="middle">update (end), MiB on y</text></svg>')
    path.write_text("\n".join(p))


if __name__ == "__main__":
    raise SystemExit(main())
