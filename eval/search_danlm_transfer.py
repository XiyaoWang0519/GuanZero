"""Paired DanLM transfer audit for a frozen B11 policy and its search wrapper."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import time

import numpy as np

from eval.danlm import arena
from eval.duplicate import bootstrap_interval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--deals", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2026092491)
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=1400)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.deals < 1 or args.chunk < 1 or args.max_seconds <= 0:
        parser.error("deals, chunk and max-seconds must be positive")

    started = time.monotonic()
    deals = arena.generate_deals(args.deals, args.seed, tribute_fraction=0.5)
    checkpoints = {"baseline": args.checkpoint, "search": f"search:{args.checkpoint}"}
    raw: dict = {"seed": args.seed, "requested_deals": args.deals,
                 "checkpoint": args.checkpoint, "danlm": arena.DEFAULT_DANLM_CHECKPOINT,
                 "records": {"baseline": [], "search": []}}
    path = args.output_prefix.with_suffix(".raw.json.gz")
    path.parent.mkdir(parents=True, exist_ok=True)

    for first in range(0, args.deals, args.chunk):
        if time.monotonic() - started >= args.max_seconds:
            break
        stop = min(first + args.chunk, args.deals)
        items = []
        for i in range(first, stop):
            deal = arena.deal_to_json(deals[i])
            items.extend([
                {"id": f"deal-{i}/a", "deal": deal, "seed": args.seed + i,
                 "seats": ["ours", "danlm", "ours", "danlm"]},
                {"id": f"deal-{i}/b", "deal": deal, "seed": args.seed + i,
                 "seats": ["danlm", "ours", "danlm", "ours"]},
            ])
        for arm, checkpoint in checkpoints.items():
            records = arena.run_jobs(items, checkpoint=checkpoint,
                                     danlm_spec=arena.DEFAULT_DANLM_CHECKPOINT,
                                     workers=1, diff=False, torch_threads=1)
            raw["records"][arm].extend(records)
        raw["completed_deals"] = stop
        payload = gzip.compress(json.dumps(raw, separators=(",", ":")).encode(),
                                compresslevel=9, mtime=0)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)
        print(f"completed {stop}/{args.deals} paired deals in {time.monotonic() - started:.1f}s",
              flush=True)

    by_arm = {arm: {r["id"]: r for r in records}
              for arm, records in raw["records"].items()}
    values = {"baseline": [], "search": [], "delta": []}
    excluded: dict[str, int] = {}
    for i in range(raw.get("completed_deals", 0)):
        rows = {arm: (by_arm[arm][f"deal-{i}/a"], by_arm[arm][f"deal-{i}/b"])
                for arm in checkpoints}
        failed = [f"{arm}:{leg['status']}" for arm, pair in rows.items()
                  for leg in pair if leg["status"] != "ok"]
        if failed:
            for reason in failed:
                excluded[reason] = excluded.get(reason, 0) + 1
            continue
        pair = {arm: (legs[0]["rewards"][0] + legs[1]["rewards"][1]) / 2
                for arm, legs in rows.items()}
        values["baseline"].append(pair["baseline"])
        values["search"].append(pair["search"])
        values["delta"].append(pair["search"] - pair["baseline"])

    summary = {"status": "paired_external_transfer_probe_not_G9",
               "seed": args.seed, "requested_deals": args.deals,
               "completed_deals": raw.get("completed_deals", 0),
               "scored_deals": len(values["delta"]), "excluded_legs": excluded,
               "wall_seconds": time.monotonic() - started,
               "checkpoint": args.checkpoint,
               "raw_results": str(path), "scores": {}}
    for arm, rows in values.items():
        summary["scores"][arm] = {
            "mean_levels_per_round": float(np.mean(rows)) if rows else None,
            "bootstrap_95_ci": list(bootstrap_interval(rows, args.seed, 2000))
            if len(rows) > 1 else None,
            "positive": sum(x > 0 for x in rows),
            "negative": sum(x < 0 for x in rows),
            "zero": sum(x == 0 for x in rows),
        }
    summary_path = args.output_prefix.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
