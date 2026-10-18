"""Summarize isolated ABBA PPO trials without interpreting component timings."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def summarize_reports(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("no trials found")
    groups = {}
    for report in reports:
        if report.get("status") != "passed":
            raise ValueError("a failed/incomplete trial cannot be summarized as a speedup")
        if report.get("backend") not in ("torch", "triton"):
            raise ValueError("unexpected backend")
        summary = report["summary"]
        rate = summary["decisions_per_wall_second"]
        if not math.isfinite(rate) or rate <= 0 or summary["learned_rows"] <= 0:
            raise ValueError("each trial needs finite positive throughput and actual PPO learning")
        group = groups.setdefault(report["case"], [])
        group.append(report)
    result = {}
    for case, trials in groups.items():
        if len(trials) % 4:
            raise ValueError("ABBA requires complete four-trial blocks per case")
        reference = trials[0]
        for trial in trials[1:]:
            for field in ("config", "environment", "source", "warmup_updates"):
                if trial[field] != reference[field]:
                    raise ValueError(f"mismatched {field}; do not pool different workloads/environments")
        blocks = []
        for i in range(0, len(trials), 4):
            block = trials[i:i+4]
            if [t["backend"] for t in block] != ["torch", "triton", "triton", "torch"]:
                raise ValueError("trials must be supplied in ABBA order")
            means = {b: statistics.mean(t["summary"]["decisions_per_wall_second"]
                                       for t in block if t["backend"] == b)
                     for b in ("torch", "triton")}
            blocks.append(means["triton"] / means["torch"])
        rates = {b: [t["summary"]["decisions_per_wall_second"] for t in trials if t["backend"] == b]
                 for b in ("torch", "triton")}
        medians = {b: statistics.median(values) for b, values in rates.items()}
        ratio = medians["triton"] / medians["torch"]
        result[case] = dict(trial_count=len(trials), median_decisions_per_wall_second=medians,
                            ratio_of_median_rates=ratio, throughput_change_percent=100*(ratio-1),
                            block_mean_rate_ratios=blocks, all_trial_rates=rates,
                            observed_max_prefix=[t["summary"]["max_prefix"] for t in trials])
    return dict(scope="fresh single-process PPO; not mature-checkpoint, multi-rank, or strength evidence",
                uncertainty="raw block variation only; no confidence interval or significance claim",
                cases=result)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = sorted(args.directory.glob("ppo-*/result.json"))
    result = summarize_reports([json.loads(path.read_text()) for path in paths])
    result["trial_files"] = [str(path.relative_to(args.directory)) for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
