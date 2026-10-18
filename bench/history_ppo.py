"""Bounded, real history-PPO throughput sweep; GPU measurements select scale.

Each case runs in a fresh process, starts randomly and logs real growing match
prefixes. Collection counts all pending decisions; learning counts unique PPO
rows and also reports epoch exposures. No CPU result can select a GPU config.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from train.history_config import HistoryPPOConfig
from train.history_ppo import HistoryTrainer


def assess(lines: list[dict], gpu_bytes: int) -> dict:
    early = [r for r in lines if 96 <= r["mean_prefix"] <= 256]
    late = [r for r in lines if r["mean_prefix"] >= 720 and r["update_samples"] > 0]
    if not early or not late:
        return dict(passed=False, reason="insufficient early/long-prefix coverage")
    first, last = min(early, key=lambda r: abs(r["mean_prefix"] - 144)), late[0]
    ratio = last["decisions_per_sec"] / first["decisions_per_sec"]
    peak = max(r["cuda_peak_reserved_bytes"] for r in lines)
    finite = all(r.get("encoder_grad_norm", 0) and r["encoder_grad_norm"] > 0
                 for r in late)
    passed = ratio >= 0.5 and peak < 0.7 * gpu_bytes and finite
    return dict(passed=bool(passed), collection_retention=ratio,
                early_prefix=first["mean_prefix"], long_prefix=last["mean_prefix"],
                long_prefix_collect_dps=last["decisions_per_sec"],
                long_prefix_learn_dps=last["learn_decisions_per_sec"],
                long_prefix_end_to_end_dps=last["update_samples"] /
                    (last["collect_seconds"] + last["learn_seconds"]),
                peak_reserved_bytes=peak,
                reason="eligible for bounded pilot" if passed else
                       "stop: profile/batched KV cache or lower memory use before pilot")


def run_case(args) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("throughput selection requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = HistoryPPOConfig(width=64, layers=2, heads=4, num_envs=args.case,
                              steps_per_update=64, updates=8, epochs=1,
                              minibatch_matches=2, torch_threads=2, num_threads=2,
                              seed=2026092601, snapshot_updates=0)
    trainer = HistoryTrainer(config, args.output, device="cuda")
    lines = []
    for _ in range(config.updates):
        line = trainer.update()
        lines.append(line)
        if line["mean_prefix"] >= 1024:
            break
    props = torch.cuda.get_device_properties(0)
    report = dict(config=asdict(config), device=props.name, gpu_bytes=props.total_memory,
                  torch=torch.__version__, cuda=torch.version.cuda,
                  assessment=assess(lines, props.total_memory))
    (Path(args.output) / "result.json").write_text(json.dumps(report, indent=2) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envs", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--case", type=int)
    parser.add_argument("--case-seconds", type=int, default=240)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case:
        run_case(args)
        return 0
    reports = []
    for envs in args.envs:
        output = args.output / f"envs-{envs}"
        output.mkdir(exist_ok=True)
        started = time.monotonic()
        with (output / "console.log").open("w") as log:
            try:
                done = subprocess.run([sys.executable, "-m", "bench.history_ppo", "--output",
                                       str(output), "--case", str(envs)], stdout=log,
                                      stderr=subprocess.STDOUT, timeout=args.case_seconds)
                code = done.returncode
            except subprocess.TimeoutExpired:
                code = 124
        path = output / "result.json"
        report = json.loads(path.read_text()) if code == 0 and path.exists() else dict(
            assessment=dict(passed=False, reason=f"case exited {code}; inspect partial metrics"))
        report.update(envs=envs, wall_seconds=time.monotonic() - started)
        reports.append(report)
    eligible = [r for r in reports if r["assessment"]["passed"]]
    selected = max(eligible, key=lambda r: r["assessment"]["long_prefix_end_to_end_dps"]) if eligible else None
    summary = dict(cases=reports, selected=selected,
                   next_step="population pilot" if selected else "stop_and_profile_or_implement_batched_cache",
                   claim="engineering throughput only; no playing-strength inference")
    (args.output / "selection.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
