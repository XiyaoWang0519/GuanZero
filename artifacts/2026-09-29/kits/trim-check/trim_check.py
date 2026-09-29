"""Pod orchestrator: short CUDA check of the allocator cache trim (one diagnostic segment).

Copied from .work/snapshot-batching-2026-09-28/snapshot_ab.py with minimal changes.
Resumes the main lineage endpoint ``init.pt`` (main-w4 latest.pt, update 844,
55.3M decisions; 4 ranks x 256 tables, 16-match minibatches per rank,
ddp_global_minibatch, snapshot cadence 2, every CUDA rollout optimization from
the checkpoint) in the same layout as the A/B. ``--resume-set`` changes only
diagnostic fields, all recorded in config_changes:

* ``batch_snapshot_policies_schedule=10:off,30:on``: 10 updates with the merged
  arm off, then 30 with it on (the environments are never restarted).
* ``rollout_trim_cuda_cache=auto``: the trim follows the arm, i.e. off for the
  first 10 updates and on (after collect, after learn, and right after the
  snapshot graphs are released at the switch) for the last 30.
* ``checkpoint_updates=1000``: no mid-run saves; rank 0 saves latest.pt at the end.
* profiling stays off.

The trainer stops itself at update 884 (``--updates`` is absolute) or by SIGTERM
at the deadline. Output ``segments/trim-check-diag/`` is a DIAGNOSTIC BRANCH:
never the main lineage endpoint, never a parent of a main segment. The main
lineage still ends at main-w4/latest.pt.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

TABLES, MINIBATCH = 1024, 64
RANKS = 4
SEGMENT = "trim-check-diag"
GATES = ("triton", "graphs", "wide", "learner", "snapshot", "trim")
stop = False


def on_term(signum, frame):
    global stop
    stop = True


def log(results: Path, event: str, **values):
    record = dict(time=time.time(), event=event, **values)
    with (results / "orchestrator.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def gates(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    return {name: bool(data.get(name, {}).get("passed")) for name in GATES}


def engine_threads(ranks: int) -> int:
    from infra.cpu_budget import usable_cpus
    usable = int(usable_cpus() or os.cpu_count() or 4)
    per_rank_tables = TABLES // ranks
    if per_rank_tables < 64:                 # local smoke
        return 1
    return max(2, min(24 // ranks, (usable - 4) // ranks, per_rank_tables // 32))


def env(src: Path) -> dict:
    values = dict(os.environ)
    values.update(PYTHONPATH=f"{src}/python:{src}/oracle:{src}", NVIDIA_TF32_OVERRIDE="0",
                  OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return values


def resume_update(path: Path) -> int:
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return int(payload["progress"]["updates"])


def command(ranks: int, resume: Path, output: Path, target: int, arms: str,
            device: str) -> list[str]:
    # The recipe and the CUDA rollout flags come from the checkpoint; the layout
    # values equal the A/B's (main-w4 layout; engine threads from the host).
    layout = {"num_envs": TABLES // ranks, "minibatch_matches": MINIBATCH // ranks,
              "num_threads": engine_threads(ranks),
              "ddp_global_minibatch": "true" if ranks > 1 else "false",
              "checkpoint_updates": 1000,
              "batch_snapshot_policies_schedule": arms,
              "rollout_trim_cuda_cache": "auto"}
    if TABLES >= 1024:   # the overnight graph budgets (2048 / 1024 MB), split over the ranks
        layout.update(rollout_graph_budget_mb=2048 // ranks,
                      rollout_graph_policy_budget_mb=1024 // ranks)
    cmd = [sys.executable, "-m", "train.history_ddp", "--world-size", str(ranks),
           "--output", str(output), "--updates", str(target), "--resume", str(resume),
           "--allow-source-change", "--torch-threads", "2", "--device", device]
    for key, value in layout.items():
        cmd += ["--resume-set", f"{key}={value}"]
    return cmd


def read_lines(path: Path) -> list[dict]:
    try:
        return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    except (OSError, ValueError):
        return []


def unhealthy(line: dict) -> str | None:
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        value = line.get(key)
        if value is not None and value != value:
            return f"non-finite {key} at update {line.get('update')}"
    return None


def main() -> int:
    global TABLES, MINIBATCH
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True, help="main-w4 latest.pt")
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ranks", type=int, default=RANKS)
    parser.add_argument("--off", type=int, default=10, help="updates with the merged arm off")
    parser.add_argument("--on", type=int, default=30, help="updates with the arm and trim on")
    parser.add_argument("--smoke", action="store_true",
                        help="local CPU check with a tiny 8-table checkpoint")
    args = parser.parse_args()
    if args.smoke:
        TABLES, MINIBATCH = 8, 4
    signal.signal(signal.SIGTERM, on_term)
    results, src = args.results, args.source_root
    results.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(src))           # infra.cpu_budget for engine threads
    ok = gates(args.gates)
    log(results, "gates", **ok)
    if not args.smoke and not all(ok.values()):
        log(results, "abort", reason="a CUDA gate failed")
        return 1
    output = results / "segments" / SEGMENT
    output.mkdir(parents=True, exist_ok=True)
    (output / "DIAGNOSTIC_BRANCH.txt").write_text(
        "Diagnostic allocator-trim check resumed from main-w4/latest.pt.\n"
        "Not the main lineage endpoint; never continue the main lineage from here.\n")
    first = resume_update(args.init)
    arms = f"{args.off}:off,{args.on}:on"
    target = first + args.off + args.on
    cmd = command(args.ranks, args.init, output, target, arms, args.device)
    log(results, "segment_start", segment=SEGMENT, resume=str(args.init), resume_update=first,
        target_update=target, schedule=arms, command=cmd, deadline=args.deadline_epoch)
    (results / "plan.json").write_text(json.dumps(dict(
        resume_update=first, target_update=target, schedule=arms, off=args.off, on=args.on),
        indent=2) + "\n")
    begin = time.time()
    status = "completed"
    seen = 0
    with (results / "segments" / f"{SEGMENT}.log").open("a") as stream, \
            (results / "update-epochs.jsonl").open("a") as epochs:
        child = subprocess.Popen(cmd, cwd=src, env=env(src), stdout=stream,
                                 stderr=subprocess.STDOUT)
        while child.poll() is None:
            time.sleep(2)
            lines = read_lines(output / "metrics.jsonl")
            if len(lines) > seen:   # wall-clock stamp of each rank-0 update (+-2 s)
                for line in lines[seen:]:
                    epochs.write(json.dumps(dict(epoch=time.time(), update=line["update"])) + "\n")
                epochs.flush()
                seen = len(lines)
            reason = unhealthy(lines[-1]) if lines else None
            if stop or reason or time.time() > args.deadline_epoch:
                status = "sigterm" if stop else (reason or "time_cap")
                child.send_signal(signal.SIGTERM)   # ranks stop together; rank 0 saves
                try:
                    child.wait(timeout=300)
                except subprocess.TimeoutExpired:
                    child.kill()
                    status += "+killed"
                break
        code = child.wait()
    if code != 0:
        status = f"{status}; exit {code}"
    lines = read_lines(output / "metrics.jsonl")
    row = dict(segment=SEGMENT, ranks=args.ranks, status=status, seconds=time.time() - begin,
               resume_update=first, target_update=target, updates=len(lines),
               last_update=lines[-1]["update"] if lines else None,
               has_checkpoint=(output / "latest.pt").exists(), diagnostic_branch=True)
    log(results, "segment_done", **row)
    (results / "segments.json").write_text(json.dumps([row], indent=2) + "\n")
    summary = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "summarize.py"),
                              str(results)], capture_output=True, text=True)
    log(results, "summarize", returncode=summary.returncode, stderr=summary.stderr[-1000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
