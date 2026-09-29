"""Pod orchestrator: one diagnostic segment that profiles where collection time goes.

Resumes the main lineage endpoint ``init.pt`` (main-w4 latest.pt, update 844,
55.3M decisions; 4 ranks x 256 tables, 16-match minibatches per rank,
ddp_global_minibatch, snapshot cadence 2, every CUDA rollout optimization) in
exactly the main-w4 layout. ``--resume-set`` changes only diagnostic fields:

* ``profile_collection=true`` and ``profile_collection_warmup=WARMUP``: the first
  WARMUP updates of the process are unprofiled (resident snapshots and history
  lengths climb back for ~25 updates after a resume), then every update records
  synchronized phase timings, the learner/snapshot split and the rows-per-call
  histogram. Profiling never changes what is collected
  (tests/test_history_collect_profile.py).
* ``checkpoint_updates=1000``: no mid-run 105 MB saves inside the window; rank 0
  still saves ``latest.pt`` when the segment ends.

The segment stops by itself after WARMUP + WINDOW updates (``--updates`` is the
absolute target), or by SIGTERM at the deadline. Its checkpoints go to
``segments/profile-diag/`` and are a DIAGNOSTIC BRANCH: never the main lineage
endpoint, never a parent of a later main segment. The main lineage still ends at
main-w4/latest.pt.
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
SEGMENT = "profile-diag"
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
    return {name: bool(data.get(name, {}).get("passed")) for name in ("triton", "graphs", "wide",
                                                                      "learner")}


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


def command(ranks: int, resume: Path, output: Path, target: int, warmup: int,
            device: str) -> list[str]:
    # The recipe and the CUDA rollout flags come from the checkpoint; the layout
    # values equal main-w4's, so they record no change.
    layout = {"num_envs": TABLES // ranks, "minibatch_matches": MINIBATCH // ranks,
              "num_threads": engine_threads(ranks),
              "ddp_global_minibatch": "true" if ranks > 1 else "false",
              "checkpoint_updates": 1000,
              "profile_collection": "true", "profile_collection_warmup": warmup}
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
    for key in ("policy_loss", "value_loss", "entropy"):
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
    parser.add_argument("--warmup", type=int, default=25, help="unprofiled updates first")
    parser.add_argument("--window", type=int, default=15, help="profiled updates")
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
        # The checkpoint's recipe turns every CUDA rollout optimization on; a failed
        # gate means that path is not trusted on this GPU, so do not run on it.
        log(results, "abort", reason="a CUDA gate failed; the lineage uses all four paths")
        return 1
    output = results / "segments" / SEGMENT
    output.mkdir(parents=True, exist_ok=True)
    (output / "DIAGNOSTIC_BRANCH.txt").write_text(
        "Diagnostic collection-profile branch resumed from main-w4/latest.pt.\n"
        "Not the main lineage endpoint; never continue the main lineage from here.\n")
    first = resume_update(args.init)
    target = first + args.warmup + args.window
    cmd = command(args.ranks, args.init, output, target, args.warmup, args.device)
    log(results, "segment_start", segment=SEGMENT, resume=str(args.init), resume_update=first,
        target_update=target, warmup=args.warmup, window=args.window, command=cmd,
        deadline=args.deadline_epoch)
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
                              str(results), "--window", str(args.window)],
                             capture_output=True, text=True)
    log(results, "summarize", returncode=summary.returncode, stderr=summary.stderr[-1000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
