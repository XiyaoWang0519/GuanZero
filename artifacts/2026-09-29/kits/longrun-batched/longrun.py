"""Pod orchestrator: overnight continuation of the MAIN lineage from main-w4 (init.pt).

Resumes ``init.pt`` (main-w4 latest.pt, update 844, 55.3M decisions) with
``train.history_ddp --world-size 4``. Model, recipe and layout (256 tables and
16-match minibatches per rank, ddp_global_minibatch, 6 engine threads, snapshot
cadence 2, recent-4 pool, checkpoint every 31 updates, graph budgets 512/256 MB,
every CUDA rollout flag) all come from the checkpoint. ``--resume-set`` changes
only the speed setting chosen at launch (``--mode``):

* ``batched``: ``batch_snapshot_policies=true``, ``rollout_trim_cuda_cache=auto``
* ``plain``:   ``batch_snapshot_policies=false``, ``rollout_trim_cuda_cache=auto``
  (the verified main-w4 setting; records no config change)

Protections:

* train for ``--train-hours`` of wall time from the first training start (restart
  gaps included), never past ``--hard-deadline-epoch``; then SIGTERM: the ranks
  stop at an update boundary and rank 0 saves latest.pt;
* crash -> resume from the segment's latest.pt (init.pt if none yet), at most
  ``--max-restarts`` restarts, exponential backoff, every restart logged;
* a CUDA out-of-memory crash in batched mode switches the next resume to
  ``batch_snapshot_policies=false`` (recorded in config_changes and here) and it
  stays off for the rest of the run;
* a non-finite loss stops training for good (no restart); low disk (< 3 GB) too.

Every attempt logs to ``segments/<SEGMENT>-attempt-N.log``. ``--simulate`` (local
smoke only) kills the trainer at a given update to exercise the crash and OOM
paths; the OOM case appends a line marked SIMULATED to the attempt log.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

RANKS = 4
SEGMENT = "main-overnight"
BASE_GATES = ("triton", "graphs", "wide", "learner")
BATCHED_GATES = BASE_GATES + ("snapshot", "trim")
MODES = {"batched": {"batch_snapshot_policies": "true", "rollout_trim_cuda_cache": "auto"},
         "plain": {"batch_snapshot_policies": "false", "rollout_trim_cuda_cache": "auto"}}
OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED",
               "CUDA error: out of memory")
MIN_FREE_BYTES = 3 * 10 ** 9
stop = False


def on_term(signum, frame):
    global stop
    stop = True


def log(results: Path, event: str, **values):
    record = dict(time=time.time(), event=event, **values)
    with (results / "orchestrator.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def gates(path: Path, names) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        data = {}
    return {name: bool(data.get(name, {}).get("passed")) for name in names}


def env(src: Path) -> dict:
    values = dict(os.environ)
    values.update(PYTHONPATH=f"{src}/python:{src}/oracle:{src}", NVIDIA_TF32_OVERRIDE="0",
                  OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    return values


def command(ranks: int, resume: Path, output: Path, device: str, settings: dict) -> list[str]:
    cmd = [sys.executable, "-m", "train.history_ddp", "--world-size", str(ranks),
           "--output", str(output), "--updates", "1000000000", "--resume", str(resume),
           "--allow-source-change", "--device", device]
    for key, value in settings.items():
        cmd += ["--resume-set", f"{key}={value}"]
    return cmd


def new_lines(path: Path, state: dict) -> list[dict]:
    """Complete rank-0 metrics lines appended since the last call."""
    out = []
    try:
        with path.open() as stream:
            stream.seek(state.get("offset", 0))
            for raw in stream:
                if not raw.endswith("\n"):
                    break
                state["offset"] = state.get("offset", 0) + len(raw.encode())
                try:
                    out.append(json.loads(raw))
                except ValueError:
                    pass
    except OSError:
        pass
    return out


def non_finite(line: dict) -> str | None:
    for key in ("policy_loss", "value_loss", "entropy", "approx_kl"):
        value = line.get(key)
        if value is not None and not math.isfinite(float(value)):
            return f"non-finite {key} at update {line.get('update')}"
    return None


def oom_in(path: Path) -> bool:
    try:
        text = path.read_text(errors="replace")[-200_000:]
    except OSError:
        return False
    return any(marker in text for marker in OOM_MARKERS)


def kill_group(child: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(child.pid, sig)
    except ProcessLookupError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--gates", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True, help="main-w4 latest.pt")
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--train-hours", type=float, default=8.0)
    parser.add_argument("--hard-deadline-epoch", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ranks", type=int, default=RANKS)
    parser.add_argument("--max-restarts", type=int, default=6)
    parser.add_argument("--backoff-seconds", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--no-restart-window", type=float, default=600.0,
                        help="no restart when a crash comes this close to the deadline")
    parser.add_argument("--smoke", action="store_true", help="local CPU check: no gate needed")
    parser.add_argument("--simulate", action="append", default=[], metavar="KIND@UPDATE",
                        help="smoke only: crash@N or oom@N kills the trainer after update N")
    args = parser.parse_args()
    if args.simulate and not args.smoke:
        raise SystemExit("--simulate is for the local smoke only")
    signal.signal(signal.SIGTERM, on_term)
    results, src = args.results, args.source_root
    results.mkdir(parents=True, exist_ok=True)
    needed = BATCHED_GATES if args.mode == "batched" else BASE_GATES
    ok = gates(args.gates, needed)
    log(results, "gates", mode=args.mode, required=list(needed), **ok)
    if not args.smoke and not all(ok.values()):
        log(results, "abort", reason="a required CUDA gate failed")
        return 1
    output = results / "segments" / SEGMENT
    output.mkdir(parents=True, exist_ok=True)
    (output / "MAIN_LINEAGE.txt").write_text(
        "Main lineage continuation of main-w4/latest.pt (update 844).\n")
    clock_path = results / "train-clock.json"
    if clock_path.exists():
        clock = json.loads(clock_path.read_text())
    else:
        start = time.time()
        clock = dict(first_train_start=start,
                     train_deadline=min(start + 3600 * args.train_hours, args.hard_deadline_epoch))
        clock_path.write_text(json.dumps(clock, indent=2) + "\n")
    deadline = clock["train_deadline"]
    simulations = []
    for item in args.simulate:
        kind, _, update = item.partition("@")
        simulations.append((kind, int(update)))
    fallback = False
    restarts, attempt, final_reason = 0, 0, None
    epochs_path = results / "update-epochs.jsonl"
    log(results, "run_start", mode=args.mode, deadline=deadline, clock=clock,
        settings=MODES[args.mode])
    while True:
        attempt += 1
        settings = dict(MODES[args.mode])
        if fallback:
            settings["batch_snapshot_policies"] = "false"
        latest = output / "latest.pt"
        resume = latest if latest.exists() else args.init
        cmd = command(args.ranks, resume, output, args.device, settings)
        attempt_log = results / "segments" / f"{SEGMENT}-attempt-{attempt}.log"
        log(results, "train_start", attempt=attempt, restarts=restarts, resume=str(resume),
            fallback=fallback, settings=settings, command=cmd)
        state = dict(offset=(output / "metrics.jsonl").stat().st_size
                     if (output / "metrics.jsonl").exists() else 0)
        reason, simulated, last_disk = None, None, 0.0
        with attempt_log.open("a") as stream:
            child = subprocess.Popen(cmd, cwd=src, env=env(src), stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                time.sleep(args.poll_seconds)
                lines = new_lines(output / "metrics.jsonl", state)
                if lines:
                    with epochs_path.open("a") as epochs:
                        for line in lines:
                            epochs.write(json.dumps(dict(epoch=time.time(), update=line["update"],
                                                         attempt=attempt)) + "\n")
                for line in lines:
                    reason = reason or non_finite(line)
                    for kind, at in list(simulations):
                        if line["update"] >= at:
                            simulated = kind
                            simulations.remove((kind, at))
                if simulated:
                    kill_group(child, signal.SIGKILL)
                    if simulated == "oom":
                        stream.write("SIMULATED (local smoke): torch.OutOfMemoryError: "
                                     "CUDA out of memory. Tried to allocate 2.00 GiB\n")
                        stream.flush()
                    log(results, "simulated_crash", kind=simulated, attempt=attempt)
                    break
                if time.time() - last_disk > 300:
                    last_disk = time.time()
                    free = shutil.disk_usage(results).free
                    log(results, "disk", free_bytes=free)
                    if free < MIN_FREE_BYTES:
                        reason = reason or f"low disk: {free} bytes free"
                if reason or stop or time.time() > deadline:
                    reason = reason or ("sigterm" if stop else "train deadline")
                    log(results, "train_stopping", attempt=attempt, reason=reason)
                    kill_group(child, signal.SIGTERM)   # launcher forwards; rank 0 saves
                    try:
                        child.wait(timeout=600)
                    except subprocess.TimeoutExpired:
                        kill_group(child, signal.SIGKILL)
                        log(results, "train_killed", attempt=attempt)
                    break
            code = child.wait()
        # Late lines of this attempt (written between the last poll and the exit).
        lines = new_lines(output / "metrics.jsonl", state)
        if lines:
            with epochs_path.open("a") as epochs:
                for line in lines:
                    epochs.write(json.dumps(dict(epoch=time.time(), update=line["update"],
                                                 attempt=attempt)) + "\n")
        oom = oom_in(attempt_log)
        log(results, "train_exit", attempt=attempt, returncode=code, reason=reason, oom=oom,
            simulated=simulated)
        if reason:
            final_reason = reason
            break
        if code == 0 and not simulated:
            final_reason = "trainer exited cleanly"
            break
        if restarts >= args.max_restarts:
            final_reason = f"crash with {restarts} restarts used; giving up"
            break
        if time.time() > deadline - args.no_restart_window:
            final_reason = "crash too close to the deadline; not restarting"
            break
        if oom and args.mode == "batched" and not fallback:
            fallback = True
            log(results, "fallback", attempt=attempt,
                change="batch_snapshot_policies=false from the next resume on (CUDA OOM)")
        restarts += 1
        wait = min(300.0, args.backoff_seconds * 2 ** (restarts - 1))
        log(results, "restart", restarts=restarts, backoff_seconds=wait, next_attempt=attempt + 1)
        time.sleep(wait)
        if stop:
            final_reason = "sigterm during backoff"
            break
    row = dict(segment=SEGMENT, mode=args.mode, fallback=fallback, restarts=restarts,
               attempts=attempt, reason=final_reason, has_checkpoint=(output / "latest.pt").exists())
    log(results, "run_done", **row)
    (results / "run.json").write_text(json.dumps(row, indent=2) + "\n")
    summary = subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "summarize.py"),
                              str(results)], capture_output=True, text=True)
    log(results, "summarize", returncode=summary.returncode, stderr=summary.stderr[-1000:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
