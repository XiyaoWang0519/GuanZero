"""Pod orchestrator: run the unchanged run_point.py on each checkpoint, 4 torch threads
each, as many in parallel as the pod's actual CPU quota allows, until a deadline.

Writes, per point, into RESULTS/out/: <name>.json (run_point output), <name>.log and
<name>.time (same format as the local run_one_export.sh: "start ..." then "exit N end ...
wall_seconds S"). status.json and host-facts.json in RESULTS describe the plan.
A point still running at the deadline is killed and recorded as such; a point never
started is listed as not_started. No point is retried.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def cpu_facts(max_cpus: int | None) -> dict:
    facts = dict(cpu_count=os.cpu_count())
    try:
        facts["affinity"] = len(os.sched_getaffinity(0))
    except AttributeError:
        facts["affinity"] = os.cpu_count()
    quota = None
    try:
        q, p = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        facts["cgroup_v2_cpu_max"] = f"{q} {p}"
        if q != "max":
            quota = int(q) / int(p)
    except (OSError, ValueError):
        pass
    if quota is None:
        try:
            q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
            p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            facts["cgroup_v1_quota_period"] = [q, p]
            if q > 0:
                quota = q / p
        except (OSError, ValueError):
            pass
    facts["quota_cpus"] = quota
    usable = facts["affinity"] if quota is None else min(facts["affinity"], math.floor(quota + 1e-6))
    if max_cpus:
        facts["max_cpus_override"] = max_cpus
        usable = min(usable, max_cpus)
    facts["usable_cpus"] = usable
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(("MemTotal", "MemAvailable")):
                facts[line.split(":")[0]] = line.split(":")[1].strip()
    except OSError:
        pass
    try:
        limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        facts["cgroup_memory_max"] = limit
    except OSError:
        pass
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                facts["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return facts


def memory_bytes(facts) -> float | None:
    limits = []
    if facts.get("MemAvailable"):
        limits.append(int(facts["MemAvailable"].split()[0]) * 1024)
    if facts.get("cgroup_memory_max", "max") not in ("max", ""):
        limits.append(int(facts["cgroup_memory_max"]))
    return min(limits) if limits else None


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--kit", type=Path, required=True, help="directory with run_point.py, ckpt/, expected.json")
    ap.add_argument("--src", type=Path, required=True, help="extracted evaluator source (b8c0c54)")
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--freeze", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--deadline-epoch", type=float, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--points", default=None, help="comma list; default: expected.json order")
    ap.add_argument("--max-cpus", type=int, default=None, help="cap usable CPUs (local smoke)")
    ap.add_argument("--gb-per-eval", type=float, default=3.0, help="memory planned per evaluation")
    args = ap.parse_args(argv)
    # The evaluations run with cwd=RESULTS: every path must be absolute.
    args.kit, args.src, args.results, args.freeze = (p.resolve() for p in
                                                     (args.kit, args.src, args.results, args.freeze))

    expected = json.loads((args.kit / "expected.json").read_text())
    names = args.points.split(",") if args.points else [p["name"] for p in expected["points"]]
    out = args.results / "out"
    out.mkdir(parents=True, exist_ok=True)
    facts = cpu_facts(args.max_cpus)
    by_cpu = max(1, facts["usable_cpus"] // args.threads)
    mem = memory_bytes(facts)
    by_mem = max(1, int(mem // (args.gb_per_eval * 2**30))) if mem else by_cpu
    parallel = max(1, min(len(names), by_cpu, by_mem))
    plan = dict(threads_per_eval=args.threads, parallel=parallel, by_cpu=by_cpu, by_memory=by_mem,
                points=names, deadline_epoch=args.deadline_epoch, started=utc(),
                waves=math.ceil(len(names) / parallel))
    (args.results / "host-facts.json").write_text(json.dumps(dict(facts, plan=plan), indent=2) + "\n")
    print(json.dumps(dict(facts=facts, plan=plan)), flush=True)

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{args.src}/python:{args.src}/oracle:{args.src}"
    # run_point.py sets torch.set_num_threads(4); these only stop other pools oversubscribing.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = str(args.threads)
    env["CUDA_VISIBLE_DEVICES"] = ""      # CPU numerics only; the GPU is never used

    status = {n: dict(state="pending") for n in names}
    pending, running = list(names), {}

    def write_status():
        (args.results / "status.json").write_text(json.dumps(dict(plan=plan, points=status,
                                                                  updated=utc()), indent=2) + "\n")

    def start(name):
        ckpt = args.kit / "ckpt" / f"{name}.pt"
        timef = out / f"{name}.time"
        timef.write_text(f"start {utc()} source=b8c0c54 (14e72581) pod parallel={parallel}\n")
        log = (out / f"{name}.log").open("w")
        proc = subprocess.Popen([args.python, str(args.kit / "run_point.py"), str(args.freeze),
                                 str(ckpt), str(out / f"{name}.json")],
                                stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(args.results),
                                start_new_session=True)
        running[name] = (proc, time.time(), log)
        status[name] = dict(state="running", pid=proc.pid, started=utc())

    def finish(name, code, killed=False):
        proc, began, log = running.pop(name)
        log.close()
        wall = int(time.time() - began)
        with (out / f"{name}.time").open("a") as f:
            if killed:
                f.write(f"killed_at_deadline end {utc()} after_seconds {wall}\n")
            else:
                f.write(f"exit {code} end {utc()} wall_seconds {wall}\n")
        status[name] = dict(state="killed_at_deadline" if killed else ("done" if code == 0 else "failed"),
                            returncode=code, wall_seconds=wall, ended=utc())

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    write_status()
    while pending or running:
        if time.time() >= args.deadline_epoch or stop["flag"]:
            for name, (proc, _, _) in list(running.items()):
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            limit = time.time() + 30
            for name, (proc, _, _) in list(running.items()):
                try:
                    proc.wait(timeout=max(1, limit - time.time()))
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                finish(name, proc.returncode, killed=True)
            for name in pending:
                status[name] = dict(state="not_started")
            pending = []
            write_status()
            break
        while pending and len(running) < parallel:
            start(pending.pop(0))
            write_status()
        for name, (proc, _, _) in list(running.items()):
            code = proc.poll()
            if code is not None:
                finish(name, code)
                write_status()
        time.sleep(5)
    plan["finished"] = utc()
    write_status()
    print(json.dumps({n: s["state"] for n, s in status.items()}), flush=True)
    return 0 if all(s["state"] == "done" for s in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
