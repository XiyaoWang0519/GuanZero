"""Attach to an explicitly created T4 pod; guard, upload, monitor, sync, delete.

Never creates a resource. The provider key remains local. The detached guard
keeps its own deadline if this monitor fails; both need this Mac and network.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from infra.history_artifacts import sha256
from infra.history_pilot import write_json
from infra.runpod import Client, load_key, read_manifest, safe_pod, teardown, verify_owned


def verify_download(root: Path, records: dict[str, str]) -> None:
    for name, digest in records.items():
        path = root / name
        if path.resolve().is_relative_to(root.resolve()) is False or not path.is_file():
            raise ValueError("artifact path missing/outside destination")
        if sha256(path) != digest:
            raise ValueError(f"artifact hash mismatch: {name}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--ssh-key", type=Path, required=True)
    args = parser.parse_args(argv)
    kit = args.kit.resolve()
    pod_path = kit / "pod.json"
    owned = read_manifest(pod_path)
    plan = json.loads((kit / "run-manifest.json").read_text())
    if (owned["gpu_id"] != plan["compute"]["proposed_gpu"]
            or owned["cloud"] != plan["compute"]["cloud"]
            or owned["budget_usd"] > plan["lifecycle"]["proposed_budget_usd"]
            or owned["max_hours"] > plan["lifecycle"]["max_hours_from_create"]
            or owned["max_hourly_usd"] > plan["lifecycle"]["max_hourly_usd"]):
        raise ValueError("owned resource differs from reviewed plan")
    if sha256(kit / "source.tar.gz") != plan["source"]["archive_sha256"]:
        raise ValueError("source archive hash mismatch")
    client = Client(load_key(args.env_file))
    progress = kit / "progress.jsonl"
    def log(event, **fields):
        record = dict(event=event, utc=time.time(), **fields)
        with progress.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    # Independent process, launched before any SSH wait/upload/build.
    with (kit / "guard.log").open("a") as guard_log:
        guard = subprocess.Popen([sys.executable, "-m", "infra.runpod", "--env-file",
                                  str(args.env_file.resolve()), "guard", "--manifest", str(pod_path)],
                                 stdout=guard_log, stderr=subprocess.STDOUT, start_new_session=True)
    write_json(kit / "guard-receipt.json", dict(pid=guard.pid, deadline_epoch=owned["deadline_epoch"]))
    ssh = scp = None
    dest = kit / "local" / "results"
    dest.mkdir(parents=True, exist_ok=True)
    complete = False
    def run(command, timeout=30):
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)

    def sync():
        # ssh argv ends in host; rsync uses the same options without host.
        run(["rsync", "-az", "--exclude=*.tmp", "--exclude=.*.pt.*", "-e", shlex.join(ssh[:-1]),
             ssh[-1] + ":/workspace/results/", str(dest) + "/"], timeout=90)

    try:
        until = min(time.time() + 900, owned["deadline_epoch"] - 180)
        while time.time() < until:
            if guard.poll() is not None:
                raise RuntimeError("independent guard exited before workload launch")
            pod = client.pod(owned["pod_id"])
            if pod is None:
                raise RuntimeError("owned pod disappeared during initialization")
            verify_owned(owned, pod)
            address = safe_pod(pod).get("ssh")
            if address:
                options = ["-i", str(args.ssh_key.resolve()), "-o", "BatchMode=yes", "-o",
                           "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new", "-o",
                           f"UserKnownHostsFile={kit / 'known_hosts'}", "-o", "ServerAliveInterval=15",
                           "-o", "ServerAliveCountMax=2"]
                host = f"root@{address['host']}"
                ssh = ["ssh", *options, "-p", str(address["port"]), host]
                scp = ["scp", *options, "-P", str(address["port"])]
                try:
                    run(ssh + ["true"])
                    break
                except (subprocess.SubprocessError, OSError):
                    pass
            time.sleep(10)
        else:
            raise RuntimeError("no SSH within bounded initialization time")
        log("ssh_ready", elapsed_since_create=time.time() - owned["created_at_epoch"])
        # Install the transfer tool before starting periodic sync. Minimal GPU
        # images may omit it; three sync failures must not race package setup.
        run(ssh + ["command -v rsync >/dev/null || (apt-get update -qq && "
                   "apt-get install -y -qq --no-install-recommends rsync)"], timeout=120)
        run(ssh + ["mkdir -p /workspace/GuanZero /workspace/payload /workspace/results"])
        run(scp + [str(kit / "source.tar.gz"), host + ":/workspace/source.tar.gz"], timeout=90)
        run(scp + ["-r", str(kit / "payload") + "/.", host + ":/workspace/payload/"], timeout=60)
        actual = run(ssh + ["sha256sum /workspace/source.tar.gz"]).stdout.split()[0]
        if actual != plan["source"]["archive_sha256"]:
            raise ValueError("uploaded archive hash mismatch")
        run(ssh + ["tar -xzf /workspace/source.tar.gz -C /workspace/GuanZero"])
        driver = "\n".join([
            "#!/usr/bin/env bash", "set +e",
            "timeout --signal=TERM --kill-after=30s 600s bash /workspace/payload/setup.sh > /workspace/results/setup.log 2>&1",
            "status=$?",
            "if [ \"$status\" -eq 0 ]; then",
            f"  export POD_DEADLINE_EPOCH={float(owned['deadline_epoch'])}",
            "  bash /workspace/payload/run.sh > /workspace/results/run.log 2>&1",
            "  status=$?", "fi",
            'printf \'{"returncode":%s}\\n\' "$status" > /workspace/results/exit.json', ""])
        (kit / "driver.sh").write_text(driver)
        run(scp + [str(kit / "driver.sh"), host + ":/workspace/payload/driver.sh"])
        run(ssh + ["nohup bash /workspace/payload/driver.sh >/workspace/driver.log 2>&1 </dev/null &"])
        log("workload_dispatched")
        failures = 0
        while time.time() < owned["deadline_epoch"] - 180:
            if guard.poll() is not None:
                raise RuntimeError("independent provider guard exited")
            try:
                sync()
                failures = 0
            except (subprocess.SubprocessError, OSError):
                failures += 1
                log("sync_failed", consecutive=failures)
                if failures >= 3:
                    raise RuntimeError("three failed artifact syncs")
                time.sleep(15)
                continue
            health_path = dest / "health.json"
            if health_path.exists():
                health = json.loads(health_path.read_text())
                age = time.time() - health_path.stat().st_mtime
                log("health", **health, age_seconds=age)
                if not health["healthy"] or age > 180:
                    raise RuntimeError("pilot health/stale-metric gate failed")
            if (dest / "exit.json").exists():
                complete = True
                break
            time.sleep(30)  # metrics + sync, not only a process-alive check
        if complete:
            script = ("import hashlib,json,pathlib; r=pathlib.Path('/workspace/results'); "
                      "print(json.dumps({str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() "
                      "for p in r.rglob('*') if p.is_file() and not p.name.endswith('.tmp')}))")
            hashes = json.loads(run(ssh + ["python -c " + shlex.quote(script)], timeout=90).stdout)
            sync()
            verify_download(dest, hashes)
            write_json(kit / "verified-artifacts.json", hashes)
            log("artifacts_verified", files=len(hashes), exit=json.loads((dest / "exit.json").read_text()))
        else:
            log("deadline_cleanup", complete=False)
    finally:
        # Make one final bounded best effort, but never extend the paid deadline.
        if ssh and time.time() < owned["deadline_epoch"] - 120:
            try:
                sync()
            except (subprocess.SubprocessError, OSError):
                log("final_sync_failed")
        result = teardown(client, pod_path)
        account, pods = client.account(), client.pods()
        receipt = dict(**result, remaining_pods=[safe_pod(p) for p in pods],
                       current_hourly_spend=account["currentSpendPerHr"])
        write_json(kit / "teardown.json", receipt)
        log("teardown", **receipt)
    return 0 if complete and json.loads((dest / "exit.json").read_text())["returncode"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
