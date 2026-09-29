"""Bounded Vast.ai lifecycle: short CUDA check of the allocator cache trim from main-w4 (init.pt). Never prints secrets.

Copied from .work/snapshot-batching-2026-09-28/lifecycle.py (supervisord start,
read-only guard verify, verified download before destroy, API absence check,
datacenter/verified/reliability offer filter). Changes: trim_check.py workload
(one diagnostic branch segment: 10 updates arm off, 30 updates merged arm + trim
on), summarize.py writes trim-summary.*, a sixth CUDA gate (tests/test_history_trim.py),
an ordered machine list (20082 first, else 67872; the rented one is recorded),
a balance check before create, and a 25-minute timeline.

    python lifecycle.py freeze --source-root DIR  # pack source + kit SHA256s (no network)
    python lifecycle.py refreeze                  # re-hash kit files, keep the packed source
    python lifecycle.py dry-run [--offline]       # every check up to, not including, create
    python lifecycle.py launch --yes-spend        # create -> run -> sync -> destroy
    python lifecycle.py status | sync | destroy

Timeline from create (seconds): the segment is stopped at 1260 if it has not
finished (40 updates should end near 1000-1100), final sync starts by 1320 (or on
DONE), the local guard destroys at 1440, the pod stops itself at 1470; hard cap
1500 (25 min). Hourly ceiling $0.80 -> compute worst case 0.80 x 1470 s = $0.327,
plus traffic <= 0.5 GB x $0.05/GB = $0.025: $0.352 <= $0.36.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
SECRETS = Path("/Users/xiyaowang/Developer/Projects/GuanZero-throughput/.work/"
               "vast-throughput-2026-09-28/secrets")
API_KEY = SECRETS / "vast_api_key"
SSH_KEY = SECRETS / "id_ed25519"
MANIFEST = ROOT / "instance.json"
LABEL = "guanzero-trim-check-20260928"
# Ordered preference: 20082 (EPYC 7B13, Maryland; every earlier baseline) if it has
# a rentable offer that passes the filters, otherwise 67872 (UK, Xeon 8173M, 26
# usable CPUs; the merged-snapshot A/B). The rented machine is recorded.
MACHINE_IDS = (20082, 67872)
MIN_RELIABILITY = 0.99
MAX_HOURLY_USD = 0.80
MAX_INET_UP_USD_PER_GB = 0.05     # pod -> Mac traffic price ceiling (20082 quoted 0.033)
EGRESS_GB_BOUND = 0.5             # logs + one final 105 MB latest.pt
BALANCE_MARGIN_USD = 0.25
COST_CAP_USD = 0.36
# main-w4 latest.pt of .work/actor-ranks-2026-09-28 (update 844), copied as init.pt
INIT_SHA256 = "b1c5503743fe69d7d997612ffeab139cb9d5ef2ecb55e02ee2c76f259e112bd7"
IMAGE = "vastai/pytorch:cuda-12.8.1-auto"
DISK_GB = 50
RUNNING_WAIT_SECONDS = 600
SWEEP_STOP_SECONDS = 4380
SYNC_STOP_SECONDS = 4560
GUARD_DESTROY_SECONDS = 5100
REMOTE_STOP_SECONDS = 5220
HARD_CAP_SECONDS = 5400
SYNC_INTERVAL_SECONDS = 90
STATIC_FILES = ("source.tar.gz", "init.pt", "trim_check.py", "summarize.py", "gpu_sampler.py",
                "remote_setup.sh", "remote_start.sh")
TIMELINES = {
    # segment stopped 21 min after create at the latest; guard destroys at 24 min,
    # the pod stops itself at 24.5 min: 0.80 * 1470 / 3600 = $0.327 (+ <= $0.025
    # traffic); hard cap 25 min.
    "25m": dict(sweep=1260, sync=1320, guard=1440, remote=1470, hard=1500, cost_cap=0.36),
}
TIMELINE = "25m"


def worst_case_usd(hourly: float, inet_up_per_gb: float) -> float:
    """Billing until the pod stops itself (the later of the two stops) plus traffic."""
    return hourly * REMOTE_STOP_SECONDS / 3600 + inet_up_per_gb * EGRESS_GB_BOUND


def use_timeline(name: str) -> None:
    global TIMELINE, SWEEP_STOP_SECONDS, SYNC_STOP_SECONDS, GUARD_DESTROY_SECONDS
    global REMOTE_STOP_SECONDS, HARD_CAP_SECONDS, COST_CAP_USD
    t = TIMELINES[name]
    TIMELINE = name
    SWEEP_STOP_SECONDS, SYNC_STOP_SECONDS, GUARD_DESTROY_SECONDS = t["sweep"], t["sync"], t["guard"]
    REMOTE_STOP_SECONDS, HARD_CAP_SECONDS, COST_CAP_USD = t["remote"], t["hard"], t["cost_cap"]
    assert worst_case_usd(MAX_HOURLY_USD, MAX_INET_UP_USD_PER_GB) <= COST_CAP_USD
    assert (SWEEP_STOP_SECONDS < SYNC_STOP_SECONDS < GUARD_DESTROY_SECONDS < REMOTE_STOP_SECONDS
            < HARD_CAP_SECONDS)


for _name in TIMELINES:
    use_timeline(_name)
use_timeline("25m")


# ---- provider API (read-only except create/destroy) -------------------------------

def api(method, path, body=None):
    key = API_KEY.read_text().strip()
    request = urllib.request.Request(
        "https://console.vast.ai" + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Vast HTTP {exc.code}: " + exc.read().decode().replace(key, "[REDACTED]")) from None


def save(path, value):
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def log(event, **values):
    record = dict(time=time.time(), event=event, **values)
    with (ROOT / "lifecycle.jsonl").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)


def instances():
    data = api("GET", "/api/v1/instances/")
    if data.get("next_token"):
        raise RuntimeError("Unexpected pagination; refuse incomplete ownership check")
    return data["instances"]


def balance():
    try:
        user = api("GET", "/api/v0/users/current/")
        return {k: user.get(k) for k in ("credit", "balance")}
    except Exception as error:
        return {"error": str(error)[:200]}


def owned(manifest):
    matches = [r for r in instances() if int(r["id"]) == int(manifest["instance_id"])]
    if not matches:
        return None
    row = matches[0]
    if row.get("label") != LABEL or int(row["machine_id"]) != manifest["machine_id"]:
        raise RuntimeError("Instance ownership check failed")
    return row


def public(row):
    return {k: row.get(k) for k in (
        "id", "machine_id", "label", "actual_status", "cur_state", "intended_status", "dph_total",
        "disk_space", "gpu_name", "num_gpus", "cpu_cores_effective", "cpu_ram", "ssh_host",
        "ssh_port", "public_ipaddr", "ports", "start_date", "image_uuid", "image_runtype",
        "status_msg", "inet_up_cost", "inet_down_cost", "inet_up_billed", "inet_down_billed")}


def destroy(manifest, reason):
    if owned(manifest) is None:
        log("absent_confirmed", instance_id=manifest["instance_id"])
        return True
    result = api("DELETE", f"/api/v0/instances/{manifest['instance_id']}/", {})
    log("destroy_requested", instance_id=manifest["instance_id"], reason=reason, result=result)
    for _ in range(12):
        if owned(manifest) is None:
            log("absent_confirmed", instance_id=manifest["instance_id"])
            return True
        time.sleep(5)
    raise RuntimeError("Destroy requested but instance absence not yet confirmed")


def guard():
    """Detached: destroy at the deadline or on a rate above the ceiling."""
    manifest = json.loads(MANIFEST.read_text())
    log("guard_started", pid=os.getpid(), deadline=manifest["guard_destroy_epoch"])
    while time.time() < manifest["guard_destroy_epoch"]:
        try:
            row = owned(manifest)
            if row is None:
                log("guard_instance_absent")
                return
            save(ROOT / "last-instance.json", public(row))
            rate = row.get("dph_total")
            if rate is not None and float(rate) > manifest["max_hourly_usd"]:
                destroy(manifest, "hourly rate exceeded")
                return
        except Exception as exc:
            log("guard_check_error", error=str(exc)[:300])
        time.sleep(max(0, min(30, manifest["guard_destroy_epoch"] - time.time())))
    for attempt in range(20):
        try:
            destroy(manifest, "guard deadline")
            return
        except Exception as exc:
            log("guard_destroy_retry", attempt=attempt, error=str(exc)[:300])
            time.sleep(10)
    raise RuntimeError("Independent guard could not confirm teardown")


def qualified(row, machine_id):
    return (int(row["machine_id"]) == machine_id and int(row["num_gpus"]) == 1
            and float(row["dph_total"]) <= MAX_HOURLY_USD
            and row.get("gpu_name") == "RTX 4090" and int(row.get("gpu_ram") or 0) < 30000
            and int(row.get("hosting_type") or 0) == 1 and row.get("verification") == "verified"
            and float(row.get("reliability2") or 0) >= MIN_RELIABILITY
            and float(row.get("inet_up_cost") or 0) <= MAX_INET_UP_USD_PER_GB)


def search():
    """Offers of the first machine in MACHINE_IDS that has a qualifying one:
    (machine_id or None, its offers sorted by price, per-machine report)."""
    report, raw = {}, {}
    chosen = (None, [])
    for machine_id in MACHINE_IDS:
        query = {"machine_id": {"eq": machine_id}, "num_gpus": {"eq": 1},
                 "rentable": {"eq": True}, "rented": {"eq": False}, "verified": {"eq": True},
                 "datacenter": {"eq": True}, "type": "on-demand", "limit": 20,
                 "allocated_storage": DISK_GB}
        data = api("POST", "/api/v0/bundles/", query)
        raw[str(machine_id)] = data
        rows = data.get("offers", [])
        offers = sorted((r for r in rows if qualified(r, machine_id)),
                        key=lambda r: float(r["dph_total"]))
        report[str(machine_id)] = dict(returned=len(rows), qualified=len(offers),
                                       dph_total=[round(float(r["dph_total"]), 4) for r in offers])
        if offers and chosen[0] is None:
            chosen = (machine_id, offers)
    save(ROOT / "offers.json", raw)
    return chosen[0], chosen[1], report


def create_body():
    return {"client_id": "me", "image": IMAGE, "disk": DISK_GB, "label": LABEL,
            "runtype": "ssh_direct", "env": {}, "onstart": "entrypoint.sh",
            "cancel_unavail": True, "target_state": "running"}


def create(offer):
    if MANIFEST.exists():
        raise RuntimeError("An owned manifest already exists; refuse duplicate rent")
    if any(r.get("label") == LABEL for r in instances()):
        raise RuntimeError("Matching instance already exists")
    if int(offer["machine_id"]) not in MACHINE_IDS or int(offer["num_gpus"]) != 1:
        raise RuntimeError("Wrong offer")
    if float(offer["dph_total"]) > MAX_HOURLY_USD:
        raise RuntimeError("Offer rate exceeds the price ceiling")
    started = time.time()
    manifest = dict(machine_id=int(offer["machine_id"]), offer_id=int(offer["id"]), label=LABEL,
                    created_at_epoch=started, offer_dph_total=float(offer["dph_total"]),
                    sweep_stop_epoch=started + SWEEP_STOP_SECONDS,
                    sync_stop_epoch=started + SYNC_STOP_SECONDS,
                    guard_destroy_epoch=started + GUARD_DESTROY_SECONDS,
                    remote_stop_epoch=started + REMOTE_STOP_SECONDS,
                    hard_cap_epoch=started + HARD_CAP_SECONDS,
                    max_hourly_usd=MAX_HOURLY_USD, cost_cap_usd=COST_CAP_USD, timeline=TIMELINE,
                    disk_gb=DISK_GB, image=IMAGE)
    save(ROOT / "create-intent.json", manifest)
    result = api("PUT", f"/api/v0/asks/{int(offer['id'])}/", create_body())
    if not result.get("success") or not result.get("new_contract"):
        raise RuntimeError("Creation failed: " + str(result)[:500])
    manifest["instance_id"] = int(result["new_contract"])
    save(MANIFEST, manifest)
    with (ROOT / "guard.log").open("a") as output:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "guard"],
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    save(ROOT / "guard-receipt.json", {"pid": process.pid,
                                       "deadline_epoch": manifest["guard_destroy_epoch"]})
    log("created", **manifest, guard_pid=process.pid)
    return manifest


# ---- kit freeze and checks ---------------------------------------------------------

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze(source_root: Path):
    sys.path.insert(0, str(source_root))
    from infra.history_artifacts import pack_source
    if sha256(ROOT / "init.pt") != INIT_SHA256:
        raise RuntimeError("init.pt is not main-w4 latest.pt")
    identity = pack_source(ROOT / "source.tar.gz", source_root)
    if identity["dirty"]:
        raise RuntimeError("source tree is dirty; commit before freezing")
    branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=source_root,
                            capture_output=True, text=True, check=True).stdout.strip()
    files = {name: sha256(ROOT / name) for name in STATIC_FILES}
    (ROOT / "kit-files.sha256").write_text("".join(f"{h}  {n}\n" for n, h in files.items()))
    manifest = dict(frozen_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    source={k: v for k, v in identity.items() if k != "files"},
                    source_branch=branch, source_root=str(source_root), init_sha256=INIT_SHA256,
                    source_files=len(identity["files"]), kit_files=files,
                    lifecycle_sha256=sha256(Path(__file__).resolve()),
                    remote_guard_template_sha256=sha256(ROOT / "remote_guard.template.py"),
                    machine_ids=list(MACHINE_IDS), label=LABEL, image=IMAGE,
                    timelines=TIMELINES, max_hourly_usd=MAX_HOURLY_USD,
                    min_reliability=MIN_RELIABILITY,
                    max_inet_up_usd_per_gb=MAX_INET_UP_USD_PER_GB)
    save(ROOT / "kit-manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def refreeze():
    """Re-hash the kit files after a kit-only edit (e.g. a new machine) without
    re-packing source.tar.gz: the packed source identity is kept as frozen."""
    old = json.loads((ROOT / "kit-manifest.json").read_text())
    if sha256(ROOT / "init.pt") != INIT_SHA256:
        raise RuntimeError("init.pt is not main-w4 latest.pt")
    if sha256(ROOT / "source.tar.gz") != old["source"]["archive_sha256"]:
        raise RuntimeError("source.tar.gz differs from the frozen archive; use freeze")
    files = {name: sha256(ROOT / name) for name in STATIC_FILES}
    (ROOT / "kit-files.sha256").write_text("".join(f"{h}  {n}\n" for n, h in files.items()))
    history = list(old.get("refreezes", []))
    history.append(dict(at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        previous_machine_ids=old.get("machine_ids"),
                        machine_ids=list(MACHINE_IDS),
                        previous_lifecycle_sha256=old.get("lifecycle_sha256"),
                        previous_kit_files={n: d for n, d in old["kit_files"].items()
                                            if files.get(n) != d}))
    manifest = dict(old, kit_files=files, lifecycle_sha256=sha256(Path(__file__).resolve()),
                    remote_guard_template_sha256=sha256(ROOT / "remote_guard.template.py"),
                    machine_ids=list(MACHINE_IDS), label=LABEL, image=IMAGE, timelines=TIMELINES,
                    max_hourly_usd=MAX_HOURLY_USD, min_reliability=MIN_RELIABILITY,
                    refreezes=history)
    save(ROOT / "kit-manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


def verify_kit():
    manifest = json.loads((ROOT / "kit-manifest.json").read_text())
    for name, digest in manifest["kit_files"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"{name} changed after freeze")
    if sha256(Path(__file__).resolve()) != manifest["lifecycle_sha256"]:
        raise RuntimeError("lifecycle.py changed after freeze")
    if sha256(ROOT / "remote_guard.template.py") != manifest["remote_guard_template_sha256"]:
        raise RuntimeError("remote guard template changed after freeze")
    if manifest["source"]["archive_sha256"] != manifest["kit_files"]["source.tar.gz"]:
        raise RuntimeError("source archive digest mismatch")
    if manifest["kit_files"]["init.pt"] != INIT_SHA256:
        raise RuntimeError("init.pt is not main-w4 latest.pt")
    if [int(m) for m in manifest["machine_ids"]] != list(MACHINE_IDS):
        raise RuntimeError("kit manifest machine list differs from MACHINE_IDS")
    return manifest


def check_secrets():
    for path in (API_KEY, SSH_KEY):
        if not path.exists():
            raise RuntimeError(f"missing secret file {path}")
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise RuntimeError(f"{path.name} permissions {oct(mode)}; require 0600")
    return {"api_key": "present, 0600", "ssh_key": "present, 0600"}


def render_remote(manifest, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    text = (ROOT / "remote_guard.template.py").read_text()
    text = text.replace("__INSTANCE__", str(int(manifest["instance_id"])))
    text = text.replace("__DEADLINE__", repr(float(manifest["remote_stop_epoch"])))
    (directory / "remote_guard.py").write_text(text)
    py_compile.compile(str(directory / "remote_guard.py"), doraise=True)
    (directory / "deadlines.env").write_text(
        f"TRAIN_DEADLINE_EPOCH={manifest['sweep_stop_epoch']:.3f}\n"
        f"REMOTE_STOP_EPOCH={manifest['remote_stop_epoch']:.3f}\n"
        f"KIT_MACHINE_ID={int(manifest['machine_id'])}\n"
        f"KIT_OFFER_ID={int(manifest.get('offer_id', 0))}\n")
    return {name: sha256(directory / name) for name in ("remote_guard.py", "deadlines.env")}


# ---- ssh helpers -------------------------------------------------------------------

def ssh_target(row):
    ports = (row.get("ports") or {}).get("22/tcp") or []
    if row.get("public_ipaddr") and ports:
        return row["public_ipaddr"].strip(), int(ports[0]["HostPort"])
    return row["ssh_host"], int(row["ssh_port"])


def ssh_base(port):
    return ["-i", str(SSH_KEY), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", f"UserKnownHostsFile={ROOT / 'known_hosts'}", "-o", "ServerAliveInterval=30"]


def ssh(target, command, timeout=120):
    host, port = target
    return subprocess.run(["ssh", *ssh_base(port), "-p", str(port), f"root@{host}", command],
                          capture_output=True, text=True, timeout=timeout)


def rsync(target):
    host, port = target
    shell = "ssh " + " ".join(ssh_base(port)) + f" -p {port}"
    destination = ROOT / "download" / "results"
    destination.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["rsync", "-az", "--timeout=60", "-e", shell,
                             f"root@{host}:/workspace/results/", str(destination) + "/"],
                            capture_output=True, text=True, timeout=900)
    log("sync", returncode=result.returncode, stderr=result.stderr[-500:])
    return result.returncode == 0


def verify_download():
    results = ROOT / "download" / "results"
    manifest_path = results / "results-manifest.json"
    if not manifest_path.exists():
        report = dict(verified=False, reason="no results-manifest.json (sweep not DONE)")
    else:
        manifest = json.loads(manifest_path.read_text())
        bad = [name for name, meta in manifest.items()
               if not (results / name).exists() or sha256(results / name) != meta["sha256"]
               or (results / name).stat().st_size != meta["bytes"]]
        report = dict(verified=not bad, files=len(manifest), mismatched=bad[:50],
                      manifest_sha256=sha256(manifest_path))
    save(ROOT / "artifact-verification.json", report)
    log("verify_download", **{k: v for k, v in report.items() if k != "mismatched"},
        mismatched=len(report.get("mismatched", [])))
    return report


# ---- commands ----------------------------------------------------------------------

def dry_run(offline: bool):
    kit = verify_kit()
    secrets = check_secrets()
    for script in ("remote_setup.sh", "remote_start.sh"):
        subprocess.run(["bash", "-n", str(ROOT / script)], check=True)
    now = time.time()
    fake = dict(instance_id=99999999, machine_id=MACHINE_IDS[0], offer_id=0,
                sweep_stop_epoch=now + SWEEP_STOP_SECONDS,
                remote_stop_epoch=now + REMOTE_STOP_SECONDS)
    rendered = render_remote(fake, ROOT / "dry-run")
    report = dict(machine_ids=list(MACHINE_IDS), kit_source=kit["source"],
                  source_branch=kit.get("source_branch"),
                  kit_files=kit["kit_files"], secrets=secrets,
                  rendered=rendered, manifest_exists=MANIFEST.exists(),
                  upload=[*STATIC_FILES, "kit-files.sha256", "remote_guard.py", "deadlines.env"],
                  timeline=TIMELINE, timeline_seconds=TIMELINES[TIMELINE],
                  worst_case_usd_at_ceiling=round(worst_case_usd(MAX_HOURLY_USD,
                                                                 MAX_INET_UP_USD_PER_GB), 3),
                  create_body=create_body())
    if MANIFEST.exists():
        raise RuntimeError("instance.json exists: a previous launch is not closed")
    if not offline:
        rows = instances()
        report["provider_instances"] = len(rows)
        report["matching_label"] = [r["id"] for r in rows if r.get("label") == LABEL]
        report["balance"] = balance()
        machine, offers, per_machine = search()
        report["search"] = per_machine
        report["would_rent_machine"] = machine
        report["offers"] = [{k: r.get(k) for k in (
            "id", "machine_id", "host_id", "gpu_name", "gpu_ram", "dph_total", "cpu_name",
            "cpu_cores_effective", "cpu_ghz", "cpu_ram", "disk_space", "inet_up", "inet_down",
            "reliability2", "hosting_type", "verification", "cuda_max_good", "driver_version",
            "pci_gen", "gpu_lanes", "geolocation", "inet_up_cost", "storage_cost")}
            for r in offers]
        report["would_create_offer"] = offers[0]["id"] if offers else None
        if offers:
            report["worst_case_usd_at_offer"] = round(worst_case_usd(
                float(offers[0]["dph_total"]), float(offers[0].get("inet_up_cost") or 0)), 3)
        report["balance_check"] = balance_check(report["balance"])
        if report["matching_label"]:
            raise RuntimeError("an instance with this kit's label already exists")
    report["stopped_before"] = "PUT /api/v0/asks/{offer}/ (create) -- dry run, nothing rented"
    save(ROOT / "dry-run.json", report)
    print(json.dumps(report, indent=2))


def balance_check(value: dict) -> dict:
    """Credit must cover the worst case at the ceiling plus a margin."""
    needed = worst_case_usd(MAX_HOURLY_USD, MAX_INET_UP_USD_PER_GB) + BALANCE_MARGIN_USD
    credit = value.get("credit")
    return dict(credit=credit, needed=round(needed, 3),
                ok=credit is not None and float(credit) >= needed)


def launch(yes_spend: bool):
    if not yes_spend:
        raise SystemExit("launch rents a GPU; pass --yes-spend after the user approves the budget")
    kit = verify_kit()
    check_secrets()
    caffeinate = None
    if sys.platform == "darwin":
        caffeinate = subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
    before = balance()
    check = balance_check(before)
    if not check["ok"]:
        raise RuntimeError(f"credit {check['credit']} below worst case + margin {check['needed']}")
    machine, offers, per_machine = search()
    if not offers:
        raise RuntimeError(f"no qualifying offer on machines {MACHINE_IDS} at <= ${MAX_HOURLY_USD}/h")
    log("launch", source=kit["source"], balance_before=before, machine_id=machine,
        search=per_machine, offer=offers[0]["id"], dph_total=offers[0]["dph_total"])
    manifest = create(offers[0])
    try:
        target = None
        while time.time() < manifest["created_at_epoch"] + RUNNING_WAIT_SECONDS:
            row = owned(manifest)
            if row is None:
                raise RuntimeError("instance disappeared before running")
            save(ROOT / "last-instance.json", public(row))
            if row.get("actual_status") == "running" and (row.get("ports") or row.get("ssh_host")):
                target = ssh_target(row)
                if ssh(target, "true", timeout=40).returncode == 0:
                    break
            time.sleep(10)
        else:
            raise RuntimeError("instance not reachable by ssh within the wait window")
        log("reachable", host_port=list(target), seconds=time.time() - manifest["created_at_epoch"])
        probe = ssh(target, "P=/venv/main/bin/python; [ -x $P ] || P=python3; "
                            "$P -c \"e=open('/proc/1/environ','rb').read().split(b'\\0');"
                            "print('present' if any(x.startswith(b'CONTAINER_API_KEY=') for x in e)"
                            " and any(x.startswith(b'VAST_CONTAINERLABEL=') for x in e) else 'absent')\"")
        if probe.stdout.strip() != "present":
            raise RuntimeError("container API key or label absent; remote self-stop impossible")
        generated = render_remote(manifest, ROOT / "generated")
        log("rendered", **generated)
        if ssh(target, "mkdir -p /workspace/kit /workspace/results").returncode != 0:
            raise RuntimeError("remote mkdir failed")
        host, port = target
        files = [str(ROOT / n) for n in (*STATIC_FILES, "kit-files.sha256")] + \
                [str(ROOT / "generated" / n) for n in ("remote_guard.py", "deadlines.env")]
        upload = subprocess.run(["scp", *ssh_base(port), "-P", str(port), *files,
                                 f"root@{host}:/workspace/kit/"],
                                capture_output=True, text=True, timeout=900)
        if upload.returncode != 0:
            raise RuntimeError("upload failed: " + upload.stderr[-300:])
        # Bare background jobs die with the SSH session on this image; supervisord
        # children survive it.
        started = ssh(target, "cat > /etc/supervisor/conf.d/gzrun.conf <<'EOF'\n"
                              "[program:gzrun]\ncommand=/bin/bash /workspace/kit/remote_start.sh\n"
                              "directory=/workspace/kit\nautostart=false\nautorestart=false\n"
                              "startsecs=0\nstopwaitsecs=300\nstopasgroup=true\nkillasgroup=true\n"
                              "stdout_logfile=/workspace/results/start.log\nredirect_stderr=true\n"
                              "EOF\nsupervisorctl reread; supervisorctl update; "
                              "supervisorctl start gzrun; sleep 3; supervisorctl status gzrun")
        log("remote_start_output", stdout=started.stdout[-800:], stderr=started.stderr[-300:],
            returncode=started.returncode)
        if "RUNNING" not in started.stdout and "EXITED" not in started.stdout:
            raise RuntimeError("remote start failed: " + started.stdout[-300:])
        log("remote_started", seconds=time.time() - manifest["created_at_epoch"])
        done = False
        while time.time() < manifest["sync_stop_epoch"]:
            time.sleep(min(SYNC_INTERVAL_SECONDS, max(1, manifest["sync_stop_epoch"] - time.time())))
            try:
                rsync(target)
                done = ssh(target, "test -f /workspace/results/DONE && echo yes || echo no",
                           timeout=40).stdout.strip() == "yes"
            except Exception as error:
                log("sync_error", error=str(error)[:300])
            if done:
                break
        log("workload_finished" if done else "sync_deadline", done=done)
        if not done:
            # Ask the pod to stop the segment and still write summary/manifest.
            # '[t]rim_check' keeps pkill from matching this very shell's command line.
            try:
                ssh(target, "pkill -TERM -f '[t]rim_check\\.py'; for i in $(seq 60); do "
                            "test -f /workspace/results/DONE && break; sleep 5; done; echo waited",
                    timeout=330)
            except Exception as error:
                log("stop_error", error=str(error)[:300])
        for _ in range(2):
            try:
                rsync(target)
            except Exception as error:
                log("sync_error", error=str(error)[:300])
        if verify_download().get("verified"):
            try:   # the pod wrote trim-summary.*; this re-derives it from the download
                subprocess.run([sys.executable, str(ROOT / "summarize.py"),
                                str(ROOT / "download" / "results"), "--out", str(ROOT)],
                               capture_output=True, text=True, timeout=120)
            except Exception as error:
                log("summarize_error", error=str(error)[:300])
    finally:
        confirmed = False
        for attempt in range(10):
            try:
                confirmed = destroy(manifest, "check finished" if attempt == 0 else "retry")
                break
            except Exception as error:
                log("destroy_retry", attempt=attempt, error=str(error)[:300])
                time.sleep(10)
        ended = time.time()
        rate = manifest["offer_dph_total"]
        try:
            rate = float(json.loads((ROOT / "last-instance.json").read_text()).get("dph_total") or rate)
        except Exception:
            pass
        elapsed = ended - manifest["created_at_epoch"]
        summary = dict(instance_id=manifest["instance_id"], machine_id=manifest["machine_id"],
                       offer_id=manifest["offer_id"], created_epoch=manifest["created_at_epoch"],
                       absence_confirmed=confirmed, ended_epoch=ended, elapsed_seconds=elapsed,
                       hourly_usd=rate, estimated_compute_disk_usd=elapsed / 3600 * rate,
                       within_hard_cap=elapsed <= HARD_CAP_SECONDS, balance_before=before,
                       balance_after=balance(), status_after=status_json(),
                       cost_scope="elapsed x quoted hourly; traffic charges not included")
        save(ROOT / "teardown-summary.json", summary)
        if confirmed:
            MANIFEST.rename(ROOT / f"instance-{manifest['instance_id']}.closed.json")
        log("teardown", **{k: v for k, v in summary.items() if k != "status_after"})
        if caffeinate is not None:
            caffeinate.terminate()


def status_json():
    if MANIFEST.exists():
        row = owned(json.loads(MANIFEST.read_text()))
        return public(row) if row else {"owned": None, "instances": len(instances())}
    return {"instances": len(instances()), "owned": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("action", choices=["freeze", "refreeze", "dry-run", "launch", "status",
                                           "sync", "destroy", "guard"])
    parser.add_argument("--offline", action="store_true", help="dry-run without read-only API calls")
    parser.add_argument("--yes-spend", action="store_true")
    parser.add_argument("--timeline", choices=sorted(TIMELINES), default="25m")
    parser.add_argument("--source-root", type=Path, default=REPO,
                        help="freeze: the committed checkout to pack")
    args = parser.parse_args()
    use_timeline(args.timeline)
    if args.action == "freeze":
        freeze(args.source_root.resolve())
    elif args.action == "refreeze":
        refreeze()
    elif args.action == "dry-run":
        dry_run(args.offline)
    elif args.action == "launch":
        launch(args.yes_spend)
    elif args.action == "guard":
        guard()
    elif args.action == "destroy":
        destroy(json.loads(MANIFEST.read_text()), "manual")
    elif args.action == "sync":
        row = owned(json.loads(MANIFEST.read_text()))
        if row is None:
            raise SystemExit("no owned instance")
        rsync(ssh_target(row))
        verify_download()
    else:
        print(json.dumps(status_json(), indent=2))


if __name__ == "__main__":
    main()
